"""FastH3 as a Reactor model: a queue of prompt-driven video-and-audio clips.

FastH3 is MiniMax-H3 distilled to four transformer forwards, and on a few
Blackwell GPUs it builds video about as fast as the video plays. This model
puts two queues in front of that: clients `enqueue` generation requests — a
prompt plus opaque metadata, each answered with a UUID — into the **generation
queue**, which builds consume front-first on their own; each finished clip
crosses into the **playout queue** (`clip_generated`), where playback is a
separate, explicit step. `play` streams one built clip on `main_video` and
`main_audio`; when it ends (or `stop` cuts it) the stream flushes to black and
holds until the next `play`. Nothing plays on its own unless autoplay is on,
and `enqueue`'s `position`, `move`, and `pop` give a client full control of
both queues' order.

The unit of work is a whole clip, not a frame, which is why this subclasses
``ReactorModel`` and owns its own ``run()`` loop rather than using
``ReactorPipeline``. Command handlers then run on their own coroutines
concurrent with ``run()``, so `enqueue` and `stop` answer immediately even
while a clip is being built or played.

Layout:
  * ``fasth3_types.py``         — everything a client sees (tracks, `ClipInfo`, messages).
  * ``fasth3_queue.py``         — the generation and playout queues and their entries.
  * ``fasth3_backend.py``       — the FastVideo engine and its worker thread.
  * ``fasth3_assets.py``        — config parsing and weights validation.
  * ``fasth3_clip_plan.py``     — clip geometry (lengths, frame counts, canvases).
  * ``fasth3_session_rules.py`` — which commands each state accepts.
  * ``fasth3.yaml``             — the generation recipe, queue size, and weight layout.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from reactor_runtime import (
    ClientInfo,
    InputField,
    ReactorModel,
    connected,
    event,
    get_weights_path,
    session_ended,
    session_started,
)
from reactor_runtime.log import get_logger

import fasth3_clip_plan as clip_plan
import fasth3_session_rules as session_rules
from fasth3_assets import load_config, require_weights, resolve_model_path
from fasth3_backend import OUTPUT_SAMPLE_RATE, ClipJob, FastH3Backend
from fasth3_queue import ClipEntry, ClipQueue, new_entry
from fasth3_types import (
    MAX_METADATA_CHARS,
    MAX_PROMPT_CHARS,
    AutoplayAccepted,
    CanvasAccepted,
    ClipFailed,
    ClipFinished,
    ClipGenerated,
    ClipLengthAccepted,
    ClipMoved,
    ClipPopped,
    ClipQueued,
    ClipStarted,
    ClipStopped,
    CommandError,
    ContinuityAccepted,
    FastH3Output,
    PromptAccepted,
    QueueUpdate,
    SeedAccepted,
    SessionReset,
    StateUpdate,
)

logger = get_logger(__name__)

FRAME_RATE = clip_plan.FPS

# The clip-length range, rendered once so the command text and the schema's own
# bounds can never disagree.
_CLIP_RANGE = f"{clip_plan.MIN_SECONDS_PUBLISHED:g} and {clip_plan.MAX_SECONDS_PUBLISHED:g}"

# Frames per emitted slice. The runtime recorder's feed queue cannot absorb
# one-second bursts, and the emitter is a metronome either way, so smaller
# slices cost nothing.
EMIT_FRAMES = 3

# How often the idle loop re-checks for a play request and a finished build.
# Runs on the event loop, so this is a scheduling granularity, not a busy-wait.
POLL_SECONDS = 0.05


class FastH3(ReactorModel):
    """Queue prompt-driven clip generations and play them back one at a time."""

    # Pinned: `_emit_clip` is a strict 24 fps metronome and every emit omits
    # `compute_time`, which is exactly the "unmeasured" path this rate tags.
    # Measuring instead re-estimates the rate from observed timing, whose wobble
    # both drops chunks while converging and drifts video against the
    # sample-clocked audio.
    fps = FRAME_RATE
    # Two seconds of transport-side tolerance at 24 fps, so a hiccup dents the
    # buffer instead of dropping frames.
    buffer_size = 48

    def __init__(self) -> None:
        """Create the model shell; everything session-scoped arrives in load()."""
        super().__init__()
        # The build in flight: its entry, its job handle, and when it was
        # submitted (monotonic), so readiness latency is a measured number.
        self._build: tuple[ClipEntry, ClipJob, float] | None = None

    # ------------------------------------------------------------------ load

    def load(self, config_path: Path | None) -> None:
        """Parse the config, validate the weights, and build the warm engine.

        Runs once at startup, before any session. The runtime marks the pod
        ready only when this returns, so the backend's warm-up means a deployed
        pod never builds a cold clip.

        Args:
            config_path: Path to ``fasth3.yaml``; its ``inference`` block is the
                generation recipe and the queue size, and its ``runtime`` block
                holds the weight layout and the engine shape.
        """
        self.config = load_config(config_path)
        weights = get_weights_path()
        model_path = resolve_model_path(self.config, weights)
        require_weights(weights, model_path)

        self.backend = FastH3Backend(self.config, model_path)
        # Session-scoped state exists before the first session, so a command
        # racing ahead of `@session_started` reads defaults, never garbage.
        self._reset_session_state()
        self.backend.load()
        logger.info("fast-h3 loaded", queue_capacity=self.config.queue_size)

    # -------------------------------------------------------- session state

    def _reset_session_state(self) -> None:
        """Return every session-scoped field to its default.

        Called once at ``load()`` and at every ``@session_started``, which is
        what keeps one session from ever observing another's queue or
        conditions. A build still in flight for the old session is cancelled;
        its result, if it completes anyway, is discarded by ``_pump_builds``
        because its entry no longer lives in the queue.
        """
        if self._build is not None:
            _entry, job, _submitted = self._build
            job.cancelled = True
            self._build = None

        # Whether this session runs the continuous take (True) or the hard-cut
        # clip queue (False). Seeded from the config default; `set_continuity`
        # flips it at runtime (only while idle), so a client picks the mode
        # per-session without a redeploy. Everything downstream reads this, not
        # the config, so the switch is a single source of truth.
        self._continuity: bool = self.config.continuity

        # Conditions newly enqueued clips snapshot. Continuity holds one length
        # (the config's continuity_clip_seconds); the queue's default governs
        # hard-cut. `set_clip_seconds` moves it only in the queue's mode.
        self._clip_frames: int = (
            self.config.continuity_clip_frames
            if self._continuity
            else self.config.clip_frames
        )
        self._seed: int = self.config.seed
        self._aspect: str = self.config.aspect
        # Off by default: playback waits for an explicit `play`.
        self._autoplay: bool = False

        # Continuity mode: the held prompt the continuous take follows, whether
        # a clip is on the wire, and the two edges the run loop watches — a
        # prompt change (re-anchor) and a stop (drain to black). Unused and
        # never read in the queue's hard-cut mode.
        self._prompt: str = ""
        self._prompt_metadata: str = ""
        self._prompt_epoch: int = 0
        self._channel_running: bool = False
        self._stop_channel: bool = False

        # The two queues, and the playout lifecycle around them: builds
        # consume `_generation` from the front and finished clips join the
        # back of `_playout`. `_play_request` is a clip taken off the playout
        # queue and armed for the run loop; `_playing` is the clip whose
        # frames are on the wire; `_stop_playout` asks the emitter to cut it.
        self._generation = ClipQueue(self.config.generation_queue_size)
        self._playout = ClipQueue(self.config.queue_size)
        self._play_request: ClipEntry | None = None
        self._playing: ClipEntry | None = None
        self._stop_playout: bool = False

        # Progress, mirrored so a `state_update` is a complete snapshot.
        self._clips_played: int = 0
        self._frames_sent: int = 0
        self._seconds_sent: float = 0.0

    def _canvas(self) -> tuple[int, int]:
        """The `(height, width)` this session generates at, at the configured tier."""
        return clip_plan.canvas_for_choice(self._aspect, self.config.canvas_short_edge)

    def _current_clip(self) -> ClipEntry | None:
        """The clip on (or headed for) the output tracks, if any."""
        return self._playing or self._play_request

    def _snapshot(self) -> StateUpdate:
        """Everything a client can observe, in one message.

        The single source of the snapshot: `state_update` broadcasts it, a
        joining client is greeted with it, and `get_state` answers with it.
        Built once here so those three can never disagree.
        """
        height, width = self._canvas()
        if self._continuity:
            # The queue fields are neutral: continuity has no queue, only the
            # single take driven by `set_prompt`. `playing` is the channel on
            # the wire; the prompt is what it is following.
            playing = self._channel_running
            return StateUpdate(
                clip_seconds=round(clip_plan.seconds_for_frames(self._clip_frames), 3),
                clip_seconds_min=clip_plan.MIN_SECONDS_PUBLISHED,
                clip_seconds_max=clip_plan.MAX_SECONDS_PUBLISHED,
                seed=self._seed,
                autoplay=False,
                aspect=self._aspect,
                width=width,
                height=height,
                continuity=True,
                prompt=self._prompt if playing else "",
                playing=playing,
                playing_clip_id=None,
                generation_queued=0,
                generation_capacity=0,
                playout_queued=0,
                playout_capacity=0,
                clips_played=self._clips_played,
                seconds_sent=round(self._seconds_sent, 2),
                valid_commands=session_rules.valid_commands(
                    playing=playing,
                    generation_queued=0,
                    generation_capacity=0,
                    playout_queued=0,
                    continuity=True,
                    prompt_set=bool(self._prompt),
                ),
            )
        current = self._current_clip()
        return StateUpdate(
            clip_seconds=round(clip_plan.seconds_for_frames(self._clip_frames), 3),
            clip_seconds_min=clip_plan.MIN_SECONDS_PUBLISHED,
            clip_seconds_max=clip_plan.MAX_SECONDS_PUBLISHED,
            seed=self._seed,
            autoplay=self._autoplay,
            aspect=self._aspect,
            width=width,
            height=height,
            continuity=False,
            prompt="",
            playing=current is not None,
            playing_clip_id=current.clip_id if current is not None else None,
            generation_queued=len(self._generation),
            generation_capacity=self._generation.capacity,
            playout_queued=len(self._playout),
            playout_capacity=self._playout.capacity,
            clips_played=self._clips_played,
            seconds_sent=round(self._seconds_sent, 2),
            valid_commands=session_rules.valid_commands(
                playing=current is not None,
                generation_queued=len(self._generation),
                generation_capacity=self._generation.capacity,
                playout_queued=len(self._playout),
            ),
        )

    async def _send_state_update(self) -> None:
        """Broadcast the snapshot to every connected client."""
        await self.send(self._snapshot())

    async def _send_queue_update(self) -> None:
        """Broadcast the queue's contents to every connected client."""
        await self.send(
            QueueUpdate(
                generation=self._generation.snapshot(),
                playout=self._playout.snapshot(),
            )
        )

    async def _refuse(self, command: str, reason: str) -> None:
        """Reject a command: tell every client, and leave its reply bodyless.

        A handler returns only the message its annotation names, and reports
        a failure by broadcasting `command_error` and returning without a
        value. The runtime answers that with a correlated bodyless
        acknowledgement, so an awaiting client resolves rather than hanging —
        and unlike a raised runtime ``CommandError``, whose failure frame is
        withheld from v0 clients, the broadcast reaches every SDK generation.

        Logged as well, so refusals are visible server-side and not only in the
        client's message.
        """
        logger.info("command refused", command=command, reason=reason)
        await self.send(CommandError(command=command, reason=reason))

    async def _refuse_queue_only(self, command: str) -> bool:
        """Refuse a queue-only command while the model runs in continuity mode.

        The two modes are disjoint surfaces; `valid_commands` already hides
        these, and this is the guard for a client that sends one anyway.
        """
        if self._continuity:
            await self._refuse(
                command,
                f"`{command}` drives the clip queue; this stream runs in "
                "continuity mode — use `set_prompt` to drive it.",
            )
            return True
        return False

    # ------------------------------------------------------------ lifecycle

    @session_started
    async def on_session_started(self) -> None:
        """Clear the queue and every condition so a new session inherits nothing."""
        self._reset_session_state()

    @session_ended
    async def on_session_ended(self) -> None:
        """Drop the session's work; the only hook guaranteed to fire on every path."""
        self._stop_playout = True
        self._stop_channel = True
        self._play_request = None
        if self._build is not None:
            _entry, job, _submitted = self._build
            job.cancelled = True
        self._generation.clear()
        self._playout.clear()

    @connected
    async def on_connect(self, client: ClientInfo) -> None:
        """Greet the joining client with the full state and the queue.

        Addressed rather than broadcast: the clients already watching have
        this, and a late joiner needs it without replaying every command.
        """
        await client.send(self._snapshot())
        await client.send(
            QueueUpdate(
                generation=self._generation.snapshot(),
                playout=self._playout.snapshot(),
            )
        )

    # ------------------------------------------------------------- commands

    @event(
        name="enqueue",
        description=(
            "Queue one clip generation. The clip enters the generation queue "
            "(at `position`, or the back), builds when its turn comes, and "
            "then joins the back of the playout queue, announced by "
            "`clip_generated`. The prompt is what the clip will show; the "
            "metadata is an opaque string echoed back on every message that "
            "references the clip, for frontends to carry their own tracking "
            "data. The clip's canvas is the session's; its length is the "
            "`seconds` passed here (snapped to what the model can produce) or "
            "the session default, and its seed is the one passed here or the "
            "session's advancing default. Builds run through the queue in "
            "order; watch `queue_update` for the clip turning ready. Replies "
            "`clip_queued` with the clip's UUID and emits `queue_update` and "
            "`state_update`, or `command_error` when the queue is full or the "
            "prompt is empty."
        ),
    )
    async def enqueue(
        self,
        prompt: str = InputField(
            default="",
            max_length=MAX_PROMPT_CHARS,
            moderate=True,
            description=(
                "What the clip should show, up to 800 characters. Fixed once "
                "enqueued; a different scene is a new `enqueue`."
            ),
        ),
        metadata: str = InputField(
            default="",
            max_length=MAX_METADATA_CHARS,
            moderate=True,
            description=(
                "Free-form string stored with the clip and echoed back on every "
                "message that references it. The model never reads it; use it "
                "to correlate clips with your own records — who asked for it, "
                "which group it belongs to, display text."
            ),
        ),
        seed: int | None = InputField(
            default=None,
            ge=0,
            description=(
                "Seed for this clip. Omitted or null, the session's default is "
                "used and advances by one; passing a seed leaves the default "
                "untouched, so explicit and automatic seeding do not interfere."
            ),
        ),
        seconds: float | None = InputField(
            default=None,
            ge=clip_plan.MIN_SECONDS_PUBLISHED,
            le=clip_plan.MAX_SECONDS_PUBLISHED,
            description=(
                f"Length of this clip in seconds, between {_CLIP_RANGE}, "
                "snapped to the nearest length the model can produce; the "
                "clip's structure reports the effective value. Omitted or "
                "null, the session default applies. A length the deployment "
                "has not built before pays a one-off compile cost on its "
                "first build."
            ),
        ),
        position: int | None = InputField(
            default=None,
            ge=0,
            description=(
                "Where the clip enters the generation queue: 0 is the front "
                "(the next build), larger values count back from there, and "
                "anything past the end — or omitted — appends. The clip "
                "already building is unaffected either way. `queue_update` "
                "reports the resulting order."
            ),
        ),
    ) -> ClipQueued:
        """Append one generation request to the queue."""
        if await self._refuse_queue_only("enqueue"):
            return None
        prompt = prompt.strip()
        if not prompt:
            await self._refuse("enqueue", "The prompt is empty; a clip needs one.")
            return None
        if self._generation.full:
            await self._refuse(
                "enqueue",
                f"The generation queue is full ({self._generation.capacity} "
                "clips); `pop` one or wait for a build to finish.",
            )
            return None
        if not isinstance(seed, int):
            # None on the wire; the InputField sentinel when called directly.
            seed = self._seed
            self._seed += 1
        frames = (
            clip_plan.frames_for_seconds(float(seconds))
            if isinstance(seconds, (int, float))
            else self._clip_frames
        )
        entry = new_entry(prompt=prompt, metadata=metadata, frames=frames, seed=seed)
        self._generation.add(
            entry,
            # None on the wire; the InputField sentinel when called directly.
            position if isinstance(position, int) else None,
        )
        await self._send_queue_update()
        await self._send_state_update()
        return ClipQueued(clip=entry.snapshot())

    @event(
        name="play",
        description=(
            "Play one clip from the playout queue. Blank `clip_id` plays the "
            "front clip; a UUID plays that specific one. Playing consumes "
            "the entry: it leaves the queue, `clip_started` marks its first "
            "frames, and when it ends the stream holds on black until the "
            "next `play`. Emits `queue_update` and `state_update`, or "
            "`command_error` when a clip is already playing, the id is "
            "unknown, or the clip is still generating."
        ),
    )
    async def play(
        self,
        clip_id: str = InputField(
            default="",
            description=(
                "UUID of the clip to play, from `clip_generated` or "
                "`queue_update`. Blank plays the playout queue's front clip."
            ),
        ),
    ) -> None:
        """Take one ready clip off the queue and hand it to the playout loop."""
        if await self._refuse_queue_only("play"):
            return
        if self._current_clip() is not None:
            await self._refuse("play", "A clip is already playing; send `stop` first.")
            return
        if clip_id:
            entry = self._playout.get(clip_id)
            if entry is None:
                if self._generation.get(clip_id) is not None:
                    await self._refuse(
                        "play",
                        "That clip is still generating; `clip_generated` will "
                        "announce it entering the playout queue.",
                    )
                else:
                    await self._refuse("play", f"No queued clip has id {clip_id!r}.")
                return
        else:
            entry = self._playout.head()
            if entry is None:
                await self._refuse(
                    "play",
                    "The playout queue is empty; `enqueue` a clip and wait "
                    "for `clip_generated`.",
                )
                return
        self._playout.remove(entry)
        self._play_request = entry
        await self._send_queue_update()
        await self._send_state_update()

    @event(
        name="pop",
        description=(
            "Remove one clip by its UUID from whichever queue holds it, "
            "freeing its slot. Works on generating and built clips alike; a "
            "build already running for it is discarded when it completes. "
            "The clip that is playing is in neither queue — `stop` is the "
            "command that cuts it. Emits `clip_popped`, `queue_update` and "
            "`state_update`, or `command_error` when no queued clip has that "
            "id."
        ),
    )
    async def pop(
        self,
        clip_id: str = InputField(
            default="",
            description="UUID of the queued clip to remove, from `clip_queued` or `queue_update`.",
        ),
    ) -> ClipPopped:
        """Take one clip out of the queue and free its slot."""
        if await self._refuse_queue_only("pop"):
            return None
        entry = None
        if clip_id:
            entry = self._generation.get(clip_id) or self._playout.get(clip_id)
        if entry is None:
            reason = (
                f"No queued clip has id {clip_id!r}."
                if clip_id
                else "Pass the `clip_id` of the queued clip to remove."
            )
            await self._refuse("pop", reason)
            return None
        self._generation.remove(entry)
        self._playout.remove(entry)
        if self._build is not None and self._build[0] is entry:
            _entry, job, _submitted = self._build
            job.cancelled = True
        await self._send_queue_update()
        await self._send_state_update()
        return ClipPopped(clip=entry.snapshot())

    @event(
        name="move",
        description=(
            "Reposition one clip within the queue that holds it — the "
            "generation queue for a clip still to build, the playout queue "
            "for a built one; clips never move between queues except by "
            "building. `position` 0 is the front: the next build, or what "
            "bare `play` and autoplay take next. Values past the end mean "
            "the back. Replies `clip_moved` with the queue and the resulting "
            "position, and emits `queue_update`; `command_error` when no "
            "queued clip has that id."
        ),
    )
    async def move(
        self,
        clip_id: str = InputField(
            default="",
            description="UUID of the queued clip to move, from any clip-referencing message.",
        ),
        position: int = InputField(
            default=0,
            ge=0,
            description="Target position in the clip's queue, 0 = front; clamped to the end.",
        ),
    ) -> ClipMoved:
        """Reposition a clip within its queue."""
        if await self._refuse_queue_only("move"):
            return None
        if not isinstance(position, int):
            # The InputField sentinel when called directly.
            position = 0
        entry = self._generation.get(clip_id) if clip_id else None
        queue, name = (
            (self._generation, "generation") if entry is not None else (None, "")
        )
        if entry is None and clip_id:
            entry = self._playout.get(clip_id)
            queue, name = self._playout, "playout"
        if entry is None:
            reason = (
                f"No queued clip has id {clip_id!r}."
                if clip_id
                else "Pass the `clip_id` of the queued clip to move."
            )
            await self._refuse("move", reason)
            return None
        landed = queue.move(entry, position)
        await self._send_queue_update()
        return ClipMoved(clip=entry.snapshot(), queue=name, position=landed)

    @event(
        name="set_continuity",
        description=(
            "Switch this session between its two modes at runtime, only while "
            "idle. On (the shipping default): one continuous FL2VA take driven "
            "by `set_prompt` — every clip is anchored on the previous clip's "
            "last frame and crossfaded, so one prompt yields an unbroken "
            "stream. Off: the hard-cut clip queue driven by `enqueue`/`play`, "
            "each clip an independent cut. The config sets the starting mode; "
            "this overrides it for the session with no redeploy, and `reset` "
            "keeps the chosen mode. Accepted only while the stream is idle — "
            "nothing playing or queued, no prompt held — so no clip or take "
            "straddles the switch; `stop` or `reset` first otherwise. Emits "
            "`continuity_accepted` and `state_update` (whose `valid_commands` "
            "then carries the other mode's surface), or `command_error` when "
            "the stream is not idle."
        ),
    )
    async def set_continuity(
        self,
        enabled: bool = InputField(
            default=True,
            description=(
                "True runs the continuous take (driven by `set_prompt`); false "
                "runs the hard-cut clip queue (driven by `enqueue`)."
            ),
        ),
    ) -> ContinuityAccepted:
        """Flip the session between continuity and the hard-cut queue, while idle."""
        enabled = bool(enabled)
        if enabled == self._continuity:
            # Already in this mode: a harmless idempotent ack the client can lean
            # on without first reading the state back.
            await self._send_state_update()
            return ContinuityAccepted(continuity=enabled)
        # A switch is only safe with nothing on the wire: the run loop is parked
        # in the current mode's idle branch, so flipping the flag lets it return
        # and `run()` re-dispatch to the other serve loop. Mirrors set_canvas.
        if self._continuity:
            if self._channel_running or self._prompt:
                await self._refuse(
                    "set_continuity",
                    "A prompt is driving the stream; `stop` or `reset` before "
                    "switching to the hard-cut queue.",
                )
                return None
        elif (
            self._current_clip() is not None
            or len(self._generation) > 0
            or len(self._playout) > 0
        ):
            await self._refuse(
                "set_continuity",
                "Clips are queued or playing; play the queue out or `reset` "
                "before switching to continuity.",
            )
            return None
        self._continuity = enabled
        # Adopt the new mode's default clip length and drop any continuity-only
        # holds. Idle, so these are already clear — set explicitly for safety.
        self._clip_frames = (
            self.config.continuity_clip_frames
            if enabled
            else self.config.clip_frames
        )
        self._prompt = ""
        self._prompt_metadata = ""
        await self._send_state_update()
        return ContinuityAccepted(continuity=enabled)

    @event(
        name="set_prompt",
        description=(
            "Set the prompt the continuous stream follows (continuity mode "
            "only). The first `set_prompt` starts the take; a later one "
            "re-anchors it — the next clip opens fresh on the new prompt and "
            "every clip after chains from it, so the picture changes over a "
            "couple of seconds rather than cutting. Holds until the next "
            "`set_prompt` or `stop`, with no re-prompting in between. Emits "
            "`prompt_accepted` and `state_update`, or `command_error` when the "
            "prompt is empty or the model runs the clip queue instead."
        ),
    )
    async def set_prompt(
        self,
        prompt: str = InputField(
            default="",
            max_length=MAX_PROMPT_CHARS,
            moderate=True,
            description=(
                "What the stream should show, up to 800 characters. The take "
                "follows it until it is changed again."
            ),
        ),
        metadata: str = InputField(
            default="",
            max_length=MAX_METADATA_CHARS,
            moderate=True,
            description=(
                "Free-form string stored with the take and echoed on "
                "`state_update`. The model never reads it; use it to carry "
                "display text or your own tracking data."
            ),
        ),
    ) -> PromptAccepted:
        """Set or change the held prompt the continuity take follows."""
        if not self._continuity:
            await self._refuse(
                "set_prompt",
                "`set_prompt` drives continuity mode; this stream uses the "
                "clip queue — `enqueue` a clip instead.",
            )
            return None
        prompt = prompt.strip()
        if not prompt:
            await self._refuse("set_prompt", "The prompt is empty; the stream needs one.")
            return None
        # A new epoch is the signal the run loop watches: it re-anchors the
        # chain on the new prompt as a fresh opener instead of carrying the old
        # prompt's last frame across.
        self._prompt = prompt
        self._prompt_metadata = metadata
        self._prompt_epoch += 1
        await self._send_state_update()
        return PromptAccepted(prompt=prompt)

    @event(
        name="stop",
        description=(
            "Cut what is playing to black within a fraction of a second. In "
            "continuity mode this ends the take and drops the held prompt, so "
            "the stream idles on black until a new `set_prompt` starts a fresh "
            "take. In queue mode whatever is queued on the output tracks is "
            "dropped and the session is back where a finished clip leaves it — "
            "the queue is untouched and the next `play` starts clean; with "
            "autoplay on this acts as a skip, so send `set_autoplay` off first "
            "to hold the stream. Emits `state_update` (and `clip_stopped` in "
            "queue mode), or `command_error` when nothing is playing."
        ),
    )
    async def stop(self) -> None:
        """Ask the playout loop to cut the current clip (or the continuity take)."""
        if self._continuity:
            if not self._channel_running:
                await self._refuse("stop", "No stream is playing.")
                return
            # End the take, not just this clip: drop the held prompt so the run
            # loop parks in idle (black) instead of re-anchoring a fresh take on
            # the still-held prompt. A new `set_prompt` starts a new take. The
            # take's own teardown broadcasts the `playing=false` state update.
            self._stop_channel = True
            self._prompt = ""
            self._prompt_metadata = ""
            return
        if self._current_clip() is None:
            await self._refuse("stop", "No clip is playing.")
            return
        self._stop_playout = True

    @event(
        name="get_queue",
        description=(
            "Return both queues' contents — `generation` (waiting to build) "
            "and `playout` (built, playable) — every clip as its full "
            "structure. The same payload the model broadcasts as "
            "`queue_update`. Valid at any time."
        ),
    )
    async def get_queue(self) -> QueueUpdate:
        """Answer with the same payload `queue_update` broadcasts."""
        return QueueUpdate(
            generation=self._generation.snapshot(),
            playout=self._playout.snapshot(),
        )

    @event(
        name="set_clip_seconds",
        description=(
            "Set the default length for enqueues that carry no `seconds` of "
            "their own. The value is snapped to the nearest length the model "
            "can produce, so read the effective one back from "
            "`clip_length_accepted`. Clips already in the queue keep the "
            "length they were enqueued with. Longer clips carry a scene "
            "further; shorter ones build faster. Emits `clip_length_accepted` "
            "and `state_update`."
        ),
    )
    async def set_clip_seconds(
        self,
        seconds: float = InputField(
            default=clip_plan.MAX_SECONDS_PUBLISHED,
            ge=clip_plan.MIN_SECONDS_PUBLISHED,
            le=clip_plan.MAX_SECONDS_PUBLISHED,
            description=(
                f"Clip length in seconds, between {_CLIP_RANGE}. Snapped to the "
                "nearest length the model can produce, so the value that takes "
                "effect can differ slightly; `state_update.clip_seconds` always "
                "carries the one in force."
            ),
        ),
    ) -> ClipLengthAccepted:
        """Set the length newly enqueued clips snapshot."""
        if await self._refuse_queue_only("set_clip_seconds"):
            return None
        self._clip_frames = clip_plan.frames_for_seconds(float(seconds))
        await self._send_state_update()
        return ClipLengthAccepted(
            clip_seconds=round(clip_plan.seconds_for_frames(self._clip_frames), 3),
            frames=self._clip_frames,
        )

    @event(
        name="set_seed",
        description=(
            "Set the default seed — the one an `enqueue` without a seed of its "
            "own uses, advancing it by one, so re-enqueuing the same prompts "
            "in the same order reproduces the same clips. Clips already in "
            "the queue keep the seed they were enqueued with. Emits "
            "`seed_accepted` and `state_update`."
        ),
    )
    async def set_seed(
        self,
        seed: int = InputField(
            default=1000,
            ge=0,
            description=(
                "Default seed for enqueues that carry none. Reproduction is "
                "close rather than exact: the deployment runs fused kernels "
                "that can reorder arithmetic."
            ),
        ),
    ) -> SeedAccepted:
        """Set the default seed for enqueues that carry none."""
        self._seed = int(seed)
        await self._send_state_update()
        return SeedAccepted(seed=self._seed)

    @event(
        name="set_autoplay",
        description=(
            "Turn autoplay on or off. On, the playout queue's front clip "
            "starts on its own whenever nothing is playing — right after a "
            "clip finishes, or the moment a build completes while the stream "
            "is idle — so a steadily fed queue plays through without a "
            "`play` per clip. Off "
            "(the default), the stream holds on black until an explicit "
            "`play`. Takes effect immediately and lasts for the session. Emits "
            "`autoplay_accepted` and `state_update`."
        ),
    )
    async def set_autoplay(
        self,
        enabled: bool = InputField(
            default=False,
            description=(
                "True plays the playout queue front-first on its own; false "
                "holds the stream after each clip until `play`."
            ),
        ),
    ) -> AutoplayAccepted:
        """Set whether ready clips start without an explicit `play`."""
        if await self._refuse_queue_only("set_autoplay"):
            return None
        self._autoplay = bool(enabled)
        await self._send_state_update()
        return AutoplayAccepted(enabled=self._autoplay)

    @event(
        name="set_canvas",
        description=(
            "Choose the aspect ratio of `main_video`. The video track keeps "
            "one size and queued clips are built at it, so this is only valid "
            "while the queue is empty and nothing is playing. Emits "
            "`canvas_accepted`, carrying the exact pixel size, and "
            "`state_update`, or `command_error` while clips are queued or "
            "playing, or when the ratio is not one this model offers."
        ),
    )
    async def set_canvas(
        self,
        aspect: str = InputField(
            default="16:9",
            choices=list(clip_plan.ASPECT_CHOICES),
            description=(
                "Aspect ratio of `main_video`. `canvas_accepted` and "
                "`state_update` report the width and height in pixels it "
                "resolves to."
            ),
        ),
    ) -> CanvasAccepted:
        """Set the session's canvas; refused while any clip depends on the old one."""
        if self._continuity:
            if self._channel_running or self._prompt:
                await self._refuse(
                    "set_canvas",
                    "The canvas is fixed while a prompt drives the stream; "
                    "`stop` or `reset` first.",
                )
                return None
        elif (
            self._current_clip() is not None
            or len(self._generation) > 0
            or len(self._playout) > 0
        ):
            await self._refuse(
                "set_canvas",
                "The canvas is fixed while clips are queued or playing; "
                "`reset` or play the queue out first.",
            )
            return None
        try:
            height, width = clip_plan.canvas_for_choice(aspect, self.config.canvas_short_edge)
        except ValueError as error:
            await self._refuse("set_canvas", str(error))
            return None
        self._aspect = aspect
        await self._send_state_update()
        return CanvasAccepted(aspect=aspect, width=width, height=height)

    @event(
        name="reset",
        description=(
            "Return every condition to its default, drop both queues' clips, "
            "and clear the output tracks. A clip that is playing is cut, with "
            "a `clip_stopped` to mark it. Valid at any time. Replies "
            "`session_reset` and emits `queue_update` and `state_update`."
        ),
    )
    async def reset(self) -> SessionReset:
        """Clear the session back to its defaults."""
        if self._continuity:
            was_playing = self._channel_running
            if self._channel_running:
                self._stop_channel = True
            # Drop the held prompt and re-anchor: a fresh take, nothing carried.
            self._prompt = ""
            self._prompt_metadata = ""
            self._prompt_epoch += 1
            self._clip_frames = self.config.continuity_clip_frames
            self._seed = self.config.seed
            self._aspect = self.config.aspect
            self.output.flush()
            await self._send_state_update()
            return SessionReset(cleared_clips=0, was_playing=was_playing)
        current = self._current_clip()
        was_playing = current is not None
        cleared = self._generation.clear() + self._playout.clear()
        # A pending play request is a clip already off the queue; dropping it
        # here counts it with the cleared ones. A clip mid-play is cut by the
        # playout loop, which owns its `clip_stopped`.
        if self._play_request is not None:
            self._play_request = None
            cleared += 1
        if self._playing is not None:
            self._stop_playout = True
        if self._build is not None:
            _entry, job, _submitted = self._build
            job.cancelled = True
        self._clip_frames = self.config.clip_frames
        self._seed = self.config.seed
        self._aspect = self.config.aspect
        self._autoplay = False
        self.output.flush()
        await self._send_queue_update()
        await self._send_state_update()
        return SessionReset(cleared_clips=cleared, was_playing=was_playing)

    @event(
        name="get_state",
        description=(
            "Return a snapshot of everything the session exposes except the "
            "queue's contents (`get_queue` carries those): the conditions in "
            "force, what is playing, progress counters, and the commands that "
            "are valid right now. The same payload the model broadcasts as "
            "`state_update`. Valid at any time."
        ),
    )
    async def get_state(self) -> StateUpdate:
        """Answer with the same snapshot `state_update` broadcasts."""
        return self._snapshot()

    # ------------------------------------------------------------- run loop

    async def run(self) -> None:
        """The model's control loop: park without an audience, serve with one.

        Nothing here may raise: an exception out of ``run()`` is an
        unrecoverable crash of the whole model loop, not the end of one
        session, so ``_serve`` owns its own failure reporting.
        """
        while True:
            await self.connected.wait()
            if self._continuity:
                await self._serve_continuity()
            else:
                await self._serve()

    # ---------------------------------------------------------- continuity loop

    async def _serve_continuity(self) -> None:
        """Drive the single continuous take while an audience is connected.

        Idle (black) until `set_prompt` holds a prompt; then run one take —
        clips built back to back, FL2VA-anchored and crossfaded into one
        stream — until `stop`, `reset`, the prompt is dropped, or the audience
        leaves. Nothing here may raise: the take owns its own failure
        reporting, and the loop parks back in ``run()`` when the audience goes.
        A runtime `set_continuity(false)` (only accepted while idle) drops this
        guard, so the loop returns and ``run()`` re-dispatches to `_serve`.
        """
        while self.connected.is_set() and self._continuity:
            try:
                if not self._prompt:
                    self._channel_running = False
                    self._stop_channel = False
                    await asyncio.sleep(POLL_SECONDS)
                    continue
                await self._run_continuity_take()
            except Exception:  # noqa: BLE001 — the model loop must survive anything
                logger.exception("error in the fast-h3 continuity loop")
                self._channel_running = False
                self.output.flush()
                await asyncio.sleep(POLL_SECONDS)

    async def _run_continuity_take(self) -> None:
        """Stream one held-prompt take, always building one clip ahead.

        Clip 0 opens plain (text-to-video); every later clip is FL2VA-anchored
        on the previous clip's colour-matched last frame, and the worker
        crossfades the boundary, so the take is one continuous stream. A
        ``set_prompt`` mid-take re-anchors: the next clip opens fresh on the new
        prompt (no first-frame carry-over) and the held seam tail dissolves the
        old scene into it. The build for clip N+1 is submitted before clip N is
        emitted, which is what keeps the stream gap-free.
        """
        import numpy as np  # noqa: F401 — kept parallel to _emit_continuity
        from PIL import Image

        height, width = self._canvas()
        seam_frames = self.config.seam_frames
        frames = self._clip_frames
        # Fresh channel state: new exposure reference, no seam tail to blend.
        self.backend.reset_continuity()
        self._channel_running = True
        self._stop_channel = False
        self._frames_sent = 0
        self._seconds_sent = 0.0
        # The emit clock is carried across clips so a seam costs no time.
        pacer = {"clock_start": None, "frames_paced": 0}

        index = 0
        prompt = self._prompt
        take_epoch = self._prompt_epoch
        started_at = time.monotonic()
        # Clip 0 opens plain: text-to-video, no first-frame anchor. The take
        # then chains every later clip on its own colour-matched last frame.
        job = self.backend.submit_continuity(
            index=0, frames=frames, prompt=prompt, seed=self._seed,
            height=height, width=width, anchor=None, seam_frames=seam_frames,
        )
        await self._send_state_update()

        try:
            while True:
                while not job.done.is_set():
                    if not self.connected.is_set():
                        break
                    await asyncio.sleep(POLL_SECONDS)
                if not self.connected.is_set() or job.cancelled:
                    break
                if job.error is not None:
                    raise job.error
                anchor_frame, emit_frames, emit_audio, clip_len = job.result
                if index == 0:
                    logger.info(
                        "continuity first clip ready",
                        ttff_s=round(time.monotonic() - started_at, 2),
                        canvas=f"{width}x{height}",
                    )

                # Decide the next clip: a prompt change opens fresh on the new
                # prompt (no anchor); otherwise chain FL2VA on this last frame.
                next_index = index + 1
                if self._prompt_epoch != take_epoch:
                    take_epoch = self._prompt_epoch
                    next_prompt = self._prompt
                    next_anchor = None
                else:
                    next_prompt = prompt
                    next_anchor = (
                        Image.fromarray(anchor_frame) if anchor_frame is not None else None
                    )

                end_take = self._stop_channel or not self._prompt
                next_job = None
                if not end_take:
                    next_job = self.backend.submit_continuity(
                        index=next_index, frames=frames, prompt=next_prompt,
                        seed=self._seed + next_index, height=height, width=width,
                        anchor=next_anchor, seam_frames=seam_frames,
                    )

                await self._emit_continuity(emit_frames, emit_audio, pacer)
                self._clips_played += 1
                await self._send_state_update()

                if next_job is None or self._stop_channel or not self.connected.is_set():
                    job = next_job
                    break
                job, prompt, index = next_job, next_prompt, next_index
        finally:
            # A build cannot be cancelled once running; skip it if the worker
            # has not reached it, then wait it out so the worker is never
            # wedged behind a take that has already ended.
            if job is not None:
                job.cancelled = True
                while not job.done.is_set() and self._worker_alive():
                    await asyncio.sleep(POLL_SECONDS)
            self._channel_running = False
            self._stop_channel = False
            self.output.flush()
            try:
                await self._send_state_update()
            except Exception:  # noqa: BLE001 — teardown must not crash the loop
                logger.exception("failed to send the continuity closing state update")

    def _worker_alive(self) -> bool:
        """Whether the backend's generation worker is still running."""
        worker = getattr(self.backend, "_worker", None)
        return bool(worker is not None and worker.is_alive())

    async def _emit_continuity(self, frames_list, samples, pacer: dict) -> None:
        """Emit one continuity clip as paced slices, the clock carried in ``pacer``.

        Identical 24 fps metronome to the queue's emitter, except the clock
        persists across clips (so a seam adds no gap) and the cut condition is
        the take's `stop`/disconnect rather than a per-clip stop flag. Never
        bursts to catch up: a late clip re-anchors the clock instead of
        overflowing the transport.
        """
        import numpy as np

        samples_per_frame = OUTPUT_SAMPLE_RATE / FRAME_RATE
        total = len(frames_list)
        for lo in range(0, total, EMIT_FRAMES):
            if self._stop_channel or not self.connected.is_set():
                return
            hi = min(lo + EMIT_FRAMES, total)
            alo = round(lo * samples_per_frame)
            ahi = round(hi * samples_per_frame)

            now = asyncio.get_running_loop().time()
            if pacer["clock_start"] is None:
                pacer["clock_start"] = now
            content_pos = pacer["frames_paced"] / FRAME_RATE
            pacer["clock_start"] = max(pacer["clock_start"], now - content_pos)
            delay = pacer["clock_start"] + content_pos - now
            if delay > 0:
                await asyncio.sleep(delay)

            pacer["frames_paced"] += hi - lo
            self._frames_sent += hi - lo
            self._seconds_sent = self._frames_sent / FRAME_RATE
            video = np.ascontiguousarray(np.stack(frames_list[lo:hi]))
            await self.emit(FastH3Output(main_video=video, main_audio=samples[:, alo:ahi]))

    async def _serve(self) -> None:
        """Pump builds and play armed clips while an audience is connected.

        Generation is gated on having an audience: with nobody connected no new
        build is submitted, and the loop parks back in ``run()``. A build
        already on the worker finishes into the queue either way, because a
        clip cannot be cancelled mid-build.
        A runtime `set_continuity(true)` (only accepted while idle) drops this
        guard, so the loop returns and ``run()`` re-dispatches to
        `_serve_continuity`.
        """
        while self.connected.is_set() and not self._continuity:
            try:
                await self._pump_builds()
                if (
                    self._autoplay
                    and self._play_request is None
                    and self._playing is None
                ):
                    # Autoplay is a standing `play`: whenever nothing is on
                    # the tracks and the playout queue has a front clip, it
                    # starts.
                    ready = self._playout.head()
                    if ready is not None:
                        self._playout.remove(ready)
                        self._play_request = ready
                        await self._send_queue_update()
                        await self._send_state_update()
                entry = self._play_request
                if entry is not None:
                    self._play_request = None
                    await self._play_clip(entry)
                else:
                    await asyncio.sleep(POLL_SECONDS)
            except Exception:  # noqa: BLE001 — the model loop must survive anything
                logger.exception("error in the fast-h3 serve loop")
                await asyncio.sleep(POLL_SECONDS)

    async def _pump_builds(self) -> None:
        """Apply a finished build and keep the worker fed, without blocking.

        Called from the idle loop and from every playout slice, so clips keep
        building while another one streams. The generation queue is consumed
        front first, always — paused only while the playout queue is full,
        since a finished build needs a slot to land in (the pause is the
        submit-time reservation). A finished build whose entry left the
        generation queue — a `pop`, a `reset`, or a session end — is
        discarded silently; the queues own what exists.
        """
        if self._build is not None:
            entry, job, submitted = self._build
            if not job.done.is_set():
                return
            self._build = None
            entry.building = False
            if job.cancelled or entry not in self._generation:
                pass
            elif job.error is not None:
                self._generation.remove(entry)
                await self.send(ClipFailed(clip=entry.snapshot(), reason=str(job.error)))
                await self._send_queue_update()
                await self._send_state_update()
            else:
                entry.video, entry.audio = job.result
                # The submit-time reservation on the playout queue is what
                # guarantees this add cannot overflow: only builds add here,
                # and playing or `pop` can only have shrunk it since.
                self._generation.remove(entry)
                self._playout.add(entry)
                logger.info(
                    f"clip generated: {entry.clip_id} ({entry.frames}f) "
                    f"{time.monotonic() - submitted:.2f}s after submit, "
                    f"{len(self._generation)} generating, {len(self._playout)} playable"
                )
                await self.send(ClipGenerated(clip=entry.snapshot()))
                await self._send_queue_update()
                await self._send_state_update()
        if self._build is None and not self._playout.full:
            pending = self._generation.next_to_build()
            if pending is not None:
                height, width = self._canvas()
                pending.building = True
                logger.info(
                    f"clip build submitted: {pending.clip_id} ({pending.frames}f), "
                    f"{len(self._generation)} generating"
                )
                self._build = (
                    pending,
                    self.backend.submit(
                        frames=pending.frames,
                        prompt=pending.prompt,
                        seed=pending.seed,
                        height=height,
                        width=width,
                    ),
                    time.monotonic(),
                )

    async def _play_clip(self, entry: ClipEntry) -> None:
        """Stream one built clip, then flush to black and report how it ended."""
        self._playing = entry
        outcome = "stopped"
        try:
            if not self._stop_playout:
                await self.send(ClipStarted(clip=entry.snapshot()))
                await self._send_state_update()
                outcome = await self._emit_clip(entry)
        finally:
            # Black between clips, always: whatever the transport still holds
            # of this clip is dropped, so the next `play` starts clean.
            self.output.flush()
            self._playing = None
            self._stop_playout = False
        self._clips_played += 1
        if outcome == "finished":
            await self.send(
                ClipFinished(clip=entry.snapshot(), seconds_sent=round(self._seconds_sent, 2))
            )
        elif outcome == "stopped":
            await self.send(
                ClipStopped(clip=entry.snapshot(), seconds_sent=round(self._seconds_sent, 2))
            )
        # A lost audience ends the playout with nobody to tell; the state
        # update below is a harmless no-op in that case.
        await self._send_state_update()

    # -------------------------------------------------------------- emitter

    async def _emit_clip(self, entry: ClipEntry) -> str:
        """Emit one clip as paced slices on a 24 fps metronome.

        - Paced by FRAMES, not slices: a clip's tail slice is short, and
          charging it a full slot would open a hole in the cadence.
        - Never burst to catch up: if the transport held a slice back,
          re-anchor instead. A catch-up burst only overflows the queue.
        - Emits omit ``compute_time``, so every slice is tagged at the pinned
          24 fps — the rate the audio is already sample-clocked against.
        - Builds keep moving: every slice pumps the worker, so a clip
          generating behind this one is ready sooner.

        Returns:
            ``"finished"`` when the whole clip went out, ``"stopped"`` when
            `stop` or `reset` cut it, ``"gone"`` when the audience left.
        """
        import numpy as np

        frames_list, samples = entry.video, entry.audio
        samples_per_frame = OUTPUT_SAMPLE_RATE / FRAME_RATE
        total = len(frames_list)
        clock_start: float | None = None
        frames_paced = 0
        for lo in range(0, total, EMIT_FRAMES):
            await self._pump_builds()
            if self._stop_playout:
                return "stopped"
            if not self.connected.is_set():
                return "gone"
            hi = min(lo + EMIT_FRAMES, total)
            alo = round(lo * samples_per_frame)
            ahi = round(hi * samples_per_frame)

            now = asyncio.get_running_loop().time()
            if clock_start is None:
                clock_start = now
            content_pos = frames_paced / FRAME_RATE
            clock_start = max(clock_start, now - content_pos)
            delay = clock_start + content_pos - now
            if delay > 0:
                await asyncio.sleep(delay)

            frames_paced += hi - lo
            self._frames_sent += hi - lo
            self._seconds_sent = self._frames_sent / FRAME_RATE
            video = np.ascontiguousarray(np.stack(frames_list[lo:hi]))
            await self.emit(FastH3Output(main_video=video, main_audio=samples[:, alo:ahi]))
        return "finished"
