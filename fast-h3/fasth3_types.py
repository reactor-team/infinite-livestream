"""Client-facing types for the FastH3 Reactor model.

Everything a client can see lives here: the outbound video and audio tracks,
the `ClipInfo` structure every clip-referencing message embeds, and the typed
messages the model sends. ``fasth3.py`` imports these; a frontend developer
reads this file to learn the whole API without opening the inference code.

The conditions behind the `set_*` commands are not here. A ``ReactorModel``
owns its session state itself, so they are plain attributes on ``FastH3`` reset
in ``_reset_session_state``; their client-facing text lives on each handler's
own ``InputField`` declaration.
"""

from __future__ import annotations

from dataclasses import dataclass

from reactor_runtime import (
    Audio,
    MessageField,
    ModelMessage,
    Output,
    Video,
)

MAX_PROMPT_CHARS = 800
MAX_METADATA_CHARS = 2000


class FastH3Output(Output):
    """The generated video and its synchronized audio, streamed per clip."""

    main_video: Video
    main_audio: Audio


@dataclass(frozen=True)
class ClipInfo:
    """One clip, as every clip-referencing message reports it.

    Whole and self-contained on purpose: `clip_queued`, `queue_update`,
    `clip_generated`, `clip_moved`, `clip_started`, `clip_finished`,
    `clip_stopped` and `clip_failed` all carry this same structure, so a
    client never has to join a clip id against an earlier message to know
    what a clip is.

    A clip lives in exactly one of two queues: the **generation queue**
    (waiting to build, ``ready: false``) and then the **playout queue**
    (built, ``ready: true``), which playing consumes. ``clip_id`` is the UUID
    the session assigned at `enqueue`; every later reference to the clip uses
    it. ``prompt`` and ``metadata`` are exactly what the client enqueued —
    the metadata is an opaque string the model never reads, for frontends to
    carry their own tracking data. ``frames`` and ``seconds`` are the clip's
    length in both units, fixed when it was enqueued. ``seed`` is the value
    this clip generates from.
    """

    clip_id: str
    prompt: str
    metadata: str
    frames: int
    seconds: float
    seed: int
    ready: bool


class StateUpdate(ModelMessage):
    """Emitted on connect and after every change to the session's state.

    One snapshot of everything observable except the queue's contents (those
    travel as `queue_update`), so a client can render its whole UI from this
    alone instead of accumulating the individual messages below.
    """

    clip_seconds: float = MessageField(
        description=(
            "Length a newly enqueued clip gets when `enqueue` carries no "
            "`seconds` of its own."
        )
    )
    clip_seconds_min: float = MessageField(
        description="Shortest clip length `set_clip_seconds` accepts."
    )
    clip_seconds_max: float = MessageField(
        description="Longest clip length `set_clip_seconds` accepts."
    )
    seed: int = MessageField(
        description=(
            "Seed the next enqueued clip will use when `enqueue` carries none; "
            "each such enqueue advances it by one."
        )
    )
    autoplay: bool = MessageField(
        description=(
            "The playout queue's front clip starts on its own whenever "
            "nothing is playing. Off by default: playback waits for an "
            "explicit `play`."
        )
    )
    aspect: str = MessageField(description="Aspect ratio in effect, e.g. `16:9`.")
    width: int = MessageField(description="Width of every frame on `main_video`.")
    height: int = MessageField(description="Height of every frame on `main_video`.")
    continuity: bool = MessageField(
        description=(
            "The stream is a single continuous take driven by `set_prompt`, not "
            "a queue of separate clips. When true, the queue commands "
            "(`enqueue`, `play`, `move`, `pop`) do not apply and `valid_commands` "
            "omits them; when false, `set_prompt` does not apply."
        )
    )
    prompt: str = MessageField(
        description=(
            "The prompt the continuous stream is currently following, set by "
            "`set_prompt`. Empty when nothing is playing, or always in the "
            "queue's clip-at-a-time mode."
        )
    )
    playing: bool = MessageField(description="A clip is streaming on the output tracks.")
    playing_clip_id: str | None = MessageField(
        description="UUID of the clip now playing, or null when the stream is idle."
    )
    generation_queued: int = MessageField(
        description="Clips in the generation queue: enqueued, not yet built."
    )
    generation_capacity: int = MessageField(
        description="Most clips the generation queue holds; `enqueue` is refused beyond it."
    )
    playout_queued: int = MessageField(
        description="Built clips in the playout queue, each playable right now."
    )
    playout_capacity: int = MessageField(
        description=(
            "Most built clips the playout queue holds. Generation pauses "
            "while it is full and resumes as playing or `pop` frees a slot."
        )
    )
    clips_played: int = MessageField(
        description="Clips that finished playing or were stopped since the session began."
    )
    seconds_sent: float = MessageField(
        description="Seconds of video and audio sent since the session began."
    )
    valid_commands: list[str] = MessageField(
        description=(
            "Names of the commands the session would accept right now. Use this "
            "to enable or grey out controls instead of re-deriving the state "
            "machine client-side; any command not listed would be rejected."
        )
    )


class QueueUpdate(ModelMessage):
    """Emitted on connect and whenever either queue changes, and answers `get_queue`.

    Both queues in full, front first, each entry a complete `ClipInfo`. A
    change is any of: a clip enqueued or `move`d, a build finishing (the clip
    crosses from `generation` to `playout`), a clip leaving to play or by
    `pop`, or the queues being cleared by `reset`.
    """

    generation: list[ClipInfo] = MessageField(
        description=(
            "Clips waiting to build, front first. Builds consume this queue "
            "from the front, one at a time, pausing only while `playout` is "
            "at capacity. `enqueue`'s `position` and `move` control the "
            "order."
        )
    )
    playout: list[ClipInfo] = MessageField(
        description=(
            "Built clips waiting to play, front first. A finished build "
            "joins at the back; bare `play` (and autoplay) takes the front; "
            "`move` reorders; playing or `pop` consumes."
        )
    )


class ClipQueued(ModelMessage):
    """Emitted when `enqueue` accepts a generation request."""

    clip: ClipInfo = MessageField(
        description=(
            "The queued clip, UUID included. `ready` is false here; "
            "`clip_generated` announces it crossing into the playout queue."
        )
    )


class ClipGenerated(ModelMessage):
    """Emitted when a clip's build completes.

    The clip has left the generation queue and joined the back of the playout
    queue, playable immediately. `queue_update` accompanies it with both
    queues' new contents.
    """

    clip: ClipInfo = MessageField(
        description="The freshly built clip, now at the back of the playout queue."
    )


class ClipMoved(ModelMessage):
    """Emitted when `move` repositions a clip within its queue."""

    clip: ClipInfo = MessageField(description="The clip that moved.")
    queue: str = MessageField(
        description="Which queue it moved within: `generation` or `playout`."
    )
    position: int = MessageField(
        description="The clip's resulting position in that queue, 0 = front."
    )


class ClipStarted(ModelMessage):
    """Emitted as a clip begins streaming on the output tracks."""

    clip: ClipInfo = MessageField(description="The clip now playing.")


class ClipFinished(ModelMessage):
    """Emitted when a clip has been fully sent on the output tracks.

    The stream then holds on black until the next `play`; nothing plays on its
    own.
    """

    clip: ClipInfo = MessageField(description="The clip that just finished.")
    seconds_sent: float = MessageField(
        description="Seconds of video and audio sent since the session began, this clip included."
    )


class ClipStopped(ModelMessage):
    """Emitted when `stop` cuts a playing clip.

    The rest of the clip is discarded — a stopped clip cannot be resumed — and
    the stream holds on black until the next `play`, exactly as after
    `clip_finished`.
    """

    clip: ClipInfo = MessageField(description="The clip that was cut.")
    seconds_sent: float = MessageField(
        description="Seconds of video and audio sent since the session began."
    )


class ClipPopped(ModelMessage):
    """Emitted when `pop` removes a clip from either queue.

    The clip's slot is free again immediately. A build already running for it
    is discarded when it completes; the GPUs cannot abandon it mid-build.
    """

    clip: ClipInfo = MessageField(description="The clip that left its queue.")


class ClipFailed(ModelMessage):
    """Emitted when a clip's generation fails.

    The clip leaves the queue and the queue moves on; nothing else is
    affected.
    """

    clip: ClipInfo = MessageField(description="The clip whose build failed.")
    reason: str = MessageField(description="What went wrong.")


class ClipLengthAccepted(ModelMessage):
    """Emitted when `set_clip_seconds` is accepted.

    The requested length is snapped to the nearest length the model can produce,
    so the value here may differ slightly from the one sent.
    """

    clip_seconds: float = MessageField(description="Clip length now in effect, in seconds.")
    frames: int = MessageField(description="Frames each newly enqueued clip will carry.")


class SeedAccepted(ModelMessage):
    """Emitted when `set_seed` is accepted."""

    seed: int = MessageField(
        description="Seed the next enqueued clip will use when `enqueue` carries none."
    )


class AutoplayAccepted(ModelMessage):
    """Emitted when `set_autoplay` is accepted."""

    enabled: bool = MessageField(
        description="Whether ready clips now start on their own when nothing is playing."
    )


class ContinuityAccepted(ModelMessage):
    """Emitted when `set_continuity` switches the session's mode.

    Continuity is on by config default, but a client can flip the session
    between the continuous take and the hard-cut clip queue at runtime (only
    while idle). `state_update.valid_commands` then carries the other mode's
    surface, so a frontend re-draws its controls from the snapshot.
    """

    continuity: bool = MessageField(
        description=(
            "True: the session runs the continuous FL2VA take driven by "
            "`set_prompt`. False: the hard-cut clip queue driven by `enqueue`."
        )
    )


class PromptAccepted(ModelMessage):
    """Emitted when `set_prompt` is accepted in continuity mode.

    The continuous stream re-anchors on the new prompt: the next clip opens
    fresh on it (no first-frame carry-over from the old prompt), and every clip
    after chains from that opener until the prompt changes again or `stop`.
    """

    prompt: str = MessageField(description="The prompt the stream now follows.")


class CanvasAccepted(ModelMessage):
    """Emitted when `set_canvas` is accepted."""

    aspect: str = MessageField(description="Aspect ratio now in effect.")
    width: int = MessageField(description="Width of every frame on `main_video`.")
    height: int = MessageField(description="Height of every frame on `main_video`.")


class SessionReset(ModelMessage):
    """Emitted when `reset` is accepted.

    Every condition is back to its default, the queue is empty, and the output
    stream is cleared.
    """

    cleared_clips: int = MessageField(
        description="Clips dropped from both queues, built and pending alike."
    )
    was_playing: bool = MessageField(
        description="A clip was playing and has been cut; a `clip_stopped` accompanies it."
    )


class CommandError(ModelMessage):
    """Emitted when a command is rejected. The command had no effect."""

    command: str = MessageField(description="Name of the command that was rejected.")
    reason: str = MessageField(description="Why it was rejected.")
