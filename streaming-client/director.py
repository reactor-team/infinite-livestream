"""The director: viewer prompts in, tagged scene groups on the model's queue.

One prompt from chat becomes one *scene group*: the upsampler expands it into
1..N self-contained scenes — a single shot, or a chunked short story — and
the director enqueues them contiguously on the fast-h3 queue. The director is
also the *playout* brain: the model's autoplay starts the playout front the
instant the stream idles (gapless), and `run_playout` curates that front with
`move` — viewer content before filler, judged purely from the metadata echo
(`pick_next` in `group_tag.py`), because who asked for a clip is client-side
knowledge the model deliberately never has. Rules that keep it coherent:

  * The director is the queue's only writer. Both writers inside it — the
    viewer worker (`run`) and the idle filler (`run_idle`) — serialize group
    enqueues through one lock, so groups can never interleave.
  * A group is only enqueued when the whole group fits the generation
    queue (after evicting waiting filler if needed), so it cannot get stuck
    half-in; a backlog already full of viewer content drops new prompts
    instead of stalling them behind a wait. Every capacity is whatever the
    connected deployment reports in `state_update`, never a constant.
  * Viewer prompts outrank filler, and stay first-come-first-served among
    themselves: viewer groups insert into the generation queue ahead of
    waiting filler and behind waiting viewer clips (`enqueue`'s `position`),
    the playout loop pops one built filler per tick when a full playout
    queue blocks a viewer's build, and the idle filler stands down whenever
    viewer work is pending.
  * On a deployment that supports continuation (`link.supports_continuation`
    — the hosted `reactor/fast-h3`), a story group's scenes are chained:
    each scene after the first is enqueued with `continue_from_clip_id`
    naming the previous scene's clip, so it opens on that clip's last frame
    and autoplay hands the pair over seamlessly — one uninterrupted story
    instead of clips separated by black. The upsampler writes every chained
    scene to open on a hard cut (see its module docstring for why); the
    director only wires the chain. A refused chained enqueue falls back to
    a standalone clip after two retries — the scene prompts are
    self-contained, so a lost link degrades the handover, never the story.

Every scene carries the group's identity in the clip's `metadata` — an opaque
string fast-h3 stores and echoes back on every message that references the
clip. That is what lets this client (or any overlay built on it) reconstruct
"scene 2/3 of *Neon Alley* by viewer_42" from a `clip_started` alone, without
joining ids against local state that a reconnect may have lost. The same tag
carries `generated: true` on filler clips, which is what makes them
recognizably evictable later — including by a director restarted with no
memory of enqueueing them.

Viewer prompts pass moderation before they reach the upsampler; the curated
idle list does not need it (see `moderator.py` for the policy).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Sequence

from chat import ChatPrompt
from group_tag import is_generated, parse_group_tag, pick_next, viewer_insert_position
from moderator import Moderator
from reactor_link import ReactorLink
from upsampler import PromptUpsampler, SceneGroup

logger = logging.getLogger(__name__)

# How many chat prompts may wait for upsampling+enqueue before new ones are
# turned away. Depth here is viewer wait time: at ~10s of video per scene the
# model plays through a full generation queue in a couple of minutes; a long
# backlog on top of that serves nobody.
_PENDING_LIMIT = 24

# Enqueue retry cadence while the model refuses (reconnect mid-command, ...).
_RETRY_DELAY_S = 3.0

# How often the idle filler re-checks whether the queue wants topping up.
_IDLE_POLL_S = 3.0

# How often the playout loop re-checks for an idle stream with a ready clip.
# state_update/queue_update broadcasts keep the mirrors fresh; polling them is
# what survives a missed message.
_PLAYOUT_POLL_S = 0.5


class Director:
    """Consume chat prompts; keep the fast-h3 queue fed with scene groups."""

    def __init__(
        self,
        link: ReactorLink,
        upsampler: PromptUpsampler,
        moderator: Moderator,
        cooldown_s: float,
        idle_prompts: Sequence[str] = (),
        idle_queue_target: int = 0,
    ) -> None:
        self._link = link
        self._upsampler = upsampler
        self._moderator = moderator
        self._cooldown_s = cooldown_s
        self._idle_prompts = list(idle_prompts)
        random.shuffle(self._idle_prompts)
        self._idle_index = 0
        self._idle_target = idle_queue_target
        self._pending: asyncio.Queue[ChatPrompt] = asyncio.Queue(_PENDING_LIMIT)
        self._last_accepted: dict[str, float] = {}  # author -> monotonic
        self._enqueue_lock = asyncio.Lock()
        link.add_listener(self._on_model_message)

    def set_idle_prompts(self, prompts: Sequence[str]) -> None:
        """Replace the idle filler's prompt list (a preset switch), reshuffled.

        Takes effect on the filler's next top-up. The preset switch that
        calls this also flushes clips queued under the old preset
        (`flush_stale_clips`). An empty list idles the filler until a later
        switch.
        """
        self._idle_prompts = list(prompts)
        random.shuffle(self._idle_prompts)
        self._idle_index = 0

    async def flush_stale_clips(self) -> None:
        """Drop queued clips after a preset switch so the new identity shows fast.

        Every clip in both queues was written for the old preset, viewer and
        filler alike, and a full queue of them keeps the old style on stream
        for minutes. One clip survives as the on-air buffer — the playout
        front when something is built, otherwise the generation front (its
        build may already be running) — to bridge the gap while the first
        new-style clips build. Pending chat prompts are untouched: they have
        not been upsampled yet, so they come out in the new style.
        """
        async with self._enqueue_lock:
            keep: str | None = None
            if self._link.playout_clips:
                keep = self._link.playout_clips[0]["clip_id"]
            elif self._link.generation_clips:
                keep = self._link.generation_clips[0]["clip_id"]
            # Built (playout) clips first, then the generation queue back to
            # front: on the extended contract `pop` refuses an unbuilt clip
            # that queued clips still continue from, so dependents must go
            # before their sources.
            doomed = [
                clip
                for clip in self._link.playout_clips
                + list(reversed(self._link.generation_clips))
                if clip["clip_id"] != keep
            ]
            popped = 0
            for clip in doomed:
                reply = await self._link.send_command(
                    "pop", {"clip_id": clip["clip_id"]}
                )
                if isinstance(reply, dict) and "clip" in reply:
                    popped += 1
            logger.info(
                "[director] preset switch: flushed %d/%d queued clips%s",
                popped, len(doomed),
                " (front kept as buffer)" if keep else "",
            )

    # -------------------------------------------------------- chat intake

    def submit(self, prompt: ChatPrompt) -> None:
        """Accept one chat prompt (called synchronously by chat sources)."""
        now = time.monotonic()
        last = self._last_accepted.get(prompt.author)
        if last is not None and now - last < self._cooldown_s:
            logger.info(
                "[director] cooldown: dropping prompt from %s (%.0fs left)",
                prompt.author, self._cooldown_s - (now - last),
            )
            return
        try:
            self._pending.put_nowait(prompt)
        except asyncio.QueueFull:
            logger.warning(
                "[director] backlog full (%d); dropping prompt from %s",
                _PENDING_LIMIT, prompt.author,
            )
            return
        self._last_accepted[prompt.author] = now
        logger.info(
            "[director] accepted from %s@%s: %s",
            prompt.author, prompt.source, prompt.text,
        )

    # ------------------------------------------------- viewer prompt loop

    def _viewer_clips_queued(self) -> int:
        """Viewer clips across both queues (anything not tagged filler)."""
        return sum(
            1
            for clip in self._link.generation_clips + self._link.playout_clips
            if not is_generated(clip)
        )

    async def run(self) -> None:
        """Moderate, upsample, and enqueue pending prompts, one group at a time."""
        while True:
            prompt = await self._pending.get()
            try:
                # A queue already full of viewer content takes no more: the
                # prompt is dropped now, before it costs a moderation and an
                # LLM call. Capacity is whatever the connected deployment
                # reports, never a constant.
                if self._viewer_clips_queued() >= self._link.playout_capacity:
                    logger.warning(
                        "[director] %d viewer clips already queued (budget %d); "
                        "dropping prompt from %s",
                        self._viewer_clips_queued(), self._link.playout_capacity,
                        prompt.author,
                    )
                    continue
                verdict = await self._moderator.review(prompt.text)
                if verdict is not None:
                    logger.warning(
                        "[director] rejected prompt from %s@%s (%s): %s",
                        prompt.author, prompt.source, verdict, prompt.text,
                    )
                    continue
                group = await self._upsampler.upsample(
                    raw_prompt=prompt.text,
                    author=prompt.author,
                    source=prompt.source,
                    min_seconds=self._link.min_seconds,
                    max_seconds=self._link.max_seconds,
                    chained=self._link.supports_continuation,
                )
                await self._enqueue_group(group)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error(
                    "[director] failed to process prompt from %s: %s",
                    prompt.author, error,
                )

    # -------------------------------------------------------- idle filler

    async def run_idle(self) -> None:
        """Keep the queue topped up with generated clips while chat is quiet.

        One clip per group, on purpose: single-scene fillers are the finest
        eviction granularity, and popping one never truncates a story.
        """
        if self._idle_target <= 0:
            logger.info("[director] idle filler disabled (target 0)")
            return
        logger.info(
            "[director] idle filler: %d prompts, queue target %d",
            len(self._idle_prompts), self._idle_target,
        )
        while True:
            await asyncio.sleep(_IDLE_POLL_S)
            # The list may be empty — at startup or after a preset switch to
            # a preset without idle prompts; keep polling so a later switch
            # revives the filler without a restart.
            if not self._idle_prompts:
                continue
            # The configured target self-clamps under the deployment's live
            # playout capacity: filler must never be what fills the playout
            # queue to the brim, because a full playout queue pauses builds
            # (leave at least one slot's headroom for a viewer clip to land).
            target = min(
                self._idle_target, max(1, self._link.playout_capacity - 1)
            )
            if (
                not self._pending.empty()
                or not self._link.connected
                or self._link.generation_queued + self._link.playout_queued
                >= target
            ):
                continue
            text = self._idle_prompts[self._idle_index % len(self._idle_prompts)]
            self._idle_index += 1
            try:
                group = await self._upsampler.upsample(
                    raw_prompt=text,
                    author="auto",
                    source="idle",
                    min_seconds=self._link.min_seconds,
                    max_seconds=self._link.max_seconds,
                    generated=True,
                    max_chunks=1,
                )
                # A viewer prompt that arrived while the LLM ran outranks the
                # filler; drop this group rather than making the viewer wait.
                if self._pending.empty():
                    await self._enqueue_group(group)
            except asyncio.CancelledError:
                raise
            except Exception as error:
                logger.error("[director] idle fill failed: %s", error)

    # ------------------------------------------------------------- playout

    async def run_playout(self) -> None:
        """Curate the playout queue's front so autoplay always starts right.

        The model's autoplay chains the playout front the instant the stream
        idles — a millisecond gap, against the half-second-plus a
        client-sent `play` costs. So the client never sends `play` in steady
        state; it keeps the *front* correct instead: whenever `pick_next`
        (viewer content before filler, the same policy the overlay
        announces) disagrees with the queue's actual front, one `move` fixes
        it. Reordering happens while a clip plays, ahead of the moment it
        matters, so autoplay's next start is already the right clip.
        """
        while True:
            await asyncio.sleep(_PLAYOUT_POLL_S)
            if not self._link.connected:
                continue
            await self._relieve_build_backpressure()
            clips = self._link.playout_clips
            desired = pick_next(clips)
            if desired is None or clips[0]["clip_id"] == desired["clip_id"]:
                continue
            await self._link.send_command(
                "move", {"clip_id": desired["clip_id"], "position": 0}
            )
            # Let the resulting queue_update land before re-evaluating.
            await asyncio.sleep(_PLAYOUT_POLL_S)

    async def _relieve_build_backpressure(self) -> None:
        """Pop one playout filler when built fillers block a viewer's build.

        Generation pauses while the playout queue is full. When what fills it
        is idle filler and a viewer clip waits to build, the newest filler is
        the right thing to lose — one per tick, so a draining queue gets
        every chance to make room by playing instead.
        """
        if self._link.playout_queued < self._link.playout_capacity:
            return
        viewer_waiting = any(
            not is_generated(clip) for clip in self._link.generation_clips
        )
        if not viewer_waiting:
            return
        for clip in reversed(self._link.playout_clips):
            if is_generated(clip):
                reply = await self._link.send_command(
                    "pop", {"clip_id": clip["clip_id"]}
                )
                if isinstance(reply, dict) and "clip" in reply:
                    logger.info(
                        "[director] popped playout filler %s to unblock a "
                        "viewer build",
                        clip["clip_id"][:8],
                    )
                return

    # ---------------------------------------------------------- enqueueing

    async def _enqueue_group(self, group: SceneGroup) -> None:
        """Put one group on the model's generation queue, or drop it and say why.

        Viewer groups enter *ahead of waiting filler and behind waiting
        viewer clips* (`viewer_insert_position`), so viewer requests stay
        first-come-first-served and idle filler just slides back — no
        popping, no waste. Filler groups append. When the generation queue
        cannot fit the group even after dropping the filler waiting in it,
        the group is dropped with the queues intact — a backlog full of
        viewer content takes no more, rather than stalling every later
        prompt behind a wait.

        On a continuation-capable deployment, each scene after the first
        continues the previous scene's clip (`continue_from_clip_id`), so
        the group plays as one uninterrupted story. The chain is dropped
        per scene after two refused attempts — a source clip lost to a
        reconnect must not stall the group — and the scene enqueues as a
        standalone clip instead (a hard cut, which its prompt already is).
        """
        scene_count = len(group.scenes)
        async with self._enqueue_lock:
            free = self._link.generation_capacity - self._link.generation_queued
            if free < scene_count and not group.generated:
                evictable = sum(
                    1 for clip in self._link.generation_clips if is_generated(clip)
                )
                if free + evictable >= scene_count:
                    await self._evict_generation_fillers(scene_count - free)
                    await asyncio.sleep(0.3)  # let the pops' queue_update land
                    free = (
                        self._link.generation_capacity - self._link.generation_queued
                    )
            if free < scene_count:
                logger.warning(
                    "[director] no room in the generation queue for %s "
                    "(%d scenes, %d free); dropping the group",
                    group.group_id, scene_count, free,
                )
                return

            position = (
                None if group.generated
                else viewer_insert_position(self._link.generation_clips)
            )
            chain = self._link.supports_continuation
            previous_clip_id: str | None = None
            for index, scene in enumerate(group.scenes, start=1):
                metadata = json.dumps(
                    {
                        "group_id": group.group_id,
                        "title": group.title[:120],
                        "scene": index,
                        "scenes": scene_count,
                        "author": group.author,
                        "source": group.source,
                        "generated": group.generated,
                        # Truncated so the whole blob stays well under fast-h3's
                        # 2000-char metadata cap.
                        "raw_prompt": group.raw_prompt[:400],
                    },
                    ensure_ascii=False,
                )
                payload = {
                    "prompt": scene.prompt,
                    "metadata": metadata,
                    "seconds": scene.seconds,
                }
                if position is not None:
                    # Consecutive positions keep the group contiguous and in
                    # scene order, ahead of the filler it displaced.
                    payload["position"] = position + index - 1
                if chain and previous_clip_id is not None:
                    # The scene opens on the previous scene's last frame and
                    # autoplay hands the pair over with no cut to black. The
                    # scene prompt itself opens on a hard cut (the
                    # upsampler's chained rules), so the chain stays sharp.
                    payload["continue_from_clip_id"] = previous_clip_id
                chained_refusals = 0
                while True:
                    reply = await self._link.send_command("enqueue", payload)
                    if isinstance(reply, dict) and "clip" in reply:
                        clip = reply["clip"]
                        logger.info(
                            "[director] queued %s scene %d/%d as %s (%.1fs, seed %s)%s%s",
                            group.group_id, index, scene_count,
                            clip["clip_id"][:8], clip["seconds"], clip["seed"],
                            " [auto]" if group.generated else "",
                            f" ← {previous_clip_id[:8]}"
                            if payload.get("continue_from_clip_id") else "",
                        )
                        previous_clip_id = clip["clip_id"]
                        break
                    # Bodyless reply = refused (command_error was logged by the
                    # link) or the session dropped mid-command; wait and retry.
                    if "continue_from_clip_id" in payload:
                        # A reconnect loses the source clip server-side, and a
                        # chained enqueue naming it is refused forever. Two
                        # strikes, then the scene enqueues standalone — its
                        # prompt is a self-contained cut, so only the seamless
                        # handover is lost, never the story.
                        chained_refusals += 1
                        if chained_refusals >= 2:
                            del payload["continue_from_clip_id"]
                            logger.warning(
                                "[director] %s scene %d/%d: dropping the "
                                "continuation after %d refusals; enqueueing "
                                "as a standalone cut",
                                group.group_id, index, scene_count,
                                chained_refusals,
                            )
                    logger.warning(
                        "[director] enqueue of %s scene %d/%d refused; retrying in %.0fs",
                        group.group_id, index, scene_count, _RETRY_DELAY_S,
                    )
                    await asyncio.sleep(_RETRY_DELAY_S)

    async def _evict_generation_fillers(self, needed: int) -> int:
        """Pop up to `needed` filler clips from the generation queue.

        Capacity relief only — order needs no eviction now that viewer
        groups insert ahead of filler positionally. Newest-queued first, and
        only clips tagged `generated: true`. Returns how many pops succeeded.
        """
        popped = 0
        for clip in reversed(self._link.generation_clips):
            if popped >= needed:
                break
            if not is_generated(clip):
                continue
            reply = await self._link.send_command(
                "pop", {"clip_id": clip["clip_id"]}
            )
            if isinstance(reply, dict) and "clip" in reply:
                popped += 1
                logger.info(
                    "[director] evicted waiting filler %s for a viewer group",
                    clip["clip_id"][:8],
                )
        return popped

    # ----------------------------------------------------- announcements

    def _on_model_message(self, kind: str, data: dict) -> None:
        """Narrate group playback from clip messages alone (via metadata)."""
        clip = data.get("clip") if isinstance(data, dict) else None
        if not isinstance(clip, dict):
            return
        tag = parse_group_tag(clip.get("metadata", ""))
        label = (
            f"'{tag['title']}' scene {tag['scene']}/{tag['scenes']} "
            f"(by {tag['author']}@{tag['source']})"
            + (" [auto]" if tag.get("generated") else "")
            if tag
            else f"clip {clip.get('clip_id', '?')[:8]}"
        )
        if kind == "clip_started":
            logger.info("[now playing] %s", label)
        elif kind == "clip_finished":
            logger.info("[finished] %s", label)
        elif kind == "clip_failed":
            logger.error(
                "[director] build failed for %s: %s — the queue moves on",
                label, data.get("reason"),
            )
