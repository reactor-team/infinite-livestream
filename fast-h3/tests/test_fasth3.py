"""FastH3's clip geometry, queue, command contract, playout loop, schema and manifest.

Everything here runs on a laptop: the GPU work sits behind the backend, which
these tests replace with a fake that builds instantly, so the real queueing,
ordering, pacing, emission and teardown all run.

Run from the model folder: ``PYTHONPATH=. python -m pytest tests/ -q``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

import fasth3_clip_plan as clip_plan
import fasth3_session_rules as session_rules
import fasth3_seam as seam
from fasth3 import EMIT_FRAMES, FastH3
from fasth3_assets import FastH3Config, load_config
from fasth3_backend import ClipJob, FastH3Backend
from fasth3_queue import ClipQueue, new_entry
from fasth3_types import ClipInfo

MODEL_DIR = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------- clip geometry
#
# The rules a clip length has to satisfy to be generatable.


def test_bounds_are_the_generatable_range():
    """5 s aligns up to 124 frames; 15 s aligns up to 362 and is out of range."""
    assert clip_plan.MIN_FRAMES == 124
    assert clip_plan.MAX_FRAMES == 345
    # The ceiling is not 15 s: 360 frames aligns up to 362, which is 15.083 s
    # and past the cap, so 345 frames (14.375 s) is the longest clip there is.
    assert clip_plan.MAX_SECONDS == pytest.approx(14.375)
    assert 362 / clip_plan.FPS > 15.0


@pytest.mark.parametrize("frames", range(1, 400))
def test_align_frames_lands_on_the_chunk_grid(frames):
    aligned = clip_plan.align_frames(frames)
    assert aligned % 17 == 5
    assert aligned >= frames
    assert aligned - frames < 17


@pytest.mark.parametrize("seconds", [5.0, 5.167, 8.0, 10.0, 14.375, 15.0, 60.0])
def test_every_accepted_length_is_generatable(seconds):
    frames = clip_plan.frames_for_seconds(seconds)
    assert frames % 17 == 5
    assert clip_plan.MIN_FRAMES <= frames <= clip_plan.MAX_FRAMES
    assert 5.0 <= clip_plan.seconds_for_frames(frames) <= 15.0


def test_published_bounds_round_inward():
    """A client reading the schema sees tidy numbers that still snap cleanly."""
    assert clip_plan.MIN_SECONDS_PUBLISHED == 5.167
    assert clip_plan.MAX_SECONDS_PUBLISHED == 14.375
    assert clip_plan.MIN_SECONDS_PUBLISHED >= clip_plan.MIN_SECONDS
    assert clip_plan.MAX_SECONDS_PUBLISHED <= clip_plan.MAX_SECONDS
    for bound in (clip_plan.MIN_SECONDS_PUBLISHED, clip_plan.MAX_SECONDS_PUBLISHED):
        assert clip_plan.frames_for_seconds(bound) % 17 == 5


def test_seconds_must_be_positive():
    with pytest.raises(ValueError):
        clip_plan.frames_for_seconds(0)


@pytest.mark.parametrize("aspect", clip_plan.ASPECT_CHOICES)
def test_every_offered_canvas_satisfies_the_checkpoint(aspect):
    height, width = clip_plan.canvas_for_choice(aspect)
    assert height % 32 == 0 and width % 32 == 0
    assert height * width <= 768 * 1344
    assert min(height, width) == 768 or height * width <= 768 * 1344
    assert 1 / 4 <= width / height <= 4


def test_default_canvas_is_the_measured_one():
    assert clip_plan.canvas_for_choice("16:9") == (768, 1344)


def test_unknown_aspect_is_rejected():
    with pytest.raises(ValueError):
        clip_plan.canvas_for_choice("32:9")


def test_upstream_constants_have_not_drifted():
    """The duplicated constants must still match FastVideo's own.

    ``fasth3_clip_plan`` copies these rather than importing them, so that the
    schema renders without torch. This is the test that stops the copy going
    stale.
    """
    packing = pytest.importorskip(
        "fastvideo.pipelines.basic.minimax_h3.packing",
        reason="fastvideo is not installed on this machine",
    )
    assert clip_plan.FPS == packing.MINIMAX_H3_FPS
    assert clip_plan._FRAMES_PER_CHUNK == packing.MINIMAX_H3_FRAMES_PER_CHUNK
    assert clip_plan._LATENTS_PER_CHUNK == packing.MINIMAX_H3_LATENTS_PER_CHUNK
    assert clip_plan._MIN_DURATION == packing.MINIMAX_H3_MIN_DURATION
    assert clip_plan._MAX_DURATION == packing.MINIMAX_H3_MAX_DURATION
    assert clip_plan._SHORT_EDGE == packing.MINIMAX_H3_SHORT_EDGE
    assert clip_plan._MAX_PIXELS == packing.MINIMAX_H3_MAX_PIXELS
    assert clip_plan._CANVAS_MULTIPLE == packing.MINIMAX_H3_CANVAS_MULTIPLE

    for frames in (1, 100, 124, 200, 345):
        assert clip_plan.align_frames(frames) == packing.align_num_frames(frames)
    for aspect in clip_plan.ASPECT_CHOICES:
        ratio = clip_plan._ASPECT_RATIOS[aspect]
        assert clip_plan.canvas_for_choice(aspect) == packing.resolve_canvas_size(*ratio)


# ------------------------------------------------------------------ the queue
#
# Pure bookkeeping: order, capacity, and one wire form for every mention.


def make_queue(capacity=3) -> ClipQueue:
    return ClipQueue(capacity)


def add(q, prompt, seed=0, position=None, frames=124):
    entry = new_entry(prompt=prompt, metadata="", frames=frames, seed=seed)
    q.add(entry, position)
    return entry


def built(entry):
    entry.video = [np.zeros((2, 2, 3), np.uint8)]
    entry.audio = np.zeros((1, 4), np.int16)
    return entry


def test_the_queue_keeps_add_order():
    q = make_queue()
    a = add(q, "a", seed=1)
    b = add(q, "b", seed=2)
    assert [entry.prompt for entry in map(q.get, [a.clip_id, b.clip_id])] == ["a", "b"]
    assert q.snapshot()[0]["clip_id"] == a.clip_id
    assert q.next_to_build() is a
    assert q.head() is a


def test_add_at_a_position_lands_there_clamped():
    q = make_queue(capacity=5)
    add(q, "a")
    add(q, "b")
    add(q, "front", position=0)
    add(q, "middle", position=2)
    add(q, "past-the-end", position=99)
    assert [e["prompt"] for e in q.snapshot()] == [
        "front", "a", "middle", "b", "past-the-end",
    ]


def test_move_repositions_within_the_queue():
    q = make_queue(capacity=5)
    a, b, c = add(q, "a"), add(q, "b"), add(q, "c")
    assert q.move(c, 0) == 0
    assert [e["prompt"] for e in q.snapshot()] == ["c", "a", "b"]
    assert q.move(c, 99) == 2  # clamped to the back
    assert [e["prompt"] for e in q.snapshot()] == ["a", "b", "c"]
    q.remove(b)
    with pytest.raises(ValueError):
        q.move(b, 0)


def test_every_clip_gets_a_distinct_uuid():
    q = make_queue()
    ids = {add(q, "p").clip_id for _ in range(3)}
    assert len(ids) == 3


def test_the_queue_is_bounded():
    q = make_queue(capacity=2)
    add(q, "a", seed=1)
    add(q, "b", seed=2)
    assert q.full
    with pytest.raises(ValueError):
        add(q, "c", seed=3)


def test_ready_is_derived_from_the_built_payload():
    q = make_queue()
    entry = add(q, "a", seed=1)
    assert entry.ready is False
    built(entry)
    assert entry.ready is True


def test_building_entries_are_not_resubmitted():
    q = make_queue()
    entry = add(q, "a", seed=1)
    entry.building = True
    assert q.next_to_build() is None
    behind = add(q, "b", seed=2)
    assert q.next_to_build() is behind


def test_the_snapshot_is_exactly_the_published_struct():
    """`ClipEntry.snapshot()` and the schema's `ClipInfo` must never drift."""
    q = make_queue()
    entry = new_entry(prompt="a", metadata="m", frames=124, seed=7)
    q.add(entry)
    snapshot = entry.snapshot()
    assert list(snapshot) == [field.name for field in dataclasses.fields(ClipInfo)]
    assert snapshot["seconds"] == pytest.approx(124 / 24, abs=1e-3)
    assert snapshot["metadata"] == "m"
    assert snapshot["ready"] is False


# --------------------------------------------------------------- session rules
#
# The command state machine clients read out of `state_update`.


def rules(*, playing=False, generation=0, playout=0, capacity=10):
    return session_rules.valid_commands(
        playing=playing,
        generation_queued=generation,
        generation_capacity=capacity,
        playout_queued=playout,
    )


def test_an_empty_idle_session_can_only_enqueue_and_configure():
    commands = rules()
    assert "enqueue" in commands
    assert "set_canvas" in commands
    assert "play" not in commands
    assert "stop" not in commands
    assert "move" not in commands


def test_a_built_clip_makes_play_valid():
    commands = rules(playout=1)
    assert "play" in commands
    assert "pop" in commands
    assert "move" in commands
    # Queued clips were built at the current canvas, so it is locked.
    assert "set_canvas" not in commands


def test_pop_and_move_need_a_queued_clip():
    commands = rules()
    assert "pop" not in commands
    assert "move" not in commands
    assert "pop" in rules(generation=1)


def test_playing_offers_stop_and_locks_the_canvas():
    commands = rules(playing=True)
    assert "stop" in commands
    assert "play" not in commands
    assert "set_canvas" not in commands


def test_a_full_generation_queue_refuses_enqueue():
    commands = rules(generation=10, capacity=10)
    assert "enqueue" not in commands


def test_conditions_and_reads_are_always_available():
    for playing, generation, playout in ((False, 0, 0), (True, 3, 1), (False, 10, 10)):
        commands = rules(playing=playing, generation=generation, playout=playout)
        assert {
            "set_clip_seconds", "set_seed", "set_autoplay", "get_queue", "get_state", "reset"
        } <= set(commands)


# --------------------------------------------------------------------- config


def test_the_shipped_config_parses(tmp_path):
    config = load_config(MODEL_DIR / "fasth3.yaml")
    assert config.queue_size == 10
    assert config.aspect == "16:9"
    assert config.clip_frames == clip_plan.MAX_FRAMES
    # The shipped config is the continuous take: continuity on, at the 640 tier,
    # holding the shortest (gap-free-with-margin) clip length.
    assert config.continuity is True
    assert config.canvas_short_edge == 640
    assert config.continuity_clip_frames == clip_plan.MIN_FRAMES
    assert config.seam_frames == 12
    # Continuity holds one length, so the warm-up warms exactly that length.
    assert config.warmup_frames == (config.continuity_clip_frames,)


def test_every_legal_length_is_aligned_and_within_bounds():
    counts = clip_plan.legal_frame_counts()
    assert counts[0] == clip_plan.MIN_FRAMES
    assert counts[-1] == clip_plan.MAX_FRAMES
    assert all(frames % 17 == 5 for frames in counts)
    assert len(counts) == 14


def test_warmup_lengths_parse_all_shapes(tmp_path):
    default = tmp_path / "default.yaml"
    default.write_text("inference: {}\n", encoding="utf-8")
    config = load_config(default)
    assert config.warmup_frames == (config.clip_frames,)

    listed = tmp_path / "listed.yaml"
    listed.write_text(
        "inference:\n  warmup_lengths: [5.167, 8.0]\n", encoding="utf-8"
    )
    config = load_config(listed)
    # Snapped to legal lengths, default length always included, ascending.
    assert config.warmup_frames == tuple(sorted({
        clip_plan.frames_for_seconds(5.167),
        clip_plan.frames_for_seconds(8.0),
        config.clip_frames,
    }))

    bad = tmp_path / "bad.yaml"
    bad.write_text("inference:\n  warmup_lengths: sometimes\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad)


def test_a_bad_aspect_or_queue_size_fails_startup(tmp_path):
    bad_aspect = tmp_path / "aspect.yaml"
    bad_aspect.write_text("inference:\n  aspect: '32:9'\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_aspect)

    bad_queue = tmp_path / "queue.yaml"
    bad_queue.write_text("inference:\n  queue_size: 0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(bad_queue)


# ------------------------------------------------------------ command contract
#
# The real handlers on a model whose ``load()`` never ran: everything they touch
# is session state and pure arithmetic, so the whole state machine — refusals
# included — is testable on a laptop.


def make_config(
    queue_size=3,
    generation_queue_size=None,
    *,
    continuity=False,
    canvas_short_edge=768,
    seam_frames=12,
) -> FastH3Config:
    return FastH3Config(
        aspect="16:9",
        clip_frames=clip_plan.frames_for_seconds(clip_plan.MAX_SECONDS),
        seed=1000,
        num_inference_steps=5,
        queue_size=queue_size,
        generation_queue_size=generation_queue_size or queue_size,
        warmup_aspects=("16:9",),
        warmup_frames=(clip_plan.frames_for_seconds(clip_plan.MAX_SECONDS),),
        canvas_short_edge=canvas_short_edge,
        continuity=continuity,
        continuity_clip_frames=clip_plan.frames_for_seconds(clip_plan.MIN_SECONDS),
        seam_frames=seam_frames,
        inference={},
        runtime={},
    )


def run(coro):
    """Drive one handler to completion."""
    return asyncio.run(coro)


def refusal(model):
    """The most recent `command_error` broadcast, or None if there is none.

    A refusal is not an exception here: a handler reports failure by
    broadcasting `command_error` and returning without a value, so that clients
    on every SDK generation can see it.
    """
    errors = [message for message in model.sent if type(message).__name__ == "CommandError"]
    return errors[-1] if errors else None


@pytest.fixture
def model():
    """A FastH3 with the attributes ``load()`` would have set, and no engine."""
    instance = FastH3()
    # Loop-bound state the runtime creates when the model loop starts.
    instance._on_loop_ready()
    instance.config = make_config(queue_size=3)
    instance._reset_session_state()

    sent: list = []

    async def capture(message):
        sent.append(message)

    instance.send = capture
    instance.sent = sent
    return instance


def make_ready(model, clip):
    """What `_pump_builds` does when a build lands: cross into playout."""
    entry = model._generation.get(clip["clip_id"])
    model._generation.remove(entry)
    entry.video, entry.audio = [np.zeros((2, 2, 3), np.uint8)], np.zeros((1, 10), np.int16)
    model._playout.add(entry)
    return entry


def test_enqueue_returns_the_full_struct(model):
    reply = run(model.enqueue(prompt="a lighthouse in fog", metadata="req-42"))
    clip = reply.clip
    assert clip["prompt"] == "a lighthouse in fog"
    assert clip["metadata"] == "req-42"
    assert clip["ready"] is False
    assert clip["frames"] == model.config.clip_frames
    assert clip["seconds"] == pytest.approx(clip["frames"] / 24, abs=1e-3)
    assert clip["clip_id"]
    # The struct is a plain mapping, because the wire encoder takes only
    # JSON-representable values; ClipInfo is its schema-side declaration.
    assert list(clip) == [field.name for field in dataclasses.fields(ClipInfo)]


def test_enqueue_position_zero_is_the_next_build(model):
    first = run(model.enqueue(prompt="waits", metadata="")).clip
    jumped = run(model.enqueue(prompt="jumps", metadata="", position=0)).clip
    order = [clip["clip_id"] for clip in model._generation.snapshot()]
    assert order == [jumped["clip_id"], first["clip_id"]]
    assert model._generation.next_to_build().clip_id == jumped["clip_id"]


def test_move_repositions_and_names_the_queue(model):
    first = run(model.enqueue(prompt="a", metadata="")).clip
    second = run(model.enqueue(prompt="b", metadata="")).clip
    reply = run(model.move(clip_id=second["clip_id"], position=0))
    assert reply.queue == "generation" and reply.position == 0
    order = [clip["clip_id"] for clip in model._generation.snapshot()]
    assert order == [second["clip_id"], first["clip_id"]]

    built = make_ready(model, first)
    reply = run(model.move(clip_id=built.clip_id, position=0))
    assert reply.queue == "playout" and reply.position == 0

    assert run(model.move(clip_id="nope", position=0)) is None
    assert refusal(model).command == "move" 


def test_enqueue_needs_a_prompt(model):
    assert run(model.enqueue(prompt="   ", metadata="")) is None
    assert refusal(model).command == "enqueue"
    assert len(model._generation) == 0


def test_enqueue_snapshots_the_conditions_in_force(model):
    run(model.set_clip_seconds(seconds=8.0))
    first = run(model.enqueue(prompt="a", metadata="")).clip
    run(model.set_clip_seconds(seconds=14.375))
    second = run(model.enqueue(prompt="b", metadata="")).clip
    assert first["frames"] == clip_plan.frames_for_seconds(8.0)
    assert second["frames"] == clip_plan.frames_for_seconds(14.375)
    # Clips already queued keep the length they were enqueued with.
    assert model._generation.get(first["clip_id"]).frames == first["frames"]


def test_each_enqueue_advances_the_seed(model):
    seeds = [run(model.enqueue(prompt="p", metadata="")).clip["seed"] for _ in range(2)]
    assert seeds == [1000, 1001]
    run(model.reset())
    run(model.set_seed(seed=7))
    assert run(model.enqueue(prompt="p", metadata="")).clip["seed"] == 7


def test_an_explicit_length_applies_to_that_clip_only(model):
    """A per-enqueue `seconds` snaps to the grid and spares the session default."""
    clip = run(model.enqueue(prompt="p", metadata="", seconds=8.3)).clip
    assert clip["frames"] == clip_plan.frames_for_seconds(8.3)
    assert clip["frames"] % 17 == 5
    assert model._clip_frames == model.config.clip_frames  # default untouched
    plain = run(model.enqueue(prompt="p", metadata="")).clip
    assert plain["frames"] == model.config.clip_frames


def test_an_explicit_seed_leaves_the_default_untouched(model):
    """Explicit and automatic seeding must not interfere with each other."""
    explicit = run(model.enqueue(prompt="p", metadata="", seed=42)).clip
    assert explicit["seed"] == 42
    assert model._seed == 1000  # the advancing default did not move
    automatic = run(model.enqueue(prompt="p", metadata="")).clip
    assert automatic["seed"] == 1000
    assert model._seed == 1001


def test_autoplay_is_a_session_condition(model):
    assert run(model.get_state()).autoplay is False
    reply = run(model.set_autoplay(enabled=True))
    assert reply.enabled is True
    assert run(model.get_state()).autoplay is True
    # `reset` returns every condition to its default, autoplay included.
    run(model.reset())
    assert run(model.get_state()).autoplay is False


def test_a_full_queue_refuses_the_next_enqueue(model):
    for index in range(3):
        run(model.enqueue(prompt=f"p{index}", metadata=""))
    assert run(model.enqueue(prompt="overflow", metadata="")) is None
    assert refusal(model).command == "enqueue"
    assert len(model._generation) == 3


def test_play_needs_a_ready_clip(model):
    assert run(model.play(clip_id="")) is None
    assert refusal(model).command == "play"

    queued = run(model.enqueue(prompt="p", metadata="")).clip
    assert run(model.play(clip_id=queued["clip_id"])) is None  # enqueued, not built
    assert refusal(model).reason.startswith("That clip is still generating")

    assert run(model.play(clip_id="not-a-real-id")) is None
    assert "not-a-real-id" in refusal(model).reason


def test_play_takes_the_playout_front(model):
    first = run(model.enqueue(prompt="a", metadata="")).clip
    second = run(model.enqueue(prompt="b", metadata="")).clip
    make_ready(model, first)
    make_ready(model, second)

    run(model.play(clip_id=""))
    assert model._play_request.clip_id == first["clip_id"]
    # Playing consumed the entry: the playout queue holds only the second.
    assert [clip["clip_id"] for clip in model._playout.snapshot()] == [second["clip_id"]]


def test_play_by_id_takes_that_clip(model):
    run(model.enqueue(prompt="a", metadata=""))
    second = run(model.enqueue(prompt="b", metadata="")).clip
    make_ready(model, second)

    run(model.play(clip_id=second["clip_id"]))
    assert model._play_request.clip_id == second["clip_id"]


def test_only_one_clip_plays_at_a_time(model):
    queued = run(model.enqueue(prompt="a", metadata="")).clip
    make_ready(model, queued)
    run(model.play(clip_id=""))

    assert run(model.play(clip_id="")) is None
    assert refusal(model).reason.startswith("A clip is already playing")


def test_pop_frees_the_slot(model):
    kept = run(model.enqueue(prompt="keep", metadata="")).clip
    victim = run(model.enqueue(prompt="drop", metadata="")).clip
    reply = run(model.pop(clip_id=victim["clip_id"]))
    assert reply.clip["clip_id"] == victim["clip_id"]
    assert [c["clip_id"] for c in model._generation.snapshot()] == [kept["clip_id"]]

    # And a built clip pops out of the playout queue the same way.
    make_ready(model, kept)
    reply = run(model.pop(clip_id=kept["clip_id"]))
    assert reply.clip["clip_id"] == kept["clip_id"]
    assert len(model._playout) == 0

    assert run(model.pop(clip_id="nope")) is None
    assert refusal(model).command == "pop"
    assert run(model.pop(clip_id="")) is None  # the id is required
    assert refusal(model).command == "pop"


def test_pop_cancels_the_build_in_flight(model):
    from fasth3_backend import ClipJob

    clip = run(model.enqueue(prompt="building", metadata="")).clip
    entry = model._generation.get(clip["clip_id"])
    entry.building = True
    job = ClipJob(None)
    model._build = (entry, job, 0.0)

    run(model.pop(clip_id=clip["clip_id"]))
    assert job.cancelled is True
    assert len(model._generation) == 0


def test_stop_needs_a_playing_clip(model):
    assert run(model.stop()) is None
    assert refusal(model).command == "stop"
    assert model._stop_playout is False


def test_stop_asks_the_playout_loop_to_cut(model):
    queued = run(model.enqueue(prompt="a", metadata="")).clip
    make_ready(model, queued)
    run(model.play(clip_id=""))

    run(model.stop())
    assert model._stop_playout is True
    # The queues are untouched: stop cuts playout, not the queues.
    assert len(model._playout) == 0  # the played clip had already left it


def test_the_canvas_is_locked_while_clips_exist(model):
    reply = run(model.set_canvas(aspect="9:16"))
    assert (reply.height, reply.width) == clip_plan.canvas_for_choice("9:16")

    run(model.enqueue(prompt="a", metadata=""))
    assert run(model.set_canvas(aspect="1:1")) is None
    assert refusal(model).command == "set_canvas"
    # The refused command had no effect.
    assert model._aspect == "9:16"

    run(model.reset())
    assert run(model.set_canvas(aspect="1:1")) is not None


def test_clip_length_snaps_to_something_generatable(model):
    reply = run(model.set_clip_seconds(seconds=8.3))
    assert reply.frames % 17 == 5
    assert reply.clip_seconds == pytest.approx(reply.frames / 24, abs=1e-3)
    assert model._clip_frames == reply.frames


def test_reset_drops_the_queue_and_restores_every_default(model):
    run(model.set_clip_seconds(seconds=10.0))
    run(model.set_seed(seed=7))
    run(model.enqueue(prompt="a", metadata=""))
    run(model.enqueue(prompt="b", metadata=""))

    reply = run(model.reset())
    assert reply.cleared_clips == 2
    assert reply.was_playing is False
    assert len(model._generation) == 0
    assert len(model._playout) == 0
    assert model._clip_frames == model.config.clip_frames
    assert model._seed == model.config.seed
    assert model._aspect == model.config.aspect


def test_get_queue_reports_the_same_payload_that_is_broadcast(model):
    run(model.enqueue(prompt="a", metadata="m"))
    direct = run(model.get_queue())
    broadcasts = [m for m in model.sent if type(m).__name__ == "QueueUpdate"]
    assert direct.generation == broadcasts[-1].generation
    assert direct.playout == broadcasts[-1].playout
    assert direct.generation[0]["metadata"] == "m"


def test_get_state_reports_the_same_snapshot_that_is_broadcast(model):
    run(model.enqueue(prompt="a", metadata=""))
    model.sent.clear()
    run(model._send_state_update())
    broadcast = model.sent[-1]
    direct = run(model.get_state())
    assert vars(direct) == vars(broadcast)


def test_the_snapshot_publishes_the_live_command_set(model):
    snapshot = run(model.get_state())
    assert "play" not in snapshot.valid_commands  # nothing built yet
    assert "enqueue" in snapshot.valid_commands
    assert snapshot.generation_queued == 0
    assert snapshot.generation_capacity == 3
    assert snapshot.playout_queued == 0
    assert snapshot.playout_capacity == 3
    assert snapshot.playing is False
    assert snapshot.playing_clip_id is None

    queued = run(model.enqueue(prompt="a", metadata="")).clip
    make_ready(model, queued)
    snapshot = run(model.get_state())
    assert "play" in snapshot.valid_commands
    assert snapshot.generation_queued == 0
    assert snapshot.playout_queued == 1


def test_a_refusal_never_masquerades_as_a_reply(model):
    """A handler must return only the type its annotation names.

    `enqueue` is annotated `-> ClipQueued`; a refusal that returned a
    `CommandError` from it would reach the client typed as the message the
    schema promised, with every guaranteed field undefined.
    """
    assert run(model.enqueue(prompt="", metadata="")) is None
    error = refusal(model)
    assert type(error).__name__ == "CommandError"
    assert error.reason


def test_the_runtime_exception_is_never_raised(model):
    """Its correlated failure frame is withheld from v0 clients, so a raise is
    silence for anyone on the 2.x SDK. The broadcast is what reaches them."""
    source = (MODEL_DIR / "fasth3.py").read_text(encoding="utf-8")
    assert "raise CommandError" not in source
    # And the name in scope is the model's own message, not the runtime's.
    from fasth3 import CommandError as in_scope
    from fasth3_types import CommandError as ours

    assert in_scope is ours


# ------------------------------------------------------------ the playout loop
#
# ``_serve`` is the concurrent part of this model — a worker building clips, an
# event loop pacing an armed clip out, and abort paths crossing both. These
# tests replace only the *backend*, so the real queueing, ordering, emission
# and teardown all run.
#
# Clips are shrunk to a few frames so a whole playout runs in well under a
# second; the pacer is a real 24 fps clock, so the numbers below are wall time.

FRAMES_PER_CLIP = 6  # 0.25 s of content at 24 fps


class FakeBackend:
    """Builds tiny clips on demand; instant by default, controllable when not."""

    def __init__(self):
        self.built: list[tuple[int, str, int]] = []
        self.fail_next: Exception | None = None
        self.hold = False
        self.held: list[ClipJob] = []

    def submit(self, *, frames, prompt, seed, height, width) -> ClipJob:
        self.built.append((frames, prompt, seed))
        job = ClipJob(None)
        if self.fail_next is not None:
            job.error, self.fail_next = self.fail_next, None
            job.done.set()
        elif self.hold:
            self.held.append(job)
        else:
            self.finish(job, frames)
        return job

    @staticmethod
    def finish(job: ClipJob, frames: int = FRAMES_PER_CLIP) -> None:
        video = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(frames)]
        samples = np.zeros((1, round(frames / 24 * 48_000)), dtype=np.int16)
        job.result = (video, samples)
        job.done.set()


@pytest.fixture
def live():
    """A FastH3 wired to a fake backend, with a connected audience."""
    instance = FastH3()
    instance._on_loop_ready()
    instance.connected.set()
    instance.config = make_config(queue_size=5)
    instance._reset_session_state()
    instance._clip_frames = FRAMES_PER_CLIP  # tiny clips keep the tests fast
    instance.backend = FakeBackend()

    emitted: list = []

    async def fake_emit(output):
        emitted.append(output)

    instance.emit = fake_emit
    instance.emitted = emitted

    messages: list = []

    async def fake_send(message):
        messages.append(message)

    instance.send = fake_send
    instance.messages = messages

    flushes: list = []
    instance.output.flush = lambda: flushes.append(time.monotonic())
    instance.flushes = flushes
    return instance


def names(messages) -> list[str]:
    return [type(m).__name__ for m in messages]


def drive(live, scenario):
    """Run `_serve` against a scenario coroutine that ends the session."""

    async def main():
        async def wrapped():
            try:
                await scenario()
            finally:
                live.connected.clear()

        await asyncio.gather(live._serve(), wrapped())

    asyncio.run(main())


async def eventually(predicate, timeout=2.0):
    """Wait until `predicate()` is true, failing the test after `timeout`."""
    deadline = time.monotonic() + timeout
    while not predicate():
        assert time.monotonic() < deadline, "condition never became true"
        await asyncio.sleep(0.005)


def test_enqueued_clips_build_in_order_and_turn_ready(live):
    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await live.enqueue(prompt="b", metadata="")
        await eventually(lambda: len(live._playout) == 2)

    drive(live, scenario)
    assert [prompt for _f, prompt, _s in live.backend.built] == ["a", "b"]
    ready = [m for m in live.messages if type(m).__name__ == "QueueUpdate"]
    assert ready, "no queue_update announced the clips turning ready"


def test_a_played_clip_streams_whole_then_holds_on_black(live):
    async def scenario():
        await live.enqueue(prompt="a", metadata="tag")
        await eventually(lambda: len(live._playout) == 1)
        await live.play(clip_id="")
        await eventually(lambda: "ClipFinished" in names(live.messages))
        # No auto-play: nothing else may start on its own.
        await asyncio.sleep(0.15)

    drive(live, scenario)
    # `clip_queued` is the enqueue reply, not a broadcast, so it is not here;
    # `clip_generated` marks the build crossing into the playout queue.
    assert names([m for m in live.messages if "Clip" in type(m).__name__]) == [
        "ClipGenerated",
        "ClipStarted",
        "ClipFinished",
    ]
    frames = sum(output.main_video.shape[0] for output in live.emitted)
    assert frames == FRAMES_PER_CLIP
    assert live.flushes, "the stream must flush to black after the clip"
    assert live._playing is None
    # The full struct rides on both playout messages, metadata included.
    started = next(m for m in live.messages if type(m).__name__ == "ClipStarted")
    finished = next(m for m in live.messages if type(m).__name__ == "ClipFinished")
    assert started.clip["metadata"] == "tag"
    assert started.clip["ready"] is True
    assert finished.clip["clip_id"] == started.clip["clip_id"]


def test_video_and_audio_stay_locked_slice_for_slice(live):
    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        await live.play(clip_id="")
        await eventually(lambda: "ClipFinished" in names(live.messages))

    drive(live, scenario)
    for output in live.emitted:
        video_frames = output.main_video.shape[0]
        audio_samples = output.main_audio.shape[1]
        assert output.main_video.dtype == np.uint8
        assert output.main_audio.dtype == np.int16
        assert output.main_audio.ndim == 2 and output.main_audio.shape[0] == 1
        assert audio_samples == pytest.approx(video_frames * 48_000 / 24, abs=1)


def test_stop_cuts_the_clip_and_keeps_the_queue(live):
    # A two-second clip, so the stop reliably lands mid-play.
    live._clip_frames = 48

    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await live.enqueue(prompt="b", metadata="")
        await eventually(lambda: len(live._playout) == 2)
        await live.play(clip_id="")
        await eventually(lambda: live.emitted)
        await live.stop()
        await eventually(lambda: "ClipStopped" in names(live.messages))

    drive(live, scenario)
    assert "ClipFinished" not in names(live.messages)
    assert live.flushes, "stop must flush the output"
    # The cut clip went out only partially.
    frames = sum(output.main_video.shape[0] for output in live.emitted)
    assert frames < 48
    # The other clip still waits, built, for the next play.
    assert len(live._playout) == 1


def test_builds_continue_while_a_clip_plays(live):
    # A two-second clip, so the enqueue reliably lands mid-play.
    live._clip_frames = 48
    built_during_play = {}

    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        await live.play(clip_id="")
        await eventually(lambda: live.emitted)
        await live.enqueue(prompt="b", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        built_during_play["value"] = live._playing is not None
        await eventually(lambda: "ClipFinished" in names(live.messages))

    drive(live, scenario)
    # The second build was submitted and finished while the first clip streamed.
    assert [prompt for _f, prompt, _s in live.backend.built] == ["a", "b"]
    assert built_during_play["value"] is True
    assert len(live._playout) == 1  # the mid-play build crossed into playout


def test_a_failing_build_reports_and_the_queue_moves_on(live):
    async def scenario():
        live.backend.fail_next = RuntimeError("the engine fell over")
        await live.enqueue(prompt="a", metadata="")
        await live.enqueue(prompt="b", metadata="")
        await eventually(lambda: "ClipFailed" in names(live.messages))
        await eventually(lambda: len(live._playout) == 1)

    drive(live, scenario)
    failed = next(m for m in live.messages if type(m).__name__ == "ClipFailed")
    assert "the engine fell over" in failed.reason
    assert failed.clip["prompt"] == "a"
    # The failed clip left the queues; the survivor crossed into playout.
    assert [clip["prompt"] for clip in live._playout.snapshot()] == ["b"]


def test_reset_discards_a_build_still_in_flight(live):
    async def scenario():
        live.backend.hold = True
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: live.backend.held)
        await live.reset()
        live.backend.finish(live.backend.held[0])
        # The finished build has no entry to land on; nothing may surface.
        await asyncio.sleep(0.15)

    drive(live, scenario)
    assert len(live._generation) == 0
    assert len(live._playout) == 0
    assert "ClipFailed" not in names(live.messages)


def test_the_pacer_holds_24_fps(live):
    elapsed = {}

    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        started = time.monotonic()
        await live.play(clip_id="")
        await eventually(lambda: "ClipFinished" in names(live.messages))
        elapsed["playout"] = time.monotonic() - started

    drive(live, scenario)
    content = FRAMES_PER_CLIP / 24
    # A slice is handed over at the instant its content is due, so the playout
    # ends one slice before the content it queued finishes playing.
    expected = content - EMIT_FRAMES / 24
    # Real-time, not faster: the metronome is what keeps the audio in sync.
    assert elapsed["playout"] >= expected * 0.95
    assert elapsed["playout"] < content + 0.3


def test_autoplay_chains_ready_clips_without_play(live):
    async def scenario():
        await live.set_autoplay(enabled=True)
        await live.enqueue(prompt="a", metadata="")
        await live.enqueue(prompt="b", metadata="")
        # No `play` anywhere in this scenario: both clips must stream on
        # their own, oldest first, once their builds complete.
        await eventually(
            lambda: names(live.messages).count("ClipFinished") == 2, timeout=5.0
        )
        # The queue is drained and nothing else may start.
        await asyncio.sleep(0.15)

    drive(live, scenario)
    started = [m.clip["prompt"] for m in live.messages if type(m).__name__ == "ClipStarted"]
    assert started == ["a", "b"]
    frames = sum(output.main_video.shape[0] for output in live.emitted)
    assert frames == 2 * FRAMES_PER_CLIP
    assert live.flushes, "the stream still flushes to black at each boundary"


def test_without_autoplay_nothing_starts_on_its_own(live):
    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        await asyncio.sleep(0.2)

    drive(live, scenario)
    assert "ClipStarted" not in names(live.messages)
    assert live.emitted == []


def test_a_lost_audience_ends_the_playout_quietly(live):
    # A two-second clip, so the disconnect reliably lands mid-play.
    live._clip_frames = 48

    async def scenario():
        await live.enqueue(prompt="a", metadata="")
        await eventually(lambda: len(live._playout) == 1)
        await live.play(clip_id="")
        await eventually(lambda: live.emitted)
        # The scenario wrapper clears `connected`, which is the audience leaving.

    drive(live, scenario)
    # Nobody was there to hear a finish or a stop for the cut clip.
    assert "ClipFinished" not in names(live.messages)
    assert "ClipStopped" not in names(live.messages)


def test_generation_is_gated_on_an_audience(live):
    """`_serve` is the only submitter, and it runs only with a client connected."""

    async def main():
        await live.enqueue(prompt="a", metadata="")
        live.connected.clear()
        await asyncio.wait_for(live._serve(), timeout=1.0)

    asyncio.run(main())
    assert live.backend.built == []
    assert len(live._playout) == 0


# ---------------------------------------------------------- published contract
#
# ``reactor schema`` compiles this document out of the model class, and
# generated SDKs are built from nothing else. A change here is a change to every
# client, so these tests exist to make an accidental one fail loudly and a
# deliberate one obvious in review — together with the version bump it needs.

# Every command a client can send, and the message each answers with. `None`
# means the command is answered with a bare acknowledgement.
EXPECTED_COMMANDS = {
    "enqueue": "ClipQueued",
    "get_queue": "QueueUpdate",
    "get_state": "StateUpdate",
    "move": "ClipMoved",
    "play": None,
    "pop": "ClipPopped",
    "reset": "SessionReset",
    "set_autoplay": "AutoplayAccepted",
    "set_canvas": "CanvasAccepted",
    "set_clip_seconds": "ClipLengthAccepted",
    "set_continuity": "ContinuityAccepted",
    "set_prompt": "PromptAccepted",
    "set_seed": "SeedAccepted",
    "set_seed_image": "SeedImageAccepted",
    "stop": None,
}

EXPECTED_MESSAGES = {
    "autoplay_accepted",
    "canvas_accepted",
    "clip_failed",
    "clip_finished",
    "clip_generated",
    "clip_length_accepted",
    "clip_moved",
    "clip_popped",
    "clip_queued",
    "clip_started",
    "clip_stopped",
    "command_error",
    "continuity_accepted",
    "prompt_accepted",
    "queue_update",
    "seed_accepted",
    "seed_image_accepted",
    "session_reset",
    "state_update",
}

# Commands that can be refused. Each one has to say so in its own summary, and
# name the message a client will actually receive.
EXPECTED_REJECTIONS = (
    "enqueue", "move", "play", "pop", "stop", "set_canvas", "set_prompt",
    "set_seed_image", "set_continuity",
)

# The struct every clip-referencing message embeds, and its JSON types.
EXPECTED_CLIP_INFO = {
    "clip_id": "string",
    "prompt": "string",
    "metadata": "string",
    "frames": "integer",
    "seconds": "number",
    "seed": "integer",
    "ready": "boolean",
}


@pytest.fixture(scope="module")
def schema():
    """Render the document exactly as the release pipeline does.

    The renderer imports the model without loading it, so this needs no weights
    and no GPU — but it does need the model's own imports to stay light, which
    is itself part of what this asserts.
    """
    from reactor_runtime.schema import render

    return render(MODEL_DIR, version="v0.2.0")


def test_the_model_publishes_two_outbound_tracks(schema):
    tracks = schema["x-reactor"]["tracks"]
    assert [(t["name"], t["kind"], t["direction"]) for t in tracks] == [
        ("main_video", "video", "out"),
        ("main_audio", "audio", "out"),
    ]


def test_the_command_set_is_exactly_what_clients_expect(schema):
    published = {path.removeprefix("/events/") for path in schema["paths"]}
    assert published == set(EXPECTED_COMMANDS)


def test_every_command_answers_with_the_type_it_promises(schema):
    for name, message in EXPECTED_COMMANDS.items():
        operation = schema["paths"][f"/events/{name}"]["post"]
        responses = operation["responses"]
        if message is None:
            # A handler that returns nothing is answered with a bare 202, so an
            # awaiting client still resolves — it just learns nothing.
            assert set(responses) == {"202"}, name
            continue
        body = responses["200"]["content"]["application/json"]["schema"]
        assert body["$ref"] == f"#/components/schemas/{message}", name


def test_every_message_is_published_once(schema):
    assert set(schema["webhooks"]) == EXPECTED_MESSAGES
    for message in EXPECTED_COMMANDS.values():
        if message is not None:
            assert message in schema["components"]["schemas"]


def test_every_clip_message_embeds_the_full_struct(schema):
    """`ClipInfo` rides whole on every message that references a clip."""
    for message in (
        "ClipQueued", "ClipGenerated", "ClipMoved", "ClipStarted",
        "ClipFinished", "ClipStopped", "ClipFailed", "ClipPopped",
    ):
        clip = schema["components"]["schemas"][message]["properties"]["clip"]
        rendered = {
            name: field["type"] for name, field in clip["properties"].items()
        }
        assert rendered == EXPECTED_CLIP_INFO, message
        assert set(clip.get("required", [])) == set(EXPECTED_CLIP_INFO), message
    # And both queues report lists of the same struct.
    for queue in ("generation", "playout"):
        items = schema["components"]["schemas"]["QueueUpdate"]["properties"][queue]["items"]
        assert {name: field["type"] for name, field in items["properties"].items()} == (
            EXPECTED_CLIP_INFO
        ), queue


def test_free_text_fields_are_marked_for_moderation(schema):
    """`enqueue` carries client free text into generated video and audio."""
    properties = schema["paths"]["/events/enqueue"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]
    assert properties["prompt"]["x-reactor-moderate"] is True
    assert properties["metadata"]["x-reactor-moderate"] is True


def test_the_enqueue_seed_and_length_are_optional_on_the_wire(schema):
    """Omitted or null means the session's defaults."""
    properties = schema["paths"]["/events/enqueue"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]
    for name in ("seed", "seconds"):
        field = properties[name]
        types = field.get("anyOf") or [field]
        assert any(entry.get("type") == "null" for entry in types), (name, field)


def test_the_clip_length_bounds_a_client_reads_are_generatable(schema):
    seconds = schema["paths"]["/events/set_clip_seconds"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]["seconds"]
    assert seconds["minimum"] == clip_plan.MIN_SECONDS_PUBLISHED
    assert seconds["maximum"] == clip_plan.MAX_SECONDS_PUBLISHED
    for bound in (seconds["minimum"], seconds["maximum"]):
        assert clip_plan.frames_for_seconds(bound) % 17 == 5


def test_the_canvas_choices_are_published_as_an_enum(schema):
    aspect = schema["paths"]["/events/set_canvas"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]["aspect"]
    assert aspect["enum"] == list(clip_plan.ASPECT_CHOICES)


def test_every_client_facing_string_is_documented(schema):
    """A frontend developer who cannot read this repo works from these alone."""
    for path, operations in schema["paths"].items():
        assert operations["post"].get("summary"), f"{path} has no description"
        body = operations["post"].get("requestBody")
        if not body:
            continue
        properties = body["content"]["application/json"]["schema"]["properties"]
        for name, field in properties.items():
            assert field.get("description"), f"{path} parameter {name} has no description"
    for name, message in schema["webhooks"].items():
        assert message["post"].get("summary"), f"message {name} has no description"
    # Only the messages this model declares. The runtime contributes its own
    # components (the upload reference, for one), and those are not ours to
    # document.
    ours = {
        message["post"]["requestBody"]["content"]["application/json"]["schema"]["$ref"].rsplit(
            "/", 1
        )[-1]
        for message in schema["webhooks"].values()
    }
    assert ours, "no message definitions were resolved from the webhooks"
    for name in ours:
        definition = schema["components"]["schemas"][name]
        for field_name, field in definition.get("properties", {}).items():
            assert field.get("description"), f"{name}.{field_name} has no description"


def test_every_message_summary_says_when_it_is_emitted(schema):
    """The house style: a message docstring opens with "Emitted ..."."""
    for name, message in schema["webhooks"].items():
        summary = message["post"]["summary"]
        assert summary.startswith("Emitted "), f"{name}: {summary!r}"


def test_no_message_name_repeats_the_model_name(schema):
    """The SDK client already identifies the model; a prefix is dead weight."""
    forbidden = ("fasth3", "fast-h3", "fast_h3", "fasth-3")
    for name in schema["webhooks"]:
        assert not name.lower().startswith(forbidden), name
    for name in schema["components"]["schemas"]:
        assert not name.lower().startswith(forbidden), name


def test_every_command_summary_names_what_it_emits(schema):
    for name in EXPECTED_COMMANDS:
        summary = schema["paths"][f"/events/{name}"]["post"]["summary"]
        if name == "get_state":
            # A pure read: it answers, it does not emit.
            assert "state_update" in summary
            continue
        if name == "get_queue":
            assert "queue_update" in summary
            continue
        if name == "move":
            # A move changes order, not state: the snapshot carries counts
            # and conditions, none of which a reposition touches.
            assert "queue_update" in summary
            continue
        assert "`state_update`" in summary, f"{name} does not say it broadcasts a snapshot"


def test_every_refusable_command_documents_its_failure(schema):
    for name in EXPECTED_REJECTIONS:
        summary = schema["paths"][f"/events/{name}"]["post"]["summary"]
        assert "`command_error`" in summary, f"{name} does not document its failure"


def test_the_idle_stream_id_is_null_not_empty(schema):
    """`None` is the no-clip value on the wire; an empty string would be ambiguous."""
    playing = schema["components"]["schemas"]["StateUpdate"]["properties"]["playing_clip_id"]
    types = playing.get("anyOf") or [playing]
    assert any(entry.get("type") == "null" for entry in types), playing


def test_clip_length_prose_matches_the_published_bounds(schema):
    """The numbers in the description are generated from the same constants."""
    summary = schema["paths"]["/events/set_clip_seconds"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["properties"]["seconds"]["description"]
    assert f"{clip_plan.MIN_SECONDS_PUBLISHED:g}" in summary
    assert f"{clip_plan.MAX_SECONDS_PUBLISHED:g}" in summary


# ------------------------------------------------------------------- manifest
#
# The workspace rules from GUIDELINES.md, checked here rather than discovered
# during a build.

# A stray checkpoint in the folder is a large, slow mistake to discover later.
WEIGHT_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}


@pytest.fixture(scope="module")
def manifest() -> dict:
    return yaml.safe_load((MODEL_DIR / "reactor.yaml").read_text(encoding="utf-8"))


def test_the_model_name_matches_the_folder(manifest):
    """The folder is the workspace, and the name is the published slug."""
    assert manifest["model"]["name"] == MODEL_DIR.name


def test_the_version_is_bare_semver(manifest):
    """The platform's release tag format: no `v` prefix, ever."""
    version = manifest["model"]["version"]
    assert isinstance(version, str), "quote the version so it is not parsed as a number"
    assert re.fullmatch(r"\d+\.\d+\.\d+", version), f"{version!r} must be bare semver"


def test_the_manifest_carries_a_complete_resource_spec(manifest):
    resources = manifest["model"]["resources"]
    assert resources["gpu"]["type"] and resources["gpu"]["count"] >= 1
    assert resources["cpu"]["request"] and resources["cpu"]["limit"]
    assert resources["memory"]["request"] and resources["memory"]["limit"]


def test_the_image_is_built_from_the_manifest_not_a_dockerfile(manifest):
    """`reactor build` owns the image; a Dockerfile here would be ignored."""
    assert not (MODEL_DIR / "Dockerfile").exists()
    build = manifest["build"]
    assert build["python_requirements"] == "requirements.txt"
    assert (MODEL_DIR / build["python_requirements"]).is_file()


def test_the_config_the_runtime_hands_to_load_exists(manifest):
    config = manifest["runtime"]["config"]
    assert (MODEL_DIR / config).is_file(), f"runtime.config points at a missing {config}"


def test_the_runtime_pin_is_current(manifest):
    assert manifest["build"]["runtime_version"] == "3.2.6"


def test_the_runtime_release_is_pinned_once(manifest):
    """`build.runtime_version` owns it; a second pin would let the two drift."""
    assert "reactor-runtime" not in (MODEL_DIR / "requirements.txt").read_text(encoding="utf-8")


def test_the_runtime_import_resolves_to_the_model_class(manifest):
    module_name, _, class_name = manifest["runtime"]["import"].partition(":")
    module = __import__(module_name)
    assert getattr(module, class_name, None) is not None, f"{module_name}.py has no {class_name}"


def test_no_weights_are_committed_alongside_the_model():
    offenders = [
        path.relative_to(MODEL_DIR)
        for path in MODEL_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in WEIGHT_SUFFIXES
        # Hidden directories (.venv, .git) are not part of the commit.
        and not any(part.startswith(".") for part in path.relative_to(MODEL_DIR).parts)
    ]
    assert not offenders, f"weights never live in git: {offenders}"


# --------------------------------------------------------------- sitecustomize
#
# The interpreter-wide fixes every container process loads via PYTHONPATH.


def test_the_vsa_arch_gate_widens_to_every_built_capability(monkeypatch):
    """The widened `_SM100` accepts exactly the arches the image compiles.

    The kernel's gate is `get_device_capability(...) != _SM100`; the patch
    must make that comparison pass for each capability in the build's
    TORCH_CUDA_ARCH_LIST and keep refusing everything else.
    """
    import sys
    import types

    import sitecustomize

    fake = types.ModuleType("fastvideo_kernel.block_sparse_attn_sm100a")
    fake._SM100 = (10, 0)
    monkeypatch.setitem(sys.modules, "fastvideo_kernel.block_sparse_attn_sm100a", fake)

    sitecustomize._widen_vsa_arch_gate()
    accepted = [cap for cap in [(9, 0), (10, 0), (10, 3), (12, 0)] if not (cap != fake._SM100)]
    assert accepted == [(10, 0), (10, 3)]

    # Idempotent: a second pass leaves the widened gate intact.
    sitecustomize._widen_vsa_arch_gate()
    assert not ((10, 3) != fake._SM100)


def test_the_built_capabilities_mirror_the_manifest_arch_list(manifest):
    """sitecustomize's capability set and TORCH_CUDA_ARCH_LIST move together."""
    import sitecustomize

    arch_list = manifest["build"]["build_env"]["TORCH_CUDA_ARCH_LIST"]
    from_manifest = {
        tuple(int(part) for part in entry.rstrip("a").split("."))
        for entry in arch_list.split(";")
    }
    assert set(sitecustomize._VSA_BUILT_CAPABILITIES) == from_manifest


# --------------------------------------------------------------------------
# Continuity mode: the 640 canvas tier, the disjoint command surface, the
# exposure lock and seam arithmetic, and the held-prompt handlers. The queue's
# hard-cut path above is unchanged — every test there runs with continuity off.


# -- the resolution tier ---------------------------------------------------


def test_the_default_short_edge_is_the_measured_tier():
    """A zero-argument resolve is the 768 tier every published number used."""
    assert clip_plan.resolve_short_edge(None) == 768
    assert clip_plan.canvas_for_choice("16:9") == (768, 1344)


def test_a_lower_short_edge_selects_a_smaller_canvas():
    """640 is a real, selectable tier: a true 16:9 rounds to 1152x640."""
    assert clip_plan.canvas_for_choice("16:9", 640) == (640, 1152)


@pytest.mark.parametrize("aspect", clip_plan.ASPECT_CHOICES)
def test_every_aspect_resolves_at_the_640_tier(aspect):
    height, width = clip_plan.canvas_for_choice(aspect, 640)
    assert min(height, width) == 640
    assert height % 32 == 0 and width % 32 == 0
    assert height * width <= 768 * 1344


def test_short_edge_must_be_a_valid_tier():
    for bad in (700, 200, 800, 0, -32):
        with pytest.raises(ValueError):
            clip_plan.resolve_short_edge(bad)
    for good in (256, 640, 768):
        assert clip_plan.resolve_short_edge(good) == good


# -- config parsing --------------------------------------------------------


def test_the_continuity_config_parses(tmp_path):
    document = (
        "inference:\n"
        "  continuity: true\n"
        "  canvas_short_edge: 640\n"
        "  continuity_clip_seconds: 5.167\n"
        "  seam_frames: 12\n"
    )
    path = tmp_path / "continuity.yaml"
    path.write_text(document, encoding="utf-8")
    config = load_config(path)
    assert config.continuity is True
    assert config.canvas_short_edge == 640
    assert config.seam_frames == 12
    assert config.continuity_clip_frames == clip_plan.MIN_FRAMES
    # Continuity holds one length, so the warm-up warms exactly it.
    assert config.warmup_frames == (clip_plan.MIN_FRAMES,)


def test_continuity_defaults_off_at_the_768_tier(tmp_path):
    """A config without the continuity keys is the unchanged hard-cut queue."""
    path = tmp_path / "plain.yaml"
    path.write_text("inference: {}\n", encoding="utf-8")
    config = load_config(path)
    assert config.continuity is False
    assert config.canvas_short_edge == 768


def test_a_too_wide_seam_fails_startup(tmp_path):
    path = tmp_path / "seam.yaml"
    path.write_text(
        "inference:\n  continuity_clip_seconds: 5.167\n  seam_frames: 200\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        load_config(path)


def test_a_bad_short_edge_fails_startup(tmp_path):
    path = tmp_path / "edge.yaml"
    path.write_text("inference:\n  canvas_short_edge: 700\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


# -- the disjoint command surface -----------------------------------------


def test_continuity_offers_the_prompt_surface_not_the_queue():
    idle = session_rules.valid_commands(
        playing=False, generation_queued=0, generation_capacity=0,
        playout_queued=0, continuity=True, prompt_set=False,
    )
    assert "set_prompt" in idle and "set_canvas" in idle
    # The take can be seeded from an uploaded image (I2V) before it starts.
    assert "set_seed_image" in idle
    # Idle: the session may be switched to the hard-cut queue.
    assert "set_continuity" in idle
    assert not ({"enqueue", "play", "move", "pop", "get_queue", "set_autoplay"} & set(idle))

    running = session_rules.valid_commands(
        playing=True, generation_queued=0, generation_capacity=0,
        playout_queued=0, continuity=True, prompt_set=True,
    )
    assert "stop" in running
    # The canvas is fixed, a seed can no longer be set, and the mode can no
    # longer be switched once the take runs.
    assert "set_canvas" not in running
    assert "set_seed_image" not in running
    assert "set_continuity" not in running


def test_the_queue_surface_never_offers_set_prompt():
    commands = session_rules.valid_commands(
        playing=False, generation_queued=0, generation_capacity=10, playout_queued=0,
    )
    assert "enqueue" in commands and "set_prompt" not in commands


def test_set_continuity_is_offered_only_while_the_queue_is_idle():
    """The mode switch shares set_canvas's idle gate — empty queues, nothing playing."""
    idle = session_rules.valid_commands(
        playing=False, generation_queued=0, generation_capacity=10, playout_queued=0,
    )
    assert "set_continuity" in idle
    # A queued clip fixes the mode until it drains, exactly like the canvas.
    with_clip = session_rules.valid_commands(
        playing=False, generation_queued=1, generation_capacity=10, playout_queued=0,
    )
    assert "set_continuity" not in with_clip
    playing = session_rules.valid_commands(
        playing=True, generation_queued=0, generation_capacity=10, playout_queued=0,
    )
    assert "set_continuity" not in playing


# -- the exposure lock (the committed float64 fix) -------------------------


def test_the_exposure_lock_pins_a_clip_to_the_reference():
    """color_match shifts a whole clip so its per-channel mean is the target."""
    rng = np.random.default_rng(0)
    clip = rng.integers(0, 200, size=(6, 40, 40, 3), dtype=np.uint8)
    target = np.array([120.0, 130.0, 140.0], np.float32)
    matched = seam.color_match_to_reference(clip, target)
    got = matched.reshape(-1, 3).mean(axis=0, dtype=np.float64)
    # One integer offset for the whole clip, then a uint8 clamp: within a level.
    assert np.allclose(got, target, atol=1.0)


def test_the_reference_is_computed_in_float64():
    """A wide, high-valued frame's mean must not collapse the way float32 does."""
    frame = np.full((4000, 4000, 3), 200, np.uint8)
    reference = seam.reference_rgb(frame)
    assert np.allclose(reference, 200.0, atol=1e-3)
    # The float32 mantissa-saturating reduction the bug used would land far off.
    assert reference.dtype == np.float32 and float(reference[0]) > 199.0


def test_the_seam_blend_has_no_midpoint_flash():
    """Complementary linear-light weights: a constant field blends to itself."""
    tail = np.full((12, 16, 16, 3), 128, np.uint8)
    head = np.full((12, 16, 16, 3), 128, np.uint8)
    blended = seam.blend_video_linear(tail, head)
    assert blended.shape == (12, 16, 16, 3)
    assert np.abs(blended.astype(int) - 128).max() <= 1


# -- seam stitch arithmetic (on the worker, GPU path falls back to numpy) ---


def continuity_backend(seam_frames=12):
    config = make_config(continuity=True, canvas_short_edge=640, seam_frames=seam_frames)
    backend = FastH3Backend(config, Path("."))
    backend.reset_continuity()
    return backend


def test_the_seam_removes_one_overlap_per_boundary():
    backend = continuity_backend(seam_frames=12)
    n, k = 124, 12
    spf = 48_000 / 24

    def clip(value):
        frames = [np.full((32, 32, 3), value, np.uint8) for _ in range(n)]
        audio = np.zeros((1, round(n * spf)), np.int16)
        return frames, audio

    first_frames, first_audio = clip(100)
    emit0, audio0 = backend._stitch_seam(first_frames, first_audio, k)
    # Clip 0 opens with its tail held back for the next boundary.
    assert len(emit0) == n - k
    assert audio0.shape[-1] == round((n - k) * spf)

    second_frames, second_audio = clip(150)
    emit1, audio1 = backend._stitch_seam(second_frames, second_audio, k)
    # Clip 1: k blended frames + the untouched middle = n - k again.
    assert len(emit1) == n - k
    assert audio1.shape[-1] == round((n - k) * spf)


# -- the held-prompt handlers ----------------------------------------------


@pytest.fixture
def continuity_model():
    """A continuity-mode FastH3 with the state ``load()`` would set, no engine."""
    instance = FastH3()
    instance._on_loop_ready()
    instance.config = make_config(continuity=True, canvas_short_edge=640)
    instance._reset_session_state()
    sent: list = []

    async def capture(message):
        sent.append(message)

    instance.send = capture
    instance.sent = sent
    return instance


def test_set_prompt_holds_the_prompt_and_reanchors(continuity_model):
    reply = run(continuity_model.set_prompt(prompt="a misty forest", metadata="m"))
    assert reply.prompt == "a misty forest"
    assert continuity_model._prompt == "a misty forest"
    epoch = continuity_model._prompt_epoch
    run(continuity_model.set_prompt(prompt="a city at night", metadata=""))
    # A changed prompt advances the epoch, the run loop's re-anchor signal.
    assert continuity_model._prompt == "a city at night"
    assert continuity_model._prompt_epoch == epoch + 1


def test_set_prompt_needs_a_prompt(continuity_model):
    assert run(continuity_model.set_prompt(prompt="   ", metadata="")) is None
    assert refusal(continuity_model).command == "set_prompt"
    assert continuity_model._prompt == ""


def test_the_queue_commands_are_refused_in_continuity(continuity_model):
    assert run(continuity_model.enqueue(prompt="p", metadata="")) is None
    assert refusal(continuity_model).command == "enqueue"
    assert run(continuity_model.play(clip_id="")) is None
    assert refusal(continuity_model).command == "play"
    assert run(continuity_model.set_autoplay(enabled=True)) is None
    assert refusal(continuity_model).command == "set_autoplay"


def test_set_prompt_is_refused_in_the_queue_mode(model):
    assert run(model.set_prompt(prompt="p", metadata="")) is None
    assert refusal(model).command == "set_prompt"


def test_the_continuity_snapshot_carries_the_prompt_and_canvas(continuity_model):
    run(continuity_model.set_prompt(prompt="a lighthouse", metadata=""))
    continuity_model._channel_running = True
    state = run(continuity_model.get_state())
    assert state.continuity is True
    assert state.prompt == "a lighthouse"
    assert (state.height, state.width) == (640, 1152)
    assert "set_prompt" in state.valid_commands
    assert "enqueue" not in state.valid_commands


def test_stop_needs_a_running_take(continuity_model):
    assert run(continuity_model.stop()) is None
    assert refusal(continuity_model).command == "stop"
    continuity_model._channel_running = True
    run(continuity_model.stop())
    assert continuity_model._stop_channel is True


def test_stop_ends_the_take_and_drops_the_prompt(continuity_model):
    # Stop must clear the held prompt, not just cut the current clip: otherwise
    # the run loop re-anchors a fresh take on the still-held prompt and the
    # stream never actually stops.
    run(continuity_model.set_prompt(prompt="a harbour at dusk", metadata="m"))
    continuity_model._channel_running = True
    run(continuity_model.stop())
    assert continuity_model._stop_channel is True
    assert continuity_model._prompt == ""
    assert continuity_model._prompt_metadata == ""


def test_reset_drops_the_held_prompt(continuity_model):
    run(continuity_model.set_prompt(prompt="a river", metadata=""))
    continuity_model._channel_running = True
    reply = run(continuity_model.reset())
    assert reply.was_playing is True
    assert continuity_model._prompt == ""
    assert continuity_model._stop_channel is True


def test_the_canvas_is_locked_while_a_prompt_drives_the_take(continuity_model):
    run(continuity_model.set_prompt(prompt="a dune", metadata=""))
    assert run(continuity_model.set_canvas(aspect="1:1")) is None
    assert refusal(continuity_model).command == "set_canvas"
    # Cleared, the canvas is selectable again.
    run(continuity_model.reset())
    reply = run(continuity_model.set_canvas(aspect="1:1"))
    assert (reply.height, reply.width) == (640, 640)


# -- runtime mode toggle (set_continuity) -----------------------------------


def test_set_continuity_switches_an_idle_take_to_the_hard_cut_queue(continuity_model):
    """An idle continuity session flips to the queue: length + surface follow."""
    assert continuity_model._continuity is True
    reply = run(continuity_model.set_continuity(enabled=False))
    assert reply.continuity is False
    assert continuity_model._continuity is False
    # The queue's default length replaces the continuity length.
    assert continuity_model._clip_frames == continuity_model.config.clip_frames
    state = run(continuity_model.get_state())
    assert state.continuity is False
    assert "enqueue" in state.valid_commands
    assert "set_prompt" not in state.valid_commands
    # And the queue commands now actually run instead of being refused.
    assert run(continuity_model.enqueue(prompt="a", metadata="")) is not None


def test_set_continuity_switches_an_idle_queue_to_continuity(model):
    """An idle queue session flips to the take: length + surface follow."""
    assert model._continuity is False
    reply = run(model.set_continuity(enabled=True))
    assert reply.continuity is True
    assert model._continuity is True
    assert model._clip_frames == model.config.continuity_clip_frames
    state = run(model.get_state())
    assert state.continuity is True
    assert "set_prompt" in state.valid_commands
    assert "enqueue" not in state.valid_commands
    # set_prompt, refused a moment ago in the queue, now drives the stream.
    assert run(model.set_prompt(prompt="a misty forest", metadata="")) is not None


def test_set_continuity_is_refused_while_a_take_runs(continuity_model):
    run(continuity_model.set_prompt(prompt="a harbour", metadata=""))
    continuity_model._channel_running = True
    assert run(continuity_model.set_continuity(enabled=False)) is None
    assert refusal(continuity_model).command == "set_continuity"
    # The mode is unchanged: the switch never straddles a running take.
    assert continuity_model._continuity is True


def test_set_continuity_is_refused_while_clips_are_queued(model):
    run(model.enqueue(prompt="a", metadata=""))
    assert run(model.set_continuity(enabled=True)) is None
    assert refusal(model).command == "set_continuity"
    assert model._continuity is False


def test_set_continuity_is_an_idempotent_ack_in_the_same_mode(continuity_model):
    """Requesting the mode already in force is a clean no-op, not a refusal."""
    reply = run(continuity_model.set_continuity(enabled=True))
    assert reply.continuity is True
    assert continuity_model._continuity is True
    assert refusal(continuity_model) is None


def test_the_queue_serve_loop_hands_off_when_the_mode_flips(model):
    """`set_continuity(true)` while idle lets `_serve` return so run() re-dispatches."""

    async def main():
        model.connected.set()
        model._continuity = True
        # If the guard did not pick up the flip, this would spin until timeout.
        await asyncio.wait_for(model._serve(), timeout=1.0)

    asyncio.run(main())


def test_the_continuity_serve_loop_hands_off_when_the_mode_flips(continuity_model):
    """`set_continuity(false)` while idle lets `_serve_continuity` return likewise."""

    async def main():
        continuity_model.connected.set()
        continuity_model._continuity = False
        await asyncio.wait_for(continuity_model._serve_continuity(), timeout=1.0)

    asyncio.run(main())


# -- seeding the take from an uploaded image, I2V (PR#2) --------------------
#
# The pure decoder is exercised on real bytes; the handler is exercised with a
# constructed UploadedFile, so both the decode and the seed lifecycle run on a
# laptop without a GPU. Video is deliberately unsupported: a client extracts the
# frame it wants and sends it as an image.


def _png_bytes(rgb: tuple[int, int, int], size=(48, 64)) -> bytes:
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, rgb).save(buf, format="PNG")
    return buf.getvalue()


def _upload(data: bytes, mime_type: str, name: str):
    from reactor_runtime import UploadedFile

    return UploadedFile(name=name, mime_type=mime_type, data=data)


def test_the_decoder_reads_a_still_image():
    import fasth3_image as image

    frame = image.decode_seed_frame(_png_bytes((10, 120, 240)), "image/png", "s.png")
    # PIL size is (width, height); the array comes back (height, width, 3).
    assert frame.shape == (64, 48, 3) and frame.dtype == np.uint8


def test_the_decoder_rejects_video_with_a_pointer_to_the_image_path():
    import fasth3_image as image

    # A video is refused before any decode, by mime and by extension, with a
    # message that names the supported path (send a frame as an image).
    for mime, name in [("video/mp4", "clip.mp4"), ("", "clip.mov")]:
        with pytest.raises(image.SeedDecodeError, match="as an image"):
            image.decode_seed_frame(b"\x00\x00\x00\x18ftyp", mime, name)


def test_the_decoder_refuses_junk():
    import fasth3_image as image

    for data, mime, name in [
        (b"", "image/png", "a.png"),
        (b"not-an-image", "image/png", "a.png"),
        (b"hello", "text/plain", "a.txt"),
    ]:
        with pytest.raises(image.SeedDecodeError):
            image.decode_seed_frame(data, mime, name)


def test_set_seed_image_fits_an_image_to_the_canvas(continuity_model):
    reply = run(
        continuity_model.set_seed_image(
            image=_upload(_png_bytes((200, 30, 30)), "image/png", "still.png")
        )
    )
    assert reply.filename == "still.png"
    # Fitted to the 640 tier's 16:9 canvas the fixture loads.
    assert (reply.height, reply.width) == continuity_model._canvas()
    seed = continuity_model._seed_anchor
    assert seed is not None and seed.dtype == np.uint8
    assert seed.shape == (reply.height, reply.width, 3)


def test_set_seed_image_refuses_a_video_upload(continuity_model):
    assert (
        run(
            continuity_model.set_seed_image(
                image=_upload(b"\x00\x00\x00\x18ftyp", "video/mp4", "clip.mp4")
            )
        )
        is None
    )
    assert refusal(continuity_model).command == "set_seed_image"
    assert continuity_model._seed_anchor is None


def test_set_seed_image_refuses_a_non_image_upload(continuity_model):
    assert (
        run(
            continuity_model.set_seed_image(
                image=_upload(b"junk", "application/pdf", "a.pdf")
            )
        )
        is None
    )
    assert refusal(continuity_model).command == "set_seed_image"
    assert continuity_model._seed_anchor is None


def test_set_seed_image_is_refused_while_a_take_runs(continuity_model):
    continuity_model._channel_running = True
    assert (
        run(
            continuity_model.set_seed_image(
                image=_upload(_png_bytes((1, 2, 3)), "image/png", "s.png")
            )
        )
        is None
    )
    assert refusal(continuity_model).command == "set_seed_image"


def test_set_seed_image_is_refused_in_the_queue_mode(model):
    assert (
        run(model.set_seed_image(image=_upload(_png_bytes((1, 2, 3)), "image/png", "s.png")))
        is None
    )
    assert refusal(model).command == "set_seed_image"


def test_reset_and_stop_drop_the_seed(continuity_model):
    run(
        continuity_model.set_seed_image(
            image=_upload(_png_bytes((9, 9, 9)), "image/png", "s.png")
        )
    )
    assert continuity_model._seed_anchor is not None
    run(continuity_model.reset())
    assert continuity_model._seed_anchor is None

    # And a stop mid-take clears any seed too.
    run(
        continuity_model.set_seed_image(
            image=_upload(_png_bytes((9, 9, 9)), "image/png", "s.png")
        )
    )
    continuity_model._channel_running = True
    run(continuity_model.stop())
    assert continuity_model._seed_anchor is None
