"""Config parsing and weights-bundle validation for FastH3.

``fasth3.yaml`` is read here and nowhere else: ``load_config`` turns it into
one validated :class:`FastH3Config`, and ``require_weights`` fails startup
loudly when the bundle on disk is incomplete. Pure file and dict work — no
torch, no fastvideo — so the schema renders and the tests run on any machine.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import fasth3_clip_plan as clip_plan

# The HF snapshot directory inside the weights bundle.
DEFAULT_CHECKPOINT_DIR = "FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree"

# Component directories the T2VA pipeline loads. An incomplete bundle must kill
# startup, not surface as a loader traceback on the first clip.
REQUIRED_COMPONENTS = (
    "transformer",
    "text_encoder",
    "tokenizer",
    "processor",
    "vae",
    "audio_vae",
    "scheduler",
    "audio_scheduler",
)


@dataclass(frozen=True)
class FastH3Config:
    """Everything ``fasth3.yaml`` configures, validated once at load.

    The session-level fields are the defaults a fresh session starts from and
    the queue's fixed capacity. ``inference`` and ``runtime`` are the raw
    blocks; the backend reads its engine knobs (attention kernels, compile
    flags, parallelism, offload policy) straight from them.
    """

    aspect: str
    clip_frames: int
    seed: int
    num_inference_steps: int
    queue_size: int
    generation_queue_size: int
    warmup_aspects: tuple[str, ...]
    warmup_frames: tuple[int, ...]
    # Resolution tier the canvas resolves at. 640 (the default) trades pixels
    # for a proportionally faster build, which is the headroom continuity's
    # gap-free chain needs; 768 is the measured full-quality tier.
    canvas_short_edge: int
    # Continuity mode: off is the client-driven hard-cut queue above; on is a
    # self-continuing single-prompt channel that FL2VA-anchors each clip on the
    # previous one's last frame and crossfades the boundary into one stream.
    continuity: bool
    # The clip length continuity holds to (shorter than the queue default: a
    # single-still FL2VA anchor re-anchors more often and drifts less), and the
    # crossfade overlap width in frames. Both are ignored when continuity is off.
    continuity_clip_frames: int
    seam_frames: int
    inference: dict[str, Any]
    runtime: dict[str, Any]


def load_config(config_path: Path | None) -> FastH3Config:
    """Parse ``fasth3.yaml`` into a validated :class:`FastH3Config`.

    Args:
        config_path: Path the runtime hands over from ``runtime.config`` in
            ``reactor.yaml``, or ``None`` when the manifest names no config.

    Raises:
        ValueError: If the configured aspect is not one this model offers, or
            the queue size is not positive.
    """
    document: dict[str, Any] = {}
    if config_path is not None:
        document = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    inference: dict[str, Any] = document.get("inference") or {}
    runtime: dict[str, Any] = document.get("runtime") or {}

    aspect = str(inference.get("aspect", "16:9"))
    if aspect not in clip_plan.ASPECT_CHOICES:
        raise ValueError(
            f"inference.aspect must be one of {list(clip_plan.ASPECT_CHOICES)}, got {aspect!r}"
        )

    queue_size = int(inference.get("queue_size", 10))
    if queue_size < 1:
        raise ValueError(f"inference.queue_size must be positive, got {queue_size}")

    generation_queue_size = int(inference.get("generation_queue_size", 20))
    if generation_queue_size < 1:
        raise ValueError(
            f"inference.generation_queue_size must be positive, got {generation_queue_size}"
        )

    clip_frames = clip_plan.frames_for_seconds(
        float(inference.get("clip_seconds", clip_plan.MAX_SECONDS))
    )

    # Absent selects the trained 768 tier, so a config without the key — every
    # hard-cut deployment — is unchanged; the shipped continuity config sets 640.
    try:
        canvas_short_edge = clip_plan.resolve_short_edge(inference.get("canvas_short_edge"))
    except ValueError as error:
        raise ValueError(f"inference.canvas_short_edge is invalid: {error}") from None

    continuity = bool(inference.get("continuity", False))
    continuity_clip_frames = clip_plan.frames_for_seconds(
        float(inference.get("continuity_clip_seconds", clip_plan.MIN_SECONDS))
    )
    seam_frames = int(inference.get("seam_frames", 12))
    if seam_frames < 0:
        raise ValueError(f"inference.seam_frames must not be negative, got {seam_frames}")
    if 2 * seam_frames > continuity_clip_frames:
        raise ValueError(
            f"inference.seam_frames ({seam_frames}) is too wide: two seam overlaps "
            f"must fit one continuity clip ({continuity_clip_frames} frames)."
        )

    return FastH3Config(
        aspect=aspect,
        clip_frames=clip_frames,
        seed=int(inference.get("seed", 1000)),
        # Sigma-grid POINTS, not transformer forwards: the distilled schedule is
        # five points and exactly four forwards.
        num_inference_steps=int(inference.get("num_inference_steps", 5)),
        queue_size=queue_size,
        generation_queue_size=generation_queue_size,
        warmup_aspects=tuple(str(a) for a in (inference.get("warmup_aspects") or [aspect])),
        warmup_frames=_parse_warmup_lengths(
            inference.get("warmup_lengths"),
            continuity_clip_frames if continuity else clip_frames,
        ),
        canvas_short_edge=canvas_short_edge,
        continuity=continuity,
        continuity_clip_frames=continuity_clip_frames,
        seam_frames=seam_frames,
        inference=inference,
        runtime=runtime,
    )


def _parse_warmup_lengths(raw: Any, clip_frames: int) -> tuple[int, ...]:
    """Resolve ``inference.warmup_lengths`` to the frame counts load() warms.

    ``"default"`` (or nothing) warms only the session's default length;
    ``"all"`` warms every length the checkpoint can generate; a list of
    seconds warms those, snapped to legal lengths. The default length is
    always included — it is the shape every plain `enqueue` uses.
    """
    if raw in (None, "", "default"):
        return (clip_frames,)
    if raw == "all":
        frames = set(clip_plan.legal_frame_counts())
    elif isinstance(raw, (list, tuple)):
        frames = {clip_plan.frames_for_seconds(float(seconds)) for seconds in raw}
    else:
        raise ValueError(
            f'inference.warmup_lengths must be "default", "all", or a list of seconds, got {raw!r}'
        )
    frames.add(clip_frames)
    return tuple(sorted(frames))


def resolve_model_path(config: FastH3Config, weights_root: Path) -> Path:
    """The checkpoint directory inside the mounted weights bundle.

    ``checkpoint_dir: "."`` means the snapshot's components sit directly under
    the weights root, which is how ``reactor weights upload`` lays a bundle out
    when the snapshot itself is uploaded.
    """
    subdir = str(config.runtime.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR))
    if subdir in ("", "."):
        return weights_root
    return weights_root / subdir


def require_weights(root: Path, model_path: Path) -> None:
    """Fail startup loudly when the weights bundle is incomplete."""
    problems: list[str] = []
    if not model_path.is_dir():
        problems.append(f"checkpoint directory is missing: {model_path}")
    else:
        index = model_path / "modular_model_index.json"
        if not index.is_file():
            problems.append(f"modular_model_index.json is missing: {index}")
        for component in REQUIRED_COMPONENTS:
            if not (model_path / component).is_dir():
                problems.append(f"component directory is missing: {model_path / component}")
    if problems:
        raise FileNotFoundError(
            f"FastH3 weights bundle under {root} is incomplete:\n  " + "\n  ".join(problems)
        )


__all__ = [
    "DEFAULT_CHECKPOINT_DIR",
    "REQUIRED_COMPONENTS",
    "FastH3Config",
    "load_config",
    "require_weights",
    "resolve_model_path",
]
