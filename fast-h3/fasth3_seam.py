"""Seam stitching for continuity mode — pure numpy, no torch, no GPU.

Off by default. When ``inference.continuity`` is set, the channel generates each
clip after the first with FL2VA anchored on the previous clip's last frame, so
consecutive clips already share a near-identical boundary frame. Two operations
then turn that sequence into one continuous stream, and both live here so they
can be tested on any machine; ``fasth3.py`` imports this lazily so rendering the
schema still needs no numpy.

* :func:`color_match_to_reference` — locks every continuation clip's mean RGB to
  a single reference (clip 0's last frame). Per-clip (one offset for the whole
  clip), so intra-clip variation is untouched; anchored to clip 0 rather than the
  running clip, so exposure cannot ratchet across a long chain.

* :func:`blend_video_linear` — the "linearfade" seam: a crossfade done in
  **linear light** with **complementary weights** (``w_out + w_in == 1``) plus a
  ramped local exposure match. Blending in sRGB with equal-power weights
  overshoots at the midpoint for the near-identical frames FL2VA produces — a
  visible brightness *flash*. Linear light with complementary weights removes the
  overshoot; the exposure offset, full at the overlap's start and ramped to zero
  by its end, keeps the dissolve monotonic with no blend-end pop.

Audio crossfades with equal-power (constant-energy) ramps, which is correct for
the decorrelated waveforms at a seam (unlike the correlated video frames).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "srgb_to_linear",
    "linear_to_srgb",
    "luma",
    "reference_rgb",
    "color_match_to_reference",
    "blend_video_linear",
    "equal_power_ramps",
    "blend_audio_equal_power",
]


def srgb_to_linear(s: np.ndarray) -> np.ndarray:
    """sRGB in [0,1] to linear light. The standard piecewise transfer curve."""
    s = np.clip(s, 0.0, 1.0)
    return np.where(s <= 0.04045, s / 12.92, ((s + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(l: np.ndarray) -> np.ndarray:
    """Linear light to sRGB in [0,1]. Inverse of :func:`srgb_to_linear`."""
    l = np.clip(l, 0.0, 1.0)
    return np.where(l <= 0.0031308, l * 12.92, 1.055 * (l ** (1.0 / 2.4)) - 0.055)


def luma(frames_u8: np.ndarray) -> np.ndarray:
    """Rec.709 luma per frame, mean over pixels, from uint8 RGB ``(N,H,W,3)``."""
    f = frames_u8.astype(np.float32)
    y = 0.2126 * f[..., 0] + 0.7152 * f[..., 1] + 0.0722 * f[..., 2]
    return y.reshape(y.shape[0], -1).mean(axis=1)


def reference_rgb(frame_u8: np.ndarray) -> np.ndarray:
    """The per-channel mean RGB of one frame — the color-match reference.

    Locked once to clip 0's last frame and held constant for the whole chain, so
    exposure stays anchored instead of ratcheting.
    """
    return frame_u8.reshape(-1, 3).mean(axis=0, dtype=np.float64).astype(np.float32)


def color_match_to_reference(frames_u8: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    """Shift a whole clip by ONE per-channel offset so its mean RGB is ``target_rgb``.

    A single offset for every frame — not per-frame — so the clip keeps its
    natural intra-clip luminance and color variation while its average is pinned
    to the reference. ``target_rgb`` is clip 0's last-frame mean, constant across
    the chain, which is what stops exposure drift from compounding.
    """
    f = frames_u8.astype(np.float32)
    # Reduce in float64: a float32 mean over a whole clip (~10^8 samples)
    # saturates the 24-bit mantissa and collapses — at 124f/768p it returns
    # ~33.6 instead of the true ~124.5, which would shove every continuation
    # clip ~90 levels brighter and blow the highlights to white.
    src = frames_u8.reshape(-1, 3).mean(axis=0, dtype=np.float64).astype(np.float32)
    f += (target_rgb - src).astype(np.float32)[None, None, None, :]
    return np.clip(f, 0.0, 255.0).astype(np.uint8)


def blend_video_linear(
    tail_u8: np.ndarray, head_u8: np.ndarray, *, exposure_match: bool = True
) -> np.ndarray:
    """Linear-light crossfade of a clip's tail with the next clip's head.

    ``tail_u8`` and ``head_u8`` are the ``(k,H,W,3)`` overlap frames — the last
    ``k`` of clip N and the first ``k`` of clip N+1. The dissolve runs the head
    weight from 0 to 1 across the window with complementary weights (``w_out =
    1 - w_in``) in linear light, so there is no midpoint brightness bump.

    ``exposure_match`` adds a per-frame, per-channel offset that pulls the head
    onto the tail's mean level at the overlap's start and **ramps to zero** by
    its end, so the head arrives at its own true level exactly as the body
    resumes — a constant pin would darken the blend then pop when the body takes
    over. Returns ``(k,H,W,3)`` uint8.
    """
    k = int(tail_u8.shape[0])
    if k == 0:
        return tail_u8[:0]
    lt = srgb_to_linear(tail_u8.astype(np.float32) / 255.0)
    lh = srgb_to_linear(head_u8.astype(np.float32) / 255.0)
    if exposure_match:
        mt = lt.reshape(k, -1, 3).mean(axis=1)
        mh = lh.reshape(k, -1, 3).mean(axis=1)
        wr = ((np.arange(k, dtype=np.float32) + 0.5) / k)[:, None]
        offset = (mt - mh) * (1.0 - wr)  # full at start, zero by the end
        lh = np.clip(lh + offset[:, None, None, :], 0.0, 1.0)
    w_in = ((np.arange(k, dtype=np.float32) + 0.5) / k)[:, None, None, None]
    blended_linear = lt * (1.0 - w_in) + lh * w_in
    return (np.clip(linear_to_srgb(blended_linear), 0.0, 1.0) * 255.0).round().astype(np.uint8)


def equal_power_ramps(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power (constant-energy) fade-out/fade-in ramps of length ``n``.

    ``fade_out**2 + fade_in**2 == 1``, the correct crossfade for the decorrelated
    waveforms at an audio seam — it holds perceived loudness flat where a linear
    fade would dip.
    """
    if n <= 0:
        return np.ones(0, np.float32), np.ones(0, np.float32)
    t = (np.arange(n, dtype=np.float32) + 0.5) / n
    return np.cos(t * (np.pi / 2.0)).astype(np.float32), np.sin(t * (np.pi / 2.0)).astype(np.float32)


def blend_audio_equal_power(tail_i16: np.ndarray, head_i16: np.ndarray) -> np.ndarray:
    """Equal-power overlap-add of two int16 mono waveforms ``(1,S)`` of equal length.

    Blends in float to avoid int16 wrap at the sum, then requantizes. Returns
    ``(1,S)`` int16.
    """
    s = int(min(tail_i16.shape[-1], head_i16.shape[-1]))
    if s == 0:
        return np.zeros((1, 0), dtype=np.int16)
    fade_out, fade_in = equal_power_ramps(s)
    tail = tail_i16[:, :s].astype(np.float32)
    head = head_i16[:, :s].astype(np.float32)
    mixed = tail * fade_out[None, :] + head * fade_in[None, :]
    return np.clip(mixed, -32768.0, 32767.0).astype(np.int16)
