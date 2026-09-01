"""Decode a client-uploaded seed image — pure, no torch, no GPU.

Off by default. When ``inference.continuity`` is set, the take can be *seeded*
from a still image the client uploads (`set_seed_image`): image-to-video, the
still becomes a moving take. This module turns those raw bytes into the one
thing the generator needs — the **seed frame**, as ``uint8`` RGB ``[h, w, 3]``
— and nothing else, so it can be tested on any machine without a GPU or the
inference stack. ``fasth3.py`` imports it lazily, so rendering the schema needs
no Pillow.

The seed frame is fed to clip 0 as the FL2VA first-frame condition, exactly the
anchor path a self-generated clip already uses. The generator VAE-encodes it for
conditioning and VAE-decodes the result, so an off-distribution external frame
(a phone photo, a different colour space) is pulled onto the model's own decode
distribution by generation itself — clip 0's opening frame is
``decode(encode(seed))``, not the raw upload.

Video is deliberately not accepted: a client that wants to *continue from* a
video extracts the frame it wants and sends that as an image, which is the same
anchor logic without a decoder for every container in this process. The frame
comes back as ``uint8`` RGB ``[h, w, 3]`` at the image's own resolution; the
caller resizes to the session canvas (the generator would anyway, but resizing
here keeps the anchor and any reference consistent).
"""

from __future__ import annotations

import io

import numpy as np

__all__ = [
    "MAX_SEED_BYTES",
    "SeedDecodeError",
    "decode_seed_frame",
]

# A generous ceiling for a single still. Guards against a decompression bomb or
# an accidental multi-GB upload wedging the worker. The runtime also bounds the
# upload; this is defence in depth.
MAX_SEED_BYTES = 64 * 1024 * 1024

# Pillow's own guard against decompression-bomb images, in pixels. 768x1344 is
# ~1M px; 64M px is ~60x the largest canvas, comfortably above any real still.
_MAX_IMAGE_PIXELS = 64 * 1024 * 1024

# Containers a client might send hoping to continue from a video. Rejected with
# a message that points at the supported path (send a frame as an image).
_VIDEO_EXTENSIONS = frozenset(
    {"mp4", "mov", "m4v", "webm", "mkv", "avi", "gif", "mpg", "mpeg", "ts"}
)


class SeedDecodeError(ValueError):
    """Raised when an upload cannot be turned into a seed frame.

    A ``ValueError`` subclass so callers can catch either; the message is
    client-safe (it never leaks a path or a stack) and is meant to be handed
    straight back as a command refusal.
    """


def decode_seed_frame(
    data: bytes, mime_type: str = "", name: str = ""
) -> np.ndarray:
    """Turn an uploaded still into one seed frame — ``uint8`` RGB ``[h, w, 3]``.

    Raises :class:`SeedDecodeError` on anything the caller should refuse: an
    empty upload, an oversized one, a video (unsupported — send a frame as an
    image), or a corrupt/unreadable file. Never raises anything else, so a bad
    upload is a clean refusal and never a worker crash.
    """
    if not data:
        raise SeedDecodeError("The upload is empty.")
    if len(data) > MAX_SEED_BYTES:
        raise SeedDecodeError(
            f"The upload is {len(data) // (1024 * 1024)} MB; the limit is "
            f"{MAX_SEED_BYTES // (1024 * 1024)} MB for a seed image."
        )
    _reject_video(mime_type, name)
    return _decode_image(data)


# ------------------------------------------------------------------ internals


def _reject_video(mime_type: str, name: str) -> None:
    """Refuse a video upload early, with a message that names the supported path."""
    mime = (mime_type or "").strip().lower()
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if mime.startswith("video/") or ext in _VIDEO_EXTENSIONS:
        raise SeedDecodeError(
            "Video is not accepted as a seed; extract the frame you want to "
            "start from and send it as an image."
        )


def _decode_image(data: bytes) -> np.ndarray:
    """Decode a still to ``uint8`` RGB ``[h, w, 3]`` with PIL."""
    from PIL import Image

    prev_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = _MAX_IMAGE_PIXELS
    try:
        with Image.open(io.BytesIO(data)) as image:
            frame = np.asarray(image.convert("RGB"), dtype=np.uint8)
    except SeedDecodeError:
        raise
    except Exception as error:  # noqa: BLE001 — any decode failure is a refusal
        raise SeedDecodeError(f"The image could not be decoded ({error}).") from None
    finally:
        Image.MAX_IMAGE_PIXELS = prev_limit

    return _validate_frame(frame)


def _validate_frame(frame: np.ndarray) -> np.ndarray:
    """Guarantee the caller gets a contiguous ``uint8`` RGB ``[h, w, 3]`` frame."""
    if frame.ndim != 3 or frame.shape[2] != 3 or frame.shape[0] < 1 or frame.shape[1] < 1:
        raise SeedDecodeError("The decoded image is not a colour image.")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(frame)
