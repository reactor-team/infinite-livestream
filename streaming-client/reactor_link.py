"""The Reactor side: one supervised connection to a served fast-h3 model.

`ReactorLink` owns everything that touches `reactor_sdk`:

  * the connect/reconnect loop — a dropped session is rebuilt from scratch
    (queue contents die with a session server-side; see the README), while
    the pacer and sink outside this class keep the broadcast alive;
  * the media path — the recvonly video and audio tracks feed the pacer;
  * a live mirror of the model's `state_update` / `queue_update` (both the
    generation and the playout queue), so the rest of the client reads state
    instead of re-deriving it;
  * a fan-out of every model message to registered listeners.

The model contract this speaks is the fast-h3 clip queue (`../fast-h3/fasth3_types.py`
is the authoritative reference): `enqueue` → `clip_queued`, builds crossing
into the playout queue on `clip_generated`. The hosted deployment
(`reactor/fast-h3`) serves an extended build of that contract — `enqueue`
additionally takes `continue_from_clip_id` (the clip opens on the named
clip's last frame), `queue_update` carries a retained `history` of built
clips that can still be continued from, and `state_update` carries
`flush_on_clip_end`. The link mirrors all of it and exposes
`supports_continuation` so the director lights the chaining path up only
when the deployment actually publishes it; against the original in-repo
model everything degrades to independent clips. Autoplay is on for gapless
chaining; the director owns the *order*, curating the playout front with
`move` from the metadata echo (viewer content before filler — see
`Director.run_playout`).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from reactor_sdk import Reactor, ReactorStatus

from config import Config
from pacer import Pacer

logger = logging.getLogger(__name__)

RECONNECT_DELAY_S = 5.0

# fast-h3's fixed output timing (fasth3_clip_plan.FPS / backend sample rate).
# The canvas (width/height) is read from state_update instead — it depends on
# the deployment's aspect — but the rates are pinned by the checkpoint.
MODEL_FPS = 24
MODEL_SAMPLE_RATE = 48_000

# Defaults used only until the first state_update arrives.
_DEFAULT_STATE: dict[str, Any] = {
    "width": 1344,
    "height": 768,
    "clip_seconds_min": 5.167,
    "clip_seconds_max": 14.375,
    "generation_queued": 0,
    "generation_capacity": 20,
    "playout_queued": 0,
    "playout_capacity": 10,
}


def payload(reply: Any) -> Any:
    """Unwrap a send_command reply envelope ({"type", "data"}) to its data."""
    if isinstance(reply, dict) and "data" in reply and "type" in reply:
        return reply["data"]
    return reply


class ReactorLink:
    """Supervised fast-h3 session: media into the pacer, commands out."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._pacer: Pacer | None = None
        self._reactor: Reactor | None = None
        self._ready = asyncio.Event()
        self._first_state = asyncio.Event()
        self._listeners: list[Callable[[str, dict], None]] = []
        self.state: dict[str, Any] = dict(_DEFAULT_STATE)
        self.generation_clips: list[dict] = []
        self.playout_clips: list[dict] = []
        self.history_clips: list[dict] = []

    # -------------------------------------------------------------- wiring

    def attach_pacer(self, pacer: Pacer) -> None:
        """Point the media path at the pacer (built after the first state)."""
        self._pacer = pacer

    def add_listener(self, listener: Callable[[str, dict], None]) -> None:
        """Register for every model message as `(kind, data)`. Must not raise."""
        self._listeners.append(listener)

    # ------------------------------------------------------- state mirror

    @property
    def min_seconds(self) -> float:
        return float(self.state.get("clip_seconds_min", 5.167))

    @property
    def max_seconds(self) -> float:
        return float(self.state.get("clip_seconds_max", 14.375))

    @property
    def generation_queued(self) -> int:
        return int(self.state.get("generation_queued", 0))

    @property
    def generation_capacity(self) -> int:
        return int(self.state.get("generation_capacity", 20))

    @property
    def playout_queued(self) -> int:
        return int(self.state.get("playout_queued", 0))

    @property
    def playout_capacity(self) -> int:
        return int(self.state.get("playout_capacity", 10))

    @property
    def canvas(self) -> tuple[int, int]:
        """(width, height) the deployment generates at."""
        return int(self.state["width"]), int(self.state["height"])

    @property
    def supports_continuation(self) -> bool:
        """Whether the connected deployment speaks the extended contract.

        The extended surface (`enqueue`'s `continue_from_clip_id`, the
        `queue_update.history` list) is detected from the fields only it
        publishes in `state_update` — never assumed from the model name.
        False against the original in-repo model, where every clip is
        independent and the client must not send continuation fields.
        """
        return "flush_on_clip_end" in self.state

    @property
    def connected(self) -> bool:
        """Whether a session is live right now (commands would go through)."""
        return self._ready.is_set()

    async def wait_first_state(self) -> None:
        """Resolve once the first session delivered its `state_update`."""
        await self._first_state.wait()

    def _on_message(self, message: Any) -> None:
        if not isinstance(message, dict):
            return
        kind = message.get("type")
        data = message.get("data") or {}
        if kind == "state_update":
            self.state = data
        elif kind == "queue_update":
            self.generation_clips = data.get("generation", [])
            self.playout_clips = data.get("playout", [])
            # Extended contract only; absent on the original model.
            self.history_clips = data.get("history", [])
        elif kind == "command_error":
            logger.warning(
                "[reactor] command refused: %s — %s",
                data.get("command"), data.get("reason"),
            )
        for listener in self._listeners:
            listener(kind, data)

    # ------------------------------------------------------------ commands

    async def send_command(self, command: str, data: dict) -> Any:
        """Send one command on the live session; None when disconnected.

        Waits for a session to exist first, so callers ride out a reconnect
        instead of failing. A None / bodyless reply means the model refused
        the command (it broadcast `command_error` with the reason).
        """
        await self._ready.wait()
        reactor = self._reactor
        if reactor is None:
            return None
        try:
            return payload(await reactor.send_command(command, data))
        except Exception as error:
            logger.warning("[reactor] %s failed: %s", command, error)
            return None

    # ----------------------------------------------------------- lifecycle

    async def run(self) -> None:
        """Connect, and keep reconnecting forever. Cancelled only at shutdown."""
        while True:
            try:
                await self._run_session()
            except asyncio.CancelledError:
                await self._teardown()
                raise
            except Exception as error:
                logger.error("[reactor] session error: %s", error)
                await self._teardown()
                # An "orphaned" refusal means a dead client's session is
                # blocking the runtime, and its reaper takes a minute or
                # more. Only local mode force-clears, and only on
                # "orphaned" — "streaming" may be someone else's live
                # session and is left alone.
                if self._config.local and "orphaned" in str(error):
                    await self._stop_local_session("orphaned session blocks connect")
                    await asyncio.sleep(1.0)
                    continue
            logger.info("[reactor] reconnecting in %.0fs", RECONNECT_DELAY_S)
            await asyncio.sleep(RECONNECT_DELAY_S)

    async def _run_session(self) -> None:
        if self._config.local:
            # The SDK honours a non-default api_url in local mode, so a
            # runtime on another port (REACTOR_LOCAL_URL) works.
            reactor = Reactor(
                self._config.model, local=True, api_url=self._config.local_url
            )
        else:
            reactor = Reactor(self._config.model, api_key=self._config.api_key)
        disconnected = asyncio.Event()

        reactor.on("message", self._on_message)
        reactor.on_status(self._make_status_handler(disconnected))
        # Registered by wire name *before* connect: the SDK allows handler
        # registration ahead of the session declaring its tracks, whereas
        # querying `reactor.tracks` right after connect races that
        # declaration (an empty list on a slow session start).
        reactor.track("main_video").on_frame(self._on_video_frame)
        reactor.track("main_audio").on_frame(self._on_audio_frame)

        logger.info(
            "[reactor] connecting to %s (%s)...",
            self._config.model, "local" if self._config.local else "hosted",
        )
        await reactor.connect()
        logger.info(
            "[reactor] connected, session=%s status=%s",
            reactor.session_id, reactor.status,
        )

        self._reactor = reactor
        state = await asyncio.wait_for(self._raw_command(reactor, "get_state"), 30)
        if isinstance(state, dict) and "width" in state:
            self.state = state
        logger.info(
            "[reactor] canvas %dx%d, clip range %.3f-%.3fs, "
            "generation %d/%d, playout %d/%d, continuation %s",
            *self.canvas, self.min_seconds, self.max_seconds,
            self.generation_queued, self.generation_capacity,
            self.playout_queued, self.playout_capacity,
            "supported" if self.supports_continuation else "unsupported",
        )

        # Autoplay on: the model chains the playout queue's front clip the
        # instant the stream idles, which is what keeps clip-to-clip gaps at
        # milliseconds instead of a client round-trip. The director still
        # owns the ORDER — it curates the playout front with `move`, viewer
        # content first (see Director.run_playout).
        await self._raw_command(reactor, "set_autoplay", {"enabled": True})
        self._first_state.set()
        self._ready.set()
        try:
            await disconnected.wait()
            logger.warning("[reactor] session disconnected")
        finally:
            await self._teardown()

    @staticmethod
    def _make_status_handler(disconnected: asyncio.Event):
        loop = asyncio.get_running_loop()

        def on_status(status: ReactorStatus) -> None:
            logger.info("[reactor] status: %s", status.value)
            if status == ReactorStatus.DISCONNECTED:
                loop.call_soon_threadsafe(disconnected.set)

        return on_status

    @staticmethod
    async def _raw_command(reactor: Reactor, command: str, data: dict | None = None) -> Any:
        return payload(await reactor.send_command(command, data or {}))

    async def _teardown(self) -> None:
        self._ready.clear()
        reactor, self._reactor = self._reactor, None
        if reactor is None:
            return
        try:
            await reactor.disconnect()
        except Exception as error:
            logger.warning("[reactor] disconnect failed: %s", error)
            # The session we owned may now linger server-side and block the
            # next connect until the runtime's reaper gets to it.
            await self._stop_local_session("teardown after failed disconnect")

    async def _stop_local_session(self, reason: str) -> None:
        """Best-effort ``POST /stop_session`` on the local runtime.

        Local mode only: the local runtime serves one session, so whatever
        session exists is either ours or a dead predecessor's. Hosted
        sessions belong to the coordinator and are never force-stopped.
        """
        if not self._config.local:
            return
        import aiohttp

        url = f"{self._config.local_url.rstrip('/')}/stop_session"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json={}, timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    logger.info(
                        "[reactor] stop_session (%s): HTTP %d", reason, response.status
                    )
        except Exception as error:
            logger.warning("[reactor] stop_session failed (%s): %s", reason, error)

    # ---------------------------------------------------------- media path

    def _on_video_frame(self, frame) -> None:
        if self._pacer is not None:
            self._pacer.submit_video(frame)

    def _on_audio_frame(self, frame, sample_rate=MODEL_SAMPLE_RATE) -> None:
        if sample_rate != MODEL_SAMPLE_RATE:
            logger.warning(
                "[reactor] audio at %dHz, expected %d — timing will drift",
                sample_rate, MODEL_SAMPLE_RATE,
            )
        if self._pacer is not None:
            self._pacer.submit_audio(frame)
