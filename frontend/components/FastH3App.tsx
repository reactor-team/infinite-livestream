"use client";

import {
  ReactorProvider,
  ReactorView,
  useReactor,
  useReactorMessage,
} from "@reactor-team/js-sdk";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import type { FrontendConfig } from "@/lib/config";

const TRACKS = [
  { name: "main_video", kind: "video", direction: "recvonly" },
  { name: "main_audio", kind: "audio", direction: "recvonly" },
] as const;

interface ClipInfo {
  clip_id: string;
  prompt: string;
  metadata: string;
  frames: number;
  seconds: number;
  seed: number;
  ready: boolean;
}

interface QueueSnapshot {
  generation: ClipInfo[];
  playout: ClipInfo[];
}

interface WorldState {
  type: "state_update";
  clip_seconds: number;
  clip_seconds_min: number;
  clip_seconds_max: number;
  seed: number;
  autoplay: boolean;
  aspect: string;
  width: number;
  height: number;
  playing: boolean;
  playing_clip_id: string | null;
  generation_queued: number;
  generation_capacity: number;
  playout_queued: number;
  playout_capacity: number;
  clips_played: number;
  seconds_sent: number;
  valid_commands: string[];
}

interface ModelEvent {
  type?: string;
  clip?: ClipInfo;
  reason?: string;
  command?: string;
  generation?: ClipInfo[];
  playout?: ClipInfo[];
  [key: string]: unknown;
}

interface GroupMetadata {
  group_id?: string;
  title?: string;
  scene?: number;
  scenes?: number;
  author?: string;
  source?: string;
  generated?: boolean;
  raw_prompt?: string;
}

interface LogItem {
  id: number;
  at: string;
  text: string;
  tone: "normal" | "error";
}

const TOKEN_REFRESH_SKEW_MS = 60_000;
let cachedToken: { jwt: string; expiresAtMs: number } | null = null;
let inflightToken: Promise<string> | null = null;

async function fetchToken(): Promise<string> {
  if (
    cachedToken &&
    Date.now() < cachedToken.expiresAtMs - TOKEN_REFRESH_SKEW_MS
  ) {
    return cachedToken.jwt;
  }
  if (inflightToken) return inflightToken;
  inflightToken = (async () => {
    try {
      const response = await fetch("/api/token", { cache: "no-store" });
      const body = (await response.json().catch(() => ({}))) as {
        jwt?: string;
        expires_at?: number;
        error?: string;
      };
      if (!response.ok || !body.jwt || !body.expires_at) {
        throw new Error(body.error ?? `Token request failed: ${response.status}`);
      }
      cachedToken = { jwt: body.jwt, expiresAtMs: body.expires_at * 1000 };
      return body.jwt;
    } finally {
      inflightToken = null;
    }
  })();
  return inflightToken;
}

function unwrap(raw: unknown): ModelEvent {
  const envelope = raw as {
    type?: string;
    data?: Record<string, unknown>;
    error?: { code?: string; message?: string };
  };
  if (envelope?.error) {
    return {
      type: "command_error",
      reason: envelope.error.message ?? envelope.error.code ?? "Command rejected",
    };
  }
  if (
    envelope &&
    typeof envelope === "object" &&
    envelope.data &&
    typeof envelope.data === "object"
  ) {
    return { ...envelope.data, type: envelope.type };
  }
  return (raw ?? {}) as ModelEvent;
}

function parseMetadata(value: string): GroupMetadata | null {
  try {
    const parsed = JSON.parse(value) as unknown;
    return parsed && typeof parsed === "object" ? (parsed as GroupMetadata) : null;
  } catch {
    return null;
  }
}

function clipTitle(clip: ClipInfo): string {
  return parseMetadata(clip.metadata)?.title?.trim() || clip.prompt;
}

function clipSource(clip: ClipInfo): string {
  const metadata = parseMetadata(clip.metadata);
  if (!metadata?.group_id) return "External client";
  if (metadata.generated) return "Automatic filler";
  const author = metadata.author?.trim() || "anonymous";
  const source = metadata.source?.trim() || "unknown";
  return `${author}@${source}`;
}

function clipScene(clip: ClipInfo): string | null {
  const metadata = parseMetadata(clip.metadata);
  if (!metadata?.scene || !metadata.scenes) return null;
  return `scene ${metadata.scene}/${metadata.scenes}`;
}

function newGroupId(): string {
  if (typeof crypto.randomUUID === "function") {
    return crypto.randomUUID().replaceAll("-", "").slice(0, 12);
  }
  return `${Date.now().toString(36)}${Math.random().toString(36).slice(2)}`.slice(0, 12);
}

function statusLabel(status: string): string {
  return (
    {
      disconnected: "Not connected",
      connecting: "Connecting",
      waiting: "Waiting for model",
      ready: "Connected",
      error: "Connection failed",
    }[status] ?? status
  );
}

function Button({
  children,
  onClick,
  disabled,
  primary = false,
  danger = false,
  compact = false,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
  danger?: boolean;
  compact?: boolean;
}) {
  const color = primary
    ? "border-accent bg-accent text-slate-950 hover:bg-white"
    : danger
      ? "border-error/50 text-error hover:bg-error/10"
      : "border-edge bg-raised text-ink hover:border-faint";
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`${compact ? "h-7 px-2 text-[10px]" : "h-9 px-3 text-xs"} rounded-md border font-semibold transition disabled:cursor-not-allowed disabled:opacity-35 ${color}`}
    >
      {children}
    </button>
  );
}

function Panel({
  title,
  children,
  action,
}: {
  title: string;
  children: React.ReactNode;
  action?: React.ReactNode;
}) {
  return (
    <section className="rounded-lg border border-edge bg-panel/95 p-4 shadow-xl shadow-black/10">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-[11px] font-semibold uppercase tracking-[0.15em] text-dim">
          {title}
        </h2>
        {action}
      </div>
      {children}
    </section>
  );
}

function Capacity({ value, limit }: { value: number; limit: number }) {
  const percent = limit > 0 ? Math.min(100, (value / limit) * 100) : 0;
  return (
    <div className="mt-2 h-1 overflow-hidden rounded-full bg-edge">
      <div
        className="h-full rounded-full bg-accent transition-[width]"
        style={{ width: `${percent}%` }}
      />
    </div>
  );
}

function QueueClip({
  clip,
  index,
  total,
  lane,
  readOnly,
  valid,
  onCommand,
}: {
  clip: ClipInfo;
  index: number;
  total: number;
  lane: "generation" | "playout";
  readOnly: boolean;
  valid: Set<string>;
  onCommand: (name: string, params?: Record<string, unknown>) => Promise<boolean>;
}) {
  const scene = clipScene(clip);
  const canMove = !readOnly && valid.has("move");
  const canPop = !readOnly && valid.has("pop");
  return (
    <article className="rounded-md border border-edge bg-raised/70 p-3">
      <div className="flex items-start gap-2">
        <span className={`mt-1.5 size-2 shrink-0 rounded-full ${lane === "playout" ? "bg-live" : "bg-wait"}`} />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="text-[10px] uppercase tracking-wider text-faint">
              {lane === "playout" ? "Ready" : "Generating"} · {index + 1}/{total}
            </span>
            {scene ? <span className="text-[10px] text-faint">{scene}</span> : null}
          </div>
          <p className="mt-1 line-clamp-2 text-xs leading-relaxed text-ink" title={clip.prompt}>
            {clipTitle(clip)}
          </p>
          <p className="mt-1 truncate text-[10px] text-faint">
            {clipSource(clip)} · {clip.seconds.toFixed(3)}s · seed {clip.seed} · {clip.clip_id.slice(0, 8)}
          </p>
        </div>
      </div>
      {!readOnly ? (
        <div className="mt-2 flex flex-wrap gap-1.5 border-t border-edge pt-2">
          {lane === "playout" ? (
            <Button
              compact
              primary
              disabled={!valid.has("play")}
              onClick={() => void onCommand("play", { clip_id: clip.clip_id })}
            >
              Play
            </Button>
          ) : null}
          <Button
            compact
            disabled={!canMove || index === 0}
            onClick={() => void onCommand("move", { clip_id: clip.clip_id, position: index - 1 })}
          >
            ↑
          </Button>
          <Button
            compact
            disabled={!canMove || index === total - 1}
            onClick={() => void onCommand("move", { clip_id: clip.clip_id, position: index + 1 })}
          >
            ↓
          </Button>
          <Button
            compact
            danger
            disabled={!canPop}
            onClick={() => void onCommand("pop", { clip_id: clip.clip_id })}
          >
            Remove
          </Button>
        </div>
      ) : null}
    </article>
  );
}

function FastH3Workspace({ config }: { config: FrontendConfig }) {
  const connect = useReactor((state) => state.connect);
  const disconnect = useReactor((state) => state.disconnect);
  const sendCommand = useReactor((state) => state.sendCommand);
  const status = useReactor((state) => state.status);
  const sessionId = useReactor((state) => state.sessionId);
  const lastError = useReactor((state) => state.lastError);
  const [world, setWorld] = useState<WorldState | null>(null);
  const [queues, setQueues] = useState<QueueSnapshot>({ generation: [], playout: [] });
  const [currentClip, setCurrentClip] = useState<ClipInfo | null>(null);
  const [prompt, setPrompt] = useState("");
  const [seconds, setSeconds] = useState(14.375);
  const [seed, setSeed] = useState(1000);
  const [aspect, setAspect] = useState("16:9");
  const [joinSessionId, setJoinSessionId] = useState(config.defaultSessionId ?? "");
  const [attachedSessionId, setAttachedSessionId] = useState<string | null>(null);
  const [logs, setLogs] = useState<LogItem[]>([]);
  const [connectedAt, setConnectedAt] = useState<number | null>(null);
  const [connectedFor, setConnectedFor] = useState(0);
  const nextLogId = useRef(1);

  const appendLog = useCallback((text: string, tone: LogItem["tone"] = "normal") => {
    setLogs((items) => [
      {
        id: nextLogId.current++,
        at: new Date().toLocaleTimeString(),
        text,
        tone,
      },
      ...items,
    ].slice(0, 60));
  }, []);

  useReactorMessage((raw: unknown) => {
    const message = unwrap(raw);
    if (message.type === "state_update") {
      const snapshot = message as unknown as WorldState;
      setWorld(snapshot);
      setSeconds(snapshot.clip_seconds);
      setSeed(snapshot.seed);
      setAspect(snapshot.aspect);
      return;
    }
    if (message.type === "queue_update") {
      setQueues({
        generation: message.generation ?? [],
        playout: message.playout ?? [],
      });
      return;
    }
    if (message.type === "clip_started" && message.clip) {
      setCurrentClip(message.clip);
      appendLog(`Playing ${clipTitle(message.clip)} · ${message.clip.clip_id.slice(0, 8)}`);
      return;
    }
    if ((message.type === "clip_finished" || message.type === "clip_stopped") && message.clip) {
      setCurrentClip(null);
      appendLog(`${message.type === "clip_finished" ? "Finished" : "Stopped"} ${clipTitle(message.clip)}`);
      return;
    }
    if (message.type === "clip_generated" && message.clip) {
      appendLog(`Ready to play · ${clipTitle(message.clip)}`);
      return;
    }
    if (message.type === "clip_failed" && message.clip) {
      appendLog(`Generation failed · ${clipTitle(message.clip)} · ${message.reason ?? "unknown error"}`, "error");
      return;
    }
    if (message.type === "clip_queued" && message.clip) {
      appendLog(`Queued · ${clipTitle(message.clip)}`);
      return;
    }
    if (message.type === "session_reset") {
      setCurrentClip(null);
      appendLog("Session reset; both queues cleared");
      return;
    }
    if (message.type === "command_error") {
      appendLog(`${message.command ?? "Command"} refused · ${message.reason ?? "unknown reason"}`, "error");
      return;
    }
    if (message.type && !message.type.endsWith("_accepted") && message.type !== "clip_moved" && message.type !== "clip_popped") {
      appendLog(message.type.replaceAll("_", " "));
    }
  });

  useEffect(() => {
    if (status === "ready" && connectedAt === null) {
      setConnectedAt(Date.now());
      appendLog(attachedSessionId ? "WebRTC monitor attached" : "WebRTC control session connected");
    } else if (status !== "ready" && connectedAt !== null) {
      appendLog(`WebRTC disconnected after ${connectedFor}s`);
      setConnectedAt(null);
      setWorld(null);
      setQueues({ generation: [], playout: [] });
      setCurrentClip(null);
    }
  }, [appendLog, attachedSessionId, connectedAt, connectedFor, status]);

  useEffect(() => {
    if (connectedAt === null) {
      setConnectedFor(0);
      return;
    }
    const update = () => setConnectedFor(Math.floor((Date.now() - connectedAt) / 1000));
    update();
    const timer = window.setInterval(update, 1000);
    return () => window.clearInterval(timer);
  }, [connectedAt]);

  useEffect(() => {
    if (lastError) appendLog(lastError.message, "error");
  }, [appendLog, lastError]);

  useEffect(() => {
    if (status !== "ready") return;
    // The connect greeting can race the browser's application-message
    // subscription. Correlated snapshots make every new or joined session
    // converge immediately even when that greeting arrived first.
    void sendCommand("get_state", {}).catch((error: unknown) => {
      appendLog(error instanceof Error ? error.message : "get_state failed", "error");
    });
    void sendCommand("get_queue", {}).catch((error: unknown) => {
      appendLog(error instanceof Error ? error.message : "get_queue failed", "error");
    });
  }, [appendLog, sendCommand, status]);

  const valid = useMemo(
    () => new Set(world?.valid_commands ?? []),
    [world?.valid_commands],
  );
  const connected = status === "ready";
  const readOnly = attachedSessionId !== null;

  const command = useCallback(
    async (name: string, params: Record<string, unknown> = {}) => {
      if (readOnly) {
        appendLog("Monitor mode is read-only", "error");
        return false;
      }
      try {
        await sendCommand(name, params);
        return true;
      } catch (error) {
        appendLog(error instanceof Error ? error.message : `${name} failed`, "error");
        return false;
      }
    },
    [appendLog, readOnly, sendCommand],
  );

  const handleConnect = useCallback(async () => {
    const existing = joinSessionId.trim();
    try {
      await connect(undefined, existing ? { sessionId: existing } : undefined);
      setAttachedSessionId(existing || null);
    } catch (error) {
      appendLog(error instanceof Error ? error.message : "Connection failed", "error");
    }
  }, [appendLog, connect, joinSessionId]);

  const handleDisconnect = useCallback(async () => {
    await disconnect();
    setAttachedSessionId(null);
  }, [disconnect]);

  const enqueue = useCallback(async () => {
    const clean = prompt.trim();
    if (!clean) return;
    const metadata = JSON.stringify({
      group_id: newGroupId(),
      title: clean.slice(0, 120),
      scene: 1,
      scenes: 1,
      author: "web",
      source: "frontend",
      generated: false,
      raw_prompt: clean.slice(0, 400),
    });
    if (await command("enqueue", { prompt: clean, metadata })) {
      setPrompt("");
    }
  }, [command, prompt]);

  const clearQueues = useCallback(async () => {
    for (const clip of [...queues.generation, ...queues.playout]) {
      await command("pop", { clip_id: clip.clip_id });
    }
  }, [command, queues]);

  const playingName = currentClip
    ? clipTitle(currentClip)
    : world?.playing_clip_id
      ? `Clip ${world.playing_clip_id.slice(0, 8)}`
      : null;

  return (
    <div className="flex h-screen min-h-[42rem] flex-col overflow-hidden">
      <header className="flex shrink-0 flex-wrap items-center gap-3 border-b border-edge bg-panel/80 px-4 py-3 backdrop-blur">
        <div className="flex items-center gap-3">
          <div className="flex size-8 items-center justify-center rounded-md bg-accent text-sm font-black text-slate-950">R</div>
          <div>
            <p className="text-sm font-semibold leading-none">FastH3 queue console</p>
            <p className="mt-1 text-[10px] uppercase tracking-[0.16em] text-faint">Infinite livestream</p>
          </div>
        </div>
        <span className="rounded-full border border-edge px-2.5 py-1 text-[11px] text-dim">
          {config.mode} · {config.apiUrl ?? config.modelName}
        </span>
        <span className={`rounded-full border px-2.5 py-1 text-[10px] font-semibold uppercase tracking-wider ${readOnly ? "border-wait/40 text-wait" : "border-live/40 text-live"}`}>
          {readOnly ? "Monitor" : "Control"}
        </span>
        <div className="ml-auto flex min-w-[18rem] flex-1 items-center justify-end gap-2 sm:flex-initial">
          {!connected ? (
            <input
              value={joinSessionId}
              onChange={(event) => setJoinSessionId(event.target.value)}
              placeholder="Existing session id (read-only)"
              disabled={status === "connecting" || status === "waiting"}
              className="h-9 min-w-0 flex-1 rounded-md border border-edge bg-raised px-3 text-[11px] outline-none focus:border-accent sm:w-72"
            />
          ) : (
            <span className="max-w-72 truncate text-[10px] text-faint" title={sessionId}>
              session {sessionId ?? "pending"}
            </span>
          )}
          <span className="flex items-center gap-2 whitespace-nowrap text-xs text-dim">
            <span className={`size-2 rounded-full ${connected ? "bg-live" : lastError ? "bg-error" : "bg-faint"}`} />
            {statusLabel(status)}{connected ? ` · ${connectedFor}s` : ""}
          </span>
          {connected ? (
            <Button onClick={() => void handleDisconnect()}>Disconnect</Button>
          ) : (
            <Button primary disabled={status === "connecting" || status === "waiting"} onClick={() => void handleConnect()}>
              {joinSessionId.trim() ? "Monitor" : "Connect"}
            </Button>
          )}
        </div>
      </header>

      <main className="flex min-h-0 flex-1 flex-col lg:flex-row">
        <aside className="flex w-full shrink-0 flex-col gap-3 overflow-y-auto border-edge p-3 lg:w-[27rem] lg:border-r">
          {readOnly ? (
            <div className="rounded-lg border border-wait/30 bg-wait/5 px-3 py-2 text-[11px] leading-relaxed text-wait">
              Monitoring an existing session. Queue writes stay disabled so the streaming director remains the only scheduler.
            </div>
          ) : null}

          <Panel title="Add scene">
            <textarea
              value={prompt}
              onChange={(event) => setPrompt(event.target.value)}
              rows={4}
              maxLength={800}
              disabled={!connected || readOnly}
              placeholder="Describe the next generated clip…"
              className="w-full resize-none rounded-md border border-edge bg-raised px-3 py-2 text-xs leading-relaxed outline-none placeholder:text-faint focus:border-accent disabled:opacity-40"
            />
            <div className="mt-2 flex items-center justify-between gap-3">
              <span className="text-[10px] text-faint">{prompt.length}/800 · {seconds.toFixed(3)}s · seed {seed}</span>
              <Button primary disabled={!connected || readOnly || !valid.has("enqueue") || !prompt.trim()} onClick={() => void enqueue()}>
                Queue scene
              </Button>
            </div>
          </Panel>

          <Panel
            title="Generation queue"
            action={<span className="text-[10px] text-faint">{world?.generation_queued ?? queues.generation.length}/{world?.generation_capacity ?? "–"}</span>}
          >
            <Capacity value={world?.generation_queued ?? queues.generation.length} limit={world?.generation_capacity ?? 0} />
            <div className="mt-3 space-y-2">
              {queues.generation.length === 0 ? (
                <p className="rounded-md border border-dashed border-edge px-3 py-4 text-center text-[11px] text-faint">No clips waiting to build.</p>
              ) : queues.generation.map((clip, index) => (
                <QueueClip key={clip.clip_id} clip={clip} index={index} total={queues.generation.length} lane="generation" readOnly={readOnly} valid={valid} onCommand={command} />
              ))}
            </div>
          </Panel>

          <Panel
            title="Playout queue"
            action={<span className="text-[10px] text-faint">{world?.playout_queued ?? queues.playout.length}/{world?.playout_capacity ?? "–"}</span>}
          >
            <Capacity value={world?.playout_queued ?? queues.playout.length} limit={world?.playout_capacity ?? 0} />
            <div className="mt-3 space-y-2">
              {queues.playout.length === 0 ? (
                <p className="rounded-md border border-dashed border-edge px-3 py-4 text-center text-[11px] text-faint">No built clips ready to play.</p>
              ) : queues.playout.map((clip, index) => (
                <QueueClip key={clip.clip_id} clip={clip} index={index} total={queues.playout.length} lane="playout" readOnly={readOnly} valid={valid} onCommand={command} />
              ))}
            </div>
          </Panel>

          <Panel title="Playback">
            <div className="grid grid-cols-2 gap-2">
              <Button primary disabled={!connected || readOnly || !valid.has("play")} onClick={() => void command("play")}>Play next</Button>
              <Button danger disabled={!connected || readOnly || !valid.has("stop")} onClick={() => void command("stop")}>Stop</Button>
              <Button disabled={!connected || readOnly} onClick={() => void command("set_autoplay", { enabled: !(world?.autoplay ?? false) })}>
                Autoplay: {world?.autoplay ? "on" : "off"}
              </Button>
              <Button disabled={!connected || readOnly || queues.generation.length + queues.playout.length === 0} onClick={() => void clearQueues()}>
                Clear queues
              </Button>
              <div className="col-span-2">
                <Button danger disabled={!connected || readOnly || !valid.has("reset")} onClick={() => void command("reset")}>Reset session</Button>
              </div>
            </div>
            <dl className="mt-3 grid grid-cols-2 gap-x-4 gap-y-1 border-t border-edge pt-3 text-[11px]">
              <dt className="text-faint">State</dt><dd className="truncate text-right text-dim">{world?.playing ? "playing" : connected ? "idle" : "offline"}</dd>
              <dt className="text-faint">Clips played</dt><dd className="text-right text-dim">{world?.clips_played ?? 0}</dd>
              <dt className="text-faint">Content sent</dt><dd className="text-right text-dim">{(world?.seconds_sent ?? 0).toFixed(2)}s</dd>
              <dt className="text-faint">Now</dt><dd className="truncate text-right text-dim" title={playingName ?? undefined}>{playingName ?? "black / idle"}</dd>
            </dl>
          </Panel>

          <Panel title="Generation settings">
            <label className="block text-[11px] text-faint">
              Default clip length · {seconds.toFixed(3)}s
              <input
                type="range"
                min={world?.clip_seconds_min ?? 5.167}
                max={world?.clip_seconds_max ?? 14.375}
                step="0.001"
                value={seconds}
                onChange={(event) => setSeconds(Number(event.target.value))}
                onPointerUp={() => void command("set_clip_seconds", { seconds })}
                disabled={!connected || readOnly}
                className="mt-2 w-full"
              />
            </label>
            <div className="mt-4 grid grid-cols-[1fr_auto] gap-2">
              <input
                type="number"
                min="0"
                value={seed}
                onChange={(event) => setSeed(Number(event.target.value))}
                disabled={!connected || readOnly}
                className="min-w-0 rounded-md border border-edge bg-raised px-3 text-xs outline-none focus:border-accent disabled:opacity-40"
              />
              <Button disabled={!connected || readOnly || !valid.has("set_seed")} onClick={() => void command("set_seed", { seed })}>Set seed</Button>
              <select
                value={aspect}
                onChange={(event) => setAspect(event.target.value)}
                disabled={!connected || readOnly || !valid.has("set_canvas")}
                className="min-w-0 rounded-md border border-edge bg-raised px-3 text-xs outline-none disabled:opacity-40"
              >
                {["16:9", "1:1", "9:16", "4:3"].map((value) => <option key={value}>{value}</option>)}
              </select>
              <Button disabled={!connected || readOnly || !valid.has("set_canvas")} onClick={() => void command("set_canvas", { aspect })}>Set canvas</Button>
            </div>
          </Panel>
        </aside>

        <section className="flex min-h-0 min-w-0 flex-1 flex-col">
          <div className="relative min-h-[18rem] flex-1 bg-black">
            <ReactorView className="absolute inset-0 size-full" track="main_video" audioTrack="main_audio" muted={false} videoObjectFit="contain" />
            {!connected || !world?.playing ? (
              <div className="pointer-events-none absolute inset-0 flex items-center justify-center p-8">
                <div className="max-w-md rounded-lg border border-white/10 bg-black/70 p-5 text-center backdrop-blur">
                  <p className="text-sm text-dim">
                    {!connected
                      ? "Connect to create a control session, or enter an existing session id to monitor it."
                      : readOnly
                        ? "The monitored session is idle. Media appears when its director starts a clip."
                        : queues.playout.length > 0
                          ? "A clip is ready. Select Play or enable autoplay."
                          : "Queue a scene. FastH3 builds it before playback begins."}
                  </p>
                </div>
              </div>
            ) : null}
            {connected && world ? (
              <div className="absolute bottom-3 left-3 max-w-[80%] rounded-md border border-white/10 bg-black/65 px-3 py-2 text-[11px] text-white/75 backdrop-blur">
                <p>{world.width}×{world.height} · 24 fps · {world.playing ? "live" : "idle"}</p>
                {playingName ? <p className="mt-1 truncate text-white">{playingName}</p> : null}
              </div>
            ) : null}
          </div>
          <div className="h-44 shrink-0 overflow-y-auto border-t border-edge bg-panel px-4 py-3">
            <h2 className="mb-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-faint">Model messages</h2>
            {logs.length === 0 ? (
              <p className="text-xs text-faint">No messages yet.</p>
            ) : logs.map((item) => (
              <p key={item.id} className={`mb-1 text-xs ${item.tone === "error" ? "text-error" : "text-dim"}`}>
                <span className="mr-2 text-faint">{item.at}</span>{item.text}
              </p>
            ))}
          </div>
        </section>
      </main>
    </div>
  );
}

/** Configure local or hosted Reactor transport for the FastH3 queue UI. */
export function FastH3App({ config }: { config: FrontendConfig }) {
  const local = config.mode === "local";
  return (
    <ReactorProvider
      modelName={config.modelName}
      modelTracks={[...TRACKS]}
      local={local}
      {...(config.apiUrl ? { apiUrl: config.apiUrl } : {})}
      {...(local ? {} : { getJwt: fetchToken })}
      connectOptions={{ autoConnect: false }}
    >
      <FastH3Workspace config={config} />
    </ReactorProvider>
  );
}
