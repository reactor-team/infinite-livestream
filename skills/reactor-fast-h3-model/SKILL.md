---
name: reactor-fast-h3-model
description: Full context for the fast-h3 model in this repo — what FastH3/MiniMax-H3 is, how the Reactor Runtime serves it, the queue contract and every design decision behind it, the measured performance profile and its load-bearing tricks, and how to build and run it with the reactor CLI or raw docker. Read before changing anything under fast-h3/, debugging generation speed or crashes, serving the model locally, or writing a client against it.
---

# The fast-h3 model: full working context

`fast-h3/` serves the FastH3 video model in one of **two modes**, defaulted by
`inference.continuity` and switchable per-session at runtime via `set_continuity`
(while idle), over the open-source
[Reactor Runtime](https://github.com/reactor-team/reactor-runtime):

- **Continuity (default on)** — one continuous take. `set_prompt` holds a prompt
  and the model builds clips back to back forever, each after the first
  FL2VA-anchored on the previous clip's last frame, its exposure locked to the
  opener's, every boundary crossfaded in linear light. One prompt → one
  uninterrupted stream until `stop` or a new `set_prompt`. Shipped at the 640
  resolution tier, where a 5.167 s clip builds in ~3.3 s on 8 B200s — under the
  playout window, so the chain never starves.
- **Queue (off)** — a **clip queue with a player**: clients `enqueue`
  prompt-driven generations, the model builds them ahead of time, nothing
  reaches the tracks until `play` (or autoplay), hard cut between clips. The
  streaming client drives this mode.

Both live in one `FastH3` class with disjoint command surfaces
(`state_update.valid_commands` names the live set). The mode is a per-session
flag (`self._continuity`, seeded from the config): `set_continuity` flips it
while the session is idle and `run()`/`_serve*` re-dispatch to the other loop,
so a client picks hard cuts or continuity without a redeploy. This file is the
context a new agent needs; the per-file detail lives in
[`fast-h3/README.md`](../../fast-h3/README.md) and the code's own docstrings.

## 1. The underlying model

[FastH3 Preview v1](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
is [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) distilled by
[FastVideo](https://github.com/hao-ai-lab/FastVideo) with data-free DMD2 down
to **four transformer forwards**, plus VSA-H3 sparse video attention at 90%
sparsity. Facts that shape everything downstream:

- **Fully bidirectional, not autoregressive.** One denoise covers the whole
  clip's token sequence at once — no KV cache, no rollout. Every clip is an
  independent sample; the queue mode leaves the boundaries as hard cuts. Only
  text-to-audio-video was distilled (the base model's first/last-frame and
  reference conditioning were not), so continuity's FL2VA anchoring is an
  **undistilled** carry: it seeds the next clip with the previous last frame and
  the seam crossfade dissolves the boundary, but it is a best-effort continuation
  the checkpoint was not trained for, not an identity lock. Exposure is held
  stable across the chain by a float64-mean colour lock (a float32 clip-mean
  saturates the mantissa at production sizes and blows highlights to white).
- **One packed sequence, video + audio + text.** The H3-Omni-Transformer is a
  33B dense single-stream DiT over a packed multimodal sequence with 3D
  RoPE; audio is generated jointly (native stereo, decoded by a separate
  audio VAE at 32 kHz), text conditioning comes from the full Qwen3-VL-32B
  (~63 GB per rank, hidden states of layer 50). The distill is
  guidance-free: `guidance_scale=1.0`, empty negative prompt, no CFG pass.
  `num_inference_steps=5` counts sigma-grid POINTS (t=999,749,500,250→0) —
  five points, exactly four forwards; the checkpoint supports nothing else.
- **Clip geometry is narrow** and encoded in `fasth3_clip_plan.py`, whose
  constants are deliberately duplicated from FastVideo (importing theirs
  drags in torch) with a drift test: 24 fps only; frame counts of the form
  `17n + 5` (the causal video VAE consumes 17-frame chunks into 5 latents);
  duration 5–15 s, which after alignment means **14 legal lengths, 124–345
  frames (5.167–14.375 s)** — 15.0 s itself is unreachable because 360
  frames aligns up to 362 and is then rejected; canvas short edge 768, area
  ≤ 768×1344, sides multiples of 32. Four aspect choices are offered
  (`16:9` → 1344×768, `1:1`, `9:16`, `4:3`).
- **GPU count must divide H3's 56 attention heads** (1, 2, 4, 7, 8, …).
  Six GPUs is not a configuration; the engine refuses at init. Four B200s
  is FastVideo's tested default; the hosted deployment runs eight B200s so
  builds outpace realtime playback and autoplay never waits on the GPUs
  (the image serves B300 identically — kernels are compiled for both).

## 2. How the Reactor Runtime serves it

The [Reactor Runtime](https://github.com/reactor-team/reactor-runtime) is the
open-source authoring layer; docs at
[docs.reactor.inc/deploy](https://docs.reactor.inc/deploy). What matters here:

- **`ReactorModel` with its own `run()` loop, not `ReactorPipeline`.** The
  pipeline base is generator-driven — one `yield` per emitted chunk — which
  suits frame-per-step models. fast-h3's unit of work is a whole clip from
  one blocking call, so it subclasses `ReactorModel`: the runtime runs the
  model's `run()` concurrently with a command-dispatch loop and a lifecycle
  loop, so `@event` handlers (commands) answer immediately even mid-build or
  mid-play. Do not "normalize" this to the pipeline shape.
- **Session state is the model's own.** No runtime-built `InputState`; plain
  attributes reset in `_reset_session_state()`, called at `load()` and from
  `@session_started` (not `@connected` — a client rejoining mid-session must
  keep the session it joined). `self.connected` is an `asyncio.Event` held
  while any client is attached; generation gates on it.
- **Emission and pacing.** `await self.emit(Output(...))` hands frames to a
  per-connection pacer; a chunk is tagged with a playout rate — measured
  (`compute_time=`) or the class's declared `fps`. fast-h3 **pins `fps = 24`
  and never passes `compute_time`**: a measured rate wobbles and drifts
  video against the sample-clocked audio. Emits go out in 3-frame slices
  (the runtime recorder's feed queue cannot absorb bursts), paced by frames
  on a monotonic clock that re-anchors rather than bursting after a stall.
  `buffer_size = 48` gives 2 s of transport tolerance. `self.output.flush()`
  drops queued media on every connection and cuts to black — used after
  every clip and on `stop`/`reset`.
- **Messages.** `self.send(msg)` broadcasts to every connection; a handler's
  return value is the correlated reply to the caller only; returning nothing
  yields a bodyless ack. **Refusals are broadcast, never raised**: handlers
  emit a `command_error` message and return bodyless, because a raised
  `CommandError`'s failure frame is withheld from older SDK generations.
- **Sessions are multi-connection.** Broadcasts and media fan out to every
  connection; a late joiner is greeted with `state_update` + `queue_update`
  from the `@connected` hook. A local runtime hosts **one session**: a
  second client must join with `connect(session_id=...)` — a plain
  `connect()` gets HTTP 409 while a session streams.
- **The schema is product surface.** Every `@event` / `InputField` /
  `MessageField` description and `ModelMessage` docstring compiles into the
  published OpenAPI schema (`python -m reactor_runtime.schema --path .`).
  Describe only what a client observes on the wire; never kernels, caches,
  config keys, or GPU counts.

## 3. The client contract (the two queues)

Authoritative detail in [`fast-h3/README.md`](../../fast-h3/README.md) and
`fasth3_types.py`; the shape in brief. A clip passes three stages: enqueued
(**generation queue**), built (**playout queue**), consumed (played, or
popped).

- `enqueue(prompt, metadata, seed?, seconds?, position?)` → immediate
  `clip_queued` reply carrying the full **`ClipInfo`** struct: `clip_id`
  (UUID), `prompt`, `metadata` (opaque, echoed untouched — the client's
  correlation channel), `frames`, `seconds`, `seed`, `ready` (`false` =
  generation queue, `true` = playout queue). The clip enters the generation
  queue at `position` (0 = next build; omitted = back). Omitted seed → the
  session's advancing default (explicit seeds leave it untouched); omitted
  seconds → the session default, snapped to the `17n+5` grid. Bounds:
  `inference.generation_queue_size` (default 20) prompts waiting;
  `inference.queue_size` (default 10) built clips, each ~1 GB of host RAM.
- **Builds consume the generation queue front-first, always**, one at a
  time, also while a clip plays — pausing only while the playout queue is
  full. A finished build crosses to the playout queue's back, announced by
  `clip_generated` (+ `queue_update`, which always carries both queues in
  full, front first).
- **The playout queue is the client's to schedule**: bare `play` (or
  autoplay, toggleable) takes the *front*; `play(clip_id)` any built clip;
  `move(clip_id, position)` reorders within either queue; `pop(clip_id)`
  removes from either (an in-flight build is discarded on completion).
  Playing consumes the entry; time-to-first-frame ~0.25 s since the clip is
  prebuilt. On finish: flush to black, `clip_finished`, hold. `stop` cuts
  playout in ~0.13 s; `reset` drops both queues and restores defaults.
  `set_clip_seconds`, `set_seed`, `set_canvas` (locked while clips exist),
  `get_queue`, `get_state` complete the surface;
  `state_update.valid_commands` tells a client exactly what is legal right
  now (`fasth3_session_rules.py`).
- **The two bounds are independent, and each gates only its own entrance.**
  `generation_queue_size` (default 20) refuses `enqueue` with
  `command_error` when the generation queue is full — nothing else is
  affected, builds and playback continue. `queue_size` (default 10) never
  refuses anything a client sends: it pauses *build submission* while the
  playout queue is full, and building resumes the moment `play` or `pop`
  frees a slot. Any ratio of the two is valid; there is no ordering
  constraint between them, and no combination deadlocks (a full playout
  queue is always drainable by `play`, `pop`, or `reset`). Their meanings
  differ: the playout bound is the host-memory budget (~1 GB per built
  clip — sized together with `resources.memory`), the generation bound just
  protects the build backlog, prompts being nearly free. A session holds at
  most `generation + playout + 1` clips (the playing clip is in neither
  queue). Both capacities are published live in `state_update`
  (`generation_capacity`, `playout_capacity`); a client must read them from
  there, never assume the defaults.
- Every clip-referencing message embeds the whole `ClipInfo`. On the wire it
  travels as a plain mapping (the transport encoder accepts only
  JSON-representable values); the `ClipInfo` dataclass in `fasth3_types.py`
  is the schema-side declaration of that exact shape, and
  `ClipEntry.snapshot()` in `fasth3_queue.py` is the single producer —
  a test pins the two together.

Design decisions worth knowing before "improving" things: the model never
reorders a queue on its own — `position`, `move`, and `pop` are the client's
levers, and a build crossing queues is the only movement the model makes;
playout deliberately never auto-advances without autoplay; the playing clip
is in neither queue (pop-on-play), so `pop` cannot touch it and `stop` is the
only cut; who or what a clip is *for* travels only in the metadata echo,
keeping the model ignorant of client-side scheduling policy; handlers return
exactly their annotated message type or nothing; `state_update` is one
complete snapshot built in one place so `get_state`, the connect greeting and
the broadcast can never disagree; audio is downmixed to mono int16 48 kHz in
the backend because the transport downmixes anyway and stereo would corrupt
runtime recordings.

## 4. The performance profile — every piece is load-bearing

Measured end state on four B200s: **14.4 s per 14.375 s clip (1.0x
realtime), flat across prompts and lengths**; conditioning ~1 s, decode
~2.5 s, the rest denoise. Each element below was established by measurement,
and removing any one of them regresses badly:

1. **The sm100a VSA kernel is compiled from source at image build** (pinned
   git commit in `requirements.txt`, `TORCH_CUDA_ARCH_LIST=10.0a;10.3a` in
   `build_env` — B200 and B300, since arch-specific binaries do not run
   forward even within a family). The published fastvideo-kernel wheel's
   sm_100a binary fails every launch on driver 595 with `invalid argument`;
   the identical source built by the image's CUDA 13.1 nvcc is correct. The
   PyPI sdist lacks its CUTLASS/ThunderKittens submodules, so the git tree
   is the only workable source. The `triton` VSA route is the portable
   fallback at ~2.5x the build time. **On B300 the compiled binary is not
   enough**: the package's `is_supported` gate accepts capability `(10, 0)`
   by equality only, so a `(10, 3)` device silently drops to Triton *and*
   loses regional compile (the compile gate asks the same predicate) —
   measured 0.74x realtime on eight B300s. `sitecustomize.py` widens the
   gate to every arch in `TORCH_CUDA_ARCH_LIST` (a test pins the two lists
   together); the regression signature in the pod log is `falling back to
   the Triton-64 kernels` + `inference_torch_compile requested but
   disabled`, once per worker.
2. **Prompts are padded to exactly 256 tokens** (`PROMPT_TOKENS`,
   `fasth3_backend.py`) with the bundle's own tokenizer and a filler
   calibrated to cost exactly one token. Regional torch.compile keys its
   capture on the packed sequence length, prompt tokens included; unpadded,
   every novel prompt length recompiled (~23 s per clip). The client-facing
   prompt stays the original text.
3. **Every legal clip length is warmed at load** (`inference.warmup_lengths:
   "all"` — 14 throwaway builds, several minutes of boot) so a feed of
   arbitrary `seconds` values never pays the ~20 s first-build compile
   stall mid-session.
4. **Dynamo's recompile limit is raised in every container interpreter** by
   `fast-h3/sitecustomize.py` (reached via `PYTHONPATH=/app` in
   `runtime_env`; `FASTH3_DYNAMO_RECOMPILE_LIMIT`, default 64). The default
   limit of 8, combined with the fullgraph regional-compile route, was a
   hard crash of the engine workers once enough distinct lengths had been
   enqueued — and the limit cannot be set via torch env vars in torch 2.12,
   nor from the parent process: the spawned workers import their own torch,
   and FastVideo's own imports re-cap it, hence the import hook.
5. **Replicated DiT + text encoder offloaded to pinned host memory**
   (`replicated_dit: true`, `offload_text_encoder: true`,
   `pin_cpu_memory: true`). FSDP sharding halves per-GPU weights but roughly
   doubled the denoise; the offloaded Qwen3-VL costs ~63 GB of pinned host
   RAM per rank and ~1 s page-in per clip (unpinned: ~15 s).
   `resources.memory` is sized for four host copies plus the built-clip
   buffer.
6. **The CPU allocation is part of the profile.** Nine processes (runtime +
   eight spawned workers) plus NCCL host threads and the decode gather
   need real cores: at request 8 / limit 32 on a 192-vCPU hosted node the
   cgroup throttled 38% of CFS periods and builds ran 0.85x against 1.16x
   on identical unthrottled silicon (latent prep 1.65 s vs 0.3 s, denoise
   9.5 s vs 5 s — the NVSwitch fabric was fine). `resources.cpu` pins
   request and limit at the account's model CPU quota (64 today — the
   registry 429s any higher limit), and `OMP_NUM_THREADS=8` caps
   per-process threadpools, since `nproc` in the container reports the
   whole node and torch otherwise sizes for it.
7. **flash-attn-4 coexists with the runtime via a resolver override.** The
   runtime needs `protobuf>=7.35.1` (its generated bindings hard-reject an
   older runtime); FA4's pinned `nvidia-cutlass-dsl` caps protobuf `<7` — a
   stale cap, since protobuf accepts old gencode on a newer runtime.
   `build_env` sets `UV_OVERRIDE=/app/requirements.txt`, feeding the
   requirements back as overrides so the higher floor wins. Because
   overrides also strip torch coupling, torchvision (0.27.x) and torchaudio
   (2.11.0; only `functional.resample` is used — pure torch ops) are pinned
   to torch-2.12-matched builds explicitly.

The number to watch is the per-clip log line —
`clip built: 345f (14.38s content) in 14.4s = 1.00x realtime on 4 gpus,
stages={...}` — figures live in the message text because the runtime's log
formatter drops structured extras.

## 5. Running it

Weights: one Hugging Face snapshot (~148 GB), components directly under the
weights root (`runtime.weights_path` in `reactor.yaml`;
`checkpoint_dir: "."`). `load()` validates every component directory up
front. Nothing downloads at load (`HF_HUB_OFFLINE=1`).

**With the [reactor CLI](https://docs.reactor.inc/deploy)** (from `fast-h3/`):

```sh
reactor build --no-dockerfile       # image from reactor.yaml's build: block
reactor run --gpus '"device=0,1,2,3"' --port 8080
```

Caveat: the engine workers hand decoded frames back over torch shared
memory, and warm-up never exercises that path — docker's default 64 MB
`/dev/shm` kills the first real clip. `reactor run` has no shm flag today,
so for real serving use the documented docker equivalent:

```sh
W=~/.cache/reactor_registry/fast-h3
docker run --rm -d --name fast-h3 --shm-size=32g --gpus '"device=0,1,2,3"' \
  -p 8080:8080 -v "$W:$W" -e REACTOR_WEIGHTS_PATH="$W" -e PORT=8080 \
  reactor-local/fast-h3:dev run --port 8080
```

Load takes minutes (weights + warm-up builds; watch `docker logs -f fast-h3`
for `session ready`). Pick GPUs with ~90 GB free each on the current
profile. CPU-only checks need no GPU:

```sh
python -m reactor_runtime.schema --path . --out /tmp/schema.json
PYTHONPATH=. python -m pytest tests/ -q
```

**Connecting** with the public Python SDK
([`reactor-sdk`](https://pypi.org/project/reactor-sdk/) on PyPI):

```python
from reactor_sdk import Reactor
reactor = Reactor("fast-h3", local=True)                                # :8080
reactor = Reactor("fast-h3", local=True, api_url="http://localhost:8082")  # custom port
await reactor.connect()                       # second client: connect(session_id=...)
```

`fast-h3/client/client.py` is the reference walkthrough — it exercises the
whole contract and writes received .mp4s, the message log, and a timing
report; use it as the smoke test after any serving change.

## 6. Keeping this skill true

This file is the context handoff between agents. When work on `fast-h3/`
changes the contract, the profile, the serving mechanics, or resolves an
open item, update this skill **in the same change** — a stale skill poisons
the next session's assumptions. The same rule AGENTS.md applies to itself.
