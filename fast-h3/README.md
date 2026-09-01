# FastH3

Two queues of prompt-driven video clips with synchronized audio. Clients
enqueue generation requests — a prompt plus their own metadata, each answered
with a UUID — into the **generation queue**, which builds consume front-first
on their own; each finished clip crosses into the **playout queue**
(`clip_generated`), where playback is a separate, explicit step: `play`
streams one built clip at 768p over WebRTC, and when it ends the stream holds
on black until the next `play`. Nothing plays on its own unless autoplay is
on, and `enqueue`'s `position`, `move`, and `pop` give a client full control
of both queues' order.

Reach for this when something else decides what plays and when: a frontend
that lets people submit prompts and curates the order, a playlist that is
assembled faster than it is watched, a controller that wants clips ready
before they are needed. The model is the queue handler and the renderer; the
scheduling brain sits on the client side of the API. If you want one finished
clip returned as a file, this is the wrong shape — clips are streamed, not
returned.

[FastH3 Preview v1](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
is MiniMax-H3 (35B) distilled by FastVideo with data-free DMD2 down to **four
transformer forwards**, with 90% sparse video attention on Blackwell. It
generates video and stereo audio jointly from text. Text-to-video-and-audio is
the only task this checkpoint was distilled for; first/last-frame and reference
conditioning are not.

## Prerequisites

- **Four NVIDIA B200s** — FastVideo's tested default for this checkpoint. A
  15 s clip builds in about 15.5 s on four and 12.9 s on eight; playback is
  explicit and clips are pre-built, so the GPU count only moves the
  enqueue-to-ready wait. The count must divide H3's 56 attention heads
  (1, 2, 4, 7, 8 …), and each rank holds its own ~63 GB text encoder plus a
  66/N GB transformer shard, so fewer than four wants offloading.
- **CUDA 13.** The VSA-H3 sparse kernel and the FA4 CuTe kernels are both cu130
  builds, which is why this model's `build.cuda_version` differs from every
  other model here.
- **The weights bundle**, roughly 148 GB, placed under `runtime.weights_path`.
  Nothing is downloaded at load — see below.

## Weights

One Hugging Face snapshot carries every component:

```
~/.cache/reactor_registry/fast-h3/
└── FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree/   # ~148 GB
    ├── modular_model_index.json
    ├── transformer/        # ~70 GB, the 35B DiT
    ├── text_encoder/       # ~69 GB, Qwen3-VL
    ├── vae/  audio_vae/
    ├── scheduler/  audio_scheduler/
    └── tokenizer/  processor/
```

FastVideo resolves each component as a subdirectory of `model_path` and ignores
the repo ids inside `modular_model_index.json`, so the bundle loads fully
offline — which is what `HF_HUB_OFFLINE=1` in the manifest relies on. `load()`
checks every component directory up front, so an incomplete bundle stops
startup rather than surfacing as a loader traceback on the first clip.

## Run it

```sh
# Render the client-facing contract — no weights, no GPU.
python -m reactor_runtime.schema --path . --out /tmp/schema.json

# The CPU-only structural tests.
PYTHONPATH=. python -m pytest tests/ -q

# Build the image and serve (needs the weights and four GPUs).
reactor build
reactor run

# Drive the served model end to end with the reference client (saves .mp4s).
python client/client.py
```

`load()` warms one throwaway clip per configured canvas
(`inference.warmup_aspects`) and per configured clip length
(`inference.warmup_lengths`) before the pod reports ready, so real builds run
at warm speed. Every distinct frame count and canvas is a separate one-time
compile cost. The shipped config warms **all 14 legal lengths** at the
primary canvas — several extra minutes of boot, bought so a feed that
enqueues arbitrary `seconds` values (the streaming client's upsampler does)
never pays a mid-session compile stall; set `warmup_lengths: "default"` for
the old fast boot when every enqueue uses one length.

Before blaming the adapter for slow builds, baseline the recipe itself with
FastVideo's own `examples/inference/basic/basic_fasth3.py` at the same settings.
Its median is the number this model should match; a gap is the adapter's fault,
not the model's.

## The mental model

```
enqueue ──► [ generation queue ] ──build──► [ playout queue ] ──play──► tracks
             front consumed 1-at-a-time,      built clips in host        │
             always (pauses only while        memory; front is what      ▼
             playout is full)                 bare play / autoplay   clip ends or
             position/move/pop reorder        take; move/pop reorder stop: flush →
                                                                     black, wait
```

- **`enqueue` is the only way in.** Each request snapshots the session's
  conditions (`set_clip_seconds`, `set_seed`, `set_canvas`) as they stand,
  gets a UUID, and enters the generation queue — at the back, or at
  `position` (0 = the next build). The generation queue is bounded
  (`inference.generation_queue_size`, default 20); full, it refuses further
  enqueues.
- **Builds consume the generation queue front-first, always**, one at a
  time, whenever an audience is connected — including while another clip is
  playing — pausing only while the playout queue is at capacity (a finished
  build needs a slot to land in). A finished build crosses into the playout
  queue's back, announced by `clip_generated` and `queue_update`; a client
  that wants it sooner `move`s it forward.
- **`play` is the only way out** — unless autoplay is on. Bare `play` takes
  the playout queue's front; a `clip_id` takes that specific clip. Playing
  consumes the entry. When the clip ends — or `stop` cuts it — the output
  flushes to black and the session waits for the next `play`. With
  `set_autoplay` on, the playout front starts on its own whenever nothing is
  playing, so a steadily fed queue plays through hands-free; `stop` then
  acts as a skip.
- **Order is the client's, at every stage.** `enqueue`'s `position` places a
  clip in the generation queue, `move` repositions a clip within whichever
  queue holds it, and `pop` removes it — the model never reorders anything
  on its own; a build crossing queues is the only movement it makes.
- **Everything a clip is travels with every mention of it.** `clip_queued`,
  `queue_update`, `clip_started`, `clip_finished`, `clip_stopped` and
  `clip_failed` all embed the full `ClipInfo` structure, so a client never has
  to join a UUID against an earlier message.

## The `ClipInfo` structure

| Field | Type | Meaning |
|---|---|---|
| `clip_id` | string | UUID assigned at `enqueue`; every later reference uses it. |
| `prompt` | string | What the clip shows, exactly as enqueued. |
| `metadata` | string | Opaque client string, echoed back untouched — see below. |
| `frames` | int | Clip length in frames, fixed at enqueue time. |
| `seconds` | float | The same length in seconds (`frames / 24`). |
| `seed` | int | Seed this clip generates from. |
| `ready` | bool | Which queue holds it: `false` = generation, `true` = playout (playable now). |

**Metadata is for the frontend, not the model.** The model stores it and echoes
it back on every message that references the clip; it never parses it. Use it to
carry whatever your application needs to track — which request produced the
clip, who asked for it, which group of enqueues it belongs to, text to show
while it plays. Up to 2000 characters; JSON fits if you want structure.

## Tracks

| Track | Direction | Kind | Rate | Payload |
|---|---|---|---|---|
| `main_video` | out | video | 24 fps, fixed | RGB frames at the session's canvas, e.g. 1344×768 |
| `main_audio` | out | audio | 48 kHz | Mono, synchronized frame-for-frame with `main_video` |

There are no inbound tracks: the model reads no camera and no microphone. The
video track keeps one size — `set_canvas` chooses it and is only accepted while
the queue is empty and nothing is playing, since queued clips are built at the
size in force.

## Commands

The two modes offer disjoint surfaces; `state_update.valid_commands` names the
live set. `set_prompt` is continuity-only;
`enqueue`/`play`/`move`/`pop`/`get_queue`/`set_autoplay` are queue-only. The rest
are shared. `set_continuity` is the runtime toggle between the two — offered
while the session is idle so a client picks the mode without a config change.

| Command | Parameters | Effect | Rejected when |
|---|---|---|---|
| `set_prompt` | `prompt` (≤ 800 chars), `metadata` (≤ 2000 chars) | **Continuity mode.** Sets the prompt the continuous take follows; the first starts the take, a later one re-anchors it. Replies `prompt_accepted`. | empty prompt, or the model runs the queue |
| `enqueue` | `prompt` (≤ 800 chars), `metadata` (≤ 2000 chars), `seed` (optional, ≥ 0), `seconds` (optional, 5.167–14.375), `position` (optional, ≥ 0) | **Queue mode.** Enters the generation queue at `position` (0 = next build; omitted = the back); replies `clip_queued` with the full `ClipInfo`. Without a seed the session's advancing default is used; without `seconds` the session's default length. | generation queue full, empty prompt, or the model runs continuity |
| `move` | `clip_id` (UUID), `position` (≥ 0) | Repositions the clip within whichever queue holds it; 0 = front, clamped to the back. Replies `clip_moved` with the queue's name and the landing position. | unknown or missing id |
| `play` | `clip_id` (optional UUID) | Streams the playout queue's front clip, or the named one. Emits `clip_started` as frames begin. | already playing, unknown id, clip still generating |
| `pop` | `clip_id` (UUID) | Removes that clip from whichever queue holds it, freeing its slot; a build in flight for it is discarded. Replies `clip_popped`. | unknown or missing id |
| `stop` | — | Cuts the stream to black: the playing clip in queue mode (queues untouched; a skip with autoplay on, emits `clip_stopped`), or the whole continuity take (drops the held prompt back to idle). | nothing playing |
| `get_queue` | — | Replies with both queues — the same payload as `queue_update`. | — |
| `set_autoplay` | `enabled` (bool) | On, the playout queue's front clip starts on its own whenever nothing is playing. Off (default), the stream holds until `play`. Replies `autoplay_accepted`. | — |
| `set_clip_seconds` | `seconds` (5.167–14.375) | Default length for enqueues that carry no `seconds`, snapped to what the model can produce; the effective value returns in `clip_length_accepted`. | — |
| `set_seed` | `seed` (≥ 0) | Default seed for enqueues that carry none; each such enqueue advances it by one. Replies `seed_accepted`. | — |
| `set_canvas` | `aspect` (`16:9`, `1:1`, `9:16`, `4:3`) | Video size for the session. Replies `canvas_accepted`. | clips queued or playing, unsupported aspect |
| `set_continuity` | `enabled` (bool) | Switches this session between the continuous take (`true`, the config default) and the hard-cut queue (`false`) at runtime; the config sets the starting mode and `reset` keeps the chosen one. Replies `continuity_accepted`; `state_update.valid_commands` then carries the other mode's surface. | not idle — a clip playing/queued or a prompt held |
| `reset` | — | Drops both queues, cuts any playing clip, restores every default (keeping the current mode). Replies `session_reset`. | — |
| `get_state` | — | Replies with the full `state_update` snapshot. | — |

A rejected command has no effect and is answered by a broadcast
`command_error` naming the command and the reason.

## Messages

| Message | Reaches | When |
|---|---|---|
| `state_update` | everyone | On connect, and after every change. A complete snapshot minus the queue's contents — render from this plus `queue_update` alone. |
| `queue_update` | everyone | On connect, and whenever either queue changes: an enqueue or `move`, a build crossing into playout, a clip leaving to play or by `pop`, a reset. Carries both queues in full, front first. |
| `clip_queued` | the caller | Reply to `enqueue`. The full `ClipInfo`, UUID included. |
| `clip_generated` | everyone | A build finished: the clip left the generation queue and joined the back of the playout queue, playable immediately. |
| `clip_moved` | the caller | Reply to `move`. Names the queue and the landing position. |
| `clip_started` | everyone | A clip's first frames reach the tracks. |
| `clip_finished` | everyone | A clip was fully sent; the stream is now black until the next `play`. |
| `clip_stopped` | everyone | `stop` (or `reset`) cut the clip; the rest of it is discarded. |
| `clip_failed` | everyone | A build failed; the clip left the generation queue and builds move on. |
| `clip_popped` | the caller | Reply to `pop`. The clip left its queue and the slot is free. |
| `clip_length_accepted` | the caller | Reply to `set_clip_seconds`. Carries the snapped value. |
| `seed_accepted` | the caller | Reply to `set_seed`. |
| `autoplay_accepted` | the caller | Reply to `set_autoplay`. |
| `canvas_accepted` | the caller | Reply to `set_canvas`. Carries the exact pixel size. |
| `prompt_accepted` | the caller | Reply to `set_prompt` (continuity). Carries the prompt the take now follows. |
| `continuity_accepted` | the caller | Reply to `set_continuity`. Carries the mode now in force. |
| `session_reset` | the caller | Reply to `reset`. Says how many clips were dropped. |

## Session lifecycle

```
  session starts (no clients yet)
    |
    v
  client connects       -> state_update + queue_update (to this client)
    |
  ┌───────────────────────────────────────────────────────────────┐
  │ IDLE (black screen)                                           │
  │ Valid: enqueue, set_clip_seconds, set_seed, reset, get_queue, │
  │        get_state; set_canvas while the queue is empty;        │
  │        play once a clip is ready                              │
  │ Builds consume the generation queue in the background        │
  └───────────────────────────┬───────────────────────────────────┘
                              v  play
  ┌───────────────────────────────────────────────────────────────┐
  │ PLAYING one clip                                              │
  │ Valid: enqueue, set_clip_seconds, set_seed, stop, reset,      │
  │        get_queue, get_state                                   │
  │ Messages: clip_started, then clip_finished or clip_stopped    │
  │ Builds keep running behind the playout                        │
  └───────────────────────────┬───────────────────────────────────┘
                              v  clip ends / stop / reset
                    (flush to black, back to IDLE)
```

**Single session, shared state.** Several clients may attach to one session and
they all see the same queue and the same stream: an `enqueue` or a `stop` from
any client affects everyone, and every client receives every `state_update` and
`queue_update`. Generation is gated on having an audience — with nobody
connected no new build starts, though a build already running finishes into the
queue.

`state_update.valid_commands` names exactly what the session would accept at
that moment, so a frontend enables and greys out controls from the snapshot
instead of re-deriving these rules.

## What to expect from the timing

- **Enqueue-to-ready** is one build, plus the wait behind earlier queued
  builds; `queue_update` reports the clip turning `ready`. On the shipped
  profile (source-built sm100a kernel, regional compile, replicated DiT,
  pinned offloaded text encoder) a 14.375 s clip builds in **14.4 s on four
  B200s — 1.0x realtime**, flat across prompts. A `seconds` value the
  deployment did not warm adds a one-off ~20 s compile on its first build —
  the shipped `warmup_lengths: "all"` prevents that for every legal length;
  see the recompile notes under Deployment learnings.
- **Play-to-first-frame** is near-instant for a ready clip — the frames are
  already in host memory; the only latency is the transport.
- **`stop`** cuts to black within a fraction of a second: the emitter checks the
  flag every slice (about an eighth of a second) and whatever the transport
  still holds is flushed. A build in flight for another clip is unaffected —
  and cannot be cancelled, so `reset` may keep the GPUs busy for a few more
  seconds finishing a clip it will then discard.

Playout is a strict 24 fps metronome and the audio is sample-clocked against
the same rate, so the two tracks stay locked for the length of any clip.

## Two modes: continuous take, or hard-cut queue

The model runs one of two modes, fixed at load by `inference.continuity`.

**Continuity (`continuity: true`, the shipped default)** is one continuous
take. `set_prompt` holds a prompt and the model builds clips back to back
forever, each after the first FL2VA-anchored on the previous clip's last frame,
its exposure locked to the opener's, and every boundary crossfaded in linear
light — so subject, framing and voice carry across as one uninterrupted stream
until `stop` or a new `set_prompt`. A new `set_prompt` re-anchors: the next clip
opens fresh on the new prompt and the held seam tail dissolves the old scene
into it. There is no queue and no `play`; the take starts the moment a prompt is
set and an audience is connected. The FL2VA and Ref2VA conditioning this uses
were not distilled into the checkpoint, so the anchored continuation is a
best-effort carry rather than a trained one — good enough to dissolve a seam,
not a guaranteed identity lock.

**Queue (`continuity: false`)** is the clip-at-a-time channel described above:
every clip generated independently, `enqueue`d and `play`ed, the stream holding
on black between plays with a hard cut at each boundary — no continuity of
subject, framing, or voice, even with identical prompts. `runtime.recording` is
left disabled because a recording would carry those cuts. The streaming client
drives this mode.

The two are **disjoint command surfaces** (`fasth3_session_rules.py`):
continuity offers `set_prompt`/`stop`/`reset`/`set_seed`/`set_canvas`, the queue
offers `enqueue`/`play`/`move`/`pop`/… — `state_update.valid_commands` names the
live set, and the other mode's commands are refused. Everything else (tracks,
canvas, seed, clip length, warm-up, the engine profile) is shared.

The mode is **config-defaulted but runtime-switchable**: `inference.continuity`
sets the starting mode, and `set_continuity` flips a session between the two
while it is idle (nothing playing, queued, or a prompt held — the same gate as
`set_canvas`), so a client chooses hard cuts or continuity per session without a
redeploy. `reset` keeps whichever mode is in force. The switch is offered only
while idle so no clip or take straddles it; the run loop, parked in the current
mode's idle branch, returns and re-dispatches to the other on the flip.

The geometry it accepts is narrow, and [`fasth3_clip_plan.py`](./fasth3_clip_plan.py)
encodes it: 24 fps, a frame count of the form `17n + 5`, a duration between 5
and 15 seconds, at most 768×1344 pixels, and both sides a multiple of 32. The
duration cap has a sharp edge — 15.0 s is 360 frames, which aligns *up* to 362
(15.083 s) and is then rejected, so **the longest clip this model can make is 345
frames, 14.375 s**.

The **short edge is a selectable resolution tier**, `inference.canvas_short_edge`
(a multiple of 32, 256–768), and every `set_canvas` aspect resolves at it. 768 is
the trained tier every published single-clip number was measured at; 640 (the
shipped continuity default) makes a 16:9 clip 1152×640 — ~2.9% fewer pixels,
built proportionally faster, which is the headroom that keeps continuity's
back-to-back chain ahead of playout. The default is 768 when the key is absent,
so a queue deployment is unchanged.

## Determinism

Each enqueued clip carries its own seed — passed explicitly on `enqueue`, or
taken from the session's advancing default (`set_seed` fixes it; each seedless
enqueue advances it by one, and explicit seeds leave it untouched).
Re-enqueuing the same prompts with the same seeds, clip length and canvas
reproduces the same clips. Reproduction is approximate rather than bit-exact:
the deployment runs fused and compiled kernels that can reorder floating-point
operations.

## Notes on the code

**`ReactorModel`, not `ReactorPipeline`.** The generator base is shaped for
frame-per-step models, where a `yield` is the natural unit of work. Here the
unit is a whole clip produced by one blocking call, so the generator would buy
nothing. Under `ReactorModel`, command handlers and lifecycle hooks run on
background coroutines *concurrent* with `run()`, so:

- session state is plain attributes reset in `_reset_session_state`, called from
  `@session_started` (not `@connected`: a client rejoining mid-session keeps the
  queue it joined);
- one persistent worker thread serialises every build and gives teardown a
  single handle to wait on;
- refusals and accepts answer immediately, even mid-build and mid-play.

**Built clips live in host memory.** A built clip is about 1 GB of uint8
pixels at the 16:9 canvas and the longest length, and the playout queue can
hold `inference.queue_size` of them — size `resources.memory` in the manifest
together with that knob. Generation pauses while the playout queue is full,
which is what keeps that budget a hard bound.

**Refusals are broadcast, not raised.** A handler returns only the message its
annotation names and reports failure by broadcasting `command_error`. A raised
runtime `CommandError` has its failure frame withheld from v0 clients, so the
broadcast is what reaches every SDK generation.

**Nothing is vendored.** FastVideo is a published package, so the whole upstream
tree arrives through `requirements.txt` and an upgrade is a one-line bump.

## Layout

| Path | What it is |
|---|---|
| `fasth3.py` | The `ReactorModel`: commands, lifecycle, the queue/playout loop |
| `fasth3_types.py` | Everything a client sees — tracks, `ClipInfo`, messages |
| `fasth3_queue.py` | The bounded clip queue and its entries |
| `fasth3_backend.py` | The FastVideo engine, its worker thread, warm-up, audio conversion |
| `fasth3_assets.py` | Config parsing and weights-bundle validation |
| `fasth3_clip_plan.py` | Clip geometry: valid lengths, frame counts, canvases, resolution tiers |
| `fasth3_seam.py` | Continuity's pure-numpy exposure lock and linear-light seam crossfade (no torch) |
| `fasth3_session_rules.py` | Which commands each session state accepts, per mode |
| `fasth3.yaml` | `inference:` the recipe, queue size and warm-up plan, `runtime:` weight layout and engine shape |
| `reactor.yaml` | The manifest: identity, version, resources, runtime, image build |
| `sitecustomize.py` | Interpreter-wide fixes in every container process, spawned engine workers included (via `PYTHONPATH=/app`): raises dynamo's recompile limits, widens the VSA kernel's device gate to every built arch |
| `tests/` | Structural tests that need no GPU |
| `client/` | Reference SDK client that drives the whole queue contract and saves what it receives |

## Deployment learnings

Everything below was established empirically on a 4×B200 box (driver 595,
CUDA 13.1 image, torch 2.12+cu130) while bringing this model to its published
speed. Read this before touching `fasth3.yaml`, `requirements.txt`, or the
engine seam.

### The profile that hits 1.0x realtime, and why each piece is there

Measured end state: **14.4 s per 14.375 s clip on four B200s**, level with
FastVideo's published 15.5 s for this configuration (theirs includes file
muxing; this deployment streams instead). Conditioning ~1 s, denoise ~8.5 s
equivalent, decode ~2.5 s. Play-to-first-frame 0.22–0.25 s; `stop`-to-black
~0.13 s; load-to-serving ~3.5 min, of which ~90 s is the warm-up build.

- **The sm100a kernel is built from source at image build** (see
  `requirements.txt`). The published fastvideo-kernel 0.3.5 wheel's sm_100a
  binary fails *every* launch on this driver with `invalid argument`, eager
  and compiled alike; the identical source compiled by the image's CUDA 13.1
  nvcc is correct and fast. The PyPI sdist cannot stand in — it ships without
  its CUTLASS/ThunderKittens submodules — so the pin is the git release
  commit, with `TORCH_CUDA_ARCH_LIST=10.0a;10.3a` from `build_env` (B200 and
  B300; arch-specific binaries do not run forward even within a family). The
  triton fallback route works everywhere but is ~2.5x slower.
- **Prompts are padded to exactly 256 tokens** (`fasth3_backend.py`,
  `PROMPT_TOKENS`) using the bundle's own tokenizer. Regional torch.compile
  keys its capture on the packed sequence length, prompt tokens included, so
  a novel prompt length recompiled the transformer — ~23 s per clip, on
  almost every clip of a real feed. Upstream's benchmark never sees this
  because it reuses one prompt for warm-up and every measured request. The
  client-facing prompt in `ClipInfo` stays the original text.
- **Replicated DiT + text encoder offloaded to pinned host memory** is what
  fits four ranks and stays fast: FSDP sharding saves VRAM but roughly
  doubled the denoise; the offloaded Qwen3-VL costs ~63 GB of pinned host
  memory per rank and ~1 s page-in per clip (unpinned it was ~15 s).
  `resources.memory` in `reactor.yaml` is sized for those four host copies
  plus the built-clip buffer (`queue_size` x ~1 GB of uint8 pixels).
- **flash-attn-4 and the runtime cannot be installed together naively**: the
  runtime requires `protobuf>=7.35.1` (its generated bindings hard-reject an
  older runtime at import) while FA4's pinned `nvidia-cutlass-dsl` caps
  protobuf below 7 — a stale cap, since protobuf accepts old gencode on a
  newer runtime. `build_env` sets `UV_OVERRIDE=/app/requirements.txt`, which
  feeds the requirements back as resolver overrides and lets the higher floor
  win. Side effect handled: overrides strip torch coupling from resolution,
  so torchvision (0.27.x) and torchaudio (2.11.0, the newest cu130 build;
  its `functional.resample` is pure torch ops) are pinned to torch-2.12
  matched builds explicitly.

### B300 and the kernel's device gate (resolved)

Compiling the kernels for `10.3a` is necessary but not sufficient on a B300:
`fastvideo_kernel.block_sparse_attn_sm100a.is_supported` compares the
device's compute capability against a `(10, 0)` constant **by equality**
(upstream still does, as of the 0.3.5 pin), so a B300 — capability
`(10, 3)`, with a correct sm_103a binary sitting in the same extension — is
refused. The failure is silent and double: the engine falls back to the
Triton-64 VSA kernels *and*, because regional compile asks the same
predicate, the whole transformer stays **eager** (the pod logs
`falling back to the Triton-64 kernels` and `inference_torch_compile
requested but disabled` once per worker). Measured on eight B300s: 19.3 s
per 14.38 s clip — 0.74x realtime — with denoise at ~10.3 s.

The fix rides `sitecustomize.py` (the same import hook that raises the
recompile limits, so it lands in every spawned engine worker): after the
kernel module executes, its `_SM100` constant is replaced with a value equal
to every capability in the image's `TORCH_CUDA_ARCH_LIST`. The two lists are
kept in sync by a test. Watch the per-clip log line after any kernel or
FastVideo bump: those two warnings coming back is the regression signature.

### CPU quota and thread sizing (resolved)

The same image and config that hit 1.16x on an unthrottled HGX B200 box
built at 0.85x on the hosted pod — identical GPUs, identical NVSwitch
fabric (`nvidia-smi topo -m` showed NV18 on every pair). The variable was
CPU: with `requests.cpu: 8, limits.cpu: 32` on a 192-vCPU node, the cgroup
throttled 38% of CFS periods, and runnable threads waited on quota ~3x
longer than they ran. The stage split wears the signature: latent prep
1.65 s vs 0.3 s (the CPU-bound stage eating the throttle directly) and
denoise 9.5 s vs 5 s (NCCL host threads and kernel-launch threads stalling
between all-to-alls — the fabric was never the problem).

Two fixes, both in `reactor.yaml`: `resources.cpu` raised as far as the
account's model CPU quota allows — the registry 429s any limit above the
quota (64 today), so request and limit are both pinned there, the full
quota guaranteed and none of it burstable — and `OMP_NUM_THREADS=8` in
`runtime_env`, because `nproc` inside the container reports the whole
node, so every torch process otherwise sizes its threadpools for 192
cores and nine processes pile up against whatever the quota is. Diagnose
a recurrence from inside the pod: `cpu.max` and the throttle counters in
`cpu.stat` (cgroup v2), against the per-stage clip log line.

### Varied clip lengths and the recompile limit (resolved)

Every distinct `seconds` value is a new compile shape. torch dynamo's
`recompile_limit` defaults to **8**, and the regional-compile route runs
fullgraph, where exceeding the limit is a **hard failure**
(`FailOnRecompileLimitHit`) that kills the engine workers and takes the whole
serving process down — observed live after enqueueing many different lengths.
The fix has to respect two facts, both verified: torch 2.12 maps no
environment variable onto the limit, and it must be raised inside the *engine
worker* processes — they are spawned, never forked, so parent-process
settings do not carry over, and FastVideo's own imports re-cap the limits
in two of its modules (`layers/lora/linear.py` sets 16,
`third_party/longcat_video/.../bsa_interface.py` sets 32).

The shipped mechanism is [`sitecustomize.py`](./sitecustomize.py):
`runtime_env` in `reactor.yaml` sets `PYTHONPATH=/app`, so every Python
interpreter in the container — the runtime and each spawned worker — imports
it at start. It installs an import hook that re-raises the limits *after*
`torch._dynamo.config` and after each of FastVideo's two lowering modules
execute, so the highest setting wins regardless of import order. The limit is
`FASTH3_DYNAMO_RECOMPILE_LIMIT` (default 64 — 14 legal lengths times the
warmed canvases, with headroom); the backend also raises the parent process
directly at `load()`. The compile *stall* on a first-seen length is a
separate cost, covered by `inference.warmup_lengths` below.

### Serving mechanics worth knowing

- **The container needs a large `/dev/shm`** (`--shm-size=32g` in the local
  invocation): the engine workers hand decoded frames back over torch shared
  memory, warm-up never exercises that path (`return_frames=False`), and
  docker's 64 MB default kills the first real clip. `reactor run` has no
  shm flag today, hence the documented raw `docker run` equivalent.
- **GPU count must divide H3's 56 attention heads** (1, 2, 4, 7, 8 …) — six
  GPUs is not a configuration, the engine refuses at init.
- **One session per local runtime.** A second client must join with
  `connect(session_id=...)`; a plain `connect()` gets a 409 while a session
  streams. Sessions are multi-connection: broadcasts and media fan out to
  every connection, command replies go to the caller only.
- **The SDK's local mode honours a custom port** via
  `Reactor("fast-h3", local=True, api_url="http://localhost:<port>")` — only
  the default URL is rewritten to 8080.
- **Audio is deliberately mono int16 at 48 kHz**: the transport downmixes
  anyway and the runtime recorder corrupts stereo by concatenation, so the
  downmix happens once, in float, before quantization.
- The per-clip log line (`clip built: … = N.NNx realtime … stages={…}`) is
  the number to watch; the runtime's log formatter drops structured extras,
  which is why the figures live in the message text.

### Still open

- **Post-decode cost.** `return_frames=True` also allocates an fp32 mirror of
  the decoded video that nothing reads — several GB per clip. If
  `PostDecodeFrameProcessStage` ever dominates the stage split, the fix
  belongs upstream in FastVideo.
- **The dependency closure is not exact.** `requirements.txt` lists what the
  model imports plus the pins above; regenerate it from a `pip freeze` of a
  built image if byte-comparable rebuilds start mattering.
