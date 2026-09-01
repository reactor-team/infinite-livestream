# Agent instructions for infinite-livestream

> **Keep this file current — it is the map, and it must not lag the code.**
> When a change adds a component, an env var, a command, an invariant, or
> moves behaviour any document describes, update that document **in the same
> change**: this file for the system picture, invariants, and routing table;
> `streaming-client/README.md` for client detail; the `base.py` docstrings
> for interface contracts; `.env.example` for configuration. A stale map is
> worse than none — the next agent trusts it and drifts. `CLAUDE.md` is a
> symlink to this file; never let them diverge into two copies.

This repo is a complete, working system: a chat-driven infinite AI video
broadcast. `fast-h3/` is the model — a queue of prompt-driven clip
generations, served by the [Reactor Runtime](https://github.com/reactor-team/reactor-runtime)
as a `reactor` CLI workspace. `streaming-client/` is the client — it reads
`!prompt` ideas from Twitch/YouTube chat, upsamples them into styled scene
sequences with an LLM, feeds the model's queue over `reactor-sdk`, and pushes
the output to an RTMP ingest as one uninterrupted stream. They meet only on
the wire; `fast-h3/fasth3_types.py` is that contract.

**Contribution policy:** never commit, push, or open PRs without explicit
permission from a human maintainer in the current conversation. Commits are
signed off (`git commit -s`), imperative title, body explaining the why as
end state (no iteration narration).

## How the system works — read this before changing anything

The model is **two queues plus a player**. `enqueue` takes a prompt
(≤ 800 chars), an opaque `metadata` string (≤ 2000 chars), and optionally
`seed`, `seconds` (snapped into 5.167–14.375 s), and `position` in the
**generation queue**; it replies immediately with the clip's full structure
and UUID. Builds consume the generation queue front-first, always (pausing
only while the playout queue is full); each finished clip crosses into the
**playout queue** (`clip_generated`), where `play`, `move`, and `pop` give
the client full positional control. Nothing plays until `play` (this client
keeps `set_autoplay` off and drives playout itself). Playback streams 24 fps
video (`main_video`, 1344×768 at the default
16:9 canvas) and 48 kHz mono int16 audio (`main_audio`), then flushes to
black and holds. The metadata is echoed untouched on every message that
references the clip — it is how a client correlates clips with its own
records without local joins.

The client is a straight pipeline:

```
chat (Twitch IRC / YouTube API) → Director → Moderator → PromptUpsampler (LLM)
  presets/<n>.json (filler) ───↗
       → ReactorLink (enqueue + play) → fast-h3
       → Pacer (constant-rate clock) → StreamSink (rtmp | noop | yours)
```

One chat prompt becomes one **scene group**: a single scene, or a chunked
short story of up to `MAX_CHUNKS` short clips — the upsampling LLM picks the
shape — enqueued contiguously, each clip tagged with a JSON group id in the
metadata. The director drives playout itself (autoplay off): `pick_next`
plays viewer content before filler from the playout queue, judged purely
from the metadata echo, and viewer groups insert into the generation queue
ahead of waiting filler and behind waiting viewers (`enqueue`'s `position`)
so the GPUs build them next, FIFO among viewers. A viewer backlog at the
clip budget drops new prompts, and every capacity is read live from
`state_update`. Viewer prompts pass the OpenAI
moderations API first (own endpoint, fail-closed; see `moderator.py`).
Chatters on the `ADMIN_USERS` list can also send admin commands, routed to
`streaming-client/admin.py` ahead of the director: `!switch <preset>` swaps
the creative preset live, resolved against `presets/` at switch time, and
flushes both model queues down to one buffer clip so the new identity shows
on stream fast. While
chat is quiet, an idle filler tops the queue up to `IDLE_QUEUE_TARGET` with
single-scene groups tagged `generated: true`; viewer groups evict those
(`pop`) when they need the room. The **pacer** converts clip-shaped output
(24 fps while playing, nothing between clips) into the frame-every-period
stream RTMP requires, filling gaps with repeated frames and silence, and
passes every outgoing frame through the **overlay** (queue depth, playing
scene title/author, coming up — all reconstructed from the metadata echo).

### Load-bearing invariants — violating any of these breaks the product

1. **The two systems stay decoupled through the metadata.** The model knows
   queue positions (`enqueue`'s `position`, `move`) and playback, never who
   asked for a clip; viewer-vs-filler lives only in the metadata echo, and
   the client's `pick_next` (in `group_tag.py`) is the single playout
   policy — autoplay chains the playout front for gapless transitions, the
   director's `run_playout` curates that front with `move`, and the
   overlay's "coming up" reads the same function. Group sequencing rests on the
Director being the queues' only writer (viewer worker, idle filler, and
 the preset-switch flush serialize through one lock), viewer FIFO on
 positional inserts ahead of
   filler (`viewer_insert_position` → `enqueue`'s `position`), and a group
   being enqueued only when all its scenes fit the generation queue —
   waiting-filler eviction (`pop` on `generated: true` clips only) makes
   that room, one built filler per tick is popped when a full playout
   queue blocks a viewer build, and a viewer backlog at the clip budget
   drops new prompts. Do not add orchestration beyond this, and do not
   break any of these legs.
2. **The pacer and sink survive Reactor reconnects; the queue does not.**
   Sink + pacer are created once and never torn down mid-run — that is what
   keeps the platform-side broadcast unbroken. Server-side session state
   (the queue included) dies with a session.
3. **Sinks never block the event loop** and receive a perfectly regular,
   pre-paced stream. The contract is `streaming-client/sinks/base.py`'s
   docstrings; same for chat sources in `chat/base.py`. Those docstrings are
   authoritative — READMEs only summarize them.
4. **Prompts are hard-truncated to 800 chars and scene lengths clamped to
   the live bounds** from `state_update` — never trust the LLM's counting,
   never hardcode bounds the deployment publishes.
5. **The schema is product surface.** Every `@event`/`InputField`/
   `MessageField` description and `ModelMessage` docstring in `fast-h3/` is
   compiled into the published schema. Describe only what a client can
   observe on the wire (commands, messages, tracks, by wire name in
   backticks) — never internals (kernels, caches, config keys, GPU counts).

### Where each kind of change goes

| You are changing… | Edit | Then also |
| --- | --- | --- |
| Model behaviour (queue, playout, engine) | `fast-h3/fast-h3*.py` along the file seams below | tests; bump `model.version` if the surface moved |
| The wire contract (commands, messages, fields) | `fast-h3/fasth3_types.py` + the handler | `streaming-client/reactor_link.py` + `director.py` mirror it; both READMEs; version bump sized to schema impact |
| Stream delivery (encoding, destinations) | `streaming-client/sinks/` | register in `make_sink`, `.env.example`, README sink table |
| Prompt sources | `streaming-client/chat/` | `build_chat_sources` in `main.py`, `.env.example`, README |
| Admin chat commands (`!switch`, the `ADMIN_USERS` list) | `streaming-client/admin.py` (routed ahead of the director in `main.py`) | `.env.example`, README's Admin commands section |
| Upsampling behaviour / the style prompt | `streaming-client/upsampler.py` | keep the constraint rules intact — the rationale is in the module docstring |
| Moderation policy | `streaming-client/moderator.py` | keep it fail-closed; README's Moderation section |
| Idle filler / eviction | `streaming-client/director.py` (`run_idle`, `_evict_fillers`) | README's Idle filler section |
| A stream's creative identity (style + premade prompts) | `streaming-client/presets/<name>.json` (format in `config.py`'s `load_preset`; only `default.json` is tracked; admins swap presets live with `!switch`) | README's Idle filler + Prompt upsampling sections |
| What the broadcast shows on top of the video | `streaming-client/overlay/` (contract in `base.py`; shipped overlay in `status.py`) | README's Overlay section; keep compose non-mutating and per-frame cheap |

## Model rules (`fast-h3/`) — distilled from the Reactor cookbook

- **Start from the skill:**
  [`skills/reactor-fast-h3-model/SKILL.md`](./skills/reactor-fast-h3-model/SKILL.md)
  is the full context handoff — what FastH3/MiniMax-H3 is, how the Reactor
  Runtime serves it, the queue contract and the decisions behind it, the
  measured 1.0x-realtime profile and its load-bearing pieces (source-built
  sm100a kernel, 256-token prompt padding, warmed lengths plus the
  `sitecustomize.py` recompile-limit raise, replicated DiT with the pinned
  offloaded text encoder, the protobuf resolver override), and how to build
  and run with the reactor CLI or raw docker. **Maintain it in the same
  change whenever model work moves any of that** — contract, profile,
  serving mechanics, or an open item closing — exactly as this file demands
  of itself. [`fast-h3/README.md`](./fast-h3/README.md) stays the per-file
  detail and Deployment learnings record.
- **fast-h3 deliberately subclasses `ReactorModel` with its own `run()` loop**
  (not `ReactorPipeline`): its unit of work is a whole clip, and command
  handlers must answer while a clip builds or plays. Do not "normalize" this.
- **File seams are fixed.** `fasth3.py` owns commands + both run loops (the
  queue's playout and continuity's held-prompt take); `fasth3_types.py`
  everything a client sees; `fasth3_queue.py` the bounded queue;
  `fasth3_backend.py` the FastVideo engine + worker thread, and continuity's
  post-decode pipeline that runs on it (FL2VA anchor, GPU exposure lock, GPU
  seam blend); `fasth3_seam.py` the pure-numpy exposure lock and linear-light
  crossfade (no torch); `fasth3_image.py` decoding a `set_seed_image` upload
  (a still, image-to-video) to the seed frame that anchors clip 0;
  `fasth3_assets.py` config parsing and weights validation
  (the only reader of `fasth3.yaml`); `fasth3_clip_plan.py` pure clip geometry
  and resolution tiers; `fasth3_session_rules.py` which commands each state
  accepts, per mode. New code goes in the seam that owns it. No `__init__.py` —
  modules import flat.
- **Two modes, one class, disjoint surfaces.** `inference.continuity` (default
  on) selects a continuous single-prompt take (`set_prompt`, FL2VA-chained,
  crossfaded into one stream) over the client-driven hard-cut queue
  (`enqueue`/`play`). Both live in `FastH3`; `run()` dispatches on the
  per-session `self._continuity` flag (seeded from the config), and the modes
  share nothing but the engine, tracks, and geometry.
  `state_update.valid_commands` names the live surface. `set_continuity` flips
  the flag at runtime while the session is idle; `run()`/`_serve*` re-dispatch
  on the change, so read `self._continuity`, never `self.config.continuity`, in
  any per-session path. Keep the queue path byte-for-byte unchanged when
  continuity is off — the streaming client drives the queue.
- **Typed contracts.** Every `@event` handler declares and returns a concrete
  `ModelMessage` (or `None`); a refusal broadcasts `command_error` and
  returns bodyless. State-changing commands also broadcast a full
  `state_update`.
- **Moderation marks.** Every free-text `InputField` a client fills (prompt,
  metadata) sets `moderate=True`. Enum/bounded fields never do.
- **No ghost surface.** No undecorated command-shaped methods, no message
  classes nothing sends, no write-only attributes. Git history is the archive.
- **Manifest.** `fast-h3/reactor.yaml` orders `model:`, `runtime:`, `build:`;
  `model.version` is `v`-prefixed semver bumped with every shipped change,
  sized to the schema impact; `build.runtime_version` pins the Reactor
  Runtime release. Weights never live in git. CUDA 13 and the source-built
  fastvideo-kernel are requirements, not preferences — the comments in
  `reactor.yaml` / `requirements.txt` explain each pin; read them before
  touching versions.
- **Comments and docstrings describe the end state.** No "previously", "no
  longer", no narrating what the code visibly does.

## Client rules (`streaming-client/`)

- `streaming-client/README.md` is the detailed documentation: architecture,
  the RTMP/ffmpeg learnings (raw-video geometry, writer threads, dual-pipe
  A/V, restart policy), and maintenance notes. Keep it current in the same
  change that moves behaviour it describes.
- Built on `reactor-sdk >= 1.1.1` (`reactor.on(...)`, `tracks...one()`,
  `track.on_frame`). Do not copy patterns from pre-1.0 SDK examples.
- Config comes only through `config.py` (env / `.env` / CLI). Never commit
  `.env` — it holds real API and stream keys; `.env.example` is the template.
  Never print keys; the RTMP sink redacts the stream key in logs.

## Documentation: who owns what, and the self-maintenance rule

Four documents, four scopes — edit the owner, never a copy, and edit it in
the same change as the code it describes:

| Document | Owns | Update when… |
| --- | --- | --- |
| `AGENTS.md` (this file; `CLAUDE.md` symlinks here) | System picture, load-bearing invariants, change routing, verification | any invariant, component, or workflow moves |
| `streaming-client/README.md` | Client architecture, the ffmpeg/RTMP learnings, moderation & idle-filler behaviour, run instructions | client behaviour it describes moves |
| `streaming-client/{sinks,chat,overlay}/base.py` docstrings | The sink, chat-source, and overlay interface contracts | the contract itself changes (READMEs only summarize these) |
| `skills/*/SKILL.md` | Deep context handoffs: `reactor-fast-h3-model` (the model, its profile, serving), `reactor-streaming-client` (the client's pipeline and policies) | anything they narrate moves — same change, per their own closing sections |
| `streaming-client/.env.example` + `fast-h3/README.md` | Every knob, with its default; the model's own story | a knob or model surface is added/renamed |

A PR that changes behaviour without touching the document that describes it
is incomplete — flag it in review, and as an agent, fix it before finishing
the task. When you find the docs and the code disagreeing, the code is the
truth; correct the document in the same change and say so.

## Verifying a change

```sh
# Model: the contract renders, and only the intended surface moved.
cd fast-h3
python -m reactor_runtime.schema --path . --out /tmp/schema.json   # diff before/after
PYTHONPATH=. python -m pytest tests/ -q

# Client: compiles, and the pipeline runs end to end without an ingest.
cd streaming-client
python -m py_compile main.py config.py pacer.py reactor_link.py director.py upsampler.py sinks/*.py chat/*.py
python main.py --local --sink noop        # against a local `reactor run`

# Raw queue contract smoke test (writes .mp4s + timing report):
python fast-h3/client/client.py            # or --api-key rk_... for hosted
```
