# fast-h3 streaming client

A chat-driven livestream client for the fast-h3 clip-queue model. It reads
`!prompt` requests from Twitch and/or YouTube chat, moderates them, upsamples
each one into a styled sequence of one or more video scenes with an LLM,
enqueues those scenes on a served fast-h3 model (local `reactor run` or hosted
with an API key), and forwards the model's video+audio output as one
uninterrupted broadcast to a pluggable **sink** — RTMP (Twitch / YouTube
Live / Kick) today, a no-op sink for dry runs, and an interface designed for
LiveKit / SFU sinks tomorrow. While chat is quiet, an **idle filler** keeps
the queue topped up from a curated prompt list so the stream never runs dry.

Built on `reactor-sdk >= 1.1.1` (the current SDK surface: `reactor.on(...)`,
`reactor.tracks...one()`, `track.on_frame`). The pre-1.0 examples in the
py-sdk repo (`@reactor.on_frame`, `reactor.get_status()`) are an older API —
do not copy patterns from them into this client; their hard-won *ffmpeg*
learnings are already baked into `sinks/rtmp.py`.

## Architecture

```
Twitch IRC ─┐
            ├─▶ Director ──▶ Moderator (OpenAI moderations API)
YouTube API ┘      │   └───▶ PromptUpsampler (OpenAI-compatible LLM)
                   │              ▲
presets/<n>.json ──┘ (filler)     │
                   ▼  enqueue: scene groups, tagged via clip metadata
              ReactorLink ◀──▶ fast-h3 (clip queue; director sends play)
                   │  24 fps video + 48 kHz audio (per clip, black between)
                   ▼
                 Pacer  ──── constant-rate clock: fills gaps with
                   │         repeated frames + silence
                   │ ──── Overlay (queue badge, now playing, coming up)
                   ▼
              StreamSink ──▶ rtmp (ffmpeg) | noop | (yours)
```

| File | Owns |
| --- | --- |
| `main.py` | Wiring and task lifecycle; nothing else. |
| `config.py` | The only reader of `.env` / environment / CLI. |
| `reactor_link.py` | Everything that touches `reactor_sdk`: connect/reconnect loop, media → pacer, `state_update`/`queue_update` mirror, command sending, message fan-out. |
| `director.py` | Chat prompt → moderate → upsample → enqueue scene groups; idle filler + eviction; per-author cooldown; now-playing narration from metadata. |
| `admin.py` | Admin chat commands from the `ADMIN_USERS` list: `!switch <preset>` swaps the creative preset live. Routed ahead of the director in `main.py`. |
| `upsampler.py` | The LLM call and the system prompt; scene validation (char cap, length clamp, chunk-count cap). |
| `moderator.py` | The moderations call and the fail-closed policy. |
| `presets/` | Creative presets: one JSON per stream identity — the style block plus the premade idle prompts. `default.json` ships; other presets stay untracked. |
| `pacer.py` | The 24 fps metronome between bursty/clip-shaped model output and the sink's need for a frame + audio every period, forever. Hands each outgoing frame to the overlay. |
| `overlay/` | The per-frame decoration interface (`base.py`) and the shipped status overlay (`status.py`). |
| `sinks/` | The output interface (`base.py`), ffmpeg RTMP (`rtmp.py`), no-op (`noop.py`), factory (`__init__.py`). |
| `chat/` | The chat-source interface (`base.py`), anonymous Twitch IRC (`twitch.py`), YouTube Data API poller (`youtube.py`). |

## The model side, in one paragraph

fast-h3 (see `../fast-h3/fasth3_types.py`, the authoritative client-facing contract) is
**two queues**: `enqueue` takes a prompt (≤ 800 chars), an opaque `metadata`
string (≤ 2000 chars, echoed back on every message that references the
clip), and optionally `seconds` (snapped into 5.167–14.375 s), `seed`, and
`position` (where it enters the **generation queue**; 0 = the next build).
Builds consume the generation queue front-first on their own; each finished
clip crosses into the **playout queue** (`clip_generated`), which is
entirely the client's to schedule: `play {clip_id}`, `move`, and `pop` work
on both queues. This client keeps autoplay **on** — the model chains the
playout front the instant the stream idles, which is what keeps clip-to-clip
gaps at milliseconds — and curates that front with `move`; a playing clip
streams on `main_video` (1344×768 @ 24 fps
at the default 16:9 canvas) and `main_audio` (48 kHz mono int16), then
flushes to black until the next `play`. Both queues are bounded
(`generation_capacity` / `playout_capacity` in `state_update`; generation
pauses while playout is full) and a full generation queue refuses `enqueue`
with a `command_error`.

### The hosted deployment's extended contract

The hosted deployment, **`reactor/fast-h3`**, extends that contract, and
this client uses the extension when it is there:

- **Clip continuation.** `enqueue` also takes `continue_from_clip_id`: the
  new clip opens on the named clip's last frame, and when autoplay's next
  clip continues the one just finished the handover is **seamless** — no
  black, no cut, one uninterrupted video. The source may be any clip in
  either queue or in `queue_update.history` (built clips stay
  referenceable for a while after playing, evicted oldest first); a clip
  continuing an unbuilt source just waits its turn. `pop` refuses an
  unbuilt clip that queued clips still continue from — pop dependents
  first (the preset-switch flush does).
- **Starting images.** `enqueue` also takes a `starting_frame` upload
  (image-to-video). This client deliberately does not use it yet — it is
  deferred to a separate feature where an episode contributes the image.
- **`set_flush_on_clip_end`.** Chooses whether non-continuing clip
  boundaries cut to black (default) or hold the last frame. This client
  keeps the default.

The client **detects** the extension from `state_update`
(`link.supports_continuation`) rather than assuming it from the model
name: against the original in-repo model every clip stays independent, no
continuation field is ever sent, and scene groups play back-to-back with a
flush between — the pre-extension behaviour, intact.

**The one rule that keeps chains watchable: every chained scene opens on a
described hard cut.** A chained clip re-generates from a generated frame;
a chain written as one continuous take compounds those generation errors
link over link until the picture visibly smears ("cooks"). A hard cut to a
fully described new shot re-establishes the whole image, so the chain
stays sharp indefinitely. The upsampler's chained rules enforce this (see
"Prompt upsampling"); never write or prompt a chained scene as "the camera
continues...".

## Scene groups

One chat prompt becomes one **scene group**. The upsampler picks the shape
the idea calls for: a **single scene** of any legal length, or a **chunked
short story** — up to `MAX_CHUNKS` short clips (each near the model's
minimum length) that read as one story with a setup, development, and
payoff. The director enqueues the group contiguously and also owns
**playout order**: the model's autoplay starts the playout front the instant
the stream idles (millisecond gaps, no client round-trip), and `run_playout`
curates that front with `move` — viewer content before filler, from the
metadata echo (`pick_next` in `group_tag.py`; the overlay's "coming up" uses
the same function, so what is announced is what plays). The rules that keep
it coherent:

1. The model builds generation-queue order and knows nothing about viewers
   vs filler — **who asked for a clip travels only in the metadata**, and
   both build priority (`enqueue`'s `position`, via
   `viewer_insert_position`) and play priority (`pick_next` over the
   playout queue) are the client's decisions. The two systems stay
   decoupled.
2. The director is the **only writer** to the queue — both its viewer worker
   and its idle filler serialize group enqueues through one lock — so groups
   never interleave, and within each class `pick_next` follows queue order,
   which keeps a group's scenes sequential.
3. A group is enqueued only once **all** its scenes fit in the remaining
   capacity, so it can't wedge half-in. Viewer groups may make that room by
   evicting filler clips (see "Idle filler" below).
4. On a continuation-capable deployment, a group's scenes are **chained**:
   each scene after the first is enqueued with `continue_from_clip_id`
   naming the previous scene's clip, so the story plays as one
   uninterrupted video instead of clips separated by black. Every chained
   scene's prompt opens on a hard cut (the upsampler's chained rules — see
   below), which is what keeps a chain from degrading. A chained enqueue
   refused twice (a reconnect loses the source clip server-side) falls
   back to a standalone clip: the scene prompts are self-contained, so a
   lost link costs the seamless handover, never the story.

Each scene's `metadata` carries the group tag as JSON:

```json
{"group_id": "9f2c4e81a0b3", "title": "Neon Alley", "scene": 2, "scenes": 3,
 "author": "viewer_42", "source": "twitch", "generated": false,
 "raw_prompt": "a neon alley..."}
```

The model never reads metadata; it echoes it back on `clip_queued`,
`queue_update`, `clip_started`, `clip_finished`, `clip_failed`, ... — which is
why the director can narrate "scene 2/3 of *Neon Alley*" from a
`clip_started` alone, with no local join. Anything downstream (an overlay, a
chat bot announcing scenes) should be built the same way: read the metadata
echo, not client-side state that a reconnect can lose.

## Prompt upsampling

`upsampler.py` calls one OpenAI-compatible endpoint (`OPENAI_BASE_URL` +
`OPENAI_API_KEY`, so a proxy / vLLM / OpenRouter all work) with a system
prompt that embeds the **preset's style/character** (`PRESET` names a
JSON bundle in `presets/`; format in `config.py`'s `load_preset`). The
prompt's rules mirror how fast-h3 actually behaves — keep them intact when
editing (rationale in the module docstring):

- the model reads **only the scene's own text**, so every scene prompt must
  re-describe the full setting/subjects/style from scratch — chained or
  not, anything the text omits vanishes or mutates;
- **chained stories are written as cuts**: when the director will chain a
  group (`chained=True`), the multi-scene rules switch to the chained set —
  each scene after the first begins on the previous scene's final frame, so
  its prompt **must open on a hard cut to a fully described new shot**
  (different camera angle, distance, or location) and must never extend the
  previous take ("the camera continues...", "still on..."). Holding one
  take across chained clips compounds generation errors until the image
  degrades; a described cut re-establishes it. This rule is load-bearing —
  never soften it;
- the LLM is asked for < 750 chars but `_sanitize` **hard-truncates to 800**
  (`MAX_PROMPT_CHARS`, the model's server-side cap) because LLMs can't count;
- fast-h3 renders audio including clear spoken language, so the prompt
  asks for explicit quoted dialogue (who speaks, the exact words, the
  voice's tone) whenever the idea implies speech, plus a soundscape clause
  per scene — plain descriptive staging, no special markup;
- **a single-scene generation always runs the maximum clip length**
  (enforced in code, not just asked of the LLM); short lengths are reserved
  for transition chunks inside multi-scene stories, and `seconds` are
  always clamped to the live bounds from `state_update`;
- safety is the moderator's job, upstream — the upsampler stages the idea
  faithfully and never softens or reinterprets it.

Any LLM failure degrades to a single scene made of style + the raw prompt —
the stream never stalls on the upsampler.

## Moderation

Viewer prompts pass the OpenAI moderations API (`moderator.py`) **before**
they reach the upsampler. Moderation deliberately has its own endpoint and
key (`MODERATION_API_KEY` / `MODERATION_BASE_URL`, falling back to the
`OPENAI_*` values): inference gateways often do not expose `/moderations` —
Reactor's corp gateway answers it with `provider_not_allowed` — so moderation
typically points at api.openai.com while upsampling goes through a gateway.

The policy is **fail closed**: a prompt that cannot be checked is rejected,
so a broken moderation endpoint never silently turns moderation off. To run
without moderation, set `MODERATION_ENABLED=0` explicitly — main.py then
warns at startup. Only the raw viewer prompt is checked, and it is the
**only** safety gate: the upsampler deliberately stages ideas faithfully
rather than softening them, so what passes moderation is what gets rendered.
The idle list is curated in this repo and skips the check.

## Idle filler

When chat is quiet, the director's `run_idle` task keeps the model's queues
(generation + playout together) topped up to `IDLE_QUEUE_TARGET` clips
(default 6) from the preset's `idle_prompts` list (shuffled, then
rotated). The target **self-clamps to one below the deployment's
live playout capacity**: filler must never fill the playout queue to the
brim, because a full playout queue pauses builds, and the headroom slot is
where the next viewer clip lands. Filler prompts run through the same
upsampler and style but are forced to **one scene per group** — the finest
eviction granularity, and popping one never truncates a story — and their
metadata carries `generated: true`.

Viewers always outrank filler — while staying first-come-first-served among
themselves — in four ways:

- viewer groups **insert into the generation queue ahead of waiting filler
  and behind waiting viewer clips** (`enqueue`'s `position`, computed by
  `viewer_insert_position`): the GPUs build them next, filler just slides
  back, and nothing is popped or wasted;
- the playout loop **plays** viewer clips before filler (`pick_next` over
  the playout queue), so even an already-built filler waits;
- the filler stands down whenever a viewer prompt is pending (including one
  that arrived while the filler's LLM call was in flight — the group is
  dropped before enqueueing);
- when a **full playout queue of built filler blocks a viewer's build**
  (generation pauses while playout is at capacity), the playout loop pops
  one filler per tick, newest first, until the build resumes. Only clips
  tagged `generated: true` are ever popped; a playing clip is in neither
  queue, so playback is never cut.

When the viewer backlog alone reaches the deployment's clip budget
(`playout_capacity`), new prompts are **dropped**, with the drop logged: one
prompt never stalls the pipeline waiting for room. Every capacity is read
live from `state_update`, never assumed. Everything above reads the metadata
echo, so it survives client restarts and works on clips this process has no
memory of enqueueing.

The default target (6) sits under the default playout capacity (10) for the
same reason the clamp exists: the gap is headroom a viewer group can take
without any eviction at all.

## Overlay

Every outgoing frame — live, repeated, or black — passes through the overlay
before the sink, so the broadcast carries live status even between clips.
`OVERLAY_ENABLED` (default on) is the only configuration; *which* overlay
runs is a code decision in `main.py`, and building a different one means
implementing `overlay/base.py`'s `Overlay` (its docstring is the contract:
per-frame budget, never mutate the input frame, state via link listeners
only).

The shipped `StreamStatusOverlay` keeps to the frame's edges (small type on
thin translucent plates):

- **top-left, while playing**: `NOW <title> — scene 2/3 · by <author>`, with
  a dimmer `COMING UP <next>` beneath it (when the next clip is the same
  group it reads `COMING UP scene 3/3` instead of repeating the title);
- **top-left, while idle**: `UP NEXT <title> · by <author>` — or, with an
  empty queue, an invitation to type the chat command;
- **top-right**: `READY n · BUILDING m` — the playout queue (built, playable
  now) and the generation queue, separately. READY pinned at 0 while
  BUILDING holds a backlog is the signature of builds running slower than
  playback — the on-stream diagnostic for deployment speed.

Everything it shows is reconstructed from the wire — the metadata group tags
(title, author, scene numbering) and the link's queue mirror — so it survives
client restarts, credits idle filler clips to `auto`, and degrades untagged
clips (enqueued by some other client) to their prompt text. Rendering is
cached: Pillow rasterizes a panel only when its text changes; the per-frame
cost is numpy alpha blends (~0.6 ms at 1344×768).

## Sinks

`sinks/base.py` is the contract: by the time a sink sees data, the pacer has
already made it a perfectly regular stream — `send_video` once per period with
one fixed-size rgb24 frame, `send_audio` once per period with exactly one
period of int16 samples, gaps already filled. A sink only encodes and
forwards; it must never block the event loop, owns its own recovery, and
reports health via `alive`.

| `SINK=` | Class | Notes |
| --- | --- | --- |
| `rtmp` | `RtmpSink` | ffmpeg → RTMP(S). Twitch, YouTube Live, Kick are all just ingest URLs. |
| `noop` | `NoOpSink` | Discards everything, logs a heartbeat. Full pipeline dry-run. |

To add one (LiveKit, SFU, file recorder): implement `StreamSink`, register it
in `make_sink` (`sinks/__init__.py`), add its config to `.env.example`, and
extend this table.

## Chat sources

`chat/base.py` is the contract: a long-lived `run(on_prompt)` coroutine that
reconnects internally, delivers each message at most once, never replays
backlog from before startup, and strips the command word. Prompts are
messages starting with `CHAT_COMMAND` (default `!prompt`); sources also
match the admin commands below, and each delivered message records which
command it hit.

- **Twitch** (`TWITCH_CHANNEL`): anonymous read-only IRC (`justinfan` login) —
  no OAuth, no app registration, no token rotation.
- **YouTube** (`YOUTUBE_VIDEO_ID` + `YOUTUBE_API_KEY`): Data API v3 polling at
  the interval the API requests. The video id must be a **live** broadcast
  (it resolves `activeLiveChatId`). Polling costs quota; don't shorten the
  interval.

Both can run at once. Per-author cooldown (`CHAT_COOLDOWN_S`) and a bounded
backlog in the director keep spam from monopolizing the queue.

### Admin commands

Usernames listed in `ADMIN_USERS` (comma-separated, case-insensitive; scope
an entry as `twitch:name` / `youtube:name` when the same display name could
be different people on different platforms) may send admin commands.
`main.py` routes them to `admin.py` *before* the director, so they never
cost a cooldown slot, a moderation call, or an LLM call; the same word from
a non-admin is consumed and logged, never treated as a prompt.

- `!switch <preset>` — swap the creative preset live. The name is resolved
  against the `presets/` folder at switch time (only bare names, never
  paths), so dropping a new JSON into the folder makes it switchable with no
  restart. The upsampler's style and the idle filler's prompt list change
  immediately, and both model queues are flushed down to one buffer clip —
  the playout front, or the clip already building when nothing is built —
  so the new identity reaches the stream in about one clip's time instead
  of draining a whole queue of old-style clips. Chat prompts still waiting
  to be upsampled survive the flush; they come out in the new style.

New admin commands go in `admin.py`: add the word to `AdminControl.commands`
and a branch in `handle` — the chat sources and router pick it up from
there.

## Running

```sh
pip install -r requirements.txt   # ffmpeg must be on PATH for SINK=rtmp
cp .env.example .env              # fill in keys, style, sink, chat

# Dry run against a local `reactor run` of ../fast-h3, throwing frames away
# (local serves the in-repo model as `fast-h3` — set REACTOR_MODEL, or pass it):
python main.py --local --sink noop --model fast-h3

# Hosted model (`reactor/fast-h3`, the default in .env.example), streaming
# to Twitch, prompts from Twitch chat:
#   .env: REACTOR_API_KEY=rk_..., TWITCH_CHANNEL=yourchannel, PRESET=...
python main.py --sink rtmp --rtmp-url rtmp://live.twitch.tv/app/STREAM_KEY

# YouTube: RTMP_URL=rtmp://a.rtmp.youtube.com/live2/KEY plus
#   YOUTUBE_VIDEO_ID + YOUTUBE_API_KEY for chat.
```

Then type `!prompt a lighthouse in a storm` in chat. Expect: an upsampler log
with the scenes, `queued ... scene 1/n` lines, and — after roughly the clip's
own duration of build time per scene — `[now playing]` lines as the playout
loop runs the group. With the idle filler on (the default), `[auto]`-tagged clips fill
the queue within the first minute and the stream shows content instead of
black between viewer groups; with it off, black between groups is the model's
contract, not a bug.

## Learnings baked into this client (do not re-learn these)

From the earlier RTMP clients (py-sdk `rtmp_app` / `story_livestream_app`,
which took many iterations to stabilize) and from driving the fast-h3 queue:

- **Raw-video geometry is unforgiving.** One frame whose bytes disagree with
  ffmpeg's `-s WxH` (wrong size, or non-C-contiguous `tobytes()` including
  row padding) shifts every following scanline → "TV static". The pacer
  letterboxes odd sizes onto a fixed canvas; the RTMP sink refuses mismatched
  frames outright rather than corrupting the stream.
- **Never write to a pipe from the event loop.** `stdin.write` blocks when
  ffmpeg stalls; a blocked loop starves WebRTC and everything snowballs. Each
  ffmpeg pipe has its own writer thread behind a bounded drop-oldest queue.
- **Feed audio and video in lockstep, on separate pipes.** fast-h3 has real
  synchronized audio (`anullsrc` silence is not enough), and starving one
  ffmpeg input while pushing the other is the classic two-pipe deadlock. The
  pacer delivers both every tick.
- **An audio track is mandatory** — YouTube/Twitch won't take video-only FLV.
  The pacer emits silence when the model is idle, so the encoder never runs dry.
- **ffmpeg dies; the broadcast must not.** The sink restarts it lazily with a
  cooldown and a failure cap, keeping the last stderr lines for the log.
- **The sink outlives Reactor reconnects.** Sink + pacer are created once and
  the connection loop runs behind them, so the platform sees one continuous
  stream while the client rebuilds a session.
- **A constant-rate pacer is not optional for this model.** fast-h3's output is
  clip-shaped: 24 fps while a clip plays, *nothing* while the queue idles.
  RTMP needs a frame every period forever. The pacer (FIFO-buffered video and
  audio with the same shallow cap, repeats + silence on underflow, drop-oldest
  on overflow) is what converts one into the other — and buffering both media
  types symmetrically is what keeps A/V sync.
- **The queue dies with the session.** fast-h3 resets all session state on a
  new session: after a reconnect, clips that were queued but unplayed are
  gone. Chat prompts still waiting in the director survive (they live
  client-side); a group lost mid-flight is lost. Re-enqueueing on reconnect
  is a possible extension — if added, dedupe via the metadata group tag.
- **Refusals are broadcast, not raised.** A refused command answers with a
  bodyless reply and a `command_error` broadcast carrying the reason. Treat
  "reply without `clip`" as refusal and retry with patience (the director
  does).
- **A dead client leaves the local runtime "orphaned"** and its reaper takes
  a minute or more, during which every connect gets a 409. The link
  force-clears it with a bare `POST /stop_session` — local mode only, and
  only on the `orphaned` refusal, never on `streaming` (that may be someone
  else's live session). Teardown does the same when its own disconnect
  fails, so restarts don't inherit the wait.

## Maintenance notes (for agents)

- **Invariants to preserve:** prompt ≤ 800 chars after sanitization; metadata
  JSON well under 2000 chars; scene groups enqueued contiguously and only
  when they fully fit; all enqueues (viewer and filler) serialized through
  the director's one lock; eviction pops only `generated: true` clips;
  moderation fails closed; autoplay stays on and the director never sends
  `play` in steady state — it curates the playout front with `move`, always
  through `pick_next`; viewer groups insert positionally ahead of filler (viewer
  FIFO), and new prompts drop when the viewer backlog reaches the clip
  budget; every capacity read from `state_update`; pacer/sink never torn
  down on reconnect; sinks never block the event loop; overlays never
  mutate the pacer's frame and stay within a few ms per compose.
- **Continuation invariants:** the extended surface is gated on
  `link.supports_continuation` (detected from `state_update`, never assumed
  from the model name); a continuation field is never sent to a deployment
  that did not publish the surface; every chained scene's prompt opens on a
  described hard cut (the upsampler's chained rules — the anti-degradation
  rule above); a refused chained enqueue degrades to a standalone clip
  rather than stalling; mass pops (the preset-switch flush) remove
  dependents before their unbuilt sources.
- `../fast-h3/fasth3_types.py` is the wire contract. If the model's schema moves
  (new fields, renamed messages), update `reactor_link.py`'s mirror and the
  director's message handling together, and re-check this README's model
  paragraph.
- New sink → `sinks/` + `make_sink` + `.env.example` + the sink table above.
  New chat platform → `chat/` + `build_chat_sources` in `main.py` + docs.
  Keep the interfaces in `base.py` files authoritative — their docstrings are
  the contract, this README only summarizes them.
- There are no tests here yet; the cheap smoke test is
  `python main.py --local --sink noop` against a local `reactor run` (or the
  reference `../fast-h3/client/client.py` for the raw queue contract).
