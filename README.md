# infinite-livestream

An end-to-end, chat-driven, never-ending AI video broadcast. Viewers type
`!prompt <idea>` in Twitch or YouTube chat; an LLM expands each idea into a
styled sequence of scenes; the fast-h3 model generates them as 768p video
clips with synchronized audio; and the stream goes out over RTMP as one
uninterrupted broadcast.

The model and its clients share one contract:

| Folder | What it is | Runs on |
| --- | --- | --- |
| [`fast-h3/`](./fast-h3) | The model: a queue of prompt-driven clip generations with explicit/auto playback. A `reactor` CLI workspace. | [Reactor Runtime](https://github.com/reactor-team/reactor-runtime), 8x B200 |
| [`streaming-client/`](./streaming-client) | The client: chat → prompt upsampling → scene groups → the model's queue → paced RTMP output. | [`reactor-sdk`](https://pypi.org/project/reactor-sdk/) (Python), any box with ffmpeg |
| [`frontend/`](./frontend) | The browser console: WebRTC playback and complete generation/playout queue UI. It controls a session it creates, or monitors a streaming-client session read-only. | Next.js and [`@reactor-team/js-sdk`](https://www.npmjs.com/package/@reactor-team/js-sdk) |

They meet on the wire: `fast-h3/fasth3_types.py` is the client-facing
contract (commands, messages, tracks), and the streaming client speaks
exactly that — through [`reactor-sdk`](https://pypi.org/project/reactor-sdk/),
the [Reactor Python SDK](https://docs.reactor.inc), which carries the
session, the commands and messages, and the WebRTC media tracks.

The browser console is adapted from the Reactor Cookbook FastH3 demo. It can
create and control its own queue session, or join a session owned by
`streaming-client` as a read-only monitor.

## The model

The video generator is
[FastH3 Preview v1](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree):
MiniMax-H3 (35B) distilled by the
[FastVideo](https://github.com/hao-ai-lab/FastVideo) project down to four
transformer forwards with 90% sparse video attention, generating video and
audio jointly from text. The model, the distillation, and the inference
engine this repo serves are FastVideo's work — `fast-h3/` wraps them in a
queue-and-playout contract for the Reactor platform.

## Quickstart

Serve the model (locally with `reactor run` from `fast-h3/`, or deploy it).
To run the automated livestream:

```sh
cd streaming-client
pip install -r requirements.txt     # ffmpeg must be on PATH for RTMP
cp .env.example .env                # keys, style, sink, chat channel
python main.py --local --sink noop  # dry run against a local runtime
python main.py                      # everything from .env
```

To use the interactive browser console instead:

```sh
cd frontend
pnpm install --frozen-lockfile
pnpm dev
```

Open <http://localhost:3000>. Leave the session field blank to create a
frontend-owned session, or enter the streaming client's session id to monitor
an existing stream without changing its queues. See
[`frontend/README.md`](./frontend/README.md) for hosted and remote setup.

## Documentation

- [`STARTUP.md`](./STARTUP.md) — run the native FastH3 browser console on a
  remote GPU server through SSH and TURN-over-TCP.
- [`fast-h3/README.md`](./fast-h3/README.md) — the model: the queue contract,
  weights layout, GPU/CUDA prerequisites, performance profile, and deployment
  learnings.
- [`streaming-client/README.md`](./streaming-client/README.md) — the client:
  architecture, sinks and chat sources, moderation, idle filler, presets, and
  the RTMP/ffmpeg learnings.
- [`frontend/README.md`](./frontend/README.md) — the browser console: local and
  hosted setup, queue controls, and safe read-only monitoring of a live stream.
- [`fast-h3/client/README.md`](./fast-h3/client/README.md) — a minimal
  `reactor-sdk` smoke-test client that walks the raw queue contract once.
- [`AGENTS.md`](./AGENTS.md) — the map for coding agents: system picture,
  load-bearing invariants, and where each kind of change goes. Read it before
  changing anything.

## License

The code is Apache License 2.0 — see [LICENSE](./LICENSE) and
[NOTICE](./NOTICE). The
[model weights](https://huggingface.co/FastVideo/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree)
are licensed separately under the
[MiniMax H3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE),
inherited from the base model
[MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3).
