# Browser console

This Next.js application adapts the Reactor Cookbook FastH3 demo to this
repository's two-queue model contract. It plays synchronized WebRTC video and
audio and shows both the generation and playout queues.

The interface supports two safe operating modes:

- **Control mode:** leave the session id blank. The page creates a session and
  is the queue's writer. It can enqueue, reorder, remove and play clips; toggle
  autoplay; stop playback; reset the session; and set duration, seed and canvas.
- **Monitor mode:** join an existing session id. The page receives its media,
  state and both queues, but leaves all writes disabled. Use this while
  `streaming-client` owns scheduling so its viewer/filler priority remains the
  only queue policy.

Metadata written in control mode uses the same scene-group shape as
`streaming-client/group_tag.py`, so the streaming overlay and logs can describe
web-submitted clips if another client later reads them.

## Run locally

Start the model from the repository root:

```sh
cd fast-h3
reactor build
reactor run
```

In another terminal, start the frontend:

```sh
cd frontend
cp .env.example .env
pnpm install --frozen-lockfile
pnpm dev
```

Open <http://localhost:3000>. Leave the session id blank to create a control
session. A local Runtime serves one session at a time, so if
`streaming-client` is already connected, enter its session id to join it in
read-only monitor mode; trying to create a second session returns HTTP 409.
`REACTOR_SESSION_ID` can set the field's initial value.

The browser connects through the same-origin `/reactor` path. Next.js proxies
that path to `REACTOR_INTERNAL_URL`, which defaults to
`http://localhost:8080`. This keeps the Runtime's HTTP signaling endpoint
private.

## Run on a remote GPU server

The Next.js proxy carries signaling, but WebRTC media does not pass through
that proxy. The Runtime must advertise an ICE server the browser can reach.
If TURN-over-TCP listens on server loopback port 8080, forward both ports:

```sh
ssh -N \
  -L 3000:127.0.0.1:3000 \
  -L 8080:127.0.0.1:8080 \
  user@gpu-server
```

Then open <http://localhost:3000>. [`../STARTUP.md`](../STARTUP.md) contains a
complete native FastH3, TURN, and frontend example for this setup.

## Production process

```sh
pnpm typecheck
pnpm build
pnpm start --hostname 0.0.0.0 --port 3000
```

## Hosted Reactor model

Set `REACTOR_API_KEY`, `REACTOR_MODEL_NAME` and optionally `REACTOR_API_URL`.
The server exchanges the API key for a stable, short-lived, model-scoped token;
the key is never sent to the browser. Joining a hosted session created by a
different token additionally requires a JWT bound to that session, so the
built-in token route is intended for control mode.
