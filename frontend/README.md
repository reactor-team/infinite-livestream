# Infinite Livestream FastH3 frontend

This Reactor-branded Next.js application adapts the Reactor Cookbook FastH3
demo to this repository's two-queue model contract. The official Reactor
lockup anchors the application header. It plays synchronized WebRTC video and
audio and renders complete generation/playout queue state.

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

## Local development

Start FastH3 on port 8080, then:

```sh
cp .env.example .env
pnpm install --frozen-lockfile
pnpm dev
```

Open <http://localhost:3000>. To monitor an already-running local livestream,
copy its Reactor session id into the connection field or set
`REACTOR_SESSION_ID` before starting Next.js.

The browser connects through the same-origin `/reactor` path. Next.js proxies
that path to `REACTOR_INTERNAL_URL` (default `http://localhost:8080`), so remote
development only needs to forward port 3000 and never exposes the Reactor
runtime directly.

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
