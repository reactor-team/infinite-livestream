# Run the native browser console remotely

This guide runs the repository's native FastH3 model and browser console on a
GPU server while the browser stays on a development machine. The Next.js
server proxies HTTP signaling to Reactor Runtime, and an authenticated
TURN-over-TCP listener carries WebRTC media through the SSH tunnel.

The example reserves two loopback ports on the server:

- `3000` for the Next.js application
- `8080` for TURN-over-TCP

Reactor Runtime uses `8081` for signaling. The browser does not need a direct
forward to that port.

## Build the model and frontend

Set the repository and weight locations for the server. The FastH3 snapshot
must contain the transformer, text encoder, VAE, audio VAE, schedulers,
tokenizer, and processor described in [`fast-h3/README.md`](./fast-h3/README.md).

```sh
REPOSITORY_ROOT=$(git rev-parse --show-toplevel)
FASTH3_SNAPSHOT=/absolute/path/to/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree

cd "$REPOSITORY_ROOT/fast-h3"
reactor build --no-dockerfile

cd "$REPOSITORY_ROOT/frontend"
pnpm install --frozen-lockfile
pnpm typecheck
pnpm build
```

Set `UV_CACHE_DIR` and `HF_HOME` before the build if their caches should live
outside the system disk. The model loads the snapshot offline and does not
download missing components at startup.

## Start the stack

Run the following in one shell so coturn and Reactor Runtime receive the same
random credential. `REACTOR_IMAGE` is the local image produced by the Reactor
build; change it if the local Reactor configuration uses a different tag.

`NCCL_NVLS_ENABLE=0` is needed only on hosts where Fabric Manager rejects the
NVLink SHARP multicast allocation. It does not disable ordinary NVLink P2P.

```sh
REPOSITORY_ROOT=$(git rev-parse --show-toplevel)
FASTH3_SNAPSHOT=/absolute/path/to/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree
REACTOR_IMAGE=reactor-local/fast-h3:dev
TURN_USER=reactor_browser
TURN_PASS=$(openssl rand -hex 24)
RELAY_IP=$(ip -4 route get 1.1.1.1 | awk '{for (i=1; i<=NF; i++) if ($i=="src") {print $(i+1); exit}}')

docker run -d --rm \
  --name infinite-livestream-turn \
  --network host \
  coturn/coturn:4.6.3 \
  --log-file=stdout \
  --listening-ip=127.0.0.1 \
  --relay-ip="$RELAY_IP" \
  --listening-port=8080 \
  --min-port=50000 \
  --max-port=50020 \
  --fingerprint \
  --lt-cred-mech \
  --realm=infinite-livestream.local \
  --user="$TURN_USER:$TURN_PASS" \
  --no-udp --no-tls --no-dtls --no-cli --no-multicast-peers \
  --allow-loopback-peers \
  --server-relay \
  --total-quota=16 --user-quota=8

docker run -d \
  --name infinite-livestream-native \
  --network host \
  --ipc=host \
  --gpus all \
  -v "$FASTH3_SNAPSHOT:$FASTH3_SNAPSHOT:ro" \
  -e REACTOR_WEIGHTS_PATH="$FASTH3_SNAPSHOT" \
  -e PORT=8081 \
  -e STUN_SERVERS= \
  -e "TURN_SERVERS=$TURN_USER;$TURN_PASS;turn:localhost:8080?transport=tcp" \
  -e ICE_TRANSPORT_POLICY=relay \
  -e NCCL_NVLS_ENABLE=0 \
  "$REACTOR_IMAGE" \
  run --port 8081

docker run -d --rm \
  --name infinite-livestream-frontend \
  --network host \
  -e REACTOR_INTERNAL_URL=http://127.0.0.1:8081 \
  -e HOME=/tmp \
  -v "$REPOSITORY_ROOT/frontend:/app" \
  -w /app \
  node:22-bookworm \
  bash -lc 'corepack pnpm start --hostname 127.0.0.1 --port 3000'
```

FastH3 warms every configured clip length before it becomes available. Follow
the model log and check health through the same proxy used by the browser:

```sh
docker logs -f infinite-livestream-native
curl -fsS http://127.0.0.1:3000/reactor/health
```

The ready response contains `"state":"available"`.

## Connect from the development machine

Forward the frontend and TURN ports from the development machine:

```sh
ssh -N \
  -L 3000:127.0.0.1:3000 \
  -L 8080:127.0.0.1:8080 \
  user@gpu-server
```

Open <http://localhost:3000>. Port 3000 carries the application and HTTP
signaling; port 8080 carries TURN-over-TCP media. Both forwards are required
for this configuration.

## Stop this stack

The container names keep cleanup scoped to this example:

```sh
docker stop infinite-livestream-frontend
docker stop infinite-livestream-native
docker stop infinite-livestream-turn
docker rm infinite-livestream-native
```
