# Native FastH3 browser console startup

These commands start the `feat/fast-h3-browser-console` stack with the
repository's native FastVideo backend. There is no vLLM-Omni process and no
`H3_ADD` dependency.

The browser uses two SSH-forwarded TCP ports:

- `localhost:3000` serves the Next.js interface and proxies Reactor HTTP
  signaling to the private backend port 8081.
- `localhost:8080` carries WebRTC media through TURN-over-TCP. Forwarding only
  the HTTP port is insufficient because ordinary WebRTC candidates use UDP.

## Build after a source change

```sh
export UV_CACHE_DIR=/opt/dlami/nvme/.cache_uv
export HF_HOME=/opt/dlami/nvme/.cache_hf

cd /opt/dlami/nvme/ruixing/infinite-livestream-frontend-fix/fast-h3
reactor build --no-dockerfile

docker run --rm \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e REACTOR_INTERNAL_URL=http://127.0.0.1:8081 \
  -v /opt/dlami/nvme/ruixing/infinite-livestream-frontend-fix/frontend:/app \
  -w /app \
  node:22-bookworm \
  bash -lc 'corepack pnpm typecheck && corepack pnpm build'
```

## Start the complete local stack

Run this block in one shell so the randomly generated TURN credential is shared
by coturn and Reactor Runtime. `NCCL_NVLS_ENABLE=0` avoids this server's Fabric
Manager rejecting NCCL's NVLink SHARP multicast allocation; regular NVLink P2P
remains enabled.

```sh
DEMO_TURN_USER=reactor_demo
DEMO_TURN_PASS=$(openssl rand -hex 24)
DEMO_RELAY_IP=$(ip -4 route get 1.1.1.1 | awk '{for (i=1; i<=NF; i++) if ($i=="src") {print $(i+1); exit}}')
FASTH3_SNAPSHOT=/opt/dlami/nvme/.cache_hf/reactor_registry/fasth3/FastVideo-FastH3-4-step-Preview-v1-VSA-DataFree

docker run -d --rm \
  --name infinite-livestream-demo-turn \
  --network host \
  coturn/coturn:4.6.3 \
  --log-file=stdout \
  --listening-ip=127.0.0.1 \
  --relay-ip="$DEMO_RELAY_IP" \
  --listening-port=8080 \
  --min-port=50000 \
  --max-port=50020 \
  --fingerprint \
  --lt-cred-mech \
  --realm=infinite-livestream.local \
  --user="$DEMO_TURN_USER:$DEMO_TURN_PASS" \
  --no-udp --no-tls --no-dtls --no-cli --no-multicast-peers \
  --total-quota=16 --user-quota=8

docker run -d \
  --name infinite-livestream-demo-native \
  --network host \
  --ipc=host \
  --gpus all \
  -v "$FASTH3_SNAPSHOT:$FASTH3_SNAPSHOT:ro" \
  -e REACTOR_WEIGHTS_PATH="$FASTH3_SNAPSHOT" \
  -e PORT=8081 \
  -e STUN_SERVERS= \
  -e "TURN_SERVERS=$DEMO_TURN_USER;$DEMO_TURN_PASS;turn:localhost:8080?transport=tcp" \
  -e ICE_TRANSPORT_POLICY=relay \
  -e NCCL_NVLS_ENABLE=0 \
  reactor-local/fast-h3:dev \
  run --port 8081

docker run -d --rm \
  --name infinite-livestream-demo-frontend \
  --network host \
  -e REACTOR_INTERNAL_URL=http://127.0.0.1:8081 \
  -e HOME=/tmp \
  -v /opt/dlami/nvme/ruixing/infinite-livestream-frontend-fix/frontend:/app \
  -w /app \
  node:22-bookworm \
  bash -lc 'corepack pnpm start --hostname 0.0.0.0 --port 3000'
```

FastH3 loads the native 138 GB snapshot across all eight B200 GPUs and warms
all 14 legal clip lengths before becoming available. Follow only this stack's
model container and check health through the same proxy the browser uses:

```sh
docker logs -f infinite-livestream-demo-native
curl -fsS http://127.0.0.1:3000/reactor/health
```

The ready response contains `"state":"available"`. Forward server ports 3000
and 8080 to the same ports on the local machine, then open
<http://localhost:3000>.

## Stop only this stack

```sh
docker stop infinite-livestream-demo-frontend
docker stop infinite-livestream-demo-native
docker stop infinite-livestream-demo-turn
docker rm infinite-livestream-demo-native
```
