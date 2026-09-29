# Remote MNIST

Train the [MNIST example](../mnist)'s model on a GPU in another machine, then run inference with it
there. The model, data pipeline and training loop are the MNIST example's own; this example only
connects to a remote device and runs them on it.

The example shows you how to:

- Reach a server's device over Iroh or WebSocket.
- Train with burn-train on that device, metrics included.
- Load the trained model and run inference on the same device.

The GPU machine runs the [server example](../server) with its backend feature (`cuda`, `rocm`,
`vulkan`, or `webgpu` by default). The client can run anywhere.

## Iroh

Iroh finds the server by its identity, so the client can reach it from another network, behind a
NAT, without knowing its address. Here the identity derives from a topic both sides agree on.

On the GPU machine:

```bash
REMOTE_BACKEND_TOPIC=my-gpu cargo run -p server --example server --release --features cuda
```

On the client:

```bash
cargo run -p remote-mnist --example remote-mnist --release -- train my-gpu
cargo run -p remote-mnist --example remote-mnist --release -- infer my-gpu
```

Anyone who knows the topic can connect, or host a server under it. Pick one that is hard to guess.

## WebSocket

WebSocket needs the server to be reachable at a known address, such as on the same network.

On the GPU machine, listening on port 3000 unless `REMOTE_BACKEND_PORT` says otherwise:

```bash
cargo run -p server --example server --release --features cuda
```

On the client, with the server's host name or address:

```bash
cargo run -p remote-mnist --example remote-mnist --release -- train ws://gpu-host:3000
cargo run -p remote-mnist --example remote-mnist --release -- infer ws://gpu-host:3000
```

## Where the model goes

`train` saves the model on the client, under `/tmp/burn-example-mnist`. `infer` loads it from there,
sends the weights to the server, and classifies 1000 test images on its GPU.
