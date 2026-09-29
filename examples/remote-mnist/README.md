# Remote MNIST

The [MNIST example](../mnist), trained and run on a GPU in another machine through the remote
backend. The model, data pipeline and training loop are the MNIST example's own; this example only
connects to the server and runs them there.

The example shows you how to:

- Reach a server's device over Iroh, by topic, or over WebSocket, by URL.
- Train with burn-train on that device, metrics included.
- Load the trained model and run inference on the same device.

## Start the server

On the machine with the GPU, start the [server example](../server) with its backend. With
`REMOTE_BACKEND_TOPIC` it serves over Iroh, reachable from any network by the topic; without it,
it listens on WebSocket port `REMOTE_BACKEND_PORT`, 3000 by default.

```bash
REMOTE_BACKEND_TOPIC=my-gpu cargo run -p server --example server --release --features cuda
```

## Train and infer

On any other machine:

```bash
cargo run -p remote-mnist --example remote-mnist --release -- train my-gpu
cargo run -p remote-mnist --example remote-mnist --release -- infer my-gpu
```

Pass the server's URL instead of a topic to use WebSocket, as in `train ws://gpu-host:3000`.
Training saves the model under `/tmp/burn-example-mnist` on the client; `infer` loads it from
there, sends the weights to the server and classifies test images on its GPU.
