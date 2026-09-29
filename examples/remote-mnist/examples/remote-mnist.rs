use clap::{Parser, ValueEnum};
use remote_mnist::Server;

/// Train the MNIST model on another machine's GPU, or classify test images with it there.
#[derive(Parser)]
struct Cli {
    mode: Mode,
    /// The topic the server was started with, or its `ws://` URL.
    server: Server,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    /// Train the model on the server's GPU and save it here.
    Train,
    /// Classify test images on the server's GPU with the saved model.
    Infer,
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let device = cli.server.connect().await;

    // Training and inference block for as long as they run, so they stay off the async workers.
    tokio::task::spawn_blocking(move || match cli.mode {
        Mode::Train => mnist::training::run(device),
        Mode::Infer => remote_mnist::infer(&device),
    })
    .await
    .expect("The run completes");
}
