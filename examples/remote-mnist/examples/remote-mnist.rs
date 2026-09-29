use std::str::FromStr;

use burn::tensor::Device;
use clap::{CommandFactory, Parser, ValueEnum, error::ErrorKind};
use iroh::EndpointId;
use remote_mnist::{IrohServer, Relay};

/// Train the MNIST model on another machine's GPU, or classify test images with it there.
#[derive(Parser)]
struct Cli {
    mode: Mode,
    /// The server's Iroh id, printed when it starts, or its `ws://` URL.
    server: Server,
    /// The Iroh server's `REMOTE_BACKEND_TOKEN`.
    #[arg(long, env = "REMOTE_BACKEND_TOKEN", hide_env_values = true)]
    token: Option<String>,
    /// The Iroh server's relays: `public`, `off`, or the URL of a relay you run.
    #[arg(long, default_value = "public")]
    relay: Relay,
    /// The Iroh server's `host:port`, required with `--relay off`.
    #[arg(long, required_if_eq("relay", "off"))]
    address: Option<String>,
}

#[derive(Clone, Copy, ValueEnum)]
enum Mode {
    /// Train the model on the server's GPU and save it here.
    Train,
    /// Classify test images on the server's GPU with the saved model.
    Infer,
}

#[derive(Clone)]
enum Server {
    WebSocket(String),
    Iroh(EndpointId),
}

impl FromStr for Server {
    type Err = String;

    fn from_str(server: &str) -> Result<Self, Self::Err> {
        if server.starts_with("ws://") || server.starts_with("wss://") {
            return Ok(Self::WebSocket(server.to_string()));
        }
        server
            .parse()
            .map(Self::Iroh)
            .map_err(|_| format!("{server} is neither a ws:// URL nor an Iroh id"))
    }
}

#[tokio::main]
async fn main() {
    let cli = Cli::parse();
    let device = match cli.server {
        Server::WebSocket(url) => Device::remote_websocket(&url, 0),
        Server::Iroh(id) => {
            let Some(token) = cli.token else {
                Cli::command()
                    .error(
                        ErrorKind::MissingRequiredArgument,
                        "an Iroh server needs --token or REMOTE_BACKEND_TOKEN",
                    )
                    .exit()
            };
            let server = IrohServer {
                id,
                token,
                relay: cli.relay,
                address: cli.address,
            };
            server.connect().await
        }
    };

    // Training and inference block for as long as they run, so they stay off the async workers.
    tokio::task::spawn_blocking(move || match cli.mode {
        Mode::Train => mnist::training::run(device),
        Mode::Infer => remote_mnist::infer(&device),
    })
    .await
    .expect("The run completes");
}
