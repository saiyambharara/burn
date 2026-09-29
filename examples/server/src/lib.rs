use burn::{
    server::{Channel, RemoteSecret},
    tensor::Device,
};

/// Host `Device::default()` for remote clients.
///
/// With `REMOTE_BACKEND_TOPIC` set, the server is an Iroh peer whose identity derives from the
/// topic, so clients that know the topic can dial it. Otherwise it listens on WebSocket at
/// `REMOTE_BACKEND_PORT`, 3000 by default.
pub fn start() {
    let channel = match std::env::var("REMOTE_BACKEND_TOPIC") {
        Ok(topic) => {
            let secret = topic_secret(&topic);
            println!("listening on iroh as {}", secret.id());
            Channel::Iroh {
                secret: Box::new(secret),
            }
        }
        Err(_) => Channel::WebSocket { port: port() },
    };

    burn::server::start(Device::default(), channel);
}

fn port() -> u16 {
    std::env::var("REMOTE_BACKEND_PORT")
        .map(|port| match port.parse::<u16>() {
            Ok(val) => val,
            Err(err) => panic!("Invalid port, got {port} with error {err}"),
        })
        .unwrap_or(3000)
}

/// Anyone who knows the topic can host as this identity, which suits an example; a real deployment
/// would use `RemoteSecret::random()` and share its `id()`. Matches the derivation of the
/// `p2p-remote-training` and `remote-mnist` examples.
fn topic_secret(topic: &str) -> RemoteSecret {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    RemoteSecret::from_bytes(*hash.as_bytes())
}
