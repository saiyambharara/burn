use std::{
    fs,
    io::{ErrorKind, Write},
    path::{Path, PathBuf},
    str::FromStr,
};

use burn::{
    server::{AuthorizationRequest, BURN_REMOTE_ALPN, RemoteSecret},
    tensor::Device,
};
use iroh::{
    Endpoint, RelayMode, RelayUrl, SecretKey,
    endpoint::{Builder, QuicTransportConfig, presets},
    protocol::Router,
};
use tracing_subscriber::EnvFilter;

/// An Iroh server that only serves clients presenting its token.
pub struct IrohServer {
    secret: RemoteSecret,
    token: String,
    relay: Relay,
    port: Option<u16>,
}

impl IrohServer {
    /// Configured from the environment:
    /// - `REMOTE_BACKEND_TOKEN`, required: the secret every client must present.
    /// - `REMOTE_BACKEND_KEY`: the server's identity file, `remote-backend.key` by default,
    ///   created on first start so the server keeps its id across restarts.
    /// - `REMOTE_BACKEND_RELAY`: `public` by default, `off`, or the URL of a relay you run.
    /// - `REMOTE_BACKEND_PORT`: the UDP port to bind, which clients dial when relays are off.
    pub fn from_env() -> Self {
        let token = std::env::var("REMOTE_BACKEND_TOKEN")
            .expect("REMOTE_BACKEND_TOKEN holds the secret clients present");
        let key = std::env::var("REMOTE_BACKEND_KEY")
            .map_or_else(|_| PathBuf::from("remote-backend.key"), PathBuf::from);
        let relay = std::env::var("REMOTE_BACKEND_RELAY").map_or(Relay::Public, |relay| {
            relay.parse().unwrap_or_else(|err| panic!("{err}"))
        });

        Self {
            secret: load_or_create_secret(&key),
            token,
            relay,
            port: super::port(),
        }
    }

    /// Serve `device` until Ctrl-C.
    pub fn serve(self, device: Device) {
        // Burn reports sessions, rejected ones included, through tracing, and a server on its own
        // endpoint installs the subscriber itself.
        let filter =
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info,wgpu=warn"));
        tracing_subscriber::fmt().with_env_filter(filter).init();

        tokio::runtime::Runtime::new()
            .expect("Can build a tokio runtime")
            .block_on(self.serve_async(device));
    }

    async fn serve_async(self, device: Device) {
        let mut builder = self
            .relay
            .endpoint()
            .secret_key(SecretKey::from_bytes(&self.secret.to_bytes()))
            .alpns(vec![BURN_REMOTE_ALPN.to_vec()])
            .transport_config(transport_config());
        if let Some(port) = self.port {
            builder = builder
                .bind_addr(format!("0.0.0.0:{port}"))
                .expect("Can parse the bind address");
        }
        let endpoint = builder.bind().await.expect("Can bind the iroh endpoint");
        println!("listening on iroh as {}", self.secret.id());

        let token = blake3::hash(self.token.as_bytes());
        let protocol = burn::server::protocol(device, &endpoint)
            .with_authorizer(move |request: AuthorizationRequest<'_>| {
                // Comparing blake3 hashes takes the same time whatever the credential, so a wrong
                // guess reveals nothing about the token.
                if blake3::hash(request.credential) == token {
                    Ok(())
                } else {
                    Err("wrong token".to_string())
                }
            })
            .build();
        let router = Router::builder(endpoint)
            .accept(BURN_REMOTE_ALPN, protocol)
            .spawn();

        tokio::signal::ctrl_c()
            .await
            .expect("Can listen for Ctrl-C");
        router.shutdown().await.expect("The router shuts down");
    }
}

/// How peers that cannot dial each other directly reach one another.
#[derive(Clone, Debug)]
pub enum Relay {
    /// n0's public relays, with n0's address lookup so peers find each other by id alone.
    Public,
    /// A relay you run; peers must be told its URL.
    Private(RelayUrl),
    /// Direct connections only; peers must be told the address.
    Off,
}

impl FromStr for Relay {
    type Err = String;

    fn from_str(relay: &str) -> Result<Self, Self::Err> {
        match relay {
            "public" => Ok(Self::Public),
            "off" => Ok(Self::Off),
            url => url
                .parse()
                .map(Self::Private)
                .map_err(|err| format!("a relay is public, off or a URL, got {url}: {err}")),
        }
    }
}

impl Relay {
    fn endpoint(&self) -> Builder {
        match self {
            Self::Public => Endpoint::builder(presets::N0),
            Self::Private(url) => {
                Endpoint::builder(presets::Minimal).relay_mode(RelayMode::custom([url.clone()]))
            }
            Self::Off => Endpoint::builder(presets::Minimal).relay_mode(RelayMode::Disabled),
        }
    }
}

fn transport_config() -> QuicTransportConfig {
    // Segmentation offload can stall transfers until
    // https://github.com/n0-computer/iroh/issues/4555 is fixed.
    QuicTransportConfig::builder()
        .enable_segmentation_offload(false)
        .build()
}

fn load_or_create_secret(path: &Path) -> RemoteSecret {
    match fs::read(path) {
        Ok(bytes) => RemoteSecret::from_bytes(
            bytes
                .try_into()
                .unwrap_or_else(|_| panic!("{} holds a 32-byte key", path.display())),
        ),
        Err(err) if err.kind() == ErrorKind::NotFound => {
            let secret = RemoteSecret::random();
            let mut options = fs::OpenOptions::new();
            options.write(true).create_new(true);
            #[cfg(unix)]
            std::os::unix::fs::OpenOptionsExt::mode(&mut options, 0o600);
            options
                .open(path)
                .and_then(|mut file| file.write_all(&secret.to_bytes()))
                .unwrap_or_else(|err| panic!("Cannot write {}: {err}", path.display()));
            secret
        }
        Err(err) => panic!("Cannot read {}: {err}", path.display()),
    }
}
