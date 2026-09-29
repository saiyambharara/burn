use std::{net::ToSocketAddrs, str::FromStr};

use burn::tensor::Device;
use iroh::{
    Endpoint, EndpointAddr, EndpointId, RelayMode, RelayUrl,
    endpoint::{Builder, QuicTransportConfig, presets},
};

/// A `server` example started with `REMOTE_BACKEND_TRANSPORT=iroh`.
pub struct IrohServer {
    /// The id it printed when it started.
    pub id: EndpointId,
    /// Its `REMOTE_BACKEND_TOKEN`.
    pub token: String,
    /// Its `REMOTE_BACKEND_RELAY`.
    pub relay: Relay,
    /// Its `host:port`, required when relays are off.
    pub address: Option<String>,
}

impl IrohServer {
    /// The server's first device.
    pub async fn connect(&self) -> Device {
        // Segmentation offload can stall transfers until
        // https://github.com/n0-computer/iroh/issues/4555 is fixed.
        let transport = QuicTransportConfig::builder()
            .enable_segmentation_offload(false)
            .build();
        let endpoint = self
            .relay
            .endpoint()
            .transport_config(transport)
            .bind()
            .await
            .expect("Can bind an iroh endpoint");
        Device::remote_iroh_authorized(&endpoint, self.addr(), 0, self.token.as_bytes().to_vec())
    }

    fn addr(&self) -> EndpointAddr {
        let mut addr = EndpointAddr::new(self.id);
        if let Relay::Private(url) = &self.relay {
            addr = addr.with_relay_url(url.clone());
        }
        if let Some(address) = &self.address {
            let address = address
                .to_socket_addrs()
                .ok()
                .and_then(|mut addresses| addresses.next())
                .unwrap_or_else(|| panic!("{address} resolves to an address"));
            addr = addr.with_ip_addr(address);
        }
        addr
    }
}

/// How peers that cannot dial each other directly reach one another. Both ends must agree.
#[derive(Clone, Debug)]
pub enum Relay {
    /// n0's public relays, with n0's address lookup so a server is found by its id alone.
    Public,
    /// A relay you run.
    Private(RelayUrl),
    /// Direct connections only.
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
