mod iroh_server;

use burn::{server::Channel, tensor::Device};
use iroh_server::IrohServer;

/// Host `Device::default()` for remote clients.
///
/// `REMOTE_BACKEND_TRANSPORT` picks how: `websocket` by default, on port `REMOTE_BACKEND_PORT` or
/// 3000, or `iroh`, configured as [`IrohServer::from_env`] describes.
pub fn start() {
    let transport = std::env::var("REMOTE_BACKEND_TRANSPORT");
    match transport.as_deref() {
        Err(_) | Ok("websocket") => {
            let port = port().unwrap_or(3000);
            println!("listening on websocket port {port}");
            burn::server::start(Device::default(), Channel::WebSocket { port });
        }
        Ok("iroh") => IrohServer::from_env().serve(Device::default()),
        Ok(other) => panic!("REMOTE_BACKEND_TRANSPORT is websocket or iroh, got {other}"),
    }
}

fn port() -> Option<u16> {
    std::env::var("REMOTE_BACKEND_PORT")
        .ok()
        .map(|port| match port.parse::<u16>() {
            Ok(val) => val,
            Err(err) => panic!("Invalid port, got {port} with error {err}"),
        })
}
