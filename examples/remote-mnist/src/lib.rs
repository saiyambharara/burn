use std::{convert::Infallible, str::FromStr};

use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, transform::Mapper, vision::MnistDataset},
    },
    prelude::*,
    store::ModuleRecord,
    tensor::Transaction,
};
use iroh::{
    Endpoint, EndpointId, SecretKey,
    endpoint::{QuicTransportConfig, presets},
};
use mnist::{
    data::{MnistBatcher, MnistMapper},
    model::Model,
    training::ARTIFACT_DIR,
};

const TEST_IMAGES: usize = 1000;
const SHOWN_PREDICTIONS: usize = 10;

/// A machine running the `server` example.
#[derive(Clone, Debug)]
pub enum Server {
    /// Started with `REMOTE_BACKEND_TOPIC` set to this topic.
    Iroh { topic: String },
    /// Started without a topic, listening at this URL.
    WebSocket { url: String },
}

impl FromStr for Server {
    type Err = Infallible;

    fn from_str(address: &str) -> Result<Self, Self::Err> {
        let server = if address.starts_with("ws://") || address.starts_with("wss://") {
            Self::WebSocket {
                url: address.to_string(),
            }
        } else {
            Self::Iroh {
                topic: address.to_string(),
            }
        };
        Ok(server)
    }
}

impl Server {
    /// The server's first device.
    pub async fn connect(&self) -> Device {
        match self {
            Self::WebSocket { url } => Device::remote_websocket(url, 0),
            Self::Iroh { topic } => {
                // Segmentation offload can stall transfers until
                // https://github.com/n0-computer/iroh/issues/4555 is fixed.
                let transport = QuicTransportConfig::builder()
                    .enable_segmentation_offload(false)
                    .build();
                let endpoint = Endpoint::builder(presets::N0)
                    .transport_config(transport)
                    .bind()
                    .await
                    .expect("Can bind an iroh endpoint");
                Device::remote_iroh(&endpoint, topic_id(topic), 0)
            }
        }
    }
}

/// Classify the first test images on `device` with the model `train` saved.
pub fn infer(device: &Device) {
    let record = ModuleRecord::load(format!("{ARTIFACT_DIR}/model"))
        .expect("A trained model exists; run train first");
    let model = Model::new(device).load_record(record);

    let dataset = MnistDataset::test();
    let mapper = MnistMapper::default();
    let items = (0..TEST_IMAGES)
        .map(|index| mapper.map(&dataset.get(index).expect("MNIST has this many test images")))
        .collect();
    let batch = MnistBatcher::default().batch(items, device);
    let predicted = model.forward(batch.images).argmax(1).flatten::<1>(0, 1);

    // Both tensors come back in one round trip to the server.
    let [predicted, expected] = Transaction::default()
        .register(predicted)
        .register(batch.targets)
        .execute()
        .try_into()
        .expect("One result per registered tensor");
    let predicted: Vec<i64> = predicted.iter().collect();
    let expected: Vec<i64> = expected.iter().collect();

    for (predicted, expected) in predicted.iter().zip(&expected).take(SHOWN_PREDICTIONS) {
        println!("predicted {predicted}, expected {expected}");
    }
    let correct = predicted
        .iter()
        .zip(&expected)
        .filter(|(predicted, expected)| predicted == expected)
        .count();
    println!(
        "accuracy on {TEST_IMAGES} test images: {:.2}%",
        100.0 * correct as f64 / TEST_IMAGES as f64
    );
}

/// Must match the `server` example's derivation, or the client dials an identity nobody hosts.
fn topic_id(topic: &str) -> EndpointId {
    let hash = blake3::hash(format!("burn-p2p:{topic}").as_bytes());
    SecretKey::from_bytes(hash.as_bytes()).public()
}
