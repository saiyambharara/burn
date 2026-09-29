use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::{Dataset, transform::Mapper, vision::MnistDataset},
    },
    prelude::*,
    store::ModuleRecord,
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

/// A machine running the `server` example.
pub enum Server {
    /// Started with `REMOTE_BACKEND_TOPIC` set to this topic.
    Iroh { topic: String },
    /// Started without a topic, listening at this URL.
    WebSocket { url: String },
}

impl Server {
    pub fn parse(address: &str) -> Self {
        match address.starts_with("ws://") || address.starts_with("wss://") {
            true => Self::WebSocket {
                url: address.to_string(),
            },
            false => Self::Iroh {
                topic: address.to_string(),
            },
        }
    }

    /// The server's first device.
    pub async fn connect(&self) -> Device {
        match self {
            Self::WebSocket { url } => Device::remote_websocket(url, 0),
            Self::Iroh { topic } => {
                // https://github.com/n0-computer/iroh/issues/4555
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

/// Classify the first test images with the model `train` saved, on `device`.
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
    let correct: i64 = predicted
        .clone()
        .equal(batch.targets.clone())
        .int()
        .sum()
        .into_scalar();

    let predicted = predicted.into_data();
    let expected = batch.targets.into_data();
    for (predicted, expected) in predicted.iter::<i64>().zip(expected.iter::<i64>()).take(10) {
        println!("predicted {predicted}, expected {expected}");
    }
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
