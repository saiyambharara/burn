use remote_mnist::Server;

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (Some(mode), Some(server)) = (args.get(1), args.get(2)) else {
        usage();
    };
    let device = Server::parse(server).connect().await;

    match mode.as_str() {
        "train" => mnist::training::run(device),
        "infer" => remote_mnist::infer(&device),
        _ => usage(),
    }
}

fn usage() -> ! {
    eprintln!("usage:");
    eprintln!("  train <server>   train the MNIST model on the server's GPU");
    eprintln!("  infer <server>   classify test images with the trained model on the server's GPU");
    eprintln!();
    eprintln!("<server> is the topic the server was started with, or its ws:// URL");
    std::process::exit(1);
}
