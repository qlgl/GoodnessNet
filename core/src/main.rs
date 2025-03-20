use {
  burn::{
    backend::{Autodiff, Candle, candle::CandleDevice},
    optim::AdamConfig,
  },
  core::{
    deeplearning::{
      model::configs::ModelConfig, train::configs::TrainingConfig,
    },
    prelude::*,
  },
};

type Backend = Candle;

fn main() {
  type MyBackend = Candle<f32, u32>;
  type MyAutoDiffBackend = Autodiff<MyBackend>;

  // let device = Default::default();
  let device: CandleDevice = CandleDevice::metal(0);
  let model = ModelConfig::new(10, 128).init::<MyBackend>(&device);

  let artifact_dir = "/tmp/guide";
  core::deeplearning::train::train::<MyAutoDiffBackend>(
    artifact_dir,
    TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
    device.clone(),
  );

  println!("{:?}", model);
}
