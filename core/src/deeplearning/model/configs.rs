use {super::Model, crate::prelude::*};

#[derive(Config, Debug)]
pub struct ModelConfig {
  num_classes: usize,
  hidden_size: usize,
  #[config(default = "0.5")]
  dropout:     f64,
}

impl ModelConfig {
  /// Returns the initialized model.
  pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
    Model {
      conv1:      Conv2dConfig::new([1, 8], [3, 3]).init(device),
      conv2:      Conv2dConfig::new([8, 16], [3, 3]).init(device),
      pool:       nn::pool::AvgPool2dConfig::new([8, 8]).init(),
      activation: Relu::new(),
      linear1:    LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
      linear2:    LinearConfig::new(self.hidden_size, self.num_classes)
        .init(device),
      dropout:    DropoutConfig::new(self.dropout).init(),
    }
  }
}

/*
 * 모델의, 초기화 및 설정을 구성함
 * 모델은, models/mod.rs 에 구성되어있으며, 여기서는 모델의 설정을 구성함
 */
