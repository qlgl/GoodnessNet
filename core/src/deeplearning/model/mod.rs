use {crate::prelude::*, burn::nn::pool::AvgPool2d};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
  conv1:      Conv2d<B>,
  conv2:      Conv2d<B>,
  pool:       AvgPool2d,
  dropout:    Dropout,
  linear1:    Linear<B>,
  linear2:    Linear<B>,
  activation: Relu,
}

impl<B: Backend> Model<B> {
  /// # Shapes
  ///   - Images [batch_size, height, width]
  ///   - Output [batch_size, class_prob]
  pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
    let [batch_size, height, width] = images.dims();

    // Create a channel.
    let x = images.reshape([batch_size, 1, height, width]);

    let x = self.conv1.forward(x); // [batch_size, 8, _, _]
    let x = self.dropout.forward(x);
    let x = self.conv2.forward(x); // [batch_size, 16, _, _]
    let x = self.dropout.forward(x);
    let x = self.activation.forward(x);

    let x = self.pool.forward(x); // [batch_size, 16, 8, 8]
    let x = x.reshape([batch_size, 16 * 8 * 8]);
    let x = self.linear1.forward(x);
    let x = self.dropout.forward(x);
    let x = self.activation.forward(x);

    self.linear2.forward(x) // [batch_size, num_classes]
  }
}

/*
 * 모델에 대하여 준비하는 부분임
 * init 함수를 통해, 모델이 초기화됨
 */

pub mod configs;
