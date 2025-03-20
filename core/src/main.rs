// use core::{networks::layer::FFNetwork, prelude::*};
//
// // ✅ 2. 데이터 생성
// fn generate_data(samples: usize, input_size: usize) -> Vec<Array1<f64>> {
//   let normal = Normal::new(0.0, 1.0).unwrap();
//   (0..samples)
//     .map(|_| {
//       Array1::<f64>::from_shape_fn(input_size, |_| {
//         normal.sample(&mut rand::rng())
//       })
//     })
//     .collect()
// }
//
// pub fn main() {
//   let log4rs_config_path = std::path::Path::new("./log4rs.yml");
//   log4rs::init_file(log4rs_config_path, Default::default()).unwrap();
//   let input_size = 10000;
//   let output_size = 100;
//   let learning_rate: f64 = 0.000001;
//   let epochs = 100000;
//
//   // ✅ 3. 신경망 생성
//   let mut network = FFNetwork::new(input_size, output_size);
//
//   // ✅ 4. "좋은 데이터" 와 "나쁜 데이터" 생성
//   let good_data = generate_data(1000, input_size); // 정상 데이터
//   let bad_data = generate_data(1000, input_size).iter()
//         .map(|x| x * 2.0) // 나쁜 데이터(변형된 데이터)
//         .collect::<Vec<_>>();
//
//   // ✅ 5. 학습 실행
//   network.train(&good_data, &bad_data, learning_rate, epochs);
// }

// use {burn::backend::Wgpu, core::model::ModelConfig};
//
// fn main() {
//   type MyBackend = Wgpu<f32, i32>;
//
//   let device = Default::default();
//   let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);
//
//   println!("{}", model);
// }

use burn::tensor::Tensor;
use burn::backend::Wgpu;

// Type alias for the backend to use.
type Backend = Wgpu;

fn main() {
    let device = Default::default();
    // Creation of two tensors, the first with explicit values and the second one with ones, with the same shape as the first
    let tensor_1 = Tensor::<Backend, 2>::from_data([[2., 3.], [4., 5.]], &device);
    let tensor_2 = Tensor::<Backend, 2>::ones_like(&tensor_1);

    // Print the element-wise addition (done with the WGPU backend) of the two tensors.
    println!("{}", tensor_1 + tensor_2);
}
