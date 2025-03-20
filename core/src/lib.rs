pub mod prelude {
  pub use burn::{
    nn::{
      Dropout, DropoutConfig, Linear, LinearConfig, Relu,
      conv::{Conv2d, Conv2dConfig},
      pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
  };
}

pub mod deeplearning;

pub mod forwardforward;
