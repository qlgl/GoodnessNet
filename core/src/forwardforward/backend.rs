use crate::prelude::*;

/// FastForward Backend Trait
pub trait FFBackend: Backend {
  type InnerBackend: Backend<
      Device = Self::Device,
      FloatElem = Self::FloatElem,
      IntElem = Self::IntElem,
    >;
}
