/// BP Network writen in rust-lang.
/// 使用Rust语言编写的BP神经网络库，自己学习使用。
extern crate serde_json;

mod base;
pub use base::*;

mod layer;
pub use layer::*;

mod neuron;
pub use neuron::*;

mod network;
pub use network::*;

mod trainer;
pub use trainer::*;