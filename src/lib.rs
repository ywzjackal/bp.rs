/// BP Network writen in rust-lang.
/// 使用Rust语言编写的BP神经网络库，自己学习使用。
extern crate serde;
extern crate serde_json;
#[macro_use]
extern crate serde_derive;
extern crate log;
extern crate env_logger;
extern crate rand;

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