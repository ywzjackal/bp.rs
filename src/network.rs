/// Neuron Network
/// 神经网络

use super::*;

pub trait NeuronNetwork {
    type HiddenLayerNeuron: Neuron;
    type HiddenLayer: Layer<Neuron = Self::HiddenLayerNeuron>;

    /// Obtain Hidden Layers
    ///
    /// 获取隐藏层列表
    #[inline]
    fn hidden_layers(&mut self) -> &mut Vec<Self::HiddenLayer>;

    /// Active
    ///
    /// 激活一次运算
    #[inline]
    fn active(&mut self, inputs: &Vec<SignalType>) -> Vec<SignalType> {
        // forward
        let mut hidden_layer_it = self.hidden_layers().iter_mut();
        let mut pre_layer_result = hidden_layer_it.next().unwrap().forward(&inputs);
        // forward input layer to first hidden layer
        for other_hidden_layer in hidden_layer_it {
            let this_hidden_layer_result = other_hidden_layer.forward(&pre_layer_result);
            pre_layer_result = this_hidden_layer_result;
        }
        pre_layer_result
    }
}

pub struct SigmoidNetwork {
    pub hidden_layers: Vec<SigmoidLayer>,
}

impl SigmoidNetwork {
    pub fn new(cfg: &[usize]) -> SigmoidNetwork {
        let mut hl = Vec::with_capacity(cfg.len());
        for i in 1..cfg.len() {
            hl.push(SigmoidLayer::new(cfg[i], cfg[i - 1]));
        }
        SigmoidNetwork { hidden_layers: hl }
    }
}

impl NeuronNetwork for SigmoidNetwork {
    type HiddenLayerNeuron = SigmoidNeuron;
    type HiddenLayer = SigmoidLayer;

    fn hidden_layers(&mut self) -> &mut Vec<SigmoidLayer> {
        &mut self.hidden_layers
    }
}