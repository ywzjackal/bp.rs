/// Neural Networks Layer.
/// 神经网络层

use super::*;

/// Neural Networks Layer.
/// 神经网络层.
pub trait Layer {
    type Neuron: Neuron;

    /// Current layer's Neurons
    ///
    /// 本层神经元
    #[inline]
    fn neurons(&mut self) -> &mut Vec<Self::Neuron>;

    /// Current layer's Neurons count
    ///
    /// 本层神经元数量
    #[inline]
    fn len(&mut self) -> usize {
        self.neurons().len()
    }

    /// Forward Calculation
    ///
    /// 前向计算
    ///
    /// ```text
    /// result: 根据前层网络与本层网络加权求值后的结果列表
    /// ```
    #[inline]
    fn forward(&mut self, pre_layer_results: &Vec<SignalType>) -> Vec<SignalType> {
        let mut results = Vec::with_capacity(self.len());
        // 遍历本网络神经元，与前层网络的输出结果进行前向计算，把每个计算结果推入results中。
        for neuron in self.neurons().iter_mut() {
            let result = neuron.function(&pre_layer_results);
            results.push(result);
        }
        results
    }
}

pub struct SigmoidLayer {
    pub neurons: Vec<SigmoidNeuron>,
}

impl SigmoidLayer {
    pub fn from_vec(vec: Vec<SigmoidNeuron>) -> SigmoidLayer {
        SigmoidLayer { neurons: vec }
    }

    pub fn new(neurons_cnt: usize, pre_layer_neurons_cnt: usize) -> SigmoidLayer {
        let mut ns = Vec::with_capacity(neurons_cnt);
        for i in 0..neurons_cnt {
            let n = SigmoidNeuron::new(pre_layer_neurons_cnt);
            ns.push(n);
        }
        SigmoidLayer { neurons: ns }
    }
}

impl Layer for SigmoidLayer {
    type Neuron = SigmoidNeuron;
    fn neurons(&mut self) -> &mut Vec<SigmoidNeuron> {
        &mut self.neurons
    }
}