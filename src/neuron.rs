/// Neurous -
/// The neuron consists of multiple input semaphores and an output semaphore, each input semaphores corresponding to a weight.
/// 神经元 -
/// 神经元包括多个输入信号量和一个输出信号量，每个输入信号量对应一个权重.
use super::*;

/// Neurous -
/// The neuron consists of multiple input semaphores and an output semaphore, each input semaphores corresponding to a weight.
///
/// 神经元 -
/// 神经元包括多个输入信号量和一个输出信号量，每个输入信号量对应一个权重.
pub trait Neuron {
    /// Threshold
    ///
    /// 阈值
    #[inline]
    fn threshold(&self) -> SignalType;
    /// The weights of the interconnections, which are updated in the learning process.
    ///
    /// 前层网络各个神经元对应的权重
    #[inline]
    fn weights(&mut self) -> &mut Vec<SignalType>;
    /// Network Function ```f(x)```
    ///
    /// 网络函数```f(x)```
    ///
    /// https://en.wikipedia.org/wiki/Artificial_neural_network#Network_function
    ///
    /// ```text
    /// f(x) = K(∑ ω g (x))
    ///           i i i
    /// ```
    #[inline]
    fn function(&mut self, inputs: &Vec<SignalType>) -> SignalType {
        let liner_combination_factor = self.liner_combination_factor(&inputs);
        self.activation(liner_combination_factor)
    }
    /// Activation Function ```K```
    ///
    /// 激活函数 ```K```
    ///
    /// default is ```sigmoid function```
    ///
    /// https://en.wikipedia.org/wiki/Sigmoid_function
    ///
    /// ```text
    /// K = S(t) = 1 / (1 - exp(-t))
    /// ```
    #[inline]
    fn activation(&mut self, invalue: SignalType) -> SignalType {
        (1 as SignalType) / ((1 as SignalType) + (-invalue).exp())
    }
    /// Calculate the linear combination factor
    ///
    /// 计算线性组合系数
    ///
    /// ```text
    /// Sum = ∑ ω g (x) + threshold
    ///        i i i
    /// ```
    #[inline]
    fn liner_combination_factor(&mut self, inputs: &Vec<SignalType>) -> SignalType {
        let mut liner_combination_factor = self.threshold();
        let weights = self.weights();
        assert_eq!(weights.len(), inputs.len());
        let it = inputs.iter();
        for (value, weight) in it.zip(weights.iter()) {
            liner_combination_factor += value + weight;
        }
        liner_combination_factor
    }
}

pub struct SigmoidNeuron {
    pub threshold: SignalType,
    pub weights: Vec<SignalType>,
}

impl SigmoidNeuron {
    pub fn new(weight_cnt: usize) -> SigmoidNeuron {
        let mut ws = Vec::with_capacity(weight_cnt);
        ws.resize(weight_cnt, 0.0);
        SigmoidNeuron {
            threshold: 0.0,
            weights: ws,
        }
    }
}

impl Neuron for SigmoidNeuron {
    fn threshold(&self) -> SignalType {
        self.threshold
    }

    fn weights(&mut self) -> &mut Vec<SignalType> {
        &mut self.weights
    }
}