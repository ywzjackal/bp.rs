/// Trainer.
/// 训练器.
use super::*;

/// Trainer.
/// 训练器.
pub trait Trainer {
    type Network: NeuronNetwork;
    ///
    ///
    /// 训练一个样本
    ///
    /// ```return : 误差（2-范数）```
    fn train_a_sample(&mut self,
                      network: &mut Self::Network,
                      inputs: &Vec<SignalType>,
                      excepts: &Vec<SignalType>,
                      rate: SignalType,
                      momentum: SignalType,
                      pre_deltas: &mut Vec<Vec<Vec<SignalType>>>)
                      -> SignalType {
        // forward, collect all layer's result to results.
        // 前向计算，并保存计算过程中，各个层得到的结果。
        // (输出层结果,输出层误差,所有层的计算结果)
        let (rt, mut next_errors, results) = {
            // results: 各个层的结果
            let results = network.all_layers_result(&inputs);
            let pre_layer_result = results[results.len() - 1].clone();
            // calculates errors
            // 计算误差 (输出误差)
            let rt = calculate_error(&results[results.len() - 1], &excepts);

            // backward collect all layer's errors.
            // 反向计算，收集输出层误差数据。(局部梯度计算)
            let mut output_layer_errors = Vec::with_capacity(excepts.len());
            assert_eq!(pre_layer_result.len(), excepts.len());
            for (exp, rt) in excepts.iter().zip(pre_layer_result.iter()) {
                let r = (exp - rt) * sigmoid_rev(*rt);
                output_layer_errors.push(r);
            }
            (rt, output_layer_errors, results)
        };
        // panic!("results: {:?}", results);
        // 其他层误差数据集合
        let mut errors = Vec::with_capacity(network.hidden_layers().len());
        errors.push(next_errors.clone());
        {
            let mut hidden_layer_it_rev = network.hidden_layers().iter_mut().rev();
            let mut layer_results_it_rev = results.iter().rev();
            let mut next_layer_neurons = hidden_layer_it_rev.next().unwrap().neurons();
            let _next_layer_results = layer_results_it_rev.next().unwrap();
            for (cur_layer, cur_layer_result) in hidden_layer_it_rev.zip(layer_results_it_rev) {
                let mut current_layer_neurons = cur_layer.neurons();
                let mut current_layer_errors = Vec::with_capacity(current_layer_neurons.len());
                for (neuron_index, (_current_layer_neuron, current_layer_neuron_result)) in
                    current_layer_neurons.iter_mut().zip(cur_layer_result.iter()).enumerate() {
                    let mut current_layer_neuron_error = 0.0 as SignalType;
                    for (next_layer_neuron, next_layer_neuron_error) in
                        next_layer_neurons.iter_mut().zip(next_errors.iter()) {
                        let weight_for_current_layer_neuron =
                            &next_layer_neuron.weights()[neuron_index];
                        current_layer_neuron_error += weight_for_current_layer_neuron *
                                                      next_layer_neuron_error;
                    }
                    current_layer_neuron_error *= sigmoid_rev(*current_layer_neuron_result);
                    current_layer_errors.push(current_layer_neuron_error);
                }
                errors.push(current_layer_errors.clone());
                next_errors = current_layer_errors;
                next_layer_neurons = current_layer_neurons;
            }
        }
        let errors = errors.iter().rev().collect::<Vec<_>>();
        // update weights
        {
            let hidden_layer_count = network.hidden_layers().len();
            for layer_index in 0..hidden_layer_count {
                let mut layer = &mut network.hidden_layers()[layer_index];
                let layer_errors = &errors[layer_index];
                let neuron_cnt = layer.neurons().len();
                for neuron_index in 0..neuron_cnt {
                    let mut neuron = &mut layer.neurons()[neuron_index];
                    let neuron_errors = layer_errors[neuron_index];
                    let weight_cnt = neuron.weights().len();
                    {
                        let threshold = neuron.threshold();
                        let prev_delta = pre_deltas[layer_index][neuron_index][weight_cnt]; // 最后一个保存threshold
                        let delta = (rate * neuron_errors) + (momentum * prev_delta);
                        neuron.set_threshold(threshold + delta);
                        pre_deltas[layer_index][neuron_index][weight_cnt] = delta;
                    }
                    for weight_index in 0..weight_cnt {
                        let prev_delta = pre_deltas[layer_index][neuron_index][weight_index];
                        let pre_result = results[layer_index][weight_index];
                        let delta = (rate * neuron_errors * pre_result) + (momentum * prev_delta);
                        neuron.weights()[weight_index] += delta;
                        pre_deltas[layer_index][neuron_index][weight_index] = delta;
                    }
                }
            }
        }
        // ************************************************************************
        // calculates MSE of output layer
        // 计算残差
        #[inline]
        fn calculate_error(last_results: &Vec<SignalType>,
                           targets: &Vec<SignalType>)
                           -> SignalType {
            let mut total = 0.0;
            for (&result, &target) in last_results.iter().zip(targets.iter()) {
                total += (target - result).powi(2) as SignalType;
            }
            total / (last_results.len() as SignalType)
        }
        // sigmoid function
        #[inline]
        fn sigmoid_rev(v: SignalType) -> SignalType {
            v * (1 as SignalType - v)
        }
        // ************************************************************************
        rt
    }
    ///
    ///
    /// 训练一组样本
    ///
    /// ```text
    /// inputs : 样本输入
    /// excepts: 样本目标预期
    /// return : 累计误差（2-范数）
    /// ```
    fn train_epoch(&mut self,
                   network: &mut Self::Network,
                   inputs: &Vec<Vec<SignalType>>,
                   excepts: &Vec<Vec<SignalType>>,
                   rate: SignalType,
                   momentum: SignalType)
                   -> SignalType {
        assert_eq!(inputs.len(), excepts.len());
        let mut total_errors = 0 as SignalType;
        let layers_cnt = network.hidden_layers().len();
        let mut deltas = Vec::with_capacity(layers_cnt);
        for l in network.hidden_layers().iter_mut() {
            let mut layer_deltas = Vec::new();
            for n in l.neurons().iter_mut() {
                let mut weight_deltas = Vec::with_capacity(n.weights().len() + 1);
                weight_deltas.resize(n.weights().len() + 1, 0 as SignalType);
                layer_deltas.push(weight_deltas);
            }
            deltas.push(layer_deltas);
        }
        for (sample, except) in inputs.iter().zip(excepts.iter()) {
            total_errors += self.train_a_sample(network, &sample, &except, rate, momentum, &mut deltas);
        }
        total_errors
    }

    ///
    ///
    /// 执行训练任务
    ///
    /// ```text
    /// inputs : 样本输入
    /// excepts: 样本目标预期
    /// return : Result<(错误率,epochs 次数),(错误率,epochs 次数)>
    /// ```
    fn train(&mut self,
             network: &mut Self::Network,
             inputs: &Vec<Vec<SignalType>>,
             excepts: &Vec<Vec<SignalType>>,
             target_errors: SignalType,
             rate: SignalType,
             momentum: SignalType,
             max_epochs: usize)
             -> Result<(SignalType, usize), (SignalType, usize)> {
        let mut errors = 0.0 as SignalType;
        for i in 0..max_epochs {
            errors = self.train_epoch(network, &inputs, &excepts, rate, momentum);
            if errors <= target_errors {
                return Ok((errors, i));
            }
        }
        Err((errors, max_epochs))
    }
}
#[derive(Serialize, Deserialize)]
pub struct SigmoidNetworkTrainer;

impl Trainer for SigmoidNetworkTrainer {
    type Network = SigmoidNetwork;
}