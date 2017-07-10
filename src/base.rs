/// Define gerneral functions, const variables and types.
/// 定义通用函数、常量、类型.

/// Signal Type.
/// 信号类型。
pub type SignalType = f64;
/// Algorithm.
/// 算法。
pub trait Algorithm: Clone {
    /// 前向算法
    fn errors(&self, liner_combination_factory: SignalType) -> SignalType;
    /// 反向求导
    fn derivative(&self, liner_combination_factor: SignalType) -> SignalType;
}
/// Sigmoid Function.
/// Sigmoid 函数。
#[derive(Clone)]
pub struct Sigmoid;

impl Algorithm for Sigmoid {
    /// 前向算法
    ///
    /// f(ν) = φ(ν） = 1 / (1 - exp(-v)）
    fn errors(&self, liner_combination_factory: SignalType) -> SignalType {
        (1 as SignalType) / ((1 as SignalType) + (-liner_combination_factory).exp())
    }
    /// 反向求导
    ///
    /// f(ν) = φ'(ν） = ν * (1 - ν)
    fn derivative(&self, liner_combination_factory: SignalType) -> SignalType {
        liner_combination_factory * ((1 as SignalType) - liner_combination_factory)
    }
}