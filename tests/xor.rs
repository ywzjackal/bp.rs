extern crate bp;
use bp::*;

#[test]
fn xor_4layers() {
    // create examples of the xor function
    let input = vec![vec![0f64, 0f64], vec![0f64, 1f64], vec![1f64, 0f64], vec![1f64, 1f64]];
    let output = vec![vec![0f64], vec![1f64], vec![1f64], vec![0f64]];

    // create a new neural network
    let mut trainer = SigmoidNetworkTrainer::new(&[2, 3, 3, 1]);
    match trainer.train(&input, &output, 0.1, 0.3, 0.1, 100000) {
        Ok((errors, count)) => {
            println!("train failed, errors:{}, count:{}", errors, count);
        }
        Err((errors, count)) => {
            println!("train successed, errors:{}, count:{}", errors, count);
        }
    }
}
