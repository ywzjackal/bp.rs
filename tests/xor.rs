extern crate bp;
extern crate serde_json;
use bp::*;

#[test]
fn xor_4layers() {
    // create examples of the xor function
    let input = vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]];
    let output = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];

    // create a new neural network
    let mut trainer = SigmoidNetworkTrainer;
    let mut network = SigmoidNetwork::new(&[2, 3, 3, 1]);
    match trainer.train(&mut network, &input, &output, 0.001, 0.3, 0.1, 100000) {
        Ok((errors, count)) => {
            println!("train successed, errors:{}, count:{}", errors, count);
        }
        Err((errors, count)) => {
            println!("train failed, errors:{}, count:{}", errors, count);
            panic!("fail to test xor 4layers!");
        }
    }

    // test
    for i in 0..input.len() {
        let rt = network.active(&input[i]);
        assert_eq!(rt[0].round(), output[i][0]);
    }
}
