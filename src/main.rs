use llm::Tensor;

fn main() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4]);
    let norm = t.layer_norm();
    println!("{:?}", norm.data);
}
