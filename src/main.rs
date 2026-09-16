#[derive(Debug, Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    pub fn from_vec(data: Vec<f32>, shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "data length != shape product");
        Self {
            data,
            shape: shape.to_vec(),
        }
    }

    pub fn get(&self, idx: &[usize]) -> f32 {
        self.data[self.flat_index(idx)]
    }

    pub fn set(&mut self, idx: &[usize], value: f32) {
        let i = self.flat_index(idx);
        self.data[i] = value;
    }

    pub fn flat_index(&self, idx: &[usize]) -> usize {
        assert_eq!(idx.len(), self.shape.len(), "dim mismatch");
        let mut flat = 0;
        let mut stride = 1;
        for i in (0..self.shape.len()).rev() {
            assert!(idx[i] < self.shape[i], "index out of bounds");
            flat += idx[i] * stride;
            stride *= self.shape[i];
        }
        flat
    }
}

fn main() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    println!("t.get([0.0]) = {}", t.get(&[0, 0]));
}

#[test]
fn test_set() {
    let mut t = Tensor::zeros(&[2, 3]);
    t.set(&[1, 2], 9.0);
    assert_eq!(t.get(&[1, 2]), 9.0);
}
