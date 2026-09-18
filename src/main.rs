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

    pub fn add(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape, "shape mismatch");
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        Tensor::from_vec(data, &self.shape)
    }

    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|a| a * scalar).collect();
        Tensor::from_vec(data, &self.shape)
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

    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape.len(), 2, "matmulは2次元のみ対応");
        assert_eq!(other.shape.len(), 2, "matmulは2次元のみ対応");
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(k, k2, "行列のサイズが合わない");

        let mut result = Tensor::zeros(&[m, n]);
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += self.get(&[i, p]) * other.get(&[p, j]);
                }
                result.set(&[i, j], sum);
            }
        }
        result
    }

    pub fn relu(&self) -> Tensor {
        let data: Vec<f32> = self.data.iter().map(|&x| x.max(0.0)).collect();
        Tensor::from_vec(data, &self.shape)
    }
}

fn main() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let t2 = Tensor::from_vec(vec![2.0, 3.0, 4.0, 5.0], &[2, 2]);
    t.matmul(&t2);
}

#[test]
fn test_set() {
    let mut t = Tensor::zeros(&[2, 3]);
    t.set(&[1, 2], 9.0);
    assert_eq!(t.get(&[1, 2]), 9.0);
}

#[test]
fn test_add() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], &[3]);

    let c = a.add(&b);
    assert_eq!(c.data, vec![11.0, 22.0, 33.0]);
}

#[test]
fn test_mul_scalar() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = a.mul_scalar(3.0);
    assert_eq!(b.data, vec![3.0, 6.0, 9.0, 12.0, 15.0, 18.0]);
}

#[test]
fn test_matmul() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]);
    let c = a.matmul(&b);
    assert_eq!(c.data, vec![19.0, 22.0, 43.0, 50.0]);
}

#[test]
fn test_relu() {
    let a = Tensor::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0], &[5]);
    let b = a.relu();
    assert_eq!(b.data, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
}
