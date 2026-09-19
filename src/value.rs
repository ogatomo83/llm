#[derive(Debug, Clone)]
pub struct Value {
    pub data: f32,
    pub grad: f32,
}

impl Value {
    pub fn new(data: f32) -> Self {
        Self { data, grad: 0.0 }
    }
}

#[test]
fn test_value_new() {
    let a = Value::new(2.0);
    assert_eq!(a.data, 2.0);
    assert_eq!(a.grad, 0.0);
}
