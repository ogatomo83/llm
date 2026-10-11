use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
pub struct ValueData {
    pub data: f32,
    pub grad: f32,
}

#[derive(Debug, Clone)]
pub struct Value(pub Rc<RefCell<ValueData>>);

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Rc::new(RefCell::new(ValueData { data, grad: 0.0 })))
    }

    pub fn data(&self) -> f32 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f32 {
        self.0.borrow().grad
    }
}

#[test]
fn test_value_new() {
    let a = Value::new(2.0);
    assert_eq!(a.data(), 2.0);
    assert_eq!(a.grad(), 0.0);
}

#[test]
fn test_value_shared() {
    let a = Value::new(2.0);
    let b = a.clone();
    b.0.borrow_mut().grad = 1.0;
    assert_eq!(a.grad(), 1.0);
}
