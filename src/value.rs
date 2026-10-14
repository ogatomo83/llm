use std::cell::RefCell;
use std::rc::Rc;

#[derive(Debug)]
pub struct ValueInner {
    pub data: f32,
    pub grad: f32,
    pub prev: Vec<Value>,
}

#[derive(Debug, Clone)]
pub struct Value(pub Rc<RefCell<ValueInner>>);

impl Value {
    pub fn new(data: f32) -> Self {
        Value(Rc::new(RefCell::new(ValueInner {
            data,
            grad: 0.0,
            prev: vec![],
        })))
    }

    pub fn data(&self) -> f32 {
        self.0.borrow().data
    }

    pub fn grad(&self) -> f32 {
        self.0.borrow().grad
    }

    pub fn add(&self, other: &Value) -> Value {
        let out = Value::new(self.data() + other.data());
        out.0.borrow_mut().prev = vec![self.clone(), other.clone()];
        out
    }

    pub fn backward(&self) {
        self.0.borrow_mut().grad += 1.0;
        let prev = self.0.borrow().prev.clone();
        for p in &prev {
            p.0.borrow_mut().grad += 1.0;
        }
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

#[test]
fn test_value_add_forward() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let c = a.add(&b);
    assert_eq!(c.data(), 5.0);
}

#[test]
fn test_value_backwarc_add() {
    let a = Value::new(2.0);
    let b = Value::new(3.0);
    let c = a.add(&b);
    c.backward();
    assert_eq!(c.grad(), 1.0);
    assert_eq!(a.grad(), 1.0);
    assert_eq!(b.grad(), 1.0);
}
