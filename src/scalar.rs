use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Display, Formatter, Result};
use std::hash::{Hash, Hasher};
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::ptr;
use std::rc::Rc;

use num_traits::Float;

struct Value<T> {
    pub data: Rc<RefCell<T>>,
    pub grad: Rc<RefCell<T>>,
    pub label: String,
    pub back_prop: Option<Box<dyn Fn()>>,
    // A list of values used to produce this value. Empty indicates a "leaf" value
    producers: Vec<Rc<RefCell<Value<T>>>>,
}

impl<T: Float + Copy + Display> Value<T> {
    pub fn new(data: T) -> Self {
        Self { 
            data: Rc::new(RefCell::new(data)),
            grad: Rc::new(RefCell::new(T::zero())),
            label: "".to_string(),
            back_prop: None,
            producers: Vec::new(),
        }
    }
}

impl<T> PartialEq for Value<T> {
    fn eq(&self, other: &Self) -> bool {
        ptr::eq(self, other)
    }
}

impl<T> Eq for Value<T> {}

impl<T> Hash for Value<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(self, state);
    }
}

pub struct Scalar<T> {
    value: Rc<RefCell<Value<T>>>,
}

impl<T: Float + Copy + Display + std::ops::AddAssign + 'static> Scalar<T> {
    pub fn new(data: T) -> Self {
        Self {
            value: Rc::new(RefCell::new(Value::new(data))),
        }
    }

    fn new_from_value(value: Rc<RefCell<Value<T>>>) -> Self {
        Self {
            value,
        }
    }

    pub fn exp(&self) -> Self {
        let self_data = *self.value.borrow().data.borrow();
        let self_grad = self.value.borrow().grad.clone();
        let producers = vec![self.clone()];
        let out_value = Rc::new(RefCell::new(Value {
            data: Rc::new(RefCell::new(self_data.exp())),
            grad: Rc::new(RefCell::new(T::zero())),
            back_prop: None,
            producers: producers.into_iter().map(|x| x.value).collect(),
            label: format!("exp({})", self.get_label()).to_string(),
        }));

        let closure_out_value = out_value.clone();
        let back_prop_closure = move || {
            let out_data = *closure_out_value.borrow().data.borrow();
            let out_grad = *closure_out_value.borrow().grad.borrow();
            *self_grad.borrow_mut() += out_data * out_grad;
        };
        out_value.borrow_mut().back_prop = Some(Box::new(back_prop_closure));
        Scalar::new_from_value(out_value)
    }

    pub fn tanh(&self) -> Self {
        let self_data = *self.value.borrow().data.borrow();
        let self_grad = self.value.borrow().grad.clone();
        let producers = vec![self.clone()];
        let out_value = Rc::new(RefCell::new(Value {
            data: Rc::new(RefCell::new(self_data.tanh())),
            grad: Rc::new(RefCell::new(T::zero())),
            back_prop: None,
            producers: producers.into_iter().map(|x| x.value).collect(),
            label: format!("tanh({})", self.get_label()).to_string(),
        }));

        let closure_out_value = out_value.clone();
        let back_prop_closure = move || {
            let t = *closure_out_value.borrow().data.borrow();
            let out_grad = *closure_out_value.borrow().grad.borrow();
            *self_grad.borrow_mut() += (T::one() - t*t) * out_grad;
        };
        out_value.borrow_mut().back_prop = Some(Box::new(back_prop_closure));
        Scalar::new_from_value(out_value)
    }

    pub fn pow(&self, power: T) -> Self {
        let self_data = *self.value.borrow().data.borrow();
        let self_grad = self.value.borrow().grad.clone();
        let producers = vec![self.clone()];
        let out_value = Rc::new(RefCell::new(Value {
            data: Rc::new(RefCell::new(self_data.powf(power))),
            grad: Rc::new(RefCell::new(T::zero())),
            back_prop: None,
            producers: producers.into_iter().map(|x| x.value).collect(),
            label: format!("({}^{})", &self.get_label(), power),
        }));

        let closure_self_value = self.value.clone();
        let closure_out_value = out_value.clone();
        let back_prop_closure = move || {
            let self_data = *closure_self_value.borrow().data.borrow();
            let out_grad = *closure_out_value.borrow().grad.borrow();
            *self_grad.borrow_mut() += power * self_data.powf(power - T::one()) * out_grad;
        };
        out_value.borrow_mut().back_prop = Some(Box::new(back_prop_closure));
        Scalar::new_from_value(out_value)
    }

    pub fn back_prop(&self) {
        // Build a topological ordering of the graph
        let mut topological_ordering = Vec::<Scalar<T>>::new();
        let mut visited = HashSet::<Scalar<T>>::new();

        fn visit<T: Float + Copy + Display + std::ops::AddAssign + 'static>(
            topological_ordering: &mut Vec<Scalar<T>>,
            visited: &mut HashSet<Scalar<T>>,
            scalar: &Scalar<T>) {
            if !visited.contains(scalar) {
                visited.insert(scalar.clone());
                for child in scalar.value.borrow().producers.clone() {
                    visit(topological_ordering, visited, &Scalar::new_from_value(child));
                }
                topological_ordering.push(scalar.clone());
            }
        }
        visit(&mut topological_ordering, &mut visited, &self);

        self.value.borrow_mut().grad.replace(T::one());
        for s in topological_ordering.iter().rev() {
            if let Some(back_prop) = &s.value.borrow().back_prop {
                back_prop();
                let grad = *s.value.borrow().grad.borrow();
                log::debug!("{}'s gradient: {:.4}", s.value.borrow().label, grad);
            }
        }
    }

    pub fn set_label(&self, label: &str) {
        self.value.borrow_mut().label = label.to_owned();
    }

    pub fn get_label(&self) -> String {
        self.value.borrow_mut().label.clone()
    }

    pub fn add_number(&self, number: T) -> Scalar<T> {
        let rhs = Scalar::new(number);
        rhs.set_label(&format!("(Constant {})", number));
        self + &rhs
    }

    pub fn mul_number(&self, number: T) -> Scalar<T> {
        let rhs = Scalar::new(number);
        rhs.set_label(&format!("(Constant {})", number));
        self * &rhs
    }
}

impl<T> Clone for Scalar<T> {
    fn clone(&self) -> Self {
        Self {
            value: self.value.clone(),
        }
    }
}

impl<T: Display> Display for Scalar<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result {
        write!(f, "{} {{ data: {:.4}, grad: {:.4} }}",
            self.value.borrow().label,
            self.value.borrow().data.borrow(),
            self.value.borrow().grad.borrow())
    }
}

impl<T> PartialEq for Scalar<T> {
    fn eq(&self, other: &Self) -> bool {
        // Two Scalars are equal if they point to the same Value
        ptr::eq(&*self.value.borrow(), &*other.value.borrow())
    }
}

impl<T> Eq for Scalar<T> {}

impl<T> Hash for Scalar<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(&*self.value.borrow(), state);
    }
}

// Operator Overides

impl<T: Float + Copy + Display + std::ops::AddAssign + 'static> Add<&Scalar<T>> for &Scalar<T> {
    type Output = Scalar<T>;

    fn add(self, rhs: &Scalar<T>) -> Scalar<T> {
        let self_data = *self.value.borrow().data.borrow();
        let rhs_data = *rhs.value.borrow().data.borrow();

        let self_grad = self.value.borrow().grad.clone();
        let rhs_grad = rhs.value.borrow().grad.clone();

        let producers = vec![self.clone(), rhs.clone()];
        let out_value = Rc::new(RefCell::new(Value {
            data: Rc::new(RefCell::new(self_data + rhs_data)),
            grad: Rc::new(RefCell::new(T::zero())),
            back_prop: None,
            producers: producers.into_iter().map(|x| x.value).collect(),
            label: format!("({} + {})", self.get_label(), rhs.get_label()),
        }));

        let closure_value = out_value.clone();
        let back_prop_closure = move || {
            *self_grad.borrow_mut() += *closure_value.borrow().grad.borrow();
            *rhs_grad.borrow_mut() += *closure_value.borrow().grad.borrow();
        };
        out_value.borrow_mut().back_prop = Some(Box::new(back_prop_closure));
        Scalar::new_from_value(out_value)
    }
}

impl<T: Float + Copy + Display + std::ops::AddAssign + 'static> Div<&Scalar<T>> for &Scalar<T> {
    type Output = Scalar<T>;

    fn div(self, rhs: &Scalar<T>) -> Scalar<T> {
        let reciprocal = &rhs.pow(T::from(-1).unwrap());
        self * reciprocal
    }
}

impl<T: Float + Copy + Display + std::ops::AddAssign + 'static> Mul<&Scalar<T>> for &Scalar<T> {
    type Output = Scalar<T>;

    fn mul(self, rhs: &Scalar<T>) -> Scalar<T> {
        let self_data = *self.value.borrow().data.borrow();
        let rhs_data = *rhs.value.borrow().data.borrow();

        let self_grad = self.value.borrow().grad.clone();
        let rhs_grad = rhs.value.borrow().grad.clone();

        let producers = vec![self.clone(), rhs.clone()];
        let out_value = Rc::new(RefCell::new(Value {
            data: Rc::new(RefCell::new(self_data * rhs_data)),
            grad: Rc::new(RefCell::new(T::zero())),
            back_prop: None,
            producers: producers.into_iter().map(|x| x.value).collect(),
            label: format!("({} * {})", self.get_label(), rhs.get_label()),
        }));

        let closure_self_value = self.value.clone();
        let closure_rhs_value = rhs.value.clone();
        let closure_out_value = out_value.clone();
        let back_prop_closure = move || {
            let self_data = *closure_self_value.borrow().data.borrow();
            let rhs_data = *closure_rhs_value.borrow().data.borrow();
            let out_grad = *closure_out_value.borrow().grad.borrow();
            *self_grad.borrow_mut() += rhs_data * out_grad;
            *rhs_grad.borrow_mut() += self_data * out_grad;
        };
        out_value.borrow_mut().back_prop = Some(Box::new(back_prop_closure));
        Scalar::new_from_value(out_value)
    }
}

impl<T: Float + Copy + Display + std::ops::AddAssign + 'static> Neg for &Scalar<T> {
    type Output = Scalar<T>;

    fn neg(self) -> Scalar<T> {
        let result = self.mul_number(T::from(-1).unwrap());
        result.set_label(&format!("(not {})", self.get_label()));
        result
    }
}

impl<T: Float + Copy + Display + std::ops::AddAssign + 'static> Sub<&Scalar<T>> for &Scalar<T> {
    type Output = Scalar<T>;

    fn sub(self, rhs: &Scalar<T>) -> Scalar<T> {
        let negation = -rhs;
        let result = self + &negation;
        result.set_label(&format!("({} - {})", self.get_label(), rhs.get_label()));
        result
    }
}