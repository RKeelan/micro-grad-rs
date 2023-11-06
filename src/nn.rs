use std::fmt::Display;

use num_traits::Float;
use rand::distributions::{Distribution, Uniform};

use crate::Scalar;

// Neuron ---------------------------------------------------------------------

pub struct Neuron<T> {
    pub weights: Vec<Scalar<T>>,
    pub bias: Scalar<T>,
}

impl<T: Float + Copy + Display + std::ops::AddAssign + 'static> Neuron<T> {
    pub fn new(num_inputs: i32) -> Self {
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);
        let weights: Vec<T> = (0..num_inputs).map(|_| T::from(uniform.sample(&mut rng)).unwrap()).collect();
        let bias = T::from(uniform.sample(&mut rng)).unwrap();
        Neuron {
            weights: weights.iter().enumerate()
                .map(|(i, w)| Scalar::new_with_label(*w, format!("w{}", i).as_str()))
                .collect(),
            bias: Scalar::new_with_label(bias, "b"),
        }
    }

    pub fn forward(&self, inputs: &Vec<Scalar<T>>) -> Result<Scalar<T>, String> {
        if inputs.len() != self.weights.len() {
            Err(format!("Expected {} inputs, not {}", self.weights.len(), inputs.len()))?
        }

        // weight * x + bias
        let mut sum = self.bias.clone();
        for (input, weight) in inputs.iter().zip(self.weights.iter()) {
            sum = &sum + &(input * weight);
        }
        Ok(sum.tanh())
    }

    pub fn forward_with_numbers(&self, inputs: &Vec<T>) -> Result<Scalar<T>, String> {
        let inputs = &inputs.iter().map(|i| Scalar::new(i.clone())).collect();
        self.forward(&inputs)
    }

    pub fn parameters(&self) -> Vec<Scalar<T>> {
        self.weights.iter().chain(std::iter::once(&self.bias)).cloned().collect()
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}

// Layer ----------------------------------------------------------------------

pub struct Layer<T> {
    pub neurons: Vec<Neuron<T>>
}

impl<T: Float + Copy + Display + std::ops::AddAssign + 'static> Layer<T> {
    pub fn new(num_inputs: i32, num_outputs: i32) -> Self {
        Layer {
            neurons: (0..num_outputs).map(|_| Neuron::new(num_inputs)).collect(),
        }
    }

    pub fn forward(&self, inputs: &Vec<Scalar<T>>) -> Vec<Scalar<T>> {
        let outputs = self.neurons.iter().map(|n| n.forward(inputs).expect("")).collect();
        outputs
    }

    pub fn forward_with_numbers(&self, inputs: &Vec<T>) -> Vec<Scalar<T>> {
        let inputs = &inputs.iter().map(|i| Scalar::new(i.clone())).collect();
        let outputs = self.neurons.iter().map(|n| n.forward(inputs).expect("")).collect();
        outputs
    }

    pub fn parameters(&self) -> Vec<Scalar<T>> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}

// MLP ------------------------------------------------------------------------

pub struct MultiLayerPerceptron<T> {
    pub layers: Vec<Layer<T>>,
}

impl<T: Float + Copy + Display + std::ops::AddAssign + 'static> MultiLayerPerceptron<T> {
    pub fn new(num_inputs: i32, layer_num_outputs: Vec<i32>) -> Self {
        let mut num_inputs = num_inputs;
        MultiLayerPerceptron {
            layers: layer_num_outputs.iter().map(|num_outputs| {
                let layer = Layer::new(num_inputs, *num_outputs);
                num_inputs = *num_outputs;
                layer
            }).collect(),
        }
    }

    pub fn forward(&self, inputs: &Vec<T>) -> Vec<Scalar<T>> {
        let mut hidden_layer = self.layers[0].forward_with_numbers(inputs);
        for layer in self.layers.iter().skip(1) {
            hidden_layer = layer.forward(&hidden_layer);
        }
        hidden_layer
    }

    pub fn parameters(&self) -> Vec<Scalar<T>> {
        self.layers.iter().flat_map(|n| n.parameters()).collect()
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}
