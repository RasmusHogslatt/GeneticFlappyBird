use std::f32::MAX;

use bevy::prelude::*;
use rand::prelude::*;

#[derive(Component, Default)]
pub struct Obstacle;

#[derive(Component, Default)]
pub struct Agent;

#[derive(Component, Default)]
pub struct Velocity {
    pub speed: Vec2,
}

#[derive(Component, Default)]
pub struct Actions {
    pub up: f32,
    pub down: f32,
    pub right: f32,
    pub left: f32,
}

#[derive(Component, Default)]
pub struct Fitness {
    pub score: f32,
}

#[derive(Component, Default)]
pub struct Sensors {
    pub sensors: Vec<Sensor>,
}

#[derive(Clone)]
pub struct Sensor {
    pub distance: f32,
    pub direction: f32,
}

impl Default for Sensor {
    fn default() -> Self {
        Self {
            distance: MAX,
            direction: 0.0,
        }
    }
}

#[derive(Component, Debug, Clone)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub activation_function: fn(f32) -> f32,
}

impl NeuralNetwork {
    pub fn initialize(sizes: Vec<usize>, activation_function: fn(f32) -> f32) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len() - 1 {
            layers.push(Layer::new(sizes[i], sizes[i + 1]));
        }
        NeuralNetwork {
            layers,
            activation_function,
        }
    }

    pub fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        let mut activations = input;
        for layer in &self.layers {
            activations = layer.forward(&activations, self.activation_function);
        }
        activations
    }

    pub fn forward_with_intermediates(
        &self,
        input: Vec<f32>,
    ) -> (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        let mut activations = input.clone();
        let mut all_activations = vec![activations.clone()];
        let mut all_weighted_sums = Vec::new();

        for layer in &self.layers {
            let (new_activations, weighted_sums) =
                layer.forward_with_intermediates(&activations, self.activation_function);
            all_activations.push(new_activations.clone());
            all_weighted_sums.push(weighted_sums);
            activations = new_activations;
        }

        (activations, all_activations, all_weighted_sums)
    }

    pub fn backward(
        &mut self,
        all_activations: &[Vec<f32>],
        all_weighted_sums: &[Vec<f32>],
        d_loss: Vec<f32>,
        learning_rate: f32,
    ) {
        let mut d_output = d_loss; // Gradient of the loss with respect to the output

        // Iterate over layers in reverse order
        for i in (0..self.layers.len()).rev() {
            let layer = &mut self.layers[i];

            // Compute the gradient of the loss with respect to the weighted sums
            let d_weighted_sum: Vec<f32> = d_output
                .iter()
                .zip(&all_weighted_sums[i])
                .map(|(d_o, z)| d_o * sigmoid_derivative(*z))
                .collect();

            for (j, (weight_row, &d_z)) in layer.weights.iter_mut().zip(&d_weighted_sum).enumerate()
            {
                for (weight, &activation) in weight_row.iter_mut().zip(&all_activations[i]) {
                    *weight -= learning_rate * d_z * activation;
                }
                layer.biases[j] -= learning_rate * d_weighted_sum[j];
            }

            // Compute the gradient of the loss with respect to the output of the previous layer
            d_output = layer
                .weights
                .iter()
                .map(|weights_row| {
                    weights_row
                        .iter()
                        .zip(&d_weighted_sum)
                        .map(|(w, d_z)| w * d_z)
                        .sum()
                })
                .collect();
        }
    }
}

#[derive(Clone, Debug)]
pub struct Layer {
    pub weights: Vec<Vec<f32>>,
    pub biases: Vec<f32>,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights = (0..output_size)
            .map(|_| (0..input_size).map(|_| rng.gen::<f32>()).collect())
            .collect();

        let biases = (0..output_size).map(|_| rng.gen::<f32>()).collect();

        Layer { weights, biases }
    }

    pub fn forward(&self, input: &Vec<f32>, activation_function: fn(f32) -> f32) -> Vec<f32> {
        self.weights
            .iter()
            .zip(&self.biases)
            .map(|(w_row, b)| {
                let sum = w_row.iter().zip(input).map(|(w, i)| w * i).sum::<f32>() + b;
                activation_function(sum)
            })
            .collect()
    }

    pub fn forward_with_intermediates(
        &self,
        input: &Vec<f32>,
        activation_function: fn(f32) -> f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let activations: Vec<f32> = self
            .weights
            .iter()
            .zip(&self.biases)
            .map(|(w_row, b)| {
                let sum = w_row.iter().zip(input).map(|(w, i)| w * i).sum::<f32>() + b;
                activation_function(sum)
            })
            .collect();

        let weighted_sums: Vec<f32> = self
            .weights
            .iter()
            .zip(&self.biases)
            .map(|(w_row, b)| w_row.iter().zip(input).map(|(w, i)| w * i).sum::<f32>() + b)
            .collect();

        (activations, weighted_sums)
    }
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn sigmoid_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}
