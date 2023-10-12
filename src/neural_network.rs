use rand::prelude::*;

// Define the structure of the Neural Network
#[derive(Clone, Debug)]
pub struct NeuralNetwork {
    layers: Vec<Layer>,
}

#[derive(Clone, Debug)]
struct Layer {
    weights: Vec<Vec<f32>>, // Matrix of weights
    biases: Vec<f32>,       // Vector of biases
}

// Activation function (Sigmoid)
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

// Derivative of the sigmoid function (used later for backpropagation) NOT USED
fn _sigmoid_derivative(x: f32) -> f32 {
    let s = sigmoid(x);
    s * (1.0 - s)
}

impl NeuralNetwork {
    // Initialize a new Neural Network
    pub fn new(sizes: &[usize]) -> Self {
        let mut rng = rand::thread_rng();

        let layers = sizes
            .windows(2)
            .map(|sizes| {
                let (in_size, out_size) = (sizes[0], sizes[1]);

                let weights = (0..out_size)
                    .map(|_| {
                        (0..in_size)
                            .map(|_| rng.gen::<f32>() * 2.0 - 1.0) // random values between -1 and 1
                            .collect()
                    })
                    .collect();

                let biases = (0..out_size)
                    .map(|_| rng.gen::<f32>() * 2.0 - 1.0)
                    .collect();

                Layer { weights, biases }
            })
            .collect();

        NeuralNetwork { layers }
    }

    // Forward propagation
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        let mut input = input.to_vec();

        for layer in &self.layers {
            input = layer.forward(&input);
        }

        input
    }
}

impl Layer {
    // Forward propagation for a layer
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        self.weights
            .iter()
            .zip(&self.biases)
            .map(|(weights, bias)| {
                let sum = weights.iter().zip(input).map(|(w, i)| w * i).sum::<f32>() + bias;
                sigmoid(sum)
            })
            .collect()
    }
}

// average of two parents
pub fn crossover_average(parent1: NeuralNetwork, parent2: NeuralNetwork) -> NeuralNetwork {
    let mut child = parent1.clone();
    for (layer_child, (layer_p1, layer_p2)) in child
        .layers
        .iter_mut()
        .zip(parent1.layers.iter().zip(&parent2.layers))
    {
        for (weight_child, (weight_p1, weight_p2)) in layer_child
            .weights
            .iter_mut()
            .zip(layer_p1.weights.iter().zip(&layer_p2.weights))
        {
            // Average weights
            for (w_child, (w_p1, w_p2)) in
                weight_child.iter_mut().zip(weight_p1.iter().zip(weight_p2))
            {
                *w_child = (*w_p1 + *w_p2) / 2.0;
            }
            // Average biases
            for (bias_child, (bias_p1, bias_p2)) in layer_child
                .biases
                .iter_mut()
                .zip(layer_p1.biases.iter().zip(&layer_p2.biases))
            {
                *bias_child = (*bias_p1 + *bias_p2) / 2.0;
            }
        }
    }
    child
}

pub fn mutate(nn: &mut NeuralNetwork, mutation_probability: f32, mutation_rate: f32) {
    let mut rng = rand::thread_rng();
    let should_mutate = rng.gen::<f32>() < mutation_probability;
    for layer in &mut nn.layers {
        for weight in &mut layer.weights {
            for w in weight.iter_mut() {
                if should_mutate {
                    *w += rng.gen_range(-mutation_rate..mutation_rate);
                }
            }
        }
        for bias in layer.biases.iter_mut() {
            if should_mutate {
                *bias += rng.gen_range(-mutation_rate..mutation_rate);
            }
        }
    }
}
