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
