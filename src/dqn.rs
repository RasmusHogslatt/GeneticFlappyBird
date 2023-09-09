// In dqn.rs
use bevy::prelude::*;

use crate::components::NeuralNetwork;
use rand::Rng; // For random number generation and sampling // Assuming components.rs is in the same crate

#[derive(Clone, Debug, Resource)]
pub struct GuiParameters {
    pub replay_buffer_size: usize, // Number of experiences to store in the replay buffer
    pub batch_size: usize,         // Number of experiences to sample from the replay buffer
    pub gamma: f32, // Discount factor (0 --> get short term reward, 1 --> get long term reward)
    pub epsilon_start: f32, // Start high
    pub epsilon_end: f32, // End low
    pub epsilon_decay: f32, // Decay over time
    pub start: bool, // Whether to spawn agent
    pub episode_counter: usize,
    pub episode_size: usize,
}

impl Default for GuiParameters {
    fn default() -> Self {
        Self {
            replay_buffer_size: 10000,
            batch_size: 32,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.9,
            epsilon_decay: 0.995,
            start: false,
            episode_counter: 0,
            episode_size: 100,
        }
    }
}

#[derive(Component, Debug, Clone)]
pub struct DeepQN {
    q_network: NeuralNetwork,
    target_network: NeuralNetwork,
    replay_buffer: Vec<Experience>,
    epsilon: f32,
    pub training_counter: usize,
}

#[derive(Debug, Clone)]
pub struct Experience {
    state: Vec<f32>,
    action: usize,
    reward: f32,
    next_state: Vec<f32>,
    done: bool,
}

impl DeepQN {
    pub fn new(
        q_network: NeuralNetwork,
        target_network: NeuralNetwork,
        params: &ResMut<GuiParameters>,
    ) -> Self {
        DeepQN {
            q_network,
            target_network,
            replay_buffer: Vec::with_capacity(params.replay_buffer_size),
            epsilon: params.epsilon_start,
            training_counter: 0,
        }
    }

    pub fn update_target_network(&mut self) {
        self.target_network = self.q_network.clone();
    }

    pub fn choose_action(&mut self, state: &[f32]) -> usize {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() < self.epsilon {
            rng.gen_range(0..4) // Randomly select one of the four directions
        } else {
            let q_values = self.q_network.forward(state.to_owned());
            q_values
                .iter()
                .enumerate()
                .filter(|(_, &val)| val.is_finite()) // Filter out NaN values
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map_or(0, |(idx, _)| idx) // Default to the first action if all are NaN or the vector is empty
        }
    }

    pub fn store_experience(
        &mut self,
        state: Vec<f32>,
        action: usize,
        reward: f32,
        next_state: Vec<f32>,
        done: bool,
        params: &Res<GuiParameters>,
    ) {
        if self.replay_buffer.len() >= params.replay_buffer_size {
            self.replay_buffer.remove(0);
        }
        self.replay_buffer.push(Experience {
            state,
            action,
            reward,
            next_state,
            done,
        });
    }

    pub fn update_epsilon(&mut self, params: &Res<GuiParameters>) {
        self.epsilon = params.epsilon_end
            + (params.epsilon_start - params.epsilon_end)
                * (params.epsilon_decay.powi(self.replay_buffer.len() as i32));
        //info!("Epsilon: {}", self.epsilon);
    }

    pub fn train(&mut self, learning_rate: f32, params: &Res<GuiParameters>) {
        if self.replay_buffer.len() < params.batch_size {
            return; // Not enough experiences to sample a batch
        }

        let rng: rand::rngs::ThreadRng = rand::thread_rng();
        let samples: Vec<&Experience> = rng
            .sample_iter(&rand::distributions::Uniform::new(
                0,
                self.replay_buffer.len(),
            ))
            .take(params.batch_size)
            .map(|i| &self.replay_buffer[i])
            .collect();

        let mut states = Vec::new();
        let mut target_q_values = Vec::new();

        for experience in &samples {
            let current_q_values = self.q_network.forward(experience.state.clone());
            let next_q_values = self.target_network.forward(experience.next_state.clone());

            let max_next_q_value = next_q_values
                .iter()
                .filter(|&&x| x.is_finite()) // Filter out NaN values
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(&0.0);

            let target_value = if experience.done {
                experience.reward
            } else {
                experience.reward + params.gamma * max_next_q_value
            };

            let mut target_q_values_for_state = current_q_values.clone();
            target_q_values_for_state[experience.action] = target_value;
            states.push(experience.state.clone());
            target_q_values.push(target_q_values_for_state);
        }

        // Loop over the batch of experiences
        for (state, target_q_values_for_state) in states.iter().zip(target_q_values.iter()) {
            // Forward pass with intermediates
            let (_, all_activations, all_weighted_sums) =
                self.q_network.forward_with_intermediates(state.clone());

            // Compute the gradient of the loss with respect to the output
            let current_q_values = self.q_network.forward(state.clone());
            let d_loss: Vec<f32> = current_q_values
                .iter()
                .zip(target_q_values_for_state)
                .map(|(q, target_q)| q - target_q)
                .collect();

            // Backward pass
            self.q_network
                .backward(&all_activations, &all_weighted_sums, d_loss, learning_rate);
        }
        self.training_counter += 1;
        println!("Training counter: {}", self.training_counter);
    }
}
