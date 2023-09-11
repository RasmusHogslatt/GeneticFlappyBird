use crate::neural_network::*;
use bevy::prelude::*;

#[derive(Clone, Debug, Component)]
pub struct Bird {
    pub velocity: f32,
    pub score: f32,   // Number of pipes passed
    pub fitness: f32, // Second alive
    pub dead: bool,
    pub neural_network: NeuralNetwork,
}

#[derive(Clone, Debug, Component)]
pub struct Pipe {
    pub velocity: f32,
    pub gap: f32,        // Width of gap
    pub gap_center: f32, // Vertical center of gap
    pub bird_passed: bool,
}
