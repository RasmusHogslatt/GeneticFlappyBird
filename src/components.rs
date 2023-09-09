use std::f32::MAX;

use bevy::prelude::*;

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
    pub right: f32,
}

#[derive(Component, Default)]
pub struct Fitness {
    pub score: f32,
}

#[derive(Component, Default)]
pub struct Sensors {
    pub sensors: Vec<Sensor>,
}

#[derive(Component)]
pub struct NeuralNetwork {
    pub weights: Vec<f32>,
    pub bias: f32,
    pub activation_function: fn(f32) -> f32,
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
