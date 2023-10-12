use bevy::prelude::*;
use bevy_egui::{
    egui::{self, Color32, RichText},
    EguiContexts,
};

use crate::neural_network::NeuralNetwork;

pub const WINDOW_WIDTH: f32 = 1280.0;
pub const WINDOW_HEIGHT: f32 = 720.0;
pub const GAP_WIDTH: f32 = 125.0;
pub const PIPE_VELOCITY: f32 = 200.0;
pub const PIPE_WIDTH: f32 = 50.0;
pub const BIRD_SIZE: f32 = 15.0;
pub const SPAWN_X_POINT: f32 = -300.0;
pub const POPULATION_SIZE: usize = 400;

#[derive(Clone, Debug, Resource)]
pub struct GuiParameters {
    pub force_scaling: f32,
    pub passed_time_since_start: f32,
    pub passed_time_since_last_pipe: f32,
    pub population_size: usize,    // Number of birds in population
    pub dead_bird_count: usize,    // Check if all birds are dead
    pub current_score: f32,        // For display
    pub generation_dead: bool, // True if entire population is dead. Then a new population can be spawned
    pub mutation_rate: f32,    // Rate of mutation (factor to scale with, positive or negative)
    pub mutation_probability: f32, // Probability of mutation happening to weight
    pub current_generation: usize,
    pub number_of_visible_bird: usize,
    pub start_training: bool,
}

impl Default for GuiParameters {
    fn default() -> Self {
        Self {
            force_scaling: 45.0,
            passed_time_since_start: 0.0,
            passed_time_since_last_pipe: 10000.0,
            population_size: POPULATION_SIZE,
            dead_bird_count: 0,
            current_score: 0.0,
            generation_dead: false,
            mutation_rate: 0.125,
            mutation_probability: 0.5,
            current_generation: 0,
            number_of_visible_bird: POPULATION_SIZE,
            start_training: false,
        }
    }
}

#[derive(Clone, Debug, Resource)]
pub struct BestBirds {
    pub best_neural_network: NeuralNetwork,
    pub second_best_neural_network: NeuralNetwork,
    pub best_score: f32,
    pub second_best_score: f32,
    pub best_fitness: f32,
    pub second_best_fitness: f32,
}

#[derive(Clone, Debug, Component)]
pub struct Environment {
    pub horizontal_distance: f32,
    pub vertical_gap_position: f32,
}

impl Default for Environment {
    fn default() -> Self {
        Self {
            horizontal_distance: 0.0,
            vertical_gap_position: 0.0,
        }
    }
}

pub fn update_gui(
    mut egui_ctx: EguiContexts,
    mut gui_parameters: ResMut<GuiParameters>,
    best_birds: ResMut<BestBirds>,
) {
    egui::Window::new("Parameters").show(egui_ctx.ctx_mut(), |ui| {
        if ui.button("Start training").clicked() {
            gui_parameters.start_training = !gui_parameters.start_training;
        }
        ui.label(format!(
            "Time since start: {:.2}",
            gui_parameters.passed_time_since_start
        ));
        ui.horizontal(|ui| {
            ui.label("Population Size");
            ui.add(egui::Slider::new(
                &mut gui_parameters.population_size,
                1..=400,
            ));
        });
        // ui.horizontal(|ui| {
        //     ui.label("Force Scaling");
        //     ui.add(egui::Slider::new(
        //         &mut gui_parameters.force_scaling,
        //         1.0..=50.0,
        //     ));
        // });

        // set population size
        ui.label(
            RichText::new(format!(
                "Dead Bird Count: {}",
                gui_parameters.dead_bird_count
            ))
            .color(Color32::RED),
        );
        ui.label(format!(
            "Current Generation: {}",
            gui_parameters.current_generation
        ));
        ui.label(
            RichText::new(format!("Best Score: {:.2}", best_birds.best_score).to_string())
                .color(Color32::GREEN),
        );
        ui.label(
            RichText::new(format!("Best Fitness: {:.2}", best_birds.best_fitness))
                .color(Color32::GREEN),
        );
        ui.label(
            RichText::new(format!(
                "Current Score: {:.2}",
                gui_parameters.current_score
            ))
            .color(Color32::YELLOW),
        );
        // set mutation rate
        ui.horizontal(|ui| {
            ui.label("Mutation Rate");
            ui.add(egui::Slider::new(
                &mut gui_parameters.mutation_rate,
                -1.0..=1.0,
            ));
        });
        //set mutation probability
        ui.horizontal(|ui| {
            ui.label("Mutation Probability");
            ui.add(egui::Slider::new(
                &mut gui_parameters.mutation_probability,
                0.0..=1.0,
            ));
        });
        // set visible bird
        // let max_bird_count = gui_parameters.population_size;
        // ui.horizontal(|ui| {
        //     ui.label("Visible Birds");
        //     ui.add(egui::Slider::new(
        //         &mut gui_parameters.number_of_visible_bird,
        //         1..=max_bird_count,
        //     ));
        // });
    });
}
