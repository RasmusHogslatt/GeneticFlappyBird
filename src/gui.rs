use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

pub const WINDOW_WIDTH: f32 = 1280.0;
pub const WINDOW_HEIGHT: f32 = 720.0;
pub const GAP_WIDTH: f32 = 100.0;
pub const PIPE_VELOCITY: f32 = 200.0;
pub const PIPE_WIDTH: f32 = 50.0;
pub const BIRD_SIZE: f32 = 25.0;
pub const SPAWN_X_POINT: f32 = -300.0;

#[derive(Clone, Debug, Resource)]
pub struct GuiParameters {
    pub force_scaling: f32,
    pub passed_time_since_start: f32,
    pub passed_time_since_last_pipe: f32,
    pub population_size: usize,
    pub dead_bird_count: usize,
    pub best_generation_score: f32,
    pub best_fitness: f32,
}

impl Default for GuiParameters {
    fn default() -> Self {
        Self {
            force_scaling: 20.0,
            passed_time_since_start: 0.0,
            passed_time_since_last_pipe: 10000.0,
            population_size: 20,
            dead_bird_count: 0,
            best_generation_score: 0.0,
            best_fitness: 0.0,
        }
    }
}

#[derive(Clone, Debug, Resource)]
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

pub fn update_gui(mut egui_ctx: EguiContexts, mut gui_parameters: ResMut<GuiParameters>) {
    egui::Window::new("Parameters").show(egui_ctx.ctx_mut(), |ui| {
        ui.horizontal(|ui| {
            ui.label("Force Scaling");
            ui.add(egui::Slider::new(
                &mut gui_parameters.force_scaling,
                1.0..=50.0,
            ));
        });
    });
}
