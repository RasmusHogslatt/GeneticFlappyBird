mod components;
mod gui;
mod neural_network;
mod systems;

use crate::components::*;
use crate::gui::*;
use crate::neural_network::*;
use crate::systems::*;
use bevy::{prelude::*, sprite::MaterialMesh2dBundle};
use bevy_egui::EguiPlugin;
use rand::prelude::*;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, EguiPlugin))
        .insert_resource(GuiParameters::default())
        .insert_resource(BestBirds {
            best_neural_network: NeuralNetwork::new(&[4, 3, 1]),
            second_best_neural_network: NeuralNetwork::new(&[4, 3, 1]),
            best_score: 0.0,
            second_best_score: 0.0,
            best_fitness: 0.0,
            second_best_fitness: 0.0,
        })
        .add_systems(Startup, (set_window_size, setup, spawn_bird))
        .add_systems(
            Update,
            (
                update_gui,
                gravity_system,
                jump_system,
                move_bird,
                spawn_pipe,
                despawn_pipe,
                update_my_time,
                move_pipes,
                update_environment_state,
                update_fitness,
                check_collision,
                generate_next_generation,
            ),
        )
        .run();
}
/* INPUTS TO NEURAL NETWORK*/
// Horizontal distance next pipe: f32 x
// Vertical gap position center of next pipe: f32 y
// Velocity of bird: f32 v
// Position of bird: f32 y

/* OUTPUT OF NEURAL NETWORK */
// Jump: bool

/* NEURAL NETWORK PARAMETERS */
// Number of inputs: 4
// Number of outputs: 1
// Number of hidden layers: 1
// Number of neurons in hidden layer: 3
// Activation function: Sigmoid

/* SCORING */
// Score += 0.5 per top and bottom half: f32 1.0 per pipe

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}

fn spawn_bird(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    gui_parameters: ResMut<GuiParameters>,
) {
    for _ in 0..gui_parameters.population_size {
        commands.spawn((
            MaterialMesh2dBundle {
                mesh: meshes.add(Mesh::from(shape::Quad::default())).into(),
                transform: Transform {
                    translation: Vec3::new(SPAWN_X_POINT, 0.0, 0.0),
                    scale: Vec3::new(BIRD_SIZE, BIRD_SIZE, 0.0),
                    ..Default::default()
                },
                material: materials.add(ColorMaterial::from(Color::RED)),
                ..default()
            },
            Bird {
                velocity: 0.0,
                score: 0.0,
                fitness: 0.0,
                dead: false,
                neural_network: NeuralNetwork::new(&[4, 3, 1]),
            },
            Environment::default(),
        ));
    }
}

pub fn update_my_time(mut params: ResMut<GuiParameters>, time: Res<Time>) {
    params.passed_time_since_start += time.delta_seconds();
    params.passed_time_since_last_pipe += time.delta_seconds();
}

pub fn update_fitness(
    mut query: Query<&mut Bird>,
    time: Res<Time>,
    mut params: ResMut<GuiParameters>,
) {
    for mut bird in query.iter_mut() {
        bird.fitness += time.delta_seconds();
    }
    params.best_fitness += time.delta_seconds();
}

fn spawn_pipe(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut params: ResMut<GuiParameters>,
) {
    if params.passed_time_since_last_pipe < 3.0 {
        return;
    }
    params.passed_time_since_last_pipe = 0.0;
    let mut rng = rand::thread_rng();
    let gc = rng.gen_range((-WINDOW_HEIGHT / 2.0 + GAP_WIDTH)..(WINDOW_HEIGHT / 2.0 - GAP_WIDTH));
    let y_pos1 = gc - WINDOW_HEIGHT / 2.0 - GAP_WIDTH / 2.0;
    let y_pos2 = gc + WINDOW_HEIGHT / 2.0 + GAP_WIDTH / 2.0;
    commands.spawn((
        Pipe {
            velocity: -PIPE_VELOCITY,
            gap: GAP_WIDTH,
            gap_center: gc,
            bird_passed: false,
        },
        MaterialMesh2dBundle {
            mesh: meshes.add(Mesh::from(shape::Quad::default())).into(),
            transform: Transform {
                translation: Vec3::new(WINDOW_WIDTH / 2.0 - PIPE_WIDTH, y_pos1, 0.0),
                scale: Vec3::new(PIPE_WIDTH, WINDOW_HEIGHT, 0.0),
                ..Default::default()
            },
            material: materials.add(ColorMaterial::from(Color::GREEN)),
            ..default()
        },
    ));
    commands.spawn((
        Pipe {
            velocity: -PIPE_VELOCITY,
            gap: GAP_WIDTH,
            gap_center: gc,
            bird_passed: false,
        },
        MaterialMesh2dBundle {
            mesh: meshes.add(Mesh::from(shape::Quad::default())).into(),
            transform: Transform {
                translation: Vec3::new(WINDOW_WIDTH / 2.0 - PIPE_WIDTH, y_pos2, 0.0),
                scale: Vec3::new(PIPE_WIDTH, WINDOW_HEIGHT, 0.0),
                ..Default::default()
            },
            material: materials.add(ColorMaterial::from(Color::GREEN)),
            ..default()
        },
    ));
    println!("Spawned pipe!");
}

pub fn despawn_pipe(
    mut commands: Commands,
    mut query: Query<(Entity, &Pipe, &Transform)>,
    mut _params: ResMut<GuiParameters>,
) {
    for (entity, _pipe, transform) in query.iter_mut() {
        if transform.translation.x < -WINDOW_WIDTH / 2.0 - PIPE_WIDTH / 2.0 {
            commands.entity(entity).despawn();
        }
    }
}

pub fn move_pipes(mut query: Query<(&mut Transform, &Pipe)>, time: Res<Time>) {
    for (mut transform, pipe) in query.iter_mut() {
        transform.translation.x += pipe.velocity * time.delta_seconds();
    }
}

fn set_window_size(mut window: Query<&mut Window>) {
    let mut w = window.single_mut();
    w.resolution.set(WINDOW_WIDTH, WINDOW_HEIGHT);
}

pub fn update_environment_state(
    mut bird_query: Query<(&Transform, &mut Bird, &mut Environment)>,
    mut pipe_query: Query<(&Transform, &mut Pipe)>,

    mut params: ResMut<GuiParameters>,
) {
    for (bird_transform, mut bird, mut environment) in bird_query.iter_mut() {
        let mut nearest_pipe_dist: f32 = f32::MAX;
        let mut nearest_pipe_vertical_center: f32 = f32::MAX;
        for (pipe_transform, mut pipe) in pipe_query.iter_mut() {
            if pipe_transform.translation.x < bird_transform.translation.x && !pipe.bird_passed {
                pipe.bird_passed = true;
                bird.score += 0.5;
                params.best_generation_score += 0.5;
                continue; // Pipe is behind bird
            }
            let temporary_dist: f32 = pipe_transform.translation.x - bird_transform.translation.x;
            if temporary_dist < nearest_pipe_dist {
                nearest_pipe_dist = temporary_dist;
                nearest_pipe_vertical_center = pipe.gap_center;
            }
        }
        environment.horizontal_distance = nearest_pipe_dist;
        environment.vertical_gap_position = nearest_pipe_vertical_center;
    }
    // println!(
    //     "Horizontal distance: {}, Vertical gap center: {}",
    //     nearest_pipe_dist, nearest_pipe_vertical_center
    // );
}
