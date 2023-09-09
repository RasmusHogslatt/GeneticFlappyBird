mod components;
mod dqn;

use crate::components::*;
use crate::dqn::*;
use bevy::{prelude::*, sprite::MaterialMesh2dBundle};
use bevy_egui::{egui, EguiContexts, EguiPlugin};
use rand::prelude::*;

pub const WINDOW_WIDTH: f32 = 1280.0;
pub const WINDOW_HEIGHT: f32 = 720.0;
pub const CENTER_REWARD_FACTOR: f32 = 100.0;
pub const N_OBSTACLES: usize = 10;
pub const SPAWN_X_POINT: f32 = 400.0;
pub const DESPAWN_X_POINT: f32 = -400.0;
pub const AGENT_SPAWN_POINT: Vec2 = Vec2::new(-300.0, 0.0);
pub const OBSTACLE_SIZE: Vec2 = Vec2::new(25.0, 25.0);
pub const AGENT_SIZE: Vec2 = Vec2::new(50.0, 50.0);
pub const SENSOR_COUNT: usize = 3;

fn main() {
    App::new()
        .add_plugins((DefaultPlugins, EguiPlugin))
        .insert_resource(GuiParameters::default())
        .add_systems(Startup, (set_window_size, setup, spawn_obstacle))
        .add_systems(
            Update,
            (
                move_obstacles,
                move_agent,
                //debug_mover,
                update_sensors,
                //debug_distance,
                dqn_logic,
                update_fitness,
                update_gui,
                spawn_agent,
            ),
        )
        .run();
}

fn setup(mut commands: Commands) {
    commands.spawn(Camera2dBundle::default());
}

fn spawn_agent(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut params: ResMut<GuiParameters>,
) {
    if !params.start {
        return;
    }
    let mut temp_sensors: Vec<Sensor> = Vec::new();
    for _ in 0..SENSOR_COUNT {
        temp_sensors.push(Sensor::default());
    }
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes.add(Mesh::from(shape::Quad::default())).into(),
            transform: Transform {
                translation: AGENT_SPAWN_POINT.extend(0.0),
                scale: AGENT_SIZE.extend(0.0),
                ..Default::default()
            },
            material: materials.add(ColorMaterial::from(Color::BLUE)),
            ..default()
        },
        Agent,
        Velocity {
            speed: Vec2::new(200.0, 200.0),
        },
        Actions::default(),
        Fitness::default(),
        Sensors {
            sensors: temp_sensors.clone(),
        },
        DeepQN::new(
            NeuralNetwork::initialize(vec![6, 24, 48, 24, 4], sigmoid),
            NeuralNetwork::initialize(vec![6, 24, 48, 24, 4], sigmoid),
            &params,
        ),
    ));
    params.start = false;
}

fn set_window_size(mut window: Query<&mut Window>) {
    let mut w = window.single_mut();
    w.resolution.set(WINDOW_WIDTH, WINDOW_HEIGHT);
}

fn move_obstacles(
    mut query: Query<(&mut Transform, &mut Velocity, With<Obstacle>)>,
    time: Res<Time>,
) {
    for (mut transform, mut velocity, _) in query.iter_mut() {
        transform.translation.x -= time.delta_seconds() * velocity.speed.x;
        transform.translation.y -= time.delta_seconds() * velocity.speed.y;
        if transform.translation.x < -1.0 * WINDOW_WIDTH / 2.0 + OBSTACLE_SIZE.x / 2.0 {
            velocity.speed.x *= -1.0;
        }
        if transform.translation.x > WINDOW_WIDTH / 2.0 - OBSTACLE_SIZE.x / 2.0 {
            velocity.speed.x *= -1.0;
        }
        if transform.translation.y < -1.0 * WINDOW_HEIGHT / 2.0 + OBSTACLE_SIZE.y / 2.0 {
            velocity.speed.y *= -1.0;
        }
        if transform.translation.y > WINDOW_HEIGHT / 2.0 - OBSTACLE_SIZE.y / 2.0 {
            velocity.speed.y *= -1.0;
        }
    }
}

fn spawn_obstacle(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
) {
    for _ in 0..N_OBSTACLES {
        let mut rng = rand::thread_rng();
        let mut x_speed_sign: f32 = 1.0;
        let mut y_speed_sign: f32 = 1.0;
        if rng.gen_range(0.0..1.0) > 0.5 {
            x_speed_sign = -1.0;
        }
        if rng.gen_range(0.0..1.0) > 0.5 {
            y_speed_sign = -1.0;
        }

        commands.spawn((
            MaterialMesh2dBundle {
                mesh: meshes.add(Mesh::from(shape::Quad::default())).into(),
                transform: Transform {
                    translation: Vec3::new(
                        rng.gen_range(-WINDOW_WIDTH / 2.0..WINDOW_WIDTH / 2.0),
                        rng.gen_range(-WINDOW_HEIGHT / 2.0..WINDOW_HEIGHT / 2.0),
                        0.0,
                    ),
                    scale: Vec3::new(50.0, 50.0, 0.0),
                    ..Default::default()
                },
                material: materials.add(ColorMaterial::from(Color::PURPLE)),
                ..default()
            },
            Obstacle,
            Velocity {
                speed: Vec2::new(
                    rng.gen_range(100.0..200.0) * x_speed_sign,
                    rng.gen_range(100.0..200.0) * y_speed_sign,
                ),
            },
        ));
    }
}

fn move_agent(
    mut query: Query<(&mut Transform, &Velocity, &Actions, With<Agent>)>,
    time: Res<Time>,
) {
    for (mut transform, velocity, actions, _) in query.iter_mut() {
        if actions.right == 1.0 && transform.translation.x < WINDOW_WIDTH / 2.0 - AGENT_SIZE.x / 2.0
        {
            transform.translation.x += velocity.speed.x * time.delta_seconds();
        }
        if actions.left == 1.0 && transform.translation.x > -WINDOW_WIDTH / 2.0 + AGENT_SIZE.x / 2.0
        {
            transform.translation.x -= velocity.speed.x * time.delta_seconds();
        }
        if actions.up == 1.0 && transform.translation.y < WINDOW_HEIGHT / 2.0 - AGENT_SIZE.y / 2.0 {
            transform.translation.y += velocity.speed.y * time.delta_seconds();
        }
        if actions.down == 1.0
            && transform.translation.y > -WINDOW_HEIGHT / 2.0 + AGENT_SIZE.y / 2.0
        {
            transform.translation.y -= velocity.speed.y * time.delta_seconds();
        }
    }
}

fn _debug_distance(query: Query<(&Sensors, With<Agent>)>) {
    for (sensors, _) in query.iter() {
        println!("D1 = {}", sensors.sensors[0].distance);
        println!("D2 = {}", sensors.sensors[1].distance);
        println!("D3 = {}", sensors.sensors[2].distance)
    }
}

fn update_gui(mut egui_ctx: EguiContexts, mut gui_parameters: ResMut<GuiParameters>) {
    egui::Window::new("Parameters").show(egui_ctx.ctx_mut(), |ui| {
        ui.horizontal(|ui| {
            ui.label("Replay Buffer Size");
            ui.add(egui::Slider::new(
                &mut gui_parameters.replay_buffer_size,
                100..=10000,
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Batch Size");
            ui.add(egui::Slider::new(&mut gui_parameters.batch_size, 1..=1000));
        });
        ui.horizontal(|ui| {
            ui.label("Gamma");
            ui.add(egui::Slider::new(&mut gui_parameters.gamma, 0.0..=1.0));
        });
        ui.horizontal(|ui| {
            ui.label("Epsilon Start");
            ui.add(egui::Slider::new(
                &mut gui_parameters.epsilon_start,
                0.0..=1.0,
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Epsilon End");
            ui.add(egui::Slider::new(
                &mut gui_parameters.epsilon_end,
                0.0..=1.0,
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Epsilon Decay");
            ui.add(egui::Slider::new(
                &mut gui_parameters.epsilon_decay,
                0.0..=1.0,
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Episode Size");
            ui.add(egui::Slider::new(
                &mut gui_parameters.episode_size,
                1..=1000,
            ));
        });
        if ui.button("Start").clicked() {
            gui_parameters.start = true;
        }
    });
}

// Updates sensors reading. Sensors are sorted by distance
fn update_sensors(
    mut agent_query: Query<(&Transform, &mut Sensors, With<Agent>)>,
    obstacle_query: Query<(&Transform, With<Obstacle>)>,
) {
    for (a_tr, mut sensors, _) in agent_query.iter_mut() {
        for i in 0..sensors.sensors.len() {
            sensors.sensors[i] = Sensor {
                distance: 10000000.0,
                direction: 0.0,
            };
        }
        for (o_tr, _) in obstacle_query.iter() {
            // Check distance
            let dist = a_tr
                .translation
                .truncate()
                .distance(o_tr.translation.truncate());

            let dir = a_tr
                .translation
                .truncate()
                .angle_between(o_tr.translation.truncate());
            for i in 0..SENSOR_COUNT {
                if dist < sensors.sensors[i].distance {
                    sensors.sensors.insert(
                        i,
                        Sensor {
                            distance: dist,
                            direction: dir,
                        },
                    );
                    sensors.sensors.pop();
                    break;
                }
            }
        }
    }
}

pub fn update_fitness(mut query: Query<(&mut Fitness, &Sensors, &mut Transform, With<Agent>)>) {
    // Constants for the fitness function
    const EDGE_PENALTY_FACTOR: f32 = 2.5;
    const OBSTACLE_PENALTY_FACTOR: f32 = 100.0;
    const EPSILON: f32 = 1e-6;

    for (mut fitness, sensors, transform, _) in query.iter_mut() {
        // Base fitness
        fitness.score = 1.0;

        // Penalty for proximity to obstacles
        for sensor in sensors.sensors.iter() {
            let obstacle_penalty = OBSTACLE_PENALTY_FACTOR / (sensor.distance + EPSILON);
            fitness.score -= obstacle_penalty;
        }

        // Penalty for proximity to edges
        let x_distance_to_edge = (WINDOW_WIDTH / 2.0 - transform.translation.x.abs()).max(EPSILON);
        let y_distance_to_edge = (WINDOW_HEIGHT / 2.0 - transform.translation.y.abs()).max(EPSILON);

        let edge_penalty_x = EDGE_PENALTY_FACTOR / x_distance_to_edge;
        let edge_penalty_y = EDGE_PENALTY_FACTOR / y_distance_to_edge;

        fitness.score -= edge_penalty_x;
        fitness.score -= edge_penalty_y;

        // Ensure the fitness score remains non-negative
        fitness.score = fitness.score.max(0.0);
        fitness.score = transform.translation.y.max(0.0);
    }
}

pub fn dqn_logic(
    mut agent_query: Query<(&Agent, &mut DeepQN, &Fitness, &mut Actions, &Sensors)>,
    time: Res<Time>,
    params: Res<GuiParameters>,
) {
    for (_agent, mut dqn, fitness, mut actions, sensors) in agent_query.iter_mut() {
        let state: Vec<f32> = sensors
            .sensors
            .iter()
            .flat_map(|s| vec![s.distance, s.direction])
            .collect();

        let action_index = dqn.choose_action(&state);
        match action_index {
            0 => {
                actions.up = 1.0;
                actions.down = 0.0;
                actions.left = 0.0;
                actions.right = 0.0;
            }
            1 => {
                actions.up = -1.0;
                actions.down = 1.0;
                actions.left = 0.0;
                actions.right = 0.0;
            }
            2 => {
                actions.up = 0.0;
                actions.down = 0.0;
                actions.left = 0.0;
                actions.right = 1.0;
            }
            3 => {
                actions.up = 0.0;
                actions.down = 0.0;
                actions.left = 1.0;
                actions.right = -1.0;
            }
            _ => {}
        }

        // Storing experience and training
        let reward = fitness.score * time.delta_seconds();
        let next_state: Vec<f32> = sensors
            .sensors
            .iter()
            .flat_map(|s| vec![s.distance, s.direction])
            .collect();

        let done = false; // This can be set to true if some end condition is met
        dqn.store_experience(state, action_index, reward, next_state, done, &params);
        dqn.train(0.001, &params);
        dqn.update_epsilon(&params);

        if dqn.training_counter % params.episode_size == 0 {
            dqn.update_target_network();
        }
    }
}
