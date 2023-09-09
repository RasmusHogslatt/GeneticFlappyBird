mod components;

use crate::components::*;
use bevy::{prelude::*, sprite::MaterialMesh2dBundle};
use rand::prelude::*;

pub const WINDOW_WIDTH: f32 = 1280.0;
pub const WINDOW_HEIGHT: f32 = 720.0;
pub const N_OBSTACLES: usize = 10;
pub const SPAWN_X_POINT: f32 = 400.0;
pub const DESPAWN_X_POINT: f32 = -400.0;
pub const AGENT_SPAWN_POINT: Vec2 = Vec2::new(-300.0, 0.0);
pub const OBSTACLE_SIZE: Vec2 = Vec2::new(25.0, 25.0);
pub const AGENT_SIZE: Vec2 = Vec2::new(50.0, 50.0);
pub const SENSOR_COUNT: usize = 3;

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(
            Startup,
            (set_window_size, setup, spawn_obstacle, spawn_agent),
        )
        .add_systems(
            Update,
            (
                move_obstacles,
                //despawn_obstacles,
                move_agent,
                debug_mover,
                update_sensors,
                debug_distance,
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
) {
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
    ));
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
        if actions.right < -0.33 {
            transform.translation.x -= velocity.speed.x * time.delta_seconds();
        } else if actions.right > 0.33 {
            transform.translation.x += velocity.speed.x * time.delta_seconds();
        }

        if actions.up < -0.33 {
            transform.translation.y -= velocity.speed.y * time.delta_seconds();
        } else if actions.up > 0.33 {
            transform.translation.y += velocity.speed.y * time.delta_seconds();
        }
    }
}

fn debug_mover(mut query: Query<(&mut Transform, &Velocity, &mut Actions, With<Agent>)>) {
    let mut rng = thread_rng();
    for (_transform, _velocity, mut actions, _) in query.iter_mut() {
        actions.up = rng.gen_range(-1.0..1.0);
        actions.right = rng.gen_range(-1.0..1.0);
    }
}

fn debug_distance(query: Query<(&Sensors, With<Agent>)>) {
    for (sensors, _) in query.iter() {
        println!("D1 = {}", sensors.sensors[0].distance);
        println!("D2 = {}", sensors.sensors[1].distance);
        println!("D3 = {}", sensors.sensors[2].distance)
    }
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
