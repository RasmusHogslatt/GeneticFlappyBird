use crate::components::*;
use crate::gui::*;
use crate::neural_network::crossover;
use crate::neural_network::mutate;
use crate::WINDOW_HEIGHT;
use bevy::prelude::*;
use bevy::sprite::collide_aabb::collide;
use bevy::sprite::collide_aabb::Collision;
use bevy::sprite::MaterialMesh2dBundle;

pub const GRAVITY: f32 = 9.82;
pub const JUMP_FORCE: f32 = 4.0;

pub fn gravity_system(mut query: Query<&mut Bird>, time: Res<Time>) {
    for mut bird in query.iter_mut() {
        bird.velocity -= GRAVITY * time.delta_seconds();
    }
}

pub fn jump_system(
    mut query: Query<(&mut Bird, &Environment, &Transform)>,
    input: Res<Input<KeyCode>>,
) {
    for (mut bird, environment, transform) in query.iter_mut() {
        if input.just_pressed(KeyCode::Space) {
            bird.velocity += JUMP_FORCE;
            println!("Jumped!");
        }
        let input = vec![
            transform.translation.y,
            bird.velocity,
            environment.horizontal_distance,
            environment.vertical_gap_position,
        ];
        let output = bird.neural_network.forward(&input);
        if output[0] > 0.5 {
            bird.velocity += JUMP_FORCE;
        }
    }
}

pub fn move_bird(
    mut query: Query<(&mut Bird, &mut Transform)>,
    time: Res<Time>,
    params: Res<GuiParameters>,
) {
    for (mut bird, mut transform) in query.iter_mut() {
        if transform.translation.y > WINDOW_HEIGHT / 2.0 - BIRD_SIZE / 2.0 {
            bird.velocity = 0.0;
            transform.translation.y = WINDOW_HEIGHT / 2.0 - BIRD_SIZE / 2.0 - 0.1;
        }
        if transform.translation.y < -WINDOW_HEIGHT / 2.0 + BIRD_SIZE / 2.0 {
            bird.velocity = 0.0;
            transform.translation.y = -WINDOW_HEIGHT / 2.0 + BIRD_SIZE / 2.0 + 0.1;
        }

        transform.translation.y += bird.velocity * time.delta_seconds() * params.force_scaling;
    }
}

pub fn check_collision(
    mut commands: Commands,
    mut bird_query: Query<(Entity, &Transform, &mut Bird)>,
    mut pipe_query: Query<(&Transform, With<Pipe>)>,
    mut params: ResMut<GuiParameters>,
    mut best_birds: ResMut<BestBirds>,
) {
    if params.dead_bird_count == params.population_size {
        return;
    }
    for (ent, bird_transform, mut bird) in bird_query.iter_mut() {
        for (pipe_transform, _pipe) in pipe_query.iter_mut() {
            let collision = collide(
                bird_transform.translation,
                Vec2::new(BIRD_SIZE, BIRD_SIZE),
                pipe_transform.translation,
                Vec2::new(PIPE_WIDTH, WINDOW_HEIGHT),
            );

            if let Some(collision) = collision {
                match collision {
                    Collision::Left => {
                        bird.dead = true;
                        params.dead_bird_count += 1;
                        println!("Dead bird count: {}", params.dead_bird_count);
                    }
                    Collision::Right => {
                        // Past pipe
                    }
                    Collision::Top => {
                        bird.dead = true;
                        params.dead_bird_count += 1;
                        println!("Dead bird count: {}", params.dead_bird_count);
                    }
                    Collision::Bottom => {
                        bird.dead = true;
                        params.dead_bird_count += 1;
                        println!("Dead bird count: {}", params.dead_bird_count);
                    }
                    Collision::Inside => {
                        // Inside pipe
                    }
                }
            }
        }
        if bird.dead {
            if params.dead_bird_count == params.population_size {
                best_birds.best_neural_network = bird.neural_network.clone();
                best_birds.best_score = bird.score;
                best_birds.best_fitness = bird.fitness;
                params.generation_dead = true;
                println!("All birds dead");
            } else if params.dead_bird_count == params.population_size - 1 {
                best_birds.second_best_neural_network = bird.neural_network.clone();
                best_birds.second_best_score = bird.score;
                best_birds.second_best_fitness = bird.fitness;
            }
            // despawn
            commands.entity(ent).despawn_recursive();
        }
    }
}

// spawn new gen if generation is dead
pub fn generate_next_generation(
    mut commands: Commands,
    mut params: ResMut<GuiParameters>,
    best_birds: ResMut<BestBirds>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    mut pipe_query: Query<(Entity, &Pipe)>,
) {
    if !params.generation_dead {
        return;
    }
    for i in 0..params.population_size {
        let mut child_neural_network = crossover(
            best_birds.best_neural_network.clone(),
            best_birds.second_best_neural_network.clone(),
        );
        mutate(
            &mut child_neural_network,
            params.mutation_probability,
            params.mutation_rate,
        );

        let visible: Visibility = if i < params.number_of_visible_bird {
            Visibility::Visible
        } else {
            Visibility::Hidden
        };
        commands.spawn((
            MaterialMesh2dBundle {
                mesh: meshes.add(Mesh::from(shape::Quad::default())).into(),
                transform: Transform {
                    translation: Vec3::new(SPAWN_X_POINT, 0.0, 0.0),
                    scale: Vec3::new(BIRD_SIZE, BIRD_SIZE, 0.0),
                    ..Default::default()
                },
                material: materials.add(ColorMaterial::from(Color::RED)),
                visibility: visible,
                ..default()
            },
            Bird {
                velocity: 0.0,
                score: 0.0,
                fitness: 0.0,
                dead: false,
                neural_network: child_neural_network.clone(),
            },
            Environment::default(),
        ));
    }
    params.generation_dead = false;
    params.dead_bird_count = 0;
    params.best_generation_score = 0.0;
    params.best_fitness = 0.0;
    params.current_generation += 1;
    println!("Generating next generation");
    // Despawn all pipes
    for (ent, _pipe) in pipe_query.iter_mut() {
        commands.entity(ent).despawn_recursive();
    }
}
