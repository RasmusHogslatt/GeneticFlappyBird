use crate::components::*;
use crate::gui::*;
use crate::WINDOW_HEIGHT;
use bevy::prelude::*;

pub const GRAVITY: f32 = 9.82;
pub const JUMP_FORCE: f32 = 8.0;

pub fn gravity_system(mut query: Query<&mut Bird>, time: Res<Time>) {
    for mut bird in query.iter_mut() {
        bird.velocity -= GRAVITY * time.delta_seconds();
    }
}

pub fn jump_system(mut query: Query<&mut Bird>, input: Res<Input<KeyCode>>) {
    for mut bird in query.iter_mut() {
        if input.just_pressed(KeyCode::Space) {
            bird.velocity += JUMP_FORCE;
            println!("Jumped!");
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
    mut bird_query: Query<(&Transform, &mut Bird)>,
    mut pipe_query: Query<(&Transform, With<Pipe>)>,
    mut params: ResMut<GuiParameters>,
) {
    for (bird_transform, mut bird) in bird_query.iter_mut() {
        for (pipe_transform, _pipe) in pipe_query.iter_mut() {
            if (bird_transform.translation.x - pipe_transform.translation.x).abs()
                < PIPE_WIDTH / 2.0 + BIRD_SIZE / 2.0
                && (bird_transform.translation.y - pipe_transform.translation.y).abs()
                    < GAP_WIDTH / 2.0 + BIRD_SIZE / 2.0
            {
                bird.dead = true;
                params.dead_bird_count += 1;
                println!("Collision!");
            }
        }
    }
}
