use cgmath::{InnerSpace, Rotation3};

use crate::{Instance, InstanceRaw};

pub struct Animator {
    frames: Vec<Vec<InstanceRaw>>,
    current_frame: usize,
    current_delta: usize,
    delta: usize,
}

impl Animator {
    pub fn new(frames: Vec<Vec<cgmath::Vector3<f32>>>, delta: usize) -> Self {
        let frames = frames.iter().map(|x| x.iter().map(|pos| {
            Instance {
                position: pos.to_owned(), 
                rotation: cgmath::Quaternion::from_axis_angle(pos.normalize(), cgmath::Deg(0.0)),
            }.to_raw()
        }).collect::<Vec<_>>()).collect::<Vec<_>>();
        Self {
            frames,
            current_frame: 0,
            current_delta: 0,
            delta,
        }
    }

    pub fn update(&mut self) {
        self.current_delta += 1;
        if self.current_delta % self.delta == 0 {
            self.current_frame += 1;
            if self.current_frame >= self.frames.len() {
                self.current_frame = 0;
            }
        }
    }

    pub fn get_current_frame(&self) -> &Vec<InstanceRaw> {
        &self.frames[self.current_frame]
    }
}
