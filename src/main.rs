use cgmath::prelude::*;
use std::iter;
use wgpu::util::DeviceExt;
use winit::{
    event::*,
    event_loop::{ControlFlow, EventLoop},
    window::Window,
};

mod camera;
mod model;
mod texture;
mod animator; // NEW!

use model::{DrawLight, DrawModel, Vertex};

use crate::animator::Animator;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    view_position: [f32; 4],
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            view_position: [0.0; 4],
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    // UPDATED!
    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.view_position = camera.position.to_homogeneous().into();
        self.view_proj = (projection.calc_matrix() * camera.calc_matrix()).into()
    }
}

struct Instance {
    position: cgmath::Vector3<f32>,
    rotation: cgmath::Quaternion<f32>,
}

impl Instance {
    fn to_raw(&self) -> InstanceRaw {
        InstanceRaw {
            model: (cgmath::Matrix4::from_translation(self.position)
                * cgmath::Matrix4::from(self.rotation))
            .into(),
            normal: cgmath::Matrix3::from(self.rotation).into(),
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
#[allow(dead_code)]
pub struct InstanceRaw {
    model: [[f32; 4]; 4],
    normal: [[f32; 3]; 3],
}

impl model::Vertex for InstanceRaw {
    fn desc<'a>() -> wgpu::VertexBufferLayout<'a> {
        use std::mem;
        wgpu::VertexBufferLayout {
            array_stride: mem::size_of::<InstanceRaw>() as wgpu::BufferAddress,
            // We need to switch from using a step mode of Vertex to Instance
            // This means that our shaders will only change to use the next
            // instance when the shader starts processing a new instance
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    // While our vertex shader only uses locations 0, and 1 now, in later tutorials we'll
                    // be using 2, 3, and 4, for Vertex. We'll start at slot 5 not conflict with them later
                    shader_location: 5,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // A mat4 takes up 4 vertex slots as it is technically 4 vec4s. We need to define a slot
                // for each vec4. We don't have to do this in code though.
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 4]>() as wgpu::BufferAddress,
                    shader_location: 6,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 7,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 12]>() as wgpu::BufferAddress,
                    shader_location: 8,
                    format: wgpu::VertexFormat::Float32x4,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 16]>() as wgpu::BufferAddress,
                    shader_location: 9,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 19]>() as wgpu::BufferAddress,
                    shader_location: 10,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: mem::size_of::<[f32; 22]>() as wgpu::BufferAddress,
                    shader_location: 11,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct LightUniform {
    position: [f32; 3],
    // Due to uniforms requiring 16 byte (4 float) spacing, we need to use a padding field here
    _padding: u32,
    color: [f32; 3],
}

struct State {
    surface: wgpu::Surface,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    render_pipeline: wgpu::RenderPipeline,
    obj_model: model::Model,
    camera: camera::Camera,
    projection: camera::Projection,
    camera_controller: camera::CameraController,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    #[allow(dead_code)]
    instance_buffer: wgpu::Buffer,
    depth_texture: texture::Texture,
    size: winit::dpi::PhysicalSize<u32>,
    light_uniform: LightUniform,
    light_buffer: wgpu::Buffer,
    light_bind_group: wgpu::BindGroup,
    light_render_pipeline: wgpu::RenderPipeline,
    #[allow(dead_code)]
    debug_material: model::Material,
    mouse_pressed: bool,
    instance_animator: Animator,
}

fn create_render_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    color_format: wgpu::TextureFormat,
    depth_format: Option<wgpu::TextureFormat>,
    vertex_layouts: &[wgpu::VertexBufferLayout],
    shader: wgpu::ShaderModuleDescriptor,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(&shader);

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{:?}", shader)),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "main",
            buffers: vertex_layouts,
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "main",
            targets: &[wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState {
                    alpha: wgpu::BlendComponent::REPLACE,
                    color: wgpu::BlendComponent::REPLACE,
                }),
                write_mask: wgpu::ColorWrites::ALL,
            }],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
            polygon_mode: wgpu::PolygonMode::Fill,
            // Requires Features::DEPTH_CLAMPING
            clamp_depth: false,
            // Requires Features::CONSERVATIVE_RASTERIZATION
            conservative: false,
        },
        depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
            format,
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        },
    })
}

impl State {
    async fn new(window: &Window) -> Self {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(wgpu::Backends::all());
        let surface = unsafe { instance.create_surface(window) };
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface.get_preferred_format(&adapter).unwrap(),
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
        };

        surface.configure(&device, &config);

        let texture_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true,
                        },
                        count: None,
                    },
                    // normal map
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            multisampled: false,
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler {
                            comparison: false,
                            filtering: true,
                        },
                        count: None,
                    },
                ],
                label: Some("texture_bind_group_layout"),
            });

        // UPDATED!
        let camera = camera::Camera::new((0.0, 5.0, 10.0), cgmath::Deg(-90.0), cgmath::Deg(-20.0));
        let projection =
            camera::Projection::new(config.width, config.height, cgmath::Deg(45.0), 0.1, 100.0);
        let camera_controller = camera::CameraController::new(4.0, 0.4);

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // let instance_data: Vec::<Instance> = vec![];
        // let instance_data = instance_data.iter().map(Instance::to_raw).collect::<Vec<InstanceRaw>>();
        let instance_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Instance Buffer"),
            contents: &[0_u8; 1024 * 256], //bytemuck::cast_slice(&instance_data),
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let res_dir = std::path::Path::new(env!("OUT_DIR")).join("res");
        let obj_model = model::Model::load(
            &device,
            &queue,
            &texture_bind_group_layout,
            res_dir.join("cube.obj"),
        )
        .unwrap();

        let light_uniform = LightUniform {
            position: [2.0, 2.0, 2.0],
            _padding: 0,
            color: [1.0, 1.0, 1.0],
        };

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Light VB"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let light_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: None,
            });

        let light_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &light_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: light_buffer.as_entire_binding(),
            }],
            label: None,
        });

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &config, "depth_texture");

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[
                    &texture_bind_group_layout,
                    &camera_bind_group_layout,
                    &light_bind_group_layout,
                ],
                push_constant_ranges: &[],
            });

        let render_pipeline = {
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Normal Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &render_pipeline_layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc(), InstanceRaw::desc()],
                shader,
            )
        };

        let light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Light Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout, &light_bind_group_layout],
                push_constant_ranges: &[],
            });
            let shader = wgpu::ShaderModuleDescriptor {
                label: Some("Light Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("light.wgsl").into()),
            };
            create_render_pipeline(
                &device,
                &layout,
                config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader,
            )
        };

        let debug_material = {
            let diffuse_bytes = include_bytes!("../res/cobble-diffuse.png");
            let normal_bytes = include_bytes!("../res/cobble-normal.png");

            let diffuse_texture = texture::Texture::from_bytes(
                &device,
                &queue,
                diffuse_bytes,
                "res/alt-diffuse.png",
                false,
            )
            .unwrap();
            let normal_texture = texture::Texture::from_bytes(
                &device,
                &queue,
                normal_bytes,
                "res/alt-normal.png",
                true,
            )
            .unwrap();

            model::Material::new(
                &device,
                "alt-material",
                diffuse_texture,
                normal_texture,
                &texture_bind_group_layout,
            )
        };

        let positions = vec![
            vec![ (2, 0, 3), (0, 1, 3), (2, 1, 3), (1, 2, 3), (2, 2, 3), (2, 0, 4), (0, 1, 4), (2, 1, 4), (1, 2, 4), (2, 2, 4), ],
            vec![ (1, 0, 3), (2, 1, 3), (3, 1, 3), (1, 2, 3), (2, 2, 3), (1, 0, 4), (2, 1, 4), (3, 1, 4), (1, 2, 4), (2, 2, 4), ],
            vec![ (2, 0, 3), (3, 1, 3), (1, 2, 3), (2, 2, 3), (3, 2, 3), (2, 0, 4), (3, 1, 4), (1, 2, 4), (2, 2, 4), (3, 2, 4), ],
            vec![ (1, 1, 3), (3, 1, 3), (2, 2, 3), (3, 2, 3), (2, 3, 3), (1, 1, 4), (3, 1, 4), (2, 2, 4), (3, 2, 4), (2, 3, 4), ],
            vec![ (3, 1, 3), (1, 2, 3), (3, 2, 3), (2, 3, 3), (3, 3, 3), (3, 1, 4), (1, 2, 4), (3, 2, 4), (2, 3, 4), (3, 3, 4), ],
            vec![ (2, 1, 3), (3, 2, 3), (4, 2, 3), (2, 3, 3), (3, 3, 3), (2, 1, 4), (3, 2, 4), (4, 2, 4), (2, 3, 4), (3, 3, 4), ],
            vec![ (3, 1, 3), (4, 2, 3), (2, 3, 3), (3, 3, 3), (4, 3, 3), (3, 1, 4), (4, 2, 4), (2, 3, 4), (3, 3, 4), (4, 3, 4), ],
            vec![ (2, 2, 3), (4, 2, 3), (3, 3, 3), (4, 3, 3), (3, 4, 3), (2, 2, 4), (4, 2, 4), (3, 3, 4), (4, 3, 4), (3, 4, 4), ],
            vec![ (4, 2, 3), (2, 3, 3), (4, 3, 3), (3, 4, 3), (4, 4, 3), (4, 2, 4), (2, 3, 4), (4, 3, 4), (3, 4, 4), (4, 4, 4), ],
            vec![ (3, 2, 3), (4, 3, 3), (5, 3, 3), (3, 4, 3), (4, 4, 3), (3, 2, 4), (4, 3, 4), (5, 3, 4), (3, 4, 4), (4, 4, 4), ],
            vec![ (4, 2, 3), (5, 3, 3), (3, 4, 3), (4, 4, 3), (5, 4, 3), (4, 2, 4), (5, 3, 4), (3, 4, 4), (4, 4, 4), (5, 4, 4), ],
            vec![ (3, 3, 3), (5, 3, 3), (4, 4, 3), (5, 4, 3), (4, 5, 3), (3, 3, 4), (5, 3, 4), (4, 4, 4), (5, 4, 4), (4, 5, 4), ],
            vec![ (5, 3, 3), (3, 4, 3), (5, 4, 3), (4, 5, 3), (5, 5, 3), (5, 3, 4), (3, 4, 4), (5, 4, 4), (4, 5, 4), (5, 5, 4), ],
            vec![ (4, 3, 3), (5, 4, 3), (6, 4, 3), (4, 5, 3), (5, 5, 3), (4, 3, 4), (5, 4, 4), (6, 4, 4), (4, 5, 4), (5, 5, 4), ],
            vec![ (5, 3, 3), (6, 4, 3), (4, 5, 3), (5, 5, 3), (6, 5, 3), (5, 3, 4), (6, 4, 4), (4, 5, 4), (5, 5, 4), (6, 5, 4), ],
            vec![ (4, 4, 3), (6, 4, 3), (5, 5, 3), (6, 5, 3), (5, 6, 3), (4, 4, 4), (6, 4, 4), (5, 5, 4), (6, 5, 4), (5, 6, 4), ],
            vec![ (6, 4, 3), (4, 5, 3), (6, 5, 3), (5, 6, 3), (6, 6, 3), (6, 4, 4), (4, 5, 4), (6, 5, 4), (5, 6, 4), (6, 6, 4), ],
            vec![ (5, 4, 3), (6, 5, 3), (7, 5, 3), (5, 6, 3), (6, 6, 3), (5, 4, 4), (6, 5, 4), (7, 5, 4), (5, 6, 4), (6, 6, 4), ],
            vec![ (6, 4, 3), (7, 5, 3), (5, 6, 3), (6, 6, 3), (7, 6, 3), (6, 4, 4), (7, 5, 4), (5, 6, 4), (6, 6, 4), (7, 6, 4), ],
            vec![ (5, 5, 3), (7, 5, 3), (6, 6, 3), (7, 6, 3), (6, 7, 3), (5, 5, 4), (7, 5, 4), (6, 6, 4), (7, 6, 4), (6, 7, 4), ],
            vec![ (7, 5, 3), (5, 6, 3), (7, 6, 3), (6, 7, 3), (7, 7, 3), (7, 5, 4), (5, 6, 4), (7, 6, 4), (6, 7, 4), (7, 7, 4), ],
            vec![ (6, 5, 3), (7, 6, 3), (8, 6, 3), (6, 7, 3), (7, 7, 3), (6, 5, 4), (7, 6, 4), (8, 6, 4), (6, 7, 4), (7, 7, 4), ],
            vec![ (7, 5, 3), (8, 6, 3), (6, 7, 3), (7, 7, 3), (8, 7, 3), (7, 5, 4), (8, 6, 4), (6, 7, 4), (7, 7, 4), (8, 7, 4), ],
            vec![ (6, 6, 3), (8, 6, 3), (7, 7, 3), (8, 7, 3), (7, 8, 3), (6, 6, 4), (8, 6, 4), (7, 7, 4), (8, 7, 4), (7, 8, 4), ],
            vec![ (8, 6, 3), (6, 7, 3), (8, 7, 3), (7, 8, 3), (8, 8, 3), (8, 6, 4), (6, 7, 4), (8, 7, 4), (7, 8, 4), (8, 8, 4), ],
            vec![ (7, 6, 3), (8, 7, 3), (9, 7, 3), (7, 8, 3), (8, 8, 3), (7, 6, 4), (8, 7, 4), (9, 7, 4), (7, 8, 4), (8, 8, 4), ],
            vec![ (8, 6, 3), (9, 7, 3), (7, 8, 3), (8, 8, 3), (9, 8, 3), (8, 6, 4), (9, 7, 4), (7, 8, 4), (8, 8, 4), (9, 8, 4), ],
            vec![ (7, 7, 3), (9, 7, 3), (8, 8, 3), (9, 8, 3), (8, 9, 3), (7, 7, 4), (9, 7, 4), (8, 8, 4), (9, 8, 4), (8, 9, 4), ],
            vec![ (9, 7, 3), (7, 8, 3), (9, 8, 3), (8, 9, 3), (9, 9, 3), (9, 7, 4), (7, 8, 4), (9, 8, 4), (8, 9, 4), (9, 9, 4), ],
            vec![ (8, 7, 3), (9, 8, 3), (10, 8, 3), (8, 9, 3), (9, 9, 3), (8, 7, 4), (9, 8, 4), (10, 8, 4), (8, 9, 4), (9, 9, 4), ],
            vec![ (9, 7, 3), (10, 8, 3), (8, 9, 3), (9, 9, 3), (10, 9, 3), (9, 7, 4), (10, 8, 4), (8, 9, 4), (9, 9, 4), (10, 9, 4), ],
            vec![ (8, 8, 3), (10, 8, 3), (9, 9, 3), (10, 9, 3), (9, 10, 3), (8, 8, 4), (10, 8, 4), (9, 9, 4), (10, 9, 4), (9, 10, 4), ],
            vec![ (10, 8, 3), (8, 9, 3), (10, 9, 3), (9, 10, 3), (10, 10, 3), (10, 8, 4), (8, 9, 4), (10, 9, 4), (9, 10, 4), (10, 10, 4), ],
            vec![ (9, 8, 3), (10, 9, 3), (11, 9, 3), (9, 10, 3), (10, 10, 3), (9, 8, 4), (10, 9, 4), (11, 9, 4), (9, 10, 4), (10, 10, 4), ],
            vec![ (10, 8, 3), (11, 9, 3), (9, 10, 3), (10, 10, 3), (11, 10, 3), (10, 8, 4), (11, 9, 4), (9, 10, 4), (10, 10, 4), (11, 10, 4), ],
            vec![ (9, 9, 3), (11, 9, 3), (10, 10, 3), (11, 10, 3), (10, 11, 3), (9, 9, 4), (11, 9, 4), (10, 10, 4), (11, 10, 4), (10, 11, 4), ],
            vec![ (11, 9, 3), (9, 10, 3), (11, 10, 3), (10, 11, 3), (11, 11, 3), (11, 9, 4), (9, 10, 4), (11, 10, 4), (10, 11, 4), (11, 11, 4), ],
            vec![ (10, 9, 3), (11, 10, 3), (12, 10, 3), (10, 11, 3), (11, 11, 3), (10, 9, 4), (11, 10, 4), (12, 10, 4), (10, 11, 4), (11, 11, 4), ],
            vec![ (11, 9, 3), (12, 10, 3), (10, 11, 3), (11, 11, 3), (12, 11, 3), (11, 9, 4), (12, 10, 4), (10, 11, 4), (11, 11, 4), (12, 11, 4), ],
            vec![ (10, 10, 3), (12, 10, 3), (11, 11, 3), (12, 11, 3), (11, 12, 3), (10, 10, 4), (12, 10, 4), (11, 11, 4), (12, 11, 4), (11, 12, 4), ],
            vec![ (12, 10, 3), (10, 11, 3), (12, 11, 3), (11, 12, 3), (12, 12, 3), (12, 10, 4), (10, 11, 4), (12, 11, 4), (11, 12, 4), (12, 12, 4), ],
            vec![ (11, 10, 3), (12, 11, 3), (13, 11, 3), (11, 12, 3), (12, 12, 3), (11, 10, 4), (12, 11, 4), (13, 11, 4), (11, 12, 4), (12, 12, 4), ],
            vec![ (12, 10, 3), (13, 11, 3), (11, 12, 3), (12, 12, 3), (13, 12, 3), (12, 10, 4), (13, 11, 4), (11, 12, 4), (12, 12, 4), (13, 12, 4), ],
            vec![ (11, 11, 3), (13, 11, 3), (12, 12, 3), (13, 12, 3), (12, 13, 3), (11, 11, 4), (13, 11, 4), (12, 12, 4), (13, 12, 4), (12, 13, 4), ],
            vec![ (13, 11, 3), (11, 12, 3), (13, 12, 3), (12, 13, 3), (13, 13, 3), (13, 11, 4), (11, 12, 4), (13, 12, 4), (12, 13, 4), (13, 13, 4), ],
            vec![ (12, 11, 3), (13, 12, 3), (14, 12, 3), (12, 13, 3), (13, 13, 3), (12, 11, 4), (13, 12, 4), (14, 12, 4), (12, 13, 4), (13, 13, 4), ],
            vec![ (13, 11, 3), (14, 12, 3), (12, 13, 3), (13, 13, 3), (14, 13, 3), (13, 11, 4), (14, 12, 4), (12, 13, 4), (13, 13, 4), (14, 13, 4), ],
            vec![ (12, 12, 3), (14, 12, 3), (13, 13, 3), (14, 13, 3), (13, 14, 3), (12, 12, 4), (14, 12, 4), (13, 13, 4), (14, 13, 4), (13, 14, 4), ],
            vec![ (14, 12, 3), (12, 13, 3), (14, 13, 3), (13, 14, 3), (14, 14, 3), (14, 12, 4), (12, 13, 4), (14, 13, 4), (13, 14, 4), (14, 14, 4), ],
            vec![ (13, 12, 3), (14, 13, 3), (15, 13, 3), (13, 14, 3), (14, 14, 3), (13, 12, 4), (14, 13, 4), (15, 13, 4), (13, 14, 4), (14, 14, 4), ],
            // vec![ (5, 3, 2), (5, 5, 2), (5, 2, 3), (4, 3, 3), (5, 3, 3), (6, 3, 3), (4, 5, 3), (5, 5, 3), (6, 5, 3), (5, 6, 3), (5, 3, 4), (5, 5, 4), ],
            // vec![ (4, 4, 1), (5, 4, 1), (6, 4, 1), (5, 2, 2), (5, 3, 2), (3, 4, 2), (7, 4, 2), (5, 5, 2), (5, 6, 2), (4, 2, 3), (6, 2, 3), (4, 3, 3), (6, 3, 3), (3, 4, 3), (7, 4, 3), (4, 5, 3), (6, 5, 3), (4, 6, 3), (6, 6, 3), (5, 2, 4), (5, 3, 4), (3, 4, 4), (7, 4, 4), (5, 5, 4), (5, 6, 4), (4, 4, 5), (5, 4, 5), (6, 4, 5), ],
            // vec![ (4, 3, 0), (6, 3, 0), (4, 4, 0), (6, 4, 0), (4, 5, 0), (6, 5, 0), (4, 2, 1), (5, 2, 1), (6, 2, 1), (3, 3, 1), (4, 3, 1), (5, 3, 1), (6, 3, 1), (7, 3, 1), (3, 4, 1), (4, 4, 1), (5, 4, 1), (6, 4, 1), (7, 4, 1), (3, 5, 1), (4, 5, 1), (5, 5, 1), (6, 5, 1), (7, 5, 1), (4, 6, 1), (5, 6, 1), (6, 6, 1), (4, 1, 2), (6, 1, 2), (3, 2, 2), (7, 2, 2), (2, 3, 2), (3, 3, 2), (5, 3, 2), (7, 3, 2), (8, 3, 2), (2, 4, 2), (3, 4, 2), (7, 4, 2), (8, 4, 2), (2, 5, 2), (3, 5, 2), (5, 5, 2), (7, 5, 2), (8, 5, 2), (3, 6, 2), (7, 6, 2), (4, 7, 2), (6, 7, 2), (3, 2, 3), (7, 2, 3), (3, 3, 3), (4, 3, 3), (6, 3, 3), (7, 3, 3), (3, 4, 3), (7, 4, 3), (3, 5, 3), (4, 5, 3), (6, 5, 3), (7, 5, 3), (3, 6, 3), (7, 6, 3), (4, 1, 4), (6, 1, 4), (3, 2, 4), (7, 2, 4), (2, 3, 4), (3, 3, 4), (5, 3, 4), (7, 3, 4), (8, 3, 4), (2, 4, 4), (3, 4, 4), (7, 4, 4), (8, 4, 4), (2, 5, 4), (3, 5, 4), (5, 5, 4), (7, 5, 4), (8, 5, 4), (3, 6, 4), (7, 6, 4), (4, 7, 4), (6, 7, 4), (4, 2, 5), (5, 2, 5), (6, 2, 5), (3, 3, 5), (4, 3, 5), (5, 3, 5), (6, 3, 5), (7, 3, 5), (3, 4, 5), (4, 4, 5), (5, 4, 5), (6, 4, 5), (7, 4, 5), (3, 5, 5), (4, 5, 5), (5, 5, 5), (6, 5, 5), (7, 5, 5), (4, 6, 5), (5, 6, 5), (6, 6, 5), (4, 3, 6), (6, 3, 6), (4, 4, 6), (6, 4, 6), (4, 5, 6), (6, 5, 6), ],
            // vec![ (4, 1, 0), (6, 1, 0), (2, 3, 0), (8, 3, 0), (2, 5, 0), (8, 5, 0), (4, 7, 0), (6, 7, 0), (5, 0, 1), (5, 1, 1), (1, 3, 1), (9, 3, 1), (1, 5, 1), (9, 5, 1), (5, 7, 1), (5, 8, 1), (5, 0, 2), (2, 1, 2), (4, 1, 2), (5, 1, 2), (6, 1, 2), (8, 1, 2), (1, 3, 2), (9, 3, 2), (1, 5, 2), (9, 5, 2), (2, 7, 2), (4, 7, 2), (5, 7, 2), (6, 7, 2), (8, 7, 2), (5, 8, 2), (3, 0, 3), (4, 0, 3), (6, 0, 3), (7, 0, 3), (3, 1, 3), (4, 1, 3), (6, 1, 3), (7, 1, 3), (1, 2, 3), (9, 2, 3), (1, 6, 3), (9, 6, 3), (3, 7, 3), (4, 7, 3), (6, 7, 3), (7, 7, 3), (3, 8, 3), (4, 8, 3), (6, 8, 3), (7, 8, 3), (5, 0, 4), (2, 1, 4), (4, 1, 4), (5, 1, 4), (6, 1, 4), (8, 1, 4), (1, 3, 4), (9, 3, 4), (1, 5, 4), (9, 5, 4), (2, 7, 4), (4, 7, 4), (5, 7, 4), (6, 7, 4), (8, 7, 4), (5, 8, 4), (5, 0, 5), (5, 1, 5), (1, 3, 5), (9, 3, 5), (1, 5, 5), (9, 5, 5), (5, 7, 5), (5, 8, 5), (4, 1, 6), (6, 1, 6), (2, 3, 6), (8, 3, 6), (2, 5, 6), (8, 5, 6), (4, 7, 6), (6, 7, 6), (5, 2, 7), (3, 3, 7), (4, 3, 7), (6, 3, 7), (7, 3, 7), (3, 5, 7), (4, 5, 7), (6, 5, 7), (7, 5, 7), (5, 6, 7), ],
            // vec![ (1, 2, 0), (2, 2, 0), (3, 2, 0), (4, 2, 0), (6, 2, 0), (7, 2, 0), (8, 2, 0), (9, 2, 0), (1, 3, 0), (9, 3, 0), (0, 4, 0), (3, 4, 0), (7, 4, 0), (10, 4, 0), (1, 5, 0), (9, 5, 0), (1, 6, 0), (2, 6, 0), (3, 6, 0), (4, 6, 0), (6, 6, 0), (7, 6, 0), (8, 6, 0), (9, 6, 0), (0, 2, 1), (10, 2, 1), (0, 3, 1), (10, 3, 1), (3, 4, 1), (7, 4, 1), (0, 5, 1), (10, 5, 1), (0, 6, 1), (10, 6, 1), (4, 9, 1), (5, 9, 1), (6, 9, 1), (1, 1, 2), (4, 1, 2), (6, 1, 2), (9, 1, 2), (2, 2, 2), (4, 2, 2), (6, 2, 2), (8, 2, 2), (2, 6, 2), (4, 6, 2), (6, 6, 2), (8, 6, 2), (1, 7, 2), (4, 7, 2), (6, 7, 2), (9, 7, 2), (3, 9, 2), (7, 9, 2), (1, 0, 3), (9, 0, 3), (2, 1, 3), (8, 1, 3), (1, 2, 3), (9, 2, 3), (1, 6, 3), (9, 6, 3), (2, 7, 3), (8, 7, 3), (1, 8, 3), (9, 8, 3), (3, 9, 3), (7, 9, 3), (1, 1, 4), (4, 1, 4), (6, 1, 4), (9, 1, 4), (2, 2, 4), (4, 2, 4), (6, 2, 4), (8, 2, 4), (2, 6, 4), (4, 6, 4), (6, 6, 4), (8, 6, 4), (1, 7, 4), (4, 7, 4), (6, 7, 4), (9, 7, 4), (3, 9, 4), (7, 9, 4), (0, 2, 5), (10, 2, 5), (0, 3, 5), (10, 3, 5), (3, 4, 5), (7, 4, 5), (0, 5, 5), (10, 5, 5), (0, 6, 5), (10, 6, 5), (4, 9, 5), (5, 9, 5), (6, 9, 5), (5, 1, 6), (1, 2, 6), (4, 2, 6), (6, 2, 6), (9, 2, 6), (1, 3, 6), (9, 3, 6), (0, 4, 6), (10, 4, 6), (1, 5, 6), (9, 5, 6), (1, 6, 6), (4, 6, 6), (6, 6, 6), (9, 6, 6), (5, 7, 6), (5, 0, 7), (4, 1, 7), (6, 1, 7), (2, 2, 7), (5, 2, 7), (8, 2, 7), (2, 3, 7), (8, 3, 7), (1, 4, 7), (9, 4, 7), (2, 5, 7), (8, 5, 7), (2, 6, 7), (5, 6, 7), (8, 6, 7), (4, 7, 7), (6, 7, 7), (5, 8, 7), (3, 2, 8), (7, 2, 8), (3, 3, 8), (7, 3, 8), (2, 4, 8), (8, 4, 8), (3, 5, 8), (7, 5, 8), (3, 6, 8), (7, 6, 8), ],
            // vec![ (0, 1, 0), (4, 1, 0), (5, 1, 0), (6, 1, 0), (10, 1, 0), (1, 2, 0), (5, 2, 0), (9, 2, 0), (11, 2, 0), (0, 3, 0), (3, 3, 0), (5, 3, 0), (7, 3, 0), (10, 3, 0), (0, 4, 0), (1, 4, 0), (4, 4, 0), (6, 4, 0), (9, 4, 0), (10, 4, 0), (0, 5, 0), (3, 5, 0), (5, 5, 0), (7, 5, 0), (10, 5, 0), (1, 6, 0), (5, 6, 0), (9, 6, 0), (11, 6, 0), (0, 7, 0), (4, 7, 0), (5, 7, 0), (6, 7, 0), (10, 7, 0), (4, 8, 0), (6, 8, 0), (4, 9, 0), (6, 9, 0), (4, 10, 0), (6, 10, 0), (5, 0, 1), (1, 1, 1), (2, 1, 1), (8, 1, 1), (9, 1, 1), (0, 2, 1), (10, 2, 1), (11, 2, 1), (0, 3, 1), (4, 3, 1), (6, 3, 1), (10, 3, 1), (0, 4, 1), (1, 4, 1), (4, 4, 1), (6, 4, 1), (9, 4, 1), (10, 4, 1), (0, 5, 1), (4, 5, 1), (6, 5, 1), (10, 5, 1), (0, 6, 1), (10, 6, 1), (11, 6, 1), (1, 7, 1), (2, 7, 1), (8, 7, 1), (9, 7, 1), (2, 8, 1), (5, 8, 1), (8, 8, 1), (3, 9, 1), (7, 9, 1), (3, 10, 1), (7, 10, 1), (0, 0, 2), (3, 0, 2), (5, 0, 2), (7, 0, 2), (10, 0, 2), (2, 1, 2), (8, 1, 2), (11, 2, 2), (4, 3, 2), (5, 3, 2), (6, 3, 2), (11, 3, 2), (0, 4, 2), (1, 4, 2), (9, 4, 2), (10, 4, 2), (11, 4, 2), (4, 5, 2), (5, 5, 2), (6, 5, 2), (11, 5, 2), (11, 6, 2), (2, 7, 2), (8, 7, 2), (0, 8, 2), (2, 8, 2), (3, 8, 2), (4, 8, 2), (5, 8, 2), (6, 8, 2), (7, 8, 2), (8, 8, 2), (10, 8, 2), (2, 10, 2), (8, 10, 2), (4, 0, 3), (6, 0, 3), (4, 3, 3), (6, 3, 3), (4, 5, 3), (6, 5, 3), (4, 8, 3), (6, 8, 3), (0, 0, 4), (3, 0, 4), (5, 0, 4), (7, 0, 4), (10, 0, 4), (2, 1, 4), (8, 1, 4), (11, 2, 4), (4, 3, 4), (5, 3, 4), (6, 3, 4), (11, 3, 4), (0, 4, 4), (1, 4, 4), (9, 4, 4), (10, 4, 4), (11, 4, 4), (4, 5, 4), (5, 5, 4), (6, 5, 4), (11, 5, 4), (11, 6, 4), (2, 7, 4), (8, 7, 4), (0, 8, 4), (2, 8, 4), (3, 8, 4), (4, 8, 4), (5, 8, 4), (6, 8, 4), (7, 8, 4), (8, 8, 4), (10, 8, 4), (2, 10, 4), (8, 10, 4), (4, 0, 5), (6, 0, 5), (0, 2, 5), (10, 2, 5), (11, 2, 5), (0, 3, 5), (10, 3, 5), (0, 4, 5), (1, 4, 5), (9, 4, 5), (10, 4, 5), (0, 5, 5), (10, 5, 5), (0, 6, 5), (10, 6, 5), (11, 6, 5), (2, 8, 5), (4, 8, 5), (6, 8, 5), (8, 8, 5), (3, 9, 5), (7, 9, 5), (3, 10, 5), (7, 10, 5), (0, 1, 6), (2, 1, 6), (4, 1, 6), (6, 1, 6), (8, 1, 6), (10, 1, 6), (11, 2, 6), (11, 6, 6), (0, 7, 6), (2, 7, 6), (4, 7, 6), (6, 7, 6), (8, 7, 6), (10, 7, 6), (3, 8, 6), (4, 8, 6), (6, 8, 6), (7, 8, 6), (4, 10, 6), (6, 10, 6), (1, 1, 7), (9, 1, 7), (0, 2, 7), (10, 2, 7), (3, 4, 7), (4, 4, 7), (6, 4, 7), (7, 4, 7), (0, 6, 7), (10, 6, 7), (1, 7, 7), (9, 7, 7), (4, 0, 8), (6, 0, 8), (2, 1, 8), (8, 1, 8), (1, 2, 8), (3, 2, 8), (7, 2, 8), (9, 2, 8), (3, 3, 8), (7, 3, 8), (3, 4, 8), (4, 4, 8), (6, 4, 8), (7, 4, 8), (3, 5, 8), (7, 5, 8), (1, 6, 8), (3, 6, 8), (7, 6, 8), (9, 6, 8), (2, 7, 8), (8, 7, 8), (4, 8, 8), (6, 8, 8), (2, 2, 9), (3, 2, 9), (4, 2, 9), (6, 2, 9), (7, 2, 9), (8, 2, 9), (4, 3, 9), (6, 3, 9), (4, 4, 9), (6, 4, 9), (4, 5, 9), (6, 5, 9), (2, 6, 9), (3, 6, 9), (4, 6, 9), (6, 6, 9), (7, 6, 9), (8, 6, 9), ],
            // vec![ (0, 0, 0), (2, 0, 0), (3, 0, 0), (7, 0, 0), (8, 0, 0), (10, 0, 0), (1, 1, 0), (3, 1, 0), (5, 1, 0), (7, 1, 0), (9, 1, 0), (12, 1, 0), (12, 2, 0), (12, 3, 0), (12, 5, 0), (12, 6, 0), (4, 7, 0), (6, 7, 0), (12, 7, 0), (0, 8, 0), (10, 8, 0), (2, 10, 0), (8, 10, 0), (3, 11, 0), (4, 11, 0), (5, 11, 0), (6, 11, 0), (7, 11, 0), (1, 0, 1), (4, 0, 1), (5, 0, 1), (6, 0, 1), (9, 0, 1), (11, 0, 1), (0, 1, 1), (2, 1, 1), (8, 1, 1), (2, 2, 1), (8, 2, 1), (2, 3, 1), (3, 3, 1), (7, 3, 1), (8, 3, 1), (12, 3, 1), (2, 4, 1), (8, 4, 1), (2, 5, 1), (3, 5, 1), (7, 5, 1), (8, 5, 1), (12, 5, 1), (2, 6, 1), (8, 6, 1), (0, 7, 1), (1, 7, 1), (9, 7, 1), (11, 8, 1), (3, 10, 1), (7, 10, 1), (2, 11, 1), (4, 11, 1), (5, 11, 1), (6, 11, 1), (8, 11, 1), (1, 1, 2), (9, 1, 2), (10, 1, 2), (12, 1, 2), (1, 2, 2), (3, 2, 2), (7, 2, 2), (9, 2, 2), (11, 2, 2), (2, 3, 2), (4, 3, 2), (5, 3, 2), (6, 3, 2), (8, 3, 2), (2, 4, 2), (8, 4, 2), (10, 4, 2), (2, 5, 2), (4, 5, 2), (5, 5, 2), (6, 5, 2), (8, 5, 2), (1, 6, 2), (3, 6, 2), (7, 6, 2), (9, 6, 2), (11, 6, 2), (4, 7, 2), (6, 7, 2), (10, 7, 2), (12, 7, 2), (4, 10, 2), (6, 10, 2), (2, 11, 2), (3, 11, 2), (7, 11, 2), (8, 11, 2), (0, 0, 3), (3, 0, 3), (4, 0, 3), (6, 0, 3), (7, 0, 3), (10, 0, 3), (11, 0, 3), (0, 1, 3), (3, 1, 3), (4, 1, 3), (6, 1, 3), (7, 1, 3), (12, 1, 3), (1, 2, 3), (2, 2, 3), (3, 2, 3), (4, 2, 3), (6, 2, 3), (7, 2, 3), (8, 2, 3), (9, 2, 3), (2, 3, 3), (4, 3, 3), (6, 3, 3), (8, 3, 3), (2, 4, 3), (8, 4, 3), (2, 5, 3), (4, 5, 3), (6, 5, 3), (8, 5, 3), (1, 6, 3), (2, 6, 3), (3, 6, 3), (4, 6, 3), (6, 6, 3), (7, 6, 3), (8, 6, 3), (9, 6, 3), (0, 7, 3), (12, 7, 3), (0, 8, 3), (10, 8, 3), (11, 8, 3), (0, 9, 3), (10, 9, 3), (11, 9, 3), (1, 10, 3), (2, 10, 3), (3, 10, 3), (7, 10, 3), (8, 10, 3), (9, 10, 3), (1, 11, 3), (2, 11, 3), (3, 11, 3), (7, 11, 3), (8, 11, 3), (9, 11, 3), (1, 0, 4), (2, 0, 4), (5, 0, 4), (8, 0, 4), (9, 0, 4), (0, 1, 4), (5, 1, 4), (12, 1, 4), (0, 2, 4), (5, 2, 4), (10, 2, 4), (11, 2, 4), (2, 3, 4), (3, 3, 4), (5, 3, 4), (7, 3, 4), (8, 3, 4), (2, 4, 4), (8, 4, 4), (10, 4, 4), (2, 5, 4), (3, 5, 4), (5, 5, 4), (7, 5, 4), (8, 5, 4), (0, 6, 4), (5, 6, 4), (10, 6, 4), (11, 6, 4), (0, 7, 4), (1, 7, 4), (4, 7, 4), (6, 7, 4), (9, 7, 4), (12, 7, 4), (2, 8, 4), (8, 8, 4), (4, 10, 4), (6, 10, 4), (2, 11, 4), (3, 11, 4), (7, 11, 4), (8, 11, 4), (0, 0, 5), (3, 0, 5), (5, 0, 5), (7, 0, 5), (10, 0, 5), (11, 0, 5), (1, 1, 5), (3, 1, 5), (5, 1, 5), (7, 1, 5), (9, 1, 5), (1, 2, 5), (2, 2, 5), (5, 2, 5), (8, 2, 5), (9, 2, 5), (2, 3, 5), (4, 3, 5), (6, 3, 5), (8, 3, 5), (12, 3, 5), (2, 4, 5), (3, 4, 5), (7, 4, 5), (8, 4, 5), (10, 4, 5), (2, 5, 5), (4, 5, 5), (6, 5, 5), (8, 5, 5), (12, 5, 5), (1, 6, 5), (2, 6, 5), (5, 6, 5), (8, 6, 5), (9, 6, 5), (0, 8, 5), (10, 8, 5), (11, 8, 5), (5, 10, 5), (2, 11, 5), (4, 11, 5), (5, 11, 5), (6, 11, 5), (8, 11, 5), (0, 0, 6), (2, 0, 6), (4, 0, 6), (6, 0, 6), (8, 0, 6), (10, 0, 6), (1, 1, 6), (9, 1, 6), (11, 1, 6), (12, 1, 6), (0, 2, 6), (2, 2, 6), (3, 2, 6), (5, 2, 6), (7, 2, 6), (8, 2, 6), (12, 2, 6), (0, 3, 6), (1, 3, 6), (2, 3, 6), (3, 3, 6), (4, 3, 6), (5, 3, 6), (6, 3, 6), (7, 3, 6), (8, 3, 6), (9, 3, 6), (12, 3, 6), (2, 4, 6), (3, 4, 6), (4, 4, 6), (5, 4, 6), (6, 4, 6), (7, 4, 6), (8, 4, 6), (0, 5, 6), (1, 5, 6), (2, 5, 6), (3, 5, 6), (4, 5, 6), (5, 5, 6), (6, 5, 6), (7, 5, 6), (8, 5, 6), (9, 5, 6), (12, 5, 6), (0, 6, 6), (2, 6, 6), (3, 6, 6), (5, 6, 6), (7, 6, 6), (8, 6, 6), (12, 6, 6), (11, 7, 6), (12, 7, 6), (0, 8, 6), (2, 8, 6), (4, 8, 6), (6, 8, 6), (8, 8, 6), (10, 8, 6), (2, 10, 6), (5, 10, 6), (8, 10, 6), (3, 11, 6), (4, 11, 6), (5, 11, 6), (6, 11, 6), (7, 11, 6), (0, 0, 7), (4, 0, 7), (6, 0, 7), (10, 0, 7), (2, 1, 7), (3, 1, 7), (7, 1, 7), (8, 1, 7), (10, 1, 7), (3, 2, 7), (5, 2, 7), (7, 2, 7), (10, 2, 7), (0, 3, 7), (1, 3, 7), (2, 3, 7), (8, 3, 7), (9, 3, 7), (11, 3, 7), (0, 5, 7), (1, 5, 7), (2, 5, 7), (8, 5, 7), (9, 5, 7), (11, 5, 7), (3, 6, 7), (5, 6, 7), (7, 6, 7), (10, 6, 7), (4, 7, 7), (6, 7, 7), (10, 7, 7), (0, 8, 7), (10, 8, 7), (5, 10, 7), (5, 11, 7), (1, 0, 8), (2, 0, 8), (3, 0, 8), (5, 0, 8), (7, 0, 8), (8, 0, 8), (9, 0, 8), (1, 1, 8), (9, 1, 8), (1, 2, 8), (4, 2, 8), (6, 2, 8), (9, 2, 8), (0, 3, 8), (10, 3, 8), (3, 4, 8), (4, 4, 8), (6, 4, 8), (7, 4, 8), (0, 5, 8), (10, 5, 8), (1, 6, 8), (4, 6, 8), (6, 6, 8), (9, 6, 8), (1, 7, 8), (9, 7, 8), (1, 8, 8), (2, 8, 8), (3, 8, 8), (5, 8, 8), (7, 8, 8), (8, 8, 8), (9, 8, 8), (5, 9, 8), (3, 0, 9), (5, 0, 9), (7, 0, 9), (2, 1, 9), (8, 1, 9), (4, 2, 9), (6, 2, 9), (1, 3, 9), (9, 3, 9), (1, 5, 9), (9, 5, 9), (4, 6, 9), (6, 6, 9), (2, 7, 9), (8, 7, 9), (3, 8, 9), (5, 8, 9), (7, 8, 9), (5, 9, 9), (2, 1, 10), (4, 1, 10), (5, 1, 10), (6, 1, 10), (8, 1, 10), (2, 2, 10), (8, 2, 10), (2, 3, 10), (3, 3, 10), (7, 3, 10), (8, 3, 10), (2, 5, 10), (3, 5, 10), (7, 5, 10), (8, 5, 10), (2, 6, 10), (8, 6, 10), (2, 7, 10), (4, 7, 10), (5, 7, 10), (6, 7, 10), (8, 7, 10), ],
            // vec![ (3, 0, 0), (4, 0, 0), (6, 0, 0), (7, 0, 0), (9, 0, 0), (12, 0, 0), (4, 1, 0), (6, 1, 0), (13, 1, 0), (0, 2, 0), (1, 2, 0), (3, 2, 0), (7, 2, 0), (2, 4, 0), (3, 4, 0), (4, 4, 0), (6, 4, 0), (7, 4, 0), (8, 4, 0), (0, 6, 0), (4, 6, 0), (5, 6, 0), (6, 6, 0), (2, 7, 0), (3, 7, 0), (5, 7, 0), (7, 7, 0), (8, 7, 0), (13, 7, 0), (5, 8, 0), (9, 8, 0), (12, 8, 0), (1, 9, 0), (2, 9, 0), (3, 9, 0), (7, 9, 0), (8, 9, 0), (9, 9, 0), (10, 9, 0), (11, 9, 0), (1, 10, 0), (9, 10, 0), (1, 11, 0), (9, 11, 0), (2, 12, 0), (4, 12, 0), (6, 12, 0), (8, 12, 0), (0, 0, 1), (3, 0, 1), (4, 0, 1), (6, 0, 1), (7, 0, 1), (11, 0, 1), (13, 0, 1), (12, 1, 1), (10, 2, 1), (13, 2, 1), (11, 3, 1), (11, 4, 1), (11, 5, 1), (13, 6, 1), (1, 7, 1), (9, 7, 1), (12, 7, 1), (3, 8, 1), (4, 8, 1), (6, 8, 1), (7, 8, 1), (11, 8, 1), (13, 8, 1), (1, 9, 1), (2, 9, 1), (4, 9, 1), (5, 9, 1), (6, 9, 1), (8, 9, 1), (9, 9, 1), (10, 9, 1), (11, 9, 1), (3, 10, 1), (7, 10, 1), (1, 12, 1), (9, 12, 1), (0, 0, 2), (2, 0, 2), (8, 0, 2), (9, 0, 2), (13, 0, 2), (9, 1, 2), (10, 1, 2), (12, 1, 2), (13, 1, 2), (0, 2, 2), (10, 2, 2), (11, 2, 2), (0, 3, 2), (12, 3, 2), (12, 4, 2), (13, 4, 2), (0, 5, 2), (12, 5, 2), (0, 6, 2), (9, 6, 2), (10, 6, 2), (11, 6, 2), (12, 7, 2), (13, 7, 2), (0, 8, 2), (1, 8, 2), (5, 8, 2), (12, 8, 2), (13, 8, 2), (5, 9, 2), (11, 9, 2), (5, 10, 2), (11, 10, 2), (0, 11, 2), (5, 11, 2), (10, 11, 2), (0, 0, 3), (2, 0, 3), (3, 0, 3), (7, 0, 3), (8, 0, 3), (9, 0, 3), (10, 0, 3), (3, 1, 3), (7, 1, 3), (9, 2, 3), (12, 2, 3), (4, 3, 3), (6, 3, 3), (11, 3, 3), (12, 3, 3), (10, 4, 3), (11, 4, 3), (4, 5, 3), (6, 5, 3), (11, 5, 3), (12, 5, 3), (9, 6, 3), (12, 6, 3), (0, 8, 3), (2, 8, 3), (4, 8, 3), (6, 8, 3), (8, 8, 3), (9, 8, 3), (12, 8, 3), (1, 9, 3), (3, 9, 3), (7, 9, 3), (9, 9, 3), (10, 9, 3), (12, 9, 3), (11, 10, 3), (0, 11, 3), (10, 11, 3), (9, 0, 4), (13, 0, 4), (0, 1, 4), (12, 1, 4), (13, 1, 4), (10, 2, 4), (11, 2, 4), (5, 3, 4), (11, 3, 4), (12, 3, 4), (12, 4, 4), (13, 4, 4), (5, 5, 4), (11, 5, 4), (12, 5, 4), (10, 6, 4), (11, 6, 4), (4, 7, 4), (6, 7, 4), (9, 7, 4), (12, 7, 4), (13, 7, 4), (3, 8, 4), (5, 8, 4), (7, 8, 4), (9, 8, 4), (12, 8, 4), (13, 8, 4), (11, 10, 4), (0, 11, 4), (2, 11, 4), (3, 11, 4), (7, 11, 4), (8, 11, 4), (10, 11, 4), (3, 0, 5), (7, 0, 5), (13, 0, 5), (5, 1, 5), (13, 2, 5), (0, 3, 5), (0, 5, 5), (9, 6, 5), (13, 6, 5), (4, 8, 5), (6, 8, 5), (9, 8, 5), (10, 8, 5), (13, 8, 5), (0, 9, 5), (1, 9, 5), (3, 9, 5), (7, 9, 5), (9, 9, 5), (2, 10, 5), (8, 10, 5), (2, 11, 5), (4, 11, 5), (6, 11, 5), (8, 11, 5), (1, 12, 5), (9, 12, 5), (0, 0, 6), (11, 1, 6), (13, 1, 6), (5, 2, 6), (0, 7, 6), (11, 7, 6), (13, 7, 6), (9, 8, 6), (1, 9, 6), (2, 9, 6), (5, 9, 6), (8, 9, 6), (9, 9, 6), (1, 10, 6), (3, 10, 6), (7, 10, 6), (9, 10, 6), (1, 11, 6), (3, 11, 6), (4, 11, 6), (6, 11, 6), (7, 11, 6), (9, 11, 6), (2, 12, 6), (8, 12, 6), (0, 0, 7), (4, 0, 7), (5, 0, 7), (6, 0, 7), (11, 0, 7), (12, 0, 7), (10, 1, 7), (13, 1, 7), (12, 2, 7), (9, 3, 7), (11, 3, 7), (13, 3, 7), (13, 4, 7), (9, 5, 7), (11, 5, 7), (13, 5, 7), (3, 6, 7), (5, 6, 7), (7, 6, 7), (10, 6, 7), (12, 6, 7), (4, 7, 7), (6, 7, 7), (13, 7, 7), (2, 8, 7), (3, 8, 7), (4, 8, 7), (6, 8, 7), (7, 8, 7), (8, 8, 7), (11, 8, 7), (12, 8, 7), (2, 9, 7), (3, 9, 7), (7, 9, 7), (8, 9, 7), (11, 9, 7), (2, 10, 7), (8, 10, 7), (2, 11, 7), (8, 11, 7), (3, 12, 7), (7, 12, 7), (11, 0, 8), (9, 1, 8), (1, 2, 8), (4, 2, 8), (5, 2, 8), (6, 2, 8), (0, 3, 8), (5, 4, 8), (12, 4, 8), (0, 5, 8), (4, 6, 8), (6, 6, 8), (9, 6, 8), (1, 8, 8), (3, 8, 8), (7, 8, 8), (11, 8, 8), (0, 9, 8), (5, 9, 8), (10, 9, 8), (4, 11, 8), (5, 11, 8), (6, 11, 8), (0, 0, 9), (1, 0, 9), (5, 0, 9), (9, 0, 9), (10, 0, 9), (2, 1, 9), (8, 1, 9), (4, 2, 9), (6, 2, 9), (1, 3, 9), (4, 3, 9), (6, 3, 9), (9, 3, 9), (5, 4, 9), (11, 4, 9), (1, 5, 9), (4, 5, 9), (6, 5, 9), (9, 5, 9), (4, 6, 9), (6, 6, 9), (2, 7, 9), (8, 7, 9), (0, 8, 9), (1, 8, 9), (9, 8, 9), (10, 8, 9), (1, 9, 9), (9, 9, 9), (4, 10, 9), (5, 10, 9), (6, 10, 9), (1, 0, 10), (9, 0, 10), (4, 1, 10), (6, 1, 10), (1, 2, 10), (5, 2, 10), (9, 2, 10), (4, 3, 10), (5, 3, 10), (6, 3, 10), (0, 4, 10), (4, 4, 10), (6, 4, 10), (10, 4, 10), (4, 5, 10), (5, 5, 10), (6, 5, 10), (1, 6, 10), (5, 6, 10), (9, 6, 10), (4, 7, 10), (6, 7, 10), (1, 8, 10), (4, 8, 10), (5, 8, 10), (6, 8, 10), (9, 8, 10), (5, 9, 10), (3, 0, 11), (4, 0, 11), (6, 0, 11), (7, 0, 11), (1, 1, 11), (2, 1, 11), (4, 1, 11), (6, 1, 11), (8, 1, 11), (9, 1, 11), (3, 2, 11), (7, 2, 11), (1, 3, 11), (9, 3, 11), (1, 4, 11), (4, 4, 11), (6, 4, 11), (9, 4, 11), (1, 5, 11), (9, 5, 11), (3, 6, 11), (7, 6, 11), (1, 7, 11), (2, 7, 11), (4, 7, 11), (6, 7, 11), (8, 7, 11), (9, 7, 11), (3, 8, 11), (4, 8, 11), (6, 8, 11), (7, 8, 11), ],
            // vec![ (2, 0, 0), (3, 0, 0), (4, 0, 0), (6, 0, 0), (7, 0, 0), (10, 0, 0), (12, 0, 0), (14, 0, 0), (9, 1, 0), (13, 1, 0), (2, 2, 0), (4, 2, 0), (5, 2, 0), (6, 2, 0), (10, 2, 0), (14, 2, 0), (0, 3, 0), (5, 3, 0), (9, 3, 0), (5, 4, 0), (1, 5, 0), (2, 5, 0), (5, 5, 0), (8, 5, 0), (10, 5, 0), (11, 5, 0), (9, 6, 0), (10, 6, 0), (11, 6, 0), (14, 6, 0), (0, 7, 0), (3, 7, 0), (7, 7, 0), (8, 7, 0), (13, 7, 0), (9, 8, 0), (14, 8, 0), (1, 9, 0), (8, 9, 0), (10, 9, 0), (12, 9, 0), (13, 9, 0), (3, 10, 0), (7, 10, 0), (12, 10, 0), (2, 11, 0), (4, 11, 0), (5, 11, 0), (6, 11, 0), (8, 11, 0), (0, 12, 0), (3, 12, 0), (5, 12, 0), (7, 12, 0), (10, 12, 0), (1, 13, 0), (2, 13, 0), (3, 13, 0), (5, 13, 0), (7, 13, 0), (8, 13, 0), (9, 13, 0), (4, 0, 1), (6, 0, 1), (9, 0, 1), (10, 0, 1), (11, 0, 1), (0, 1, 1), (2, 1, 1), (14, 1, 1), (2, 2, 1), (3, 2, 1), (4, 2, 1), (5, 2, 1), (6, 2, 1), (7, 2, 1), (8, 2, 1), (1, 3, 1), (5, 3, 1), (10, 3, 1), (14, 3, 1), (0, 4, 1), (2, 4, 1), (4, 4, 1), (5, 4, 1), (6, 4, 1), (8, 4, 1), (2, 5, 1), (5, 5, 1), (10, 5, 1), (14, 5, 1), (1, 6, 1), (10, 6, 1), (0, 7, 1), (2, 7, 1), (3, 7, 1), (7, 7, 1), (9, 7, 1), (14, 7, 1), (0, 8, 1), (0, 9, 1), (2, 9, 1), (5, 9, 1), (8, 9, 1), (11, 9, 1), (14, 9, 1), (0, 10, 1), (3, 10, 1), (7, 10, 1), (1, 11, 1), (2, 11, 1), (8, 11, 1), (9, 11, 1), (10, 11, 1), (11, 11, 1), (3, 12, 1), (4, 12, 1), (6, 12, 1), (7, 12, 1), (1, 13, 1), (2, 13, 1), (3, 13, 1), (5, 13, 1), (7, 13, 1), (8, 13, 1), (9, 13, 1), (1, 0, 2), (2, 0, 2), (5, 0, 2), (11, 0, 2), (13, 0, 2), (2, 1, 2), (5, 1, 2), (12, 1, 2), (1, 2, 2), (3, 2, 2), (4, 2, 2), (5, 2, 2), (6, 2, 2), (7, 2, 2), (9, 2, 2), (14, 2, 2), (1, 3, 2), (5, 3, 2), (14, 3, 2), (0, 4, 2), (1, 4, 2), (3, 4, 2), (4, 4, 2), (6, 4, 2), (7, 4, 2), (1, 5, 2), (5, 5, 2), (8, 5, 2), (14, 5, 2), (5, 6, 2), (11, 6, 2), (14, 6, 2), (0, 7, 2), (5, 7, 2), (13, 7, 2), (5, 8, 2), (0, 9, 2), (13, 9, 2), (14, 9, 2), (1, 10, 2), (2, 10, 2), (8, 10, 2), (11, 10, 2), (12, 10, 2), (2, 11, 2), (8, 11, 2), (10, 11, 2), (12, 11, 2), (11, 12, 2), (1, 0, 3), (4, 0, 3), (6, 0, 3), (4, 1, 3), (6, 1, 3), (3, 2, 3), (7, 2, 3), (9, 2, 3), (14, 2, 3), (0, 3, 3), (1, 3, 3), (14, 3, 3), (0, 4, 3), (1, 4, 3), (3, 4, 3), (7, 4, 3), (14, 4, 3), (0, 5, 3), (1, 5, 3), (8, 5, 3), (9, 5, 3), (14, 5, 3), (0, 6, 3), (1, 6, 3), (3, 6, 3), (5, 6, 3), (7, 6, 3), (9, 6, 3), (14, 6, 3), (1, 7, 3), (4, 7, 3), (6, 7, 3), (1, 8, 3), (2, 8, 3), (3, 8, 3), (7, 8, 3), (1, 9, 3), (2, 9, 3), (9, 9, 3), (14, 9, 3), (1, 10, 3), (4, 10, 3), (6, 10, 3), (11, 10, 3), (12, 10, 3), (2, 11, 3), (3, 11, 3), (5, 11, 3), (7, 11, 3), (8, 11, 3), (2, 12, 3), (3, 12, 3), (4, 12, 3), (6, 12, 3), (7, 12, 3), (8, 12, 3), (0, 0, 4), (11, 0, 4), (11, 1, 4), (0, 2, 4), (1, 2, 4), (3, 2, 4), (7, 2, 4), (8, 2, 4), (9, 2, 4), (10, 2, 4), (14, 2, 4), (4, 3, 4), (6, 3, 4), (14, 3, 4), (0, 4, 4), (1, 4, 4), (3, 4, 4), (7, 4, 4), (4, 5, 4), (6, 5, 4), (8, 5, 4), (14, 5, 4), (3, 6, 4), (5, 6, 4), (7, 6, 4), (14, 6, 4), (1, 7, 4), (2, 7, 4), (3, 7, 4), (4, 7, 4), (6, 7, 4), (7, 7, 4), (9, 7, 4), (12, 7, 4), (1, 8, 4), (9, 8, 4), (5, 9, 4), (13, 9, 4), (14, 9, 4), (0, 10, 4), (5, 10, 4), (2, 11, 4), (3, 11, 4), (4, 11, 4), (5, 11, 4), (6, 11, 4), (7, 11, 4), (8, 11, 4), (10, 11, 4), (12, 11, 4), (1, 12, 4), (4, 12, 4), (5, 12, 4), (6, 12, 4), (9, 12, 4), (11, 12, 4), (0, 0, 5), (1, 0, 5), (4, 0, 5), (6, 0, 5), (8, 0, 5), (10, 0, 5), (11, 0, 5), (13, 0, 5), (0, 1, 5), (1, 1, 5), (8, 1, 5), (9, 1, 5), (14, 1, 5), (0, 2, 5), (1, 2, 5), (13, 2, 5), (4, 3, 5), (5, 3, 5), (6, 3, 5), (11, 3, 5), (14, 3, 5), (0, 4, 5), (1, 4, 5), (4, 4, 5), (5, 4, 5), (6, 4, 5), (10, 4, 5), (11, 4, 5), (9, 5, 5), (11, 5, 5), (14, 5, 5), (0, 6, 5), (1, 6, 5), (4, 6, 5), (6, 6, 5), (8, 6, 5), (13, 6, 5), (5, 7, 5), (14, 7, 5), (1, 8, 5), (2, 8, 5), (3, 8, 5), (7, 8, 5), (9, 9, 5), (14, 9, 5), (5, 10, 5), (11, 10, 5), (4, 11, 5), (6, 11, 5), (10, 11, 5), (11, 11, 5), (1, 13, 5), (2, 13, 5), (8, 13, 5), (9, 13, 5), (1, 0, 6), (3, 0, 6), (7, 0, 6), (0, 1, 6), (1, 1, 6), (3, 1, 6), (4, 1, 6), (5, 1, 6), (6, 1, 6), (7, 1, 6), (11, 1, 6), (4, 2, 6), (6, 2, 6), (9, 2, 6), (13, 2, 6), (10, 3, 6), (11, 3, 6), (12, 3, 6), (0, 4, 6), (1, 4, 6), (8, 4, 6), (9, 4, 6), (11, 4, 6), (12, 4, 6), (4, 5, 6), (6, 5, 6), (12, 5, 6), (0, 6, 6), (1, 6, 6), (3, 6, 6), (7, 6, 6), (10, 6, 6), (13, 6, 6), (1, 7, 6), (11, 8, 6), (1, 9, 6), (5, 9, 6), (13, 9, 6), (0, 10, 6), (5, 10, 6), (10, 10, 6), (4, 11, 6), (6, 11, 6), (0, 12, 6), (1, 12, 6), (9, 12, 6), (10, 12, 6), (1, 13, 6), (3, 13, 6), (7, 13, 6), (9, 13, 6), (1, 0, 7), (9, 0, 7), (10, 0, 7), (11, 0, 7), (14, 0, 7), (3, 1, 7), (4, 1, 7), (6, 1, 7), (7, 1, 7), (9, 1, 7), (10, 1, 7), (14, 1, 7), (0, 2, 7), (1, 2, 7), (8, 2, 7), (10, 2, 7), (0, 3, 7), (1, 3, 7), (5, 3, 7), (10, 3, 7), (12, 3, 7), (14, 3, 7), (0, 4, 7), (1, 4, 7), (8, 4, 7), (9, 4, 7), (3, 5, 7), (7, 5, 7), (12, 5, 7), (14, 5, 7), (0, 6, 7), (1, 6, 7), (5, 6, 7), (10, 6, 7), (0, 7, 7), (2, 7, 7), (14, 7, 7), (14, 8, 7), (3, 9, 7), (7, 9, 7), (11, 9, 7), (10, 10, 7), (11, 10, 7), (0, 11, 7), (1, 11, 7), (5, 11, 7), (9, 11, 7), (10, 11, 7), (2, 12, 7), (4, 12, 7), (5, 12, 7), (6, 12, 7), (8, 12, 7), (2, 13, 7), (3, 13, 7), (7, 13, 7), (8, 13, 7), (2, 0, 8), (3, 0, 8), (7, 0, 8), (9, 0, 8), (11, 0, 8), (13, 0, 8), (1, 1, 8), (9, 1, 8), (12, 1, 8), (9, 2, 8), (10, 2, 8), (14, 2, 8), (2, 3, 8), (8, 3, 8), (9, 3, 8), (14, 3, 8), (2, 4, 8), (3, 4, 8), (7, 4, 8), (1, 5, 8), (2, 5, 8), (11, 5, 8), (14, 5, 8), (0, 6, 8), (8, 6, 8), (9, 6, 8), (10, 6, 8), (14, 6, 8), (1, 7, 8), (12, 7, 8), (0, 8, 8), (5, 8, 8), (12, 8, 8), (13, 8, 8), (0, 9, 8), (0, 10, 8), (1, 10, 8), (2, 10, 8), (8, 10, 8), (9, 10, 8), (11, 10, 8), (1, 11, 8), (3, 11, 8), (4, 11, 8), (6, 11, 8), (7, 11, 8), (9, 11, 8), (2, 12, 8), (8, 12, 8), (3, 0, 9), (4, 0, 9), (6, 0, 9), (7, 0, 9), (9, 0, 9), (10, 0, 9), (11, 0, 9), (0, 1, 9), (2, 1, 9), (8, 1, 9), (11, 1, 9), (1, 2, 9), (0, 3, 9), (1, 3, 9), (3, 3, 9), (7, 3, 9), (8, 3, 9), (12, 3, 9), (0, 4, 9), (1, 4, 9), (2, 4, 9), (3, 4, 9), (7, 4, 9), (8, 4, 9), (12, 4, 9), (2, 5, 9), (3, 5, 9), (4, 5, 9), (6, 5, 9), (7, 5, 9), (10, 5, 9), (12, 5, 9), (0, 7, 9), (11, 7, 9), (8, 8, 9), (2, 9, 9), (9, 9, 9), (0, 10, 9), (1, 10, 9), (3, 10, 9), (7, 10, 9), (9, 10, 9), (10, 10, 9), (3, 11, 9), (7, 11, 9), (4, 12, 9), (6, 12, 9), (4, 0, 10), (6, 0, 10), (0, 1, 10), (10, 1, 10), (0, 3, 10), (1, 3, 10), (2, 3, 10), (4, 3, 10), (6, 3, 10), (8, 3, 10), (9, 3, 10), (11, 3, 10), (2, 4, 10), (4, 4, 10), (6, 4, 10), (8, 4, 10), (11, 4, 10), (0, 5, 10), (1, 5, 10), (2, 5, 10), (4, 5, 10), (6, 5, 10), (8, 5, 10), (9, 5, 10), (11, 5, 10), (0, 7, 10), (4, 7, 10), (6, 7, 10), (10, 7, 10), (0, 8, 10), (10, 8, 10), (5, 9, 10), (4, 11, 10), (6, 11, 10), (0, 0, 11), (3, 0, 11), (7, 0, 11), (10, 0, 11), (2, 2, 11), (3, 2, 11), (7, 2, 11), (8, 2, 11), (2, 6, 11), (3, 6, 11), (7, 6, 11), (8, 6, 11), (0, 8, 11), (10, 8, 11), (2, 9, 11), (4, 9, 11), (6, 9, 11), (8, 9, 11), (1, 0, 12), (9, 0, 12), (1, 1, 12), (3, 1, 12), (7, 1, 12), (9, 1, 12), (0, 2, 12), (4, 2, 12), (5, 2, 12), (6, 2, 12), (10, 2, 12), (0, 3, 12), (1, 3, 12), (3, 3, 12), (4, 3, 12), (5, 3, 12), (6, 3, 12), (7, 3, 12), (9, 3, 12), (10, 3, 12), (5, 4, 12), (0, 5, 12), (1, 5, 12), (3, 5, 12), (4, 5, 12), (5, 5, 12), (6, 5, 12), (7, 5, 12), (9, 5, 12), (10, 5, 12), (0, 6, 12), (4, 6, 12), (5, 6, 12), (6, 6, 12), (10, 6, 12), (1, 7, 12), (3, 7, 12), (7, 7, 12), (9, 7, 12), (1, 8, 12), (9, 8, 12), (3, 9, 12), (4, 9, 12), (5, 9, 12), (6, 9, 12), (7, 9, 12), ],
            // vec![ (3, 0, 0), (10, 0, 0), (15, 0, 0), (1, 1, 0), (8, 1, 0), (11, 1, 0), (13, 1, 0), (2, 2, 0), (9, 2, 0), (11, 2, 0), (2, 3, 0), (9, 3, 0), (11, 3, 0), (13, 3, 0), (14, 3, 0), (15, 3, 0), (2, 4, 0), (10, 4, 0), (13, 4, 0), (14, 4, 0), (15, 4, 0), (2, 5, 0), (4, 5, 0), (6, 5, 0), (12, 5, 0), (13, 5, 0), (14, 5, 0), (15, 5, 0), (3, 6, 0), (5, 6, 0), (1, 7, 0), (4, 7, 0), (6, 7, 0), (12, 7, 0), (0, 8, 0), (2, 8, 0), (7, 8, 0), (14, 8, 0), (1, 9, 0), (8, 9, 0), (13, 9, 0), (15, 9, 0), (4, 10, 0), (6, 10, 0), (10, 10, 0), (12, 10, 0), (14, 10, 0), (1, 11, 0), (9, 11, 0), (12, 11, 0), (1, 13, 0), (5, 13, 0), (9, 13, 0), (0, 14, 0), (5, 14, 0), (10, 14, 0), (0, 0, 1), (4, 0, 1), (6, 0, 1), (15, 0, 1), (0, 2, 1), (2, 2, 1), (3, 2, 1), (10, 2, 1), (12, 2, 1), (15, 2, 1), (11, 3, 1), (2, 4, 1), (8, 4, 1), (10, 4, 1), (8, 5, 1), (1, 6, 1), (3, 6, 1), (5, 6, 1), (7, 6, 1), (10, 6, 1), (12, 6, 1), (15, 6, 1), (0, 7, 1), (4, 7, 1), (6, 7, 1), (2, 8, 1), (4, 8, 1), (6, 8, 1), (7, 8, 1), (7, 9, 1), (9, 9, 1), (14, 9, 1), (4, 10, 1), (6, 10, 1), (7, 10, 1), (15, 10, 1), (8, 11, 1), (11, 11, 1), (1, 13, 1), (5, 13, 1), (9, 13, 1), (11, 13, 1), (0, 14, 1), (5, 14, 1), (10, 14, 1), (2, 0, 2), (4, 0, 2), (6, 0, 2), (9, 0, 2), (14, 0, 2), (9, 1, 2), (13, 1, 2), (1, 2, 2), (3, 2, 2), (11, 2, 2), (9, 3, 2), (4, 4, 2), (6, 4, 2), (11, 4, 2), (5, 5, 2), (10, 5, 2), (3, 6, 2), (7, 6, 2), (12, 6, 2), (8, 7, 2), (11, 7, 2), (12, 7, 2), (5, 8, 2), (8, 8, 2), (4, 9, 2), (6, 9, 2), (7, 9, 2), (8, 9, 2), (11, 9, 2), (4, 10, 2), (6, 10, 2), (8, 10, 2), (11, 10, 2), (0, 12, 2), (11, 12, 2), (5, 13, 2), (10, 13, 2), (1, 14, 2), (3, 14, 2), (4, 14, 2), (6, 14, 2), (7, 14, 2), (9, 14, 2), (1, 0, 3), (7, 0, 3), (12, 0, 3), (13, 0, 3), (4, 1, 3), (6, 1, 3), (9, 1, 3), (11, 1, 3), (12, 1, 3), (13, 1, 3), (9, 2, 3), (10, 2, 3), (12, 2, 3), (9, 3, 3), (7, 4, 3), (9, 5, 3), (1, 6, 3), (7, 6, 3), (11, 6, 3), (1, 7, 3), (11, 7, 3), (13, 7, 3), (8, 8, 3), (2, 9, 3), (3, 9, 3), (4, 9, 3), (5, 9, 3), (6, 9, 3), (14, 9, 3), (0, 10, 3), (10, 10, 3), (14, 10, 3), (8, 11, 3), (1, 12, 3), (4, 12, 3), (6, 12, 3), (10, 12, 3), (13, 12, 3), (1, 13, 3), (5, 13, 3), (9, 13, 3), (11, 13, 3), (12, 13, 3), (7, 0, 4), (10, 0, 4), (13, 0, 4), (14, 0, 4), (3, 1, 4), (4, 1, 4), (6, 1, 4), (11, 1, 4), (12, 1, 4), (13, 1, 4), (14, 1, 4), (3, 2, 4), (4, 3, 4), (6, 3, 4), (9, 3, 4), (9, 4, 4), (10, 6, 4), (11, 6, 4), (7, 7, 4), (13, 7, 4), (7, 8, 4), (12, 8, 4), (14, 8, 4), (5, 9, 4), (8, 9, 4), (10, 9, 4), (13, 9, 4), (13, 11, 4), (0, 12, 4), (11, 12, 4), (0, 13, 4), (2, 13, 4), (3, 13, 4), (5, 13, 4), (7, 13, 4), (8, 13, 4), (1, 14, 4), (2, 14, 4), (8, 14, 4), (9, 14, 4), (4, 0, 5), (5, 0, 5), (6, 0, 5), (8, 0, 5), (11, 0, 5), (12, 0, 5), (14, 0, 5), (8, 1, 5), (13, 1, 5), (14, 1, 5), (15, 1, 5), (11, 3, 5), (14, 3, 5), (8, 4, 5), (10, 4, 5), (14, 5, 5), (4, 6, 5), (6, 6, 5), (9, 6, 5), (12, 6, 5), (13, 7, 5), (15, 7, 5), (0, 8, 5), (2, 8, 5), (3, 8, 5), (5, 8, 5), (8, 8, 5), (11, 8, 5), (14, 8, 5), (0, 9, 5), (2, 9, 5), (3, 9, 5), (5, 9, 5), (10, 9, 5), (15, 9, 5), (3, 10, 5), (7, 10, 5), (9, 10, 5), (12, 10, 5), (13, 10, 5), (15, 10, 5), (0, 11, 5), (10, 11, 5), (0, 12, 5), (0, 13, 5), (11, 13, 5), (0, 14, 5), (3, 14, 5), (7, 14, 5), (10, 14, 5), (5, 1, 6), (4, 2, 6), (6, 2, 6), (9, 2, 6), (12, 3, 6), (15, 3, 6), (2, 4, 6), (3, 5, 6), (15, 5, 6), (1, 6, 6), (3, 6, 6), (7, 6, 6), (11, 6, 6), (12, 6, 6), (3, 7, 6), (4, 7, 6), (6, 7, 6), (13, 7, 6), (14, 7, 6), (5, 8, 6), (7, 8, 6), (13, 8, 6), (14, 8, 6), (2, 9, 6), (4, 9, 6), (6, 9, 6), (7, 9, 6), (12, 9, 6), (15, 9, 6), (12, 10, 6), (13, 10, 6), (14, 10, 6), (0, 11, 6), (1, 12, 6), (0, 14, 6), (4, 14, 6), (6, 14, 6), (10, 14, 6), (0, 0, 7), (5, 0, 7), (10, 0, 7), (12, 0, 7), (15, 0, 7), (14, 1, 7), (8, 2, 7), (14, 2, 7), (4, 5, 7), (6, 5, 7), (0, 6, 7), (3, 6, 7), (9, 6, 7), (10, 6, 7), (12, 6, 7), (14, 6, 7), (0, 7, 7), (3, 7, 7), (5, 7, 7), (7, 7, 7), (9, 7, 7), (10, 7, 7), (12, 7, 7), (14, 7, 7), (2, 8, 7), (3, 8, 7), (5, 8, 7), (10, 8, 7), (15, 8, 7), (3, 10, 7), (7, 10, 7), (10, 10, 7), (5, 11, 7), (12, 11, 7), (11, 12, 7), (1, 14, 7), (4, 14, 7), (6, 14, 7), (9, 14, 7), (1, 0, 8), (4, 0, 8), (6, 0, 8), (9, 0, 8), (15, 0, 8), (4, 1, 8), (6, 1, 8), (13, 1, 8), (3, 2, 8), (10, 2, 8), (10, 4, 8), (9, 6, 8), (12, 6, 8), (4, 7, 8), (5, 7, 8), (6, 7, 8), (7, 7, 8), (9, 7, 8), (11, 7, 8), (0, 8, 8), (4, 8, 8), (6, 8, 8), (7, 8, 8), (9, 8, 8), (12, 8, 8), (13, 8, 8), (15, 8, 8), (12, 9, 8), (14, 9, 8), (9, 10, 8), (3, 11, 8), (7, 11, 8), (11, 11, 8), (12, 11, 8), (5, 13, 8), (2, 14, 8), (3, 14, 8), (7, 14, 8), (8, 14, 8), (1, 0, 9), (2, 0, 9), (7, 0, 9), (12, 0, 9), (13, 0, 9), (7, 1, 9), (12, 1, 9), (14, 1, 9), (3, 2, 9), (4, 2, 9), (5, 2, 9), (6, 2, 9), (14, 2, 9), (15, 2, 9), (4, 3, 9), (6, 3, 9), (11, 3, 9), (14, 3, 9), (15, 3, 9), (0, 4, 9), (13, 4, 9), (14, 4, 9), (15, 4, 9), (12, 5, 9), (14, 5, 9), (15, 5, 9), (8, 6, 9), (12, 6, 9), (14, 6, 9), (15, 6, 9), (4, 7, 9), (6, 7, 9), (10, 7, 9), (14, 7, 9), (2, 8, 9), (3, 8, 9), (7, 8, 9), (11, 8, 9), (2, 9, 9), (5, 9, 9), (13, 9, 9), (5, 10, 9), (11, 10, 9), (0, 11, 9), (10, 11, 9), (11, 11, 9), (1, 12, 9), (4, 12, 9), (6, 12, 9), (9, 12, 9), (3, 13, 9), (5, 13, 9), (7, 13, 9), (3, 0, 10), (7, 0, 10), (9, 0, 10), (11, 0, 10), (12, 0, 10), (4, 1, 10), (6, 1, 10), (11, 1, 10), (12, 1, 10), (5, 2, 10), (4, 3, 10), (6, 3, 10), (8, 3, 10), (13, 3, 10), (1, 5, 10), (11, 5, 10), (13, 5, 10), (4, 6, 10), (6, 6, 10), (8, 6, 10), (9, 6, 10), (10, 6, 10), (11, 6, 10), (1, 7, 10), (2, 7, 10), (5, 7, 10), (9, 7, 10), (2, 8, 10), (5, 8, 10), (10, 8, 10), (8, 9, 10), (10, 9, 10), (0, 10, 10), (2, 10, 10), (4, 10, 10), (5, 10, 10), (6, 10, 10), (7, 10, 10), (8, 10, 10), (0, 11, 10), (1, 11, 10), (9, 11, 10), (10, 11, 10), (5, 13, 10), (5, 0, 11), (11, 0, 11), (0, 1, 11), (5, 1, 11), (10, 1, 11), (0, 3, 11), (12, 3, 11), (11, 4, 11), (0, 5, 11), (12, 5, 11), (3, 6, 11), (7, 6, 11), (2, 7, 11), (3, 7, 11), (4, 7, 11), (5, 7, 11), (6, 7, 11), (7, 7, 11), (8, 7, 11), (0, 8, 11), (2, 8, 11), (8, 8, 11), (10, 8, 11), (4, 9, 11), (6, 9, 11), (11, 9, 11), (2, 10, 11), (3, 10, 11), (7, 10, 11), (8, 10, 11), (5, 11, 11), (5, 12, 11), (3, 0, 12), (4, 0, 12), (6, 0, 12), (7, 0, 12), (1, 1, 12), (3, 1, 12), (4, 1, 12), (6, 1, 12), (7, 1, 12), (9, 1, 12), (11, 1, 12), (1, 2, 12), (9, 2, 12), (11, 2, 12), (3, 3, 12), (7, 3, 12), (11, 3, 12), (11, 4, 12), (3, 5, 12), (7, 5, 12), (11, 5, 12), (1, 6, 12), (9, 6, 12), (11, 6, 12), (1, 7, 12), (2, 7, 12), (8, 7, 12), (9, 7, 12), (11, 7, 12), (2, 8, 12), (3, 8, 12), (4, 8, 12), (5, 8, 12), (6, 8, 12), (7, 8, 12), (8, 8, 12), (0, 9, 12), (5, 9, 12), (10, 9, 12), (2, 10, 12), (5, 10, 12), (8, 10, 12), (0, 0, 13), (1, 0, 13), (9, 0, 13), (10, 0, 13), (3, 1, 13), (7, 1, 13), (11, 2, 13), (2, 3, 13), (8, 3, 13), (11, 3, 13), (11, 4, 13), (2, 5, 13), (8, 5, 13), (11, 5, 13), (11, 6, 13), (3, 7, 13), (7, 7, 13), (0, 8, 13), (1, 8, 13), (9, 8, 13), (10, 8, 13), (2, 9, 13), (3, 9, 13), (7, 9, 13), (8, 9, 13), (3, 10, 13), (7, 10, 13), ],
            // vec![ (5, 5, 2), (5, 4, 3), (4, 5, 3), (5, 5, 3), (6, 5, 3), (5, 6, 3), (5, 5, 4) ],
            // vec![ (4, 4, 2), (5, 4, 2), (6, 4, 2), (4, 5, 2), (5, 5, 2), (6, 5, 2), (4, 6, 2), (5, 6, 2), (6, 6, 2), (4, 4, 3), (5, 4, 3), (6, 4, 3), (4, 5, 3), (6, 5, 3), (4, 6, 3), (5, 6, 3), (6, 6, 3), (4, 4, 4), (5, 4, 4), (6, 4, 4), (4, 5, 4), (5, 5, 4), (6, 5, 4), (4, 6, 4), (5, 6, 4), (6, 6, 4) ],
            // vec![ (4, 4, 1), (6, 4, 1), (4, 6, 1), (6, 6, 1), (4, 3, 2), (6, 3, 2), (3, 4, 2), (7, 4, 2), (3, 6, 2), (7, 6, 2), (4, 7, 2), (6, 7, 2), (4, 3, 4), (6, 3, 4), (3, 4, 4), (7, 4, 4), (3, 6, 4), (7, 6, 4), (4, 7, 4), (6, 7, 4), (4, 4, 5), (6, 4, 5), (4, 6, 5), (6, 6, 5) ],
            // vec![ (5, 5, 0), (5, 3, 1), (5, 4, 1), (3, 5, 1), (4, 5, 1), (5, 5, 1), (6, 5, 1), (7, 5, 1), (5, 6, 1), (5, 7, 1), (5, 3, 2), (5, 4, 2), (3, 5, 2), (4, 5, 2), (5, 5, 2), (6, 5, 2), (7, 5, 2), (5, 6, 2), (5, 7, 2), (5, 2, 3), (3, 3, 3), (4, 3, 3), (5, 3, 3), (6, 3, 3), (7, 3, 3), (3, 4, 3), (4, 4, 3), (5, 4, 3), (6, 4, 3), (7, 4, 3), (2, 5, 3), (3, 5, 3), (4, 5, 3), (6, 5, 3), (7, 5, 3), (8, 5, 3), (3, 6, 3), (4, 6, 3), (5, 6, 3), (6, 6, 3), (7, 6, 3), (3, 7, 3), (4, 7, 3), (5, 7, 3), (6, 7, 3), (7, 7, 3), (5, 8, 3), (5, 3, 4), (5, 4, 4), (3, 5, 4), (4, 5, 4), (5, 5, 4), (6, 5, 4), (7, 5, 4), (5, 6, 4), (5, 7, 4), (5, 3, 5), (5, 4, 5), (3, 5, 5), (4, 5, 5), (5, 5, 5), (6, 5, 5), (7, 5, 5), (5, 6, 5), (5, 7, 5), (5, 5, 6) ],
            // vec![ (5, 5, 0), (4, 3, 1), (6, 3, 1), (3, 4, 1), (7, 4, 1), (3, 6, 1), (7, 6, 1), (4, 7, 1), (6, 7, 1), (3, 3, 2), (7, 3, 2), (3, 7, 2), (7, 7, 2), (5, 2, 3), (2, 5, 3), (8, 5, 3), (5, 8, 3), (3, 3, 4), (7, 3, 4), (3, 7, 4), (7, 7, 4), (4, 3, 5), (6, 3, 5), (3, 4, 5), (7, 4, 5), (3, 6, 5), (7, 6, 5), (4, 7, 5), (6, 7, 5), (5, 5, 6) ],
            // vec![ (4, 4, 1), (6, 4, 1), (4, 6, 1), (6, 6, 1), (4, 3, 2), (6, 3, 2), (3, 4, 2), (7, 4, 2), (3, 6, 2), (7, 6, 2), (4, 7, 2), (6, 7, 2), (4, 3, 4), (6, 3, 4), (3, 4, 4), (7, 4, 4), (3, 6, 4), (7, 6, 4), (4, 7, 4), (6, 7, 4), (4, 4, 5), (6, 4, 5), (4, 6, 5), (6, 6, 5) ],
            // vec![ (5, 5, 0), (5, 3, 1), (5, 4, 1), (3, 5, 1), (4, 5, 1), (5, 5, 1), (6, 5, 1), (7, 5, 1), (5, 6, 1), (5, 7, 1), (5, 3, 2), (5, 4, 2), (3, 5, 2), (4, 5, 2), (5, 5, 2), (6, 5, 2), (7, 5, 2), (5, 6, 2), (5, 7, 2), (5, 2, 3), (3, 3, 3), (4, 3, 3), (5, 3, 3), (6, 3, 3), (7, 3, 3), (3, 4, 3), (4, 4, 3), (5, 4, 3), (6, 4, 3), (7, 4, 3), (2, 5, 3), (3, 5, 3), (4, 5, 3), (6, 5, 3), (7, 5, 3), (8, 5, 3), (3, 6, 3), (4, 6, 3), (5, 6, 3), (6, 6, 3), (7, 6, 3), (3, 7, 3), (4, 7, 3), (5, 7, 3), (6, 7, 3), (7, 7, 3), (5, 8, 3), (5, 3, 4), (5, 4, 4), (3, 5, 4), (4, 5, 4), (5, 5, 4), (6, 5, 4), (7, 5, 4), (5, 6, 4), (5, 7, 4), (5, 3, 5), (5, 4, 5), (3, 5, 5), (4, 5, 5), (5, 5, 5), (6, 5, 5), (7, 5, 5), (5, 6, 5), (5, 7, 5), (5, 5, 6) ],
            // vec![ (5, 5, 0), (4, 3, 1), (6, 3, 1), (3, 4, 1), (7, 4, 1), (3, 6, 1), (7, 6, 1), (4, 7, 1), (6, 7, 1), (3, 3, 2), (7, 3, 2), (3, 7, 2), (7, 7, 2), (5, 2, 3), (2, 5, 3), (8, 5, 3), (5, 8, 3), (3, 3, 4), (7, 3, 4), (3, 7, 4), (7, 7, 4), (4, 3, 5), (6, 3, 5), (3, 4, 5), (7, 4, 5), (3, 6, 5), (7, 6, 5), (4, 7, 5), (6, 7, 5), (5, 5, 6) ],
            // vec![ (4, 4, 1), (6, 4, 1), (4, 6, 1), (6, 6, 1), (4, 3, 2), (6, 3, 2), (3, 4, 2), (7, 4, 2), (3, 6, 2), (7, 6, 2), (4, 7, 2), (6, 7, 2), (4, 3, 4), (6, 3, 4), (3, 4, 4), (7, 4, 4), (3, 6, 4), (7, 6, 4), (4, 7, 4), (6, 7, 4), (4, 4, 5), (6, 4, 5), (4, 6, 5), (6, 6, 5) ],
            // vec![ (5, 5, 0), (5, 3, 1), (5, 4, 1), (3, 5, 1), (4, 5, 1), (5, 5, 1), (6, 5, 1), (7, 5, 1), (5, 6, 1), (5, 7, 1), (5, 3, 2), (5, 4, 2), (3, 5, 2), (4, 5, 2), (5, 5, 2), (6, 5, 2), (7, 5, 2), (5, 6, 2), (5, 7, 2), (5, 2, 3), (3, 3, 3), (4, 3, 3), (5, 3, 3), (6, 3, 3), (7, 3, 3), (3, 4, 3), (4, 4, 3), (5, 4, 3), (6, 4, 3), (7, 4, 3), (2, 5, 3), (3, 5, 3), (4, 5, 3), (6, 5, 3), (7, 5, 3), (8, 5, 3), (3, 6, 3), (4, 6, 3), (5, 6, 3), (6, 6, 3), (7, 6, 3), (3, 7, 3), (4, 7, 3), (5, 7, 3), (6, 7, 3), (7, 7, 3), (5, 8, 3), (5, 3, 4), (5, 4, 4), (3, 5, 4), (4, 5, 4), (5, 5, 4), (6, 5, 4), (7, 5, 4), (5, 6, 4), (5, 7, 4), (5, 3, 5), (5, 4, 5), (3, 5, 5), (4, 5, 5), (5, 5, 5), (6, 5, 5), (7, 5, 5), (5, 6, 5), (5, 7, 5), (5, 5, 6) ],
        ];
        let positions = positions.iter().map(|f| f.iter().map(|&(x, y, z)| cgmath::Vector3::<f32>::new((x * 2) as f32, (y * 2) as f32, (z * 2) as f32)).collect::<Vec<_>>()).collect::<Vec<_>>();
        let animator = Animator::new(positions, 100);

        Self {
            surface,
            device,
            queue,
            config,
            render_pipeline,
            obj_model,
            camera,
            projection,
            camera_controller,
            camera_buffer,
            camera_bind_group,
            camera_uniform,
            instance_buffer,
            depth_texture,
            size,
            light_uniform,
            light_buffer,
            light_bind_group,
            light_render_pipeline,
            #[allow(dead_code)]
            debug_material,
            mouse_pressed: false,
            instance_animator: animator,
        }
    }

    fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.projection.resize(new_size.width, new_size.height);
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
            self.depth_texture =
                texture::Texture::create_depth_texture(&self.device, &self.config, "depth_texture");
        }
    }

    fn input(&mut self, event: &DeviceEvent) -> bool {
        match event {
            DeviceEvent::Key(KeyboardInput {
                virtual_keycode: Some(key),
                state,
                ..
            }) => self.camera_controller.process_keyboard(*key, *state),
            DeviceEvent::MouseWheel { delta, .. } => {
                self.camera_controller.process_scroll(delta);
                true
            }
            DeviceEvent::Button {
                button: 1, // Left Mouse Button
                state,
            } => {
                self.mouse_pressed = *state == ElementState::Pressed;
                true
            }
            DeviceEvent::MouseMotion { delta } => {
                if self.mouse_pressed {
                    self.camera_controller.process_mouse(delta.0, delta.1);
                }
                true
            }
            _ => false,
        }
    }

    fn update(&mut self, dt: std::time::Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.camera_uniform
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );

        // Update the light
        // let old_position: cgmath::Vector3<_> = self.light_uniform.position.into();
        // self.light_uniform.position =
        //     (cgmath::Quaternion::from_axis_angle((0.0, 1.0, 0.0).into(), cgmath::Deg(1.0))
        //         * old_position)
        //         .into();
        // self.queue.write_buffer(
        //     &self.light_buffer,
        //     0,
        //     bytemuck::cast_slice(&[self.light_uniform]),
        // );

        self.instance_animator.update();
        self.queue.write_buffer(
            &self.instance_buffer,
            0,
            bytemuck::cast_slice(&self.instance_animator.get_current_frame()),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_frame()?.output;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: true,
                    },
                }],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
            render_pass.set_pipeline(&self.light_render_pipeline);
            render_pass.draw_light_model(
                &self.obj_model,
                &self.camera_bind_group,
                &self.light_bind_group,
            );

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.draw_model_instanced(
                &self.obj_model,
                0..self.instance_animator.get_current_frame().len() as u32,
                &self.camera_bind_group,
                &self.light_bind_group,
            );
        }
        self.queue.submit(iter::once(encoder.finish()));

        Ok(())
    }
}

fn main() {
    env_logger::init();
    let event_loop = EventLoop::new();
    let title = env!("CARGO_PKG_NAME");
    let window = winit::window::WindowBuilder::new()
        .with_title(title)
        .build(&event_loop)
        .unwrap();
    let mut state = pollster::block_on(State::new(&window)); // NEW!
    let mut last_render_time = std::time::Instant::now();
    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::MainEventsCleared => window.request_redraw(),
            Event::DeviceEvent {
                ref event,
                .. // We're not using device_id currently
            } => {
                state.input(event);
            }
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == window.id() => {
                match event {
                    WindowEvent::CloseRequested
                    | WindowEvent::KeyboardInput {
                        input:
                            KeyboardInput {
                                state: ElementState::Pressed,
                                virtual_keycode: Some(VirtualKeyCode::Escape),
                                ..
                            },
                        ..
                    } => *control_flow = ControlFlow::Exit,
                    WindowEvent::Resized(physical_size) => {
                        state.resize(*physical_size);
                    }
                    WindowEvent::ScaleFactorChanged { new_inner_size, .. } => {
                        state.resize(**new_inner_size);
                    }
                    _ => {}
                }
            }
            Event::RedrawRequested(_) => {
                let now = std::time::Instant::now();
                let dt = now - last_render_time;
                last_render_time = now;
                state.update(dt);
                match state.render() {
                    Ok(_) => {}
                    // Reconfigure the surface if lost
                    Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                    // The system is out of memory, we should probably quit
                    Err(wgpu::SurfaceError::OutOfMemory) => *control_flow = ControlFlow::Exit,
                    // All other errors (Outdated, Timeout) should be resolved by the next frame
                    Err(e) => eprintln!("{:?}", e),
                }
            }
            _ => {}
        }
    });
}
