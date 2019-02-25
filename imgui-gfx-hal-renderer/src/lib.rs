#[allow(unused_imports)]
use {
    imgui::ImVec2,
    rendy::{
        command::{Families, QueueId, RenderPassEncoder},
        factory::{Config, Factory},
        graph::{present::PresentNode, render::*, Graph, GraphBuilder, NodeBuffer, NodeImage},
        memory::MemoryUsageValue,
        mesh::{AsVertex, PosTex, AsAttribute},
        resource::buffer::Buffer,
        shader::{Shader, ShaderKind, SourceLanguage, StaticShaderInfo},
        texture::{pixel::Rgba8Srgb, Texture, TextureBuilder},
    },
    gfx_hal::{Backend, Device, pso, format, image}
};
use imgui_sys::ImU32;
use std::cmp::Ordering;


lazy_static::lazy_static! {
    static ref VERTEX: StaticShaderInfo = StaticShaderInfo::new(
        "shaders/ui.vert",
        ShaderKind::Vertex,
        SourceLanguage::GLSL,
        "main",
    );

    static ref FRAGMENT: StaticShaderInfo = StaticShaderInfo::new(
        "shaders/ui.frag",
        ShaderKind::Fragment,
        SourceLanguage::GLSL,
        "main",
    );
}

#[derive(Debug)]
struct ImguiPipeline<B: Backend> {
    texture: Texture<B>,
    vertex_buffer: Option<Buffer<B>>,
    index_buffer: Option<Buffer<B>>,
    descriptor_pool: B::DescriptorPool,
    descriptor_set: B::DescriptorSet,
}

#[derive(Debug, Default)]
struct ImguiPipelineDesc;

#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct Vec2(ImVec2);

impl PartialOrd for Vec2 {
    fn partial_cmp(&self, other: &Vec2) -> Option<Ordering> {
        let (Vec2(this), Vec2(that)) = (self, other);
        let y_ord = this.y.cmp(that.y);
        let x_ord = this.x.cmp(that.x);
        if y_ord == Ordering::Equal {
            Some(x_ord)
        } else {
            Some(y_ord)
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
struct Position(Vec2);


impl AsAttribute for Position {
    const NAME: &'static str = "position";
    const SIZE: u32 = 8;
    const FORMAT: gfx_hal::format::Format = gfx_hal::format::Format::Rg32Float;
}

#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
struct Uv(Vec2);

impl AsAttribute for Uv {
    const NAME: &'static str = "uv";
    const SIZE: u32 = 8;
    const FORMAT: gfx_hal::format::Format = gfx_hal::format::Format::Rg32Float;
}

#[derive(Copy, Clone, Debug, Default, PartialEq, PartialOrd)]
struct Color(ImU32);

impl AsAttribute for Color {
    const NAME: &'static str = "color";
    const SIZE: u32 = 4;
    const FORMAT: gfx_hal::format::Format = gfx_hal::format::Format::Rgba8Unorm;
}

impl<B, T> SimpleGraphicsPipelineDesc<B, T> for ImguiPipelineDesc
    where
        B: gfx_hal::Backend,
        T: ?Sized,
{
    type Pipeline = ImguiPipeline<B>;

    fn depth_stencil(&self) -> Option<pso::DepthStencilDesc> {
        None
    }

    fn vertices(
        &self,
    ) -> Vec<(
        Vec<gfx_hal::pso::Element<format::Format>>,
        gfx_hal::pso::ElemStride,
        gfx_hal::pso::InstanceRate,
    )> {
        vec![PosTex::VERTEX.gfx_vertex_input_desc(0)]
    }

    fn layout(&self) -> Layout {
        Layout {
            sets: vec![SetLayout {
                bindings: vec![
                    pso::DescriptorSetLayoutBinding {
                        binding: 0,
                        ty: pso::DescriptorType::CombinedImageSampler,
                        count: 1,
                        stage_flags: pso::ShaderStageFlags::FRAGMENT,
                        immutable_samplers: true,
                    }
                ],
            }],
            push_constants: Vec::new(),
        }
    }

    fn load_shader_set<'b>(
        &self,
        storage: &'b mut Vec<B::ShaderModule>,
        factory: &mut Factory<B>,
        _aux: &mut T,
    ) -> gfx_hal::pso::GraphicsShaderSet<'b, B> {
        storage.clear();

        storage.push(VERTEX.module(factory).unwrap());

        storage.push(FRAGMENT.module(factory).unwrap());

        gfx_hal::pso::GraphicsShaderSet {
            vertex: gfx_hal::pso::EntryPoint {
                entry: "main",
                module: &storage[0],
                specialization: gfx_hal::pso::Specialization::default(),
            },
            fragment: Some(gfx_hal::pso::EntryPoint {
                entry: "main",
                module: &storage[1],
                specialization: gfx_hal::pso::Specialization::default(),
            }),
            hull: None,
            domain: None,
            geometry: None,
        }
    }

    fn build<'b>(
        self,
        factory: &mut Factory<B>,
        queue: QueueId,
        _aux: &mut T,
        buffers: Vec<NodeBuffer<'b, B>>,
        images: Vec<NodeImage<'b, B>>,
        set_layouts: &[B::DescriptorSetLayout],
    ) -> Result<ImguiPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert_eq!(set_layouts.len(), 1);

        // This is how we can load an image and create a new texture.

        let (width, height) = (256, 240);

        let mut image_data = Vec::<Rgba8Srgb>::new();

        for _y in 0..height {
            for _x in 0..width {
                image_data.push(Rgba8Srgb {
                    repr: [0,0,0,0],
                });
            }
        }

        let texture_builder = TextureBuilder::new()
            .with_kind(gfx_hal::image::Kind::D2(width, height, 1, 1))
            .with_view_kind(gfx_hal::image::ViewKind::D2)
            .with_data_width(width)
            .with_data_height(height)
            .with_data(&image_data);

        let texture = texture_builder
            .build(
                queue,
                gfx_hal::image::Access::TRANSFER_WRITE,
                gfx_hal::image::Layout::TransferDstOptimal,
                factory,
            )
            .unwrap();

        let mut descriptor_pool = unsafe {

            factory.device().create_descriptor_pool(
                1,
                &[
                    gfx_hal::pso::DescriptorRangeDesc {
                        ty: gfx_hal::pso::DescriptorType::CombinedImageSampler,
                        count: 1,
                    }
                ],
            )
        }
            .unwrap();

        let descriptor_set = unsafe {
            gfx_hal::pso::DescriptorPool::allocate_set(&mut descriptor_pool, &set_layouts[0])
        }
            .unwrap();

        unsafe {
            factory.device().write_descriptor_sets(
                vec![
                    gfx_hal::pso::DescriptorSetWrite {
                        set: &descriptor_set,
                        binding: 0,
                        array_offset: 0,
                        descriptors: vec![pso::Descriptor::CombinedImageSampler(
                            texture.image_view.raw(),
                            image::Layout::ShaderReadOnlyOptimal,
                            texture.sampler.raw()
                        )],
                    }
                ],
            );
        }

        Ok(ImguiPipeline {
            texture,
            vertex_buffer: None,
            index_buffer: None,
            descriptor_pool,
            descriptor_set,
        })
    }
}

impl<B, T> SimpleGraphicsPipeline<B, T> for ImguiPipeline<B>
    where
        B: gfx_hal::Backend,
        T: ?Sized,
{
    type Desc = ImguiPipelineDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[B::DescriptorSetLayout],
        _index: usize,
        _aux: &T,
    ) -> PrepareResult {
        if self.vertex_buffer.is_some() {
            return PrepareResult::DrawReuse;
        }


        let mut vbuf = factory
            .create_buffer(
                512,
                PosTex::VERTEX.stride as u64 * 6,
                (gfx_hal::buffer::Usage::VERTEX, MemoryUsageValue::Dynamic),
            )
            .unwrap();

        let num_inds = 0; // TODO
        let mut index_buffer = factory.create_buffer(
            0 /* TODO: Correct number of indices */,
            0 /* TODO: Correct size */,
            (gfx_hal::buffer::Usage::INDEX, MemoryUsageValue::Dynamic)
        );

        unsafe {
            // Fresh buffer.
            factory
                .upload_visible_buffer(
                    &mut vbuf,
                    0,
                    &[
                        PosTex {
                            position: [-0.5, 0.33, 0.0].into(),
                            tex_coord: [0.0, 1.0].into(),
                        },
                        PosTex {
                            position: [0.5, 0.33, 0.0].into(),
                            tex_coord: [1.0, 1.0].into(),
                        },
                        PosTex {
                            position: [0.5, -0.33, 0.0].into(),
                            tex_coord: [1.0, 0.0].into(),
                        },
                        PosTex {
                            position: [-0.5, 0.33, 0.0].into(),
                            tex_coord: [0.0, 1.0].into(),
                        },
                        PosTex {
                            position: [0.5, -0.33, 0.0].into(),
                            tex_coord: [1.0, 0.0].into(),
                        },
                        PosTex {
                            position: [-0.5, -0.33, 0.0].into(),
                            tex_coord: [0.0, 0.0].into(),
                        },
                    ],
                )
                .unwrap();
        }

        self.vertex_buffer = Some(vbuf);

        return PrepareResult::DrawRecord;
    }

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        _aux: &T,
    ) {
        let vbuf = self.vertex_buffer.as_ref().unwrap();
        encoder.bind_graphics_descriptor_sets(
            layout,
            0,
            std::iter::once(&self.descriptor_set),
            std::iter::empty::<u32>(),
        );
        encoder.bind_vertex_buffers(0, Some((vbuf.raw(), 0)));
        encoder.draw(0..3, 0..1);
        encoder.draw(3..6, 0..1);
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &mut T) {}
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn run(
    event_loop: &mut EventsLoop,
    factory: &mut Factory<Backend>,
    families: &mut Families<Backend>,
    mut graph: Graph<Backend, ()>,
) -> Result<(), failure::Error> {
    let started = std::time::Instant::now();

    std::thread::spawn(move || {
        while started.elapsed() < std::time::Duration::new(30, 0) {
            std::thread::sleep(std::time::Duration::new(1, 0));
        }

        std::process::abort();
    });

    let mut frames = 0u64..;
    let mut elapsed = started.elapsed();

    for _ in &mut frames {
        factory.maintain(families);
        event_loop.poll_events(|_| ());
        graph.run(factory, families, &mut ());

        elapsed = started.elapsed();
        if elapsed >= std::time::Duration::new(5, 0) {
            break;
        }
    }

    let elapsed_ns = elapsed.as_secs() * 1_000_000_000 + elapsed.subsec_nanos() as u64;

    log::info!(
        "Elapsed: {:?}. Frames: {}. FPS: {}",
        elapsed,
        frames.start,
        frames.start * 1_000_000_000 / elapsed_ns
    );

    graph.dispose(factory, &mut ());
    Ok(())
}

#[cfg(any(feature = "dx12", feature = "metal", feature = "vulkan"))]
fn main() {
    env_logger::Builder::from_default_env()
        .filter_level(log::LevelFilter::Warn)
        .filter_module("sprite", log::LevelFilter::Trace)
        .init();

    let config: Config = Default::default();

    let (mut factory, mut families): (Factory<Backend>, _) = rendy::factory::init(config).unwrap();

    let mut event_loop = EventsLoop::new();

    let window = WindowBuilder::new()
        .with_title("Rendy example")
        .build(&event_loop)
        .unwrap();

    event_loop.poll_events(|_| ());

    let surface = factory.create_surface(window.into());

    let mut graph_builder = GraphBuilder::<Backend, ()>::new();

    let color = graph_builder.create_image(
        surface.kind(),
        1,
        gfx_hal::format::Format::Rgba8Unorm,
        MemoryUsageValue::Data,
        Some(gfx_hal::command::ClearValue::Color(
            [1.0, 1.0, 1.0, 1.0].into(),
        )),
    );

    let pass = graph_builder.add_node(
        ImguiPipeline::builder()
            .into_subpass()
            .with_color(color)
            .into_pass(),
    );

    graph_builder.add_node(PresentNode::builder(surface, color).with_dependency(pass));

    let graph = graph_builder
        .build(&mut factory, &mut families, &mut ())
        .unwrap();

    run(&mut event_loop, &mut factory, &mut families, graph).unwrap();
}

#[cfg(not(any(feature = "dx12", feature = "metal", feature = "vulkan")))]
fn main() {
    panic!("Specify feature: { dx12, metal, vulkan }");
}
