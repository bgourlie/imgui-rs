use gfx_hal::pso::Rect;
use imgui::{DrawData, ImDrawIdx};
#[allow(unused_imports)]
use {
    gfx_hal::{format, image, pso, Backend, Device, PhysicalDevice},
    imgui::{ImDrawVert, ImGui, ImVec2},
    imgui_sys::ImU32,
    rendy::{
        command::{Families, QueueId, RenderPassEncoder},
        factory::{Config, Factory},
        graph::{present::PresentNode, render::*, Graph, GraphBuilder, NodeBuffer, NodeImage},
        memory::MemoryUsageValue,
        mesh::{AsAttribute, AsVertex, Attribute, VertexFormat, WithAttribute},
        resource::buffer::Buffer,
        shader::{Shader, ShaderKind, SourceLanguage, StaticShaderInfo},
        texture::{pixel::Rgba8Srgb, Texture, TextureBuilder},
    },
    std::{
        borrow::Cow,
        cmp::Ordering,
        fmt::{Debug, Error, Formatter},
    },
};

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
struct Buffers<B: Backend> {
    vertex: Buffer<B>,
    index: Buffer<B>,
    required_vertex_capacity: usize,
    required_index_capacity: usize,
}

impl<B: Backend> Buffers<B> {
    fn new(factory: &Factory<B>, draw_data: &DrawData) -> Self {
        let align = PhysicalDevice::limits(factory.physical()).min_uniform_buffer_offset_alignment;

        let vertex_buffer = factory
            .create_buffer(
                align,
                Vertex::VERTEX.stride as u64 * draw_data.total_vtx_count() as u64,
                (gfx_hal::buffer::Usage::VERTEX, MemoryUsageValue::Dynamic),
            )
            .unwrap();

        let index_buffer = factory
            .create_buffer(
                align,
                std::mem::size_of::<ImDrawIdx>() as u64 * draw_data.total_idx_count() as u64,
                (gfx_hal::buffer::Usage::INDEX, MemoryUsageValue::Dynamic),
            )
            .unwrap();
        Buffers {
            vertex: vertex_buffer,
            index: index_buffer,
            required_vertex_capacity: draw_data.total_vtx_count(),
            required_index_capacity: draw_data.total_idx_count(),
        }
    }

    fn update(
        &mut self,
        factory: &Factory<B>,
        vertices: &[ImDrawVert],
        indices: &[ImDrawIdx],
        vertex_offset: u64,
        index_offset: u64,
    ) {
        unsafe {
            factory
                .upload_visible_buffer::<ImDrawVert>(&mut self.vertex, vertex_offset, vertices)
                .unwrap();
            factory
                .upload_visible_buffer::<ImDrawIdx>(&mut self.index, index_offset, indices)
                .unwrap()
        }
    }

    fn has_room(&self, num_vertices: usize, num_indices: usize) -> bool {
        self.required_vertex_capacity >= num_vertices && self.required_index_capacity >= num_indices
    }
}

#[derive(Debug)]
struct ImguiPipeline<B: Backend> {
    texture: Texture<B>,
    buffers: Option<(Buffers<B>)>,
    descriptor_pool: B::DescriptorPool,
    descriptor_set: B::DescriptorSet,
}

struct Gui(ImGui);

impl Default for Gui {
    fn default() -> Self {
        Gui(ImGui::init())
    }
}

impl Debug for Gui {
    fn fmt(&self, f: &mut Formatter) -> Result<(), Error> {
        f.write_str("ImGuiInstance")
    }
}

#[derive(Debug, Default)]
struct ImguiPipelineDesc {
    gui: Gui,
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
struct Vec2(ImVec2);

impl PartialOrd for Vec2 {
    fn partial_cmp(&self, other: &Vec2) -> Option<Ordering> {
        let (Vec2(this), Vec2(that)) = (self, other);
        let y_ord = this.y.partial_cmp(&that.y);
        match y_ord {
            Some(Ordering::Equal) => this.x.partial_cmp(&that.x),
            Some(_) => y_ord,
            None => None,
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

#[derive(Copy, Clone, Debug, Default)]
struct Vertex(ImDrawVert);

impl PartialOrd for Vertex {
    fn partial_cmp(&self, other: &Vertex) -> Option<Ordering> {
        let (Vertex(this), Vertex(that)) = (self, other);
        let (this_pos, that_pos) = (Position(Vec2(this.pos)), Position(Vec2(that.pos)));
        let pos_ordering = this_pos.partial_cmp(&that_pos);

        match pos_ordering {
            Some(Ordering::Equal) => {
                let (this_uv, that_uv) = (Uv(Vec2(this.uv)), Uv(Vec2(that.uv)));
                let uv_ordering = this_uv.partial_cmp(&that_uv);
                match uv_ordering {
                    Some(Ordering::Equal) => {
                        let (this_color, that_color) = (Color(this.col), Color(that.col));
                        this_color.partial_cmp(&that_color)
                    }
                    Some(_) => uv_ordering,
                    None => None,
                }
            }
            Some(_) => pos_ordering,
            None => None,
        }
    }
}

impl PartialEq for Vertex {
    fn eq(&self, other: &Vertex) -> bool {
        let (Vertex(this), Vertex(that)) = (self, other);
        let (this_pos, that_pos) = (Position(Vec2(this.pos)), Position(Vec2(that.pos)));
        let (this_uv, that_uv) = (Uv(Vec2(this.uv)), Uv(Vec2(that.uv)));
        let (this_color, that_color) = (Color(this.col), Color(that.col));
        this_pos.eq(&that_pos) && this_uv.eq(&that_uv) && this_color.eq(&that_color)
    }
}

impl AsVertex for Vertex {
    const VERTEX: VertexFormat<'static> = VertexFormat {
        attributes: Cow::Borrowed(&[
            <Self as WithAttribute<Position>>::ATTRIBUTE,
            <Self as WithAttribute<Uv>>::ATTRIBUTE,
            <Self as WithAttribute<Color>>::ATTRIBUTE,
        ]),
        stride: Position::SIZE + Uv::SIZE + Color::SIZE,
    };
}

impl WithAttribute<Position> for Vertex {
    const ATTRIBUTE: Attribute = Attribute {
        offset: 0,
        format: Position::FORMAT,
    };
}

impl WithAttribute<Uv> for Vertex {
    const ATTRIBUTE: Attribute = Attribute {
        offset: Position::SIZE,
        format: Uv::FORMAT,
    };
}

impl WithAttribute<Color> for Vertex {
    const ATTRIBUTE: Attribute = Attribute {
        offset: Position::SIZE + Uv::SIZE,
        format: Color::FORMAT,
    };
}

impl<'a, B> SimpleGraphicsPipelineDesc<B, DrawData<'a>> for ImguiPipelineDesc
where
    B: gfx_hal::Backend,
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
        vec![Vertex::VERTEX.gfx_vertex_input_desc(0)]
    }

    fn layout(&self) -> Layout {
        Layout {
            sets: vec![SetLayout {
                bindings: vec![pso::DescriptorSetLayoutBinding {
                    binding: 0,
                    ty: pso::DescriptorType::CombinedImageSampler,
                    count: 1,
                    stage_flags: pso::ShaderStageFlags::FRAGMENT,
                    immutable_samplers: true,
                }],
            }],
            push_constants: Vec::new(),
        }
    }

    fn load_shader_set<'b>(
        &self,
        storage: &'b mut Vec<B::ShaderModule>,
        factory: &mut Factory<B>,
        _aux: &mut DrawData,
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
        mut self,
        factory: &mut Factory<B>,
        queue: QueueId,
        _aux: &mut DrawData,
        buffers: Vec<NodeBuffer<'b, B>>,
        images: Vec<NodeImage<'b, B>>,
        set_layouts: &[B::DescriptorSetLayout],
    ) -> Result<ImguiPipeline<B>, failure::Error> {
        assert!(buffers.is_empty());
        assert!(images.is_empty());
        assert_eq!(set_layouts.len(), 1);

        let Gui(imgui) = &mut self.gui;
        let texture = imgui
            .prepare_texture::<_, Result<_, Error>>(|handle| {
                let (width, height) = (handle.width, handle.height);

                let texture_builder = TextureBuilder::new()
                    .with_kind(gfx_hal::image::Kind::D2(width, height, 1, 1))
                    .with_view_kind(gfx_hal::image::ViewKind::D2)
                    .with_data_width(width)
                    .with_data_height(height);

                let texture = texture_builder
                    .build(
                        queue,
                        gfx_hal::image::Access::TRANSFER_WRITE,
                        gfx_hal::image::Layout::TransferDstOptimal,
                        factory,
                    )
                    .unwrap();
                Ok(texture)
            })
            .unwrap();

        let mut descriptor_pool = unsafe {
            factory.device().create_descriptor_pool(
                1,
                &[gfx_hal::pso::DescriptorRangeDesc {
                    ty: gfx_hal::pso::DescriptorType::CombinedImageSampler,
                    count: 1,
                }],
            )
        }
        .unwrap();

        let descriptor_set = unsafe {
            gfx_hal::pso::DescriptorPool::allocate_set(&mut descriptor_pool, &set_layouts[0])
        }
        .unwrap();

        unsafe {
            factory
                .device()
                .write_descriptor_sets(vec![gfx_hal::pso::DescriptorSetWrite {
                    set: &descriptor_set,
                    binding: 0,
                    array_offset: 0,
                    descriptors: vec![pso::Descriptor::CombinedImageSampler(
                        texture.image_view.raw(),
                        image::Layout::ShaderReadOnlyOptimal,
                        texture.sampler.raw(),
                    )],
                }]);
        }

        Ok(ImguiPipeline {
            texture,
            buffers: None,
            descriptor_pool,
            descriptor_set,
        })
    }
}

impl<'a, B> SimpleGraphicsPipeline<B, DrawData<'a>> for ImguiPipeline<B>
where
    B: gfx_hal::Backend,
{
    type Desc = ImguiPipelineDesc;

    fn prepare(
        &mut self,
        factory: &Factory<B>,
        _queue: QueueId,
        _set_layouts: &[B::DescriptorSetLayout],
        _index: usize,
        draw_data: &DrawData,
    ) -> PrepareResult {
        if self
            .buffers
            .as_ref()
            .map(|buffers| {
                !buffers.has_room(draw_data.total_vtx_count(), draw_data.total_idx_count())
            })
            .unwrap_or(true)
        {
            let buffers = Buffers::new(factory, draw_data);
            if let Some(_old) = std::mem::replace(&mut self.buffers, Some(buffers)) {
                // TODO: Destroy buffers? How in rendy?
            }
        }

        return PrepareResult::DrawRecord;
    }

    fn draw(
        &mut self,
        layout: &B::PipelineLayout,
        mut encoder: RenderPassEncoder<'_, B>,
        _index: usize,
        draw_data: &DrawData,
    ) {
        let buffers = self.buffers.as_ref().unwrap();
        encoder.bind_graphics_descriptor_sets(
            layout,
            0,
            std::iter::once(&self.descriptor_set),
            std::iter::empty::<u32>(),
        );
        encoder.bind_vertex_buffers(0, Some((buffers.vertex.raw(), 0)));
        encoder.bind_index_buffer(buffers.index.raw(), 0, gfx_hal::IndexType::U16);

        // Set push constants
        let push_constants = [
            // scale
            2.0 / width,
            2.0 / height,
            //offset
            -1.0,
            -1.0,
        ];

        encoder.push_constants(layout, pso::ShaderStageFlags::VERTEX, 0, &push_constants);

        let mut vertex_offset = 0;
        let mut index_offset = 0;
        for list in draw_data {
            self.buffers.as_mut().unwrap().update(
                factory,
                list.vtx_buffer,
                list.idx_buffer,
                vertex_offset,
                index_offset,
            );

            for cmd in list.cmd_buffer.iter() {
                let scissor = Rect {
                    x: cmd.clip_rect.x as i16,
                    y: cmd.clip_rect.y as i16,
                    w: (cmd.clip_rect.z - cmd.clip_rect.x) as i16,
                    h: (cmd.clip_rect.w - cmd.clip_rect.y) as i16,
                };

                // TODO: pass.set_scissors(0, &[scissor]);
                encoder.draw_indexed(
                    index_offset as u32..index_offset as u32 + cmd.elem_count,
                    vertex_offset as i32,
                    0..1,
                );
                index_offset += cmd.elem_count as usize;
            }

            vertex_offset += list.vtx_buffer.len();
        }
    }

    fn dispose(self, _factory: &mut Factory<B>, _aux: &mut DrawData) {}
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
