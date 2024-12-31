//! This file implements the Vulkan pipeline for Cell rendering
pub const CellPipeline = @This();

const std = @import("std");
const log = std.log.scoped(.vulkan);
const Allocator = std.mem.Allocator;

const vk = @import("vulkan");
const glslang = @import("glslang");

const Vulkan = @import("../Vulkan.zig");
const Graphics = @import("Graphics.zig");
const Swapchain = @import("Swapchain.zig");
const GpuTexture = @import("GpuTexture.zig");
const GpuBuffer = @import("GpuBuffer.zig");

pub const Cell = @import("../opengl/CellProgram.zig").Cell;
pub const CellMode = @import("../opengl/CellProgram.zig").CellMode;
pub const Params = PipelineLayout.PushConstants;

// Keep in sync with both shaders.
const PipelineLayout = struct {
    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(Cell),
        .input_rate = .instance,
    };

    const attributes = [_]vk.VertexInputAttributeDescription{
        .{
            .binding = 0,
            .location = 0,
            .format = .r16g16_uint,
            .offset = @offsetOf(Cell, "grid_col"),
        },
        .{
            .binding = 0,
            .location = 1,
            .format = .r32g32_uint,
            .offset = @offsetOf(Cell, "glyph_x"),
        },
        .{
            .binding = 0,
            .location = 2,
            .format = .r32g32_uint,
            .offset = @offsetOf(Cell, "glyph_width"),
        },
        .{
            .binding = 0,
            .location = 3,
            .format = .r32g32_sint,
            .offset = @offsetOf(Cell, "glyph_offset_x"),
        },
        .{
            .binding = 0,
            .location = 4,
            .format = .r8g8b8a8_unorm,
            .offset = @offsetOf(Cell, "r"),
        },
        .{
            .binding = 0,
            .location = 5,
            .format = .r8g8b8a8_unorm,
            .offset = @offsetOf(Cell, "bg_r"),
        },
        .{
            .binding = 0,
            .location = 6,
            .format = .r8_uint,
            .offset = @offsetOf(Cell, "mode"),
        },
        .{
            .binding = 0,
            .location = 7,
            .format = .r8_uint,
            .offset = @offsetOf(Cell, "grid_width"),
        },
    };

    const PushConstants = extern struct {
        projection: [16]f32,
        grid_padding_x: f32,
        grid_padding_y: f32,
        grid_padding_z: f32,
        grid_padding_w: f32,
        grid_size_x: f32,
        grid_size_y: f32,
        cell_size_x: f32,
        cell_size_y: f32,
        min_contrast: f32,
        padding_vertical_top: u32,
        padding_vertical_bottom: u32,
    };

    const bindings = [_]vk.DescriptorSetLayoutBinding{
        // layout(set = 0, binding = 0) uniform sampler2D text
        .{
            .binding = 0,
            .descriptor_type = .combined_image_sampler,
            .descriptor_count = 1,
            .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
        },
        // layout(set = 0, binding = 1) uniform sampler2D text_color
        .{
            .binding = 1,
            .descriptor_type = .combined_image_sampler,
            .descriptor_count = 1,
            .stage_flags = .{ .vertex_bit = true, .fragment_bit = true },
        },
    };
};

descriptor_set_layout: vk.DescriptorSetLayout,
descriptor_pool: vk.DescriptorPool,
/// We may be reallocating the textures bound here, and we cannot do that
/// while a frame is still being used on the GPU. Therefore, we need to have
/// a separate descriptor set for each frame. Take care that we need to keep
/// track of pending updates per descriptor set.
descriptor_sets: [Vulkan.frames_in_flight]vk.DescriptorSet,
pipeline_layout: vk.PipelineLayout,
pipeline: vk.Pipeline,

sampler: vk.Sampler,

pub fn init(graphics: Graphics, swapchain: Swapchain) !CellPipeline {
    const dev = graphics.dev;

    var self = CellPipeline{
        .descriptor_set_layout = .null_handle,
        .descriptor_pool = .null_handle,
        .descriptor_sets = undefined,
        .pipeline = .null_handle,
        .pipeline_layout = .null_handle,
        .sampler = .null_handle,
    };
    errdefer self.deinit(graphics);

    // initialize descriptor set layout

    self.descriptor_set_layout = try dev.createDescriptorSetLayout(&.{
        .binding_count = PipelineLayout.bindings.len,
        .p_bindings = &PipelineLayout.bindings,
    }, null);

    // initialize descriptor pool

    var pool_sizes = std.BoundedArray(vk.DescriptorPoolSize, PipelineLayout.bindings.len){};
    for (PipelineLayout.bindings) |binding| {
        for (pool_sizes.slice()) |*pool_size| {
            if (pool_size.type == binding.descriptor_type) {
                pool_size.descriptor_count += binding.descriptor_count * Vulkan.frames_in_flight;
                break;
            }
        } else {
            pool_sizes.appendAssumeCapacity(.{
                .type = binding.descriptor_type,
                .descriptor_count = binding.descriptor_count * Vulkan.frames_in_flight,
            });
        }
    }

    self.descriptor_pool = try dev.createDescriptorPool(&.{
        .max_sets = Vulkan.frames_in_flight,
        .pool_size_count = pool_sizes.len,
        .p_pool_sizes = &pool_sizes.buffer,
    }, null);

    // allocate descriptor sets

    var layouts: [Vulkan.frames_in_flight]vk.DescriptorSetLayout = undefined;
    @memset(&layouts, self.descriptor_set_layout);

    try dev.allocateDescriptorSets(&.{
        .descriptor_pool = self.descriptor_pool,
        .descriptor_set_count = layouts.len,
        .p_set_layouts = &layouts,
    }, &self.descriptor_sets);

    // initialize pipeline

    const push_constant_range: vk.PushConstantRange = .{
        .stage_flags = .{
            .vertex_bit = true,
            .fragment_bit = true,
        },
        .offset = 0,
        .size = @sizeOf(PipelineLayout.PushConstants),
    };

    self.pipeline_layout = try dev.createPipelineLayout(&.{
        .set_layout_count = 1,
        .p_set_layouts = @ptrCast(&self.descriptor_set_layout),
        .push_constant_range_count = 1,
        .p_push_constant_ranges = @ptrCast(&push_constant_range),
    }, null);

    log.debug("compiling cell vertex shader", .{});
    const vert = try graphics.compileShader(@embedFile("../shaders/vk_cell.v.glsl"), .vert);
    defer dev.destroyShaderModule(vert, null);

    log.debug("compiling cell fragment shader", .{});
    const frag = try graphics.compileShader(@embedFile("../shaders/vk_cell.f.glsl"), .frag);
    defer dev.destroyShaderModule(frag, null);

    const prci: vk.PipelineRenderingCreateInfo = .{
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachment_formats = &.{
            swapchain.surface_format.format,
        },
        .depth_attachment_format = .undefined,
        .stencil_attachment_format = .undefined,
    };

    const pcbas: vk.PipelineColorBlendAttachmentState = .{
        .blend_enable = vk.TRUE,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .one_minus_src_alpha,
        .color_blend_op = .add,
        .src_alpha_blend_factor = .one,
        .dst_alpha_blend_factor = .zero,
        .alpha_blend_op = .add,
        .color_write_mask = .{ .r_bit = true, .g_bit = true, .b_bit = true, .a_bit = true },
    };

    const dynstate = [_]vk.DynamicState{ .viewport, .scissor };

    const create_info: vk.GraphicsPipelineCreateInfo = .{
        .p_next = @ptrCast(&prci),
        .stage_count = 2,
        .p_stages = &.{
            .{
                .stage = .{ .vertex_bit = true },
                .module = vert,
                .p_name = "main",
            },
            .{
                .stage = .{ .fragment_bit = true },
                .module = frag,
                .p_name = "main",
            },
        },
        .p_vertex_input_state = &.{
            .vertex_binding_description_count = 1,
            .p_vertex_binding_descriptions = @ptrCast(&PipelineLayout.binding_description),
            .vertex_attribute_description_count = PipelineLayout.attributes.len,
            .p_vertex_attribute_descriptions = &PipelineLayout.attributes,
        },
        .p_input_assembly_state = &.{
            .topology = .triangle_list,
            .primitive_restart_enable = vk.FALSE,
        },
        .p_viewport_state = &.{
            .viewport_count = 1,
            .p_viewports = undefined, // set with cmdSetViewport
            .scissor_count = 1,
            .p_scissors = undefined, // set with cmdSetScissor
        },
        .p_rasterization_state = &.{
            .depth_clamp_enable = vk.FALSE,
            .rasterizer_discard_enable = vk.FALSE,
            .polygon_mode = .fill,
            .cull_mode = .{},
            .front_face = .clockwise,
            .depth_bias_enable = vk.FALSE,
            .depth_bias_constant_factor = 0,
            .depth_bias_clamp = 0,
            .depth_bias_slope_factor = 0,
            .line_width = 1,
        },
        .p_multisample_state = &.{
            .rasterization_samples = .{ .@"1_bit" = true },
            .sample_shading_enable = vk.FALSE,
            .min_sample_shading = 1,
            .alpha_to_coverage_enable = vk.FALSE,
            .alpha_to_one_enable = vk.FALSE,
        },
        .p_depth_stencil_state = null,
        .p_color_blend_state = &.{
            .logic_op_enable = vk.FALSE,
            .logic_op = .copy,
            .attachment_count = 1,
            .p_attachments = @ptrCast(&pcbas),
            .blend_constants = [_]f32{ 0, 0, 0, 0 },
        },
        .p_dynamic_state = &.{
            .flags = .{},
            .dynamic_state_count = dynstate.len,
            .p_dynamic_states = &dynstate,
        },
        .layout = self.pipeline_layout,
        .render_pass = .null_handle,
        .subpass = 0,
        .base_pipeline_handle = .null_handle,
        .base_pipeline_index = -1,
    };

    _ = try dev.createGraphicsPipelines(
        .null_handle,
        1,
        @ptrCast(&create_info),
        null,
        @ptrCast(&self.pipeline),
    );

    self.sampler = try dev.createSampler(&.{
        .flags = .{},
        .mag_filter = .linear,
        .min_filter = .linear,
        .mipmap_mode = .linear,
        .address_mode_u = .clamp_to_edge,
        .address_mode_v = .clamp_to_edge,
        .address_mode_w = .clamp_to_edge,
        .mip_lod_bias = 0,
        .anisotropy_enable = vk.FALSE,
        .max_anisotropy = 0,
        .compare_enable = vk.FALSE,
        .compare_op = .always,
        .min_lod = 0,
        .max_lod = 0,
        .border_color = .float_opaque_black,
        .unnormalized_coordinates = vk.FALSE,
    }, null);

    return self;
}

pub fn deinit(self: *CellPipeline, graphics: Graphics) void {
    graphics.dev.destroySampler(self.sampler, null);
    // Freeing the pool also frees the descriptor sets.
    graphics.dev.destroyDescriptorPool(self.descriptor_pool, null);
    graphics.dev.destroyDescriptorSetLayout(self.descriptor_set_layout, null);
    graphics.dev.destroyPipeline(self.pipeline, null);
    graphics.dev.destroyPipelineLayout(self.pipeline_layout, null);
    self.* = undefined;
}

pub fn bindTextures(
    self: *CellPipeline,
    graphics: Graphics,
    atlas_grayscale: GpuTexture,
    atlas_color: GpuTexture,
) !void {
    const greyscale_info: vk.DescriptorImageInfo = .{
        .sampler = self.sampler,
        .image_view = atlas_grayscale.view,
        .image_layout = .general, // TODO?
    };

    const color_info: vk.DescriptorImageInfo = .{
        .sampler = self.sampler,
        .image_view = atlas_color.view,
        .image_layout = .general, // TODO?
    };

    // TODO: We need some deferred operation for this stuff...
    // For now just ignore all of that stuff, because we are
    // binding the texture each frame anyway.

    const set = self.descriptor_sets[graphics.frameIndex()];
    const writes = [_]vk.WriteDescriptorSet{
        // layout(set = 0, binding = 0) uniform sampler2D text
        .{
            .dst_set = set,
            .dst_binding = 0,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .combined_image_sampler,
            .p_image_info = @ptrCast(&greyscale_info),
            .p_buffer_info = undefined,
            .p_texel_buffer_view = undefined,
        },
        // layout(set = 0, binding = 1) uniform sampler2D text_color
        .{
            .dst_set = set,
            .dst_binding = 1,
            .dst_array_element = 0,
            .descriptor_count = 1,
            .descriptor_type = .combined_image_sampler,
            .p_image_info = @ptrCast(&color_info),
            .p_buffer_info = undefined,
            .p_texel_buffer_view = undefined,
        },
    };

    graphics.dev.updateDescriptorSets(
        @intCast(writes.len),
        &writes,
        0,
        undefined,
    );
}

pub fn draw(
    self: *CellPipeline,
    graphics: Graphics,
    cmd_buf: vk.CommandBuffer,
    cells: GpuBuffer,
    len: usize,
    params: Params,
) !void {
    const dev = graphics.dev;

    dev.cmdBindPipeline(cmd_buf, .graphics, self.pipeline);
    dev.cmdBindDescriptorSets(
        cmd_buf,
        .graphics,
        self.pipeline_layout,
        0,
        1,
        @ptrCast(&self.descriptor_sets[graphics.frameIndex()]),
        0,
        undefined,
    );
    dev.cmdPushConstants(
        cmd_buf,
        self.pipeline_layout,
        .{ .vertex_bit = true, .fragment_bit = true },
        0,
        @sizeOf(PipelineLayout.PushConstants),
        @ptrCast(&params),
    );
    dev.cmdBindVertexBuffers(cmd_buf, 0, 1, @ptrCast(&cells.buffer), &.{0});
    dev.cmdDraw(cmd_buf, 6, @intCast(len), 0, 0);
}
