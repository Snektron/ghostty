//! This file implements the Vulkan pipeline for Cell rendering
pub const CellPipeline = @This();

const std = @import("std");
const log = std.log.scoped(.vulkan);
const Allocator = std.mem.Allocator;

const vk = @import("vulkan");
const glslang = @import("glslang");

const Graphics = @import("Graphics.zig");
const Swapchain = @import("Swapchain.zig");

pub const Cell = @import("../opengl/CellProgram.zig").Cell;
pub const CellMode = @import("../opengl/CellProgram.zig").CellMode;

// Keep in sync with both shaders.
const PipelineLayout = struct {
    const binding_description = vk.VertexInputBindingDescription{
        .binding = 0,
        .stride = @sizeOf(Cell),
        .input_rate = .vertex,
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
            .format = .r32_uint,
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

    const bindings = [_]vk.DescriptorSetLayoutBinding {
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
pipeline_layout: vk.PipelineLayout,
pipeline: vk.Pipeline,

pub fn init(graphics: Graphics, swapchain: Swapchain) !CellPipeline {
    const dev = graphics.dev;

    var self = CellPipeline{
        .descriptor_set_layout = .null_handle,
        .pipeline = .null_handle,
        .pipeline_layout = .null_handle,
    };
    errdefer self.deinit(graphics);

    self.descriptor_set_layout = try dev.createDescriptorSetLayout(&.{
        .binding_count = PipelineLayout.bindings.len,
        .p_bindings = &PipelineLayout.bindings,
    }, null);

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
        .blend_enable = vk.FALSE,
        .src_color_blend_factor = .one,
        .dst_color_blend_factor = .zero,
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
            .cull_mode = .{ .back_bit = true },
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

    return self;
}

pub fn deinit(self: *CellPipeline, graphics: Graphics) void {
    graphics.dev.destroyDescriptorSetLayout(self.descriptor_set_layout, null);
    graphics.dev.destroyPipeline(self.pipeline, null);
    graphics.dev.destroyPipelineLayout(self.pipeline_layout, null);
    self.* = undefined;
}
