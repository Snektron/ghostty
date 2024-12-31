//! Much like GpuBuffer, this file wraps a Vulkan texture
//! with all the data that it requires.

const GpuTexture = @This();

const std = @import("std");
const vk = @import("vulkan");
const Graphics = @import("Graphics.zig");
const GpuBuffer = @import("GpuBuffer.zig");

image: vk.Image,
memory: vk.DeviceMemory,
view: vk.ImageView,
width: u32,
height: u32,

pub fn init(
    graphics: Graphics,
    width: u32,
    height: u32,
    format: vk.Format,
    usage: vk.ImageUsageFlags,
    property_flags: vk.MemoryPropertyFlags,
) !GpuTexture {
    const dev = graphics.dev;
    const image = try dev.createImage(&.{
        .image_type = .@"2d",
        .format = format,
        .extent = .{
            .width = width,
            .height = height,
            .depth = 1,
        },
        .mip_levels = 1,
        .array_layers = 1,
        .samples = .{ .@"1_bit" = true },
        .tiling = .optimal,
        .usage = usage,
        // We will only use images from the Graphics queue.
        .sharing_mode = .exclusive,
        .initial_layout = .undefined,
    }, null);
    errdefer dev.destroyImage(image, null);

    const mem_reqs = dev.getImageMemoryRequirements(image);
    const memory = try graphics.allocate(mem_reqs, property_flags);
    errdefer dev.freeMemory(memory, null);

    try dev.bindImageMemory(image, memory, 0);

    const view = try dev.createImageView(&.{
        .flags = .{},
        .image = image,
        .view_type = .@"2d",
        .format = format,
        .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
        .subresource_range = .{
            .aspect_mask = .{ .color_bit = true },
            .base_mip_level = 0,
            .level_count = 1,
            .base_array_layer = 0,
            .layer_count = 1,
        },
    }, null);

    return .{
        .image = image,
        .memory = memory,
        .view = view,
        .width = width,
        .height = height,
    };
}

pub fn deinit(self: *GpuTexture, graphics: Graphics) void {
    graphics.dev.destroyImageView(self.view, null);
    graphics.dev.freeMemory(self.memory, null);
    graphics.dev.destroyImage(self.image, null);
}

pub fn uploadWithStagingBuffer(
    self: *GpuTexture,
    graphics: Graphics,
    data: []const u8,
    width: u32,
    height: u32,
) !void {
    var staging_buffer = GpuBuffer.init(
        graphics,
        data.len,
        .{ .transfer_src_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true, .host_cached_bit = true },
    ) catch try GpuBuffer.init(
        graphics,
        data.len,
        .{ .transfer_src_bit = true },
        .{ .host_visible_bit = true, .host_coherent_bit = true },
    );
    defer staging_buffer.deinit(graphics);

    {
        const staging_data = try staging_buffer.map(graphics);
        defer staging_buffer.unmap(graphics);
        @memcpy(staging_data, data);
    }

    try graphics.copyBufferToImage(self.image, staging_buffer.buffer, width, height);
}
