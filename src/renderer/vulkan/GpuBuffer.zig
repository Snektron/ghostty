//! This file wraps a Vulkan buffer with some functionality.

const GpuBuffer = @This();

const std = @import("std");
const vk = @import("vulkan");
const Graphics = @import("Graphics.zig");

buffer: vk.Buffer,
memory: vk.DeviceMemory,
size: usize,

pub fn init(
    graphics: Graphics,
    size: usize,
    usage: vk.BufferUsageFlags,
    property_flags: vk.MemoryPropertyFlags,
) !GpuBuffer {
    const dev = graphics.dev;
    const buffer = try dev.createBuffer(&.{
        .size = size,
        .usage = usage,
        // We will only use buffers from the Graphics queue.
        .sharing_mode = .exclusive,
    }, null);
    errdefer dev.destroyBuffer(buffer, null);

    const mem_reqs = dev.getBufferMemoryRequirements(buffer);
    const memory = try graphics.allocate(mem_reqs, property_flags);
    errdefer dev.freeMemory(memory, null);

    try dev.bindBufferMemory(buffer, memory, 0);

    return .{
        .buffer = buffer,
        .memory = memory,
        .size = size,
    };
}

pub fn deinit(self: *GpuBuffer, graphics: Graphics) void {
    graphics.dev.freeMemory(self.memory, null);
    graphics.dev.destroyBuffer(self.buffer, null);
}

pub fn map(self: *GpuBuffer, graphics: Graphics) ![]u8 {
    const ptr: [*]u8 = @ptrCast((try graphics.dev.mapMemory(self.memory, 0, vk.WHOLE_SIZE, .{})).?);
    return ptr[0..self.size];
}

pub fn unmap(self: *GpuBuffer, graphics: Graphics) void {
    graphics.dev.unmapMemory(self.memory);
}

pub fn uploadWithStagingBuffer(
    self: *GpuBuffer,
    graphics: Graphics,
    data: []const u8,
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

    try graphics.copyBuffer(self.buffer, staging_buffer.buffer, data.len);
}
