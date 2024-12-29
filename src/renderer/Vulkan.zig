//! Rendering implementation for Vulkan.
pub const Vulkan = @This();

const std = @import("std");
const assert = std.debug.assert;
const log = std.log.scoped(.vulkan);
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const builtin = @import("builtin");
const glfw = @import("glfw");
const vk = @import("vulkan");

const configpkg = @import("../config.zig");
const apprt = @import("../apprt.zig");
const font = @import("../font/main.zig");
const renderer = @import("../renderer.zig");
const terminal = @import("../terminal/main.zig");

const Graphics = @import("vulkan/Graphics.zig");
const Swapchain = @import("vulkan/Swapchain.zig");

const frame_timeout = 1 * std.time.ns_per_s;
const frames_in_flight = 3;

a: Allocator,

config: DerivedConfig,

/// The actual foreground color. May differ from the config foreground color if
/// changed by a terminal application
foreground_color: terminal.color.RGB,

/// The actual background color. May differ from the config background color if
/// changed by a terminal application
background_color: terminal.color.RGB,

/// The actual cursor color. May differ from the config cursor color if changed
/// by a terminal application
cursor_color: ?terminal.color.RGB,

/// When `cursor_color` is null, swap the foreground and background colors of
/// the cell under the cursor for the cursor color. Otherwise, use the default
/// foreground color as the cursor color.
cursor_invert: bool,

/// Current background to draw. This may not match self.background if the
/// terminal is in reversed mode.
draw_background: terminal.color.RGB,

/// The mailbox for communicating with the window.
surface_mailbox: apprt.surface.Mailbox,

/// Vulkan GPU state.
/// Initialization of the Vulkan GPU state is deferred until `finalizeSurfaceInit`,
/// as we require the surface in order to pick the right rendering device.
gpu_state: ?GPUState = null,

const GPUState = struct {
    graphics: Graphics,
    swapchain: Swapchain,
    frames: [frames_in_flight]Frame,
    frame_nr: usize = 0,
};

pub const DerivedConfig = struct {
    arena: ArenaAllocator,

    cursor_color: ?terminal.color.RGB,
    cursor_invert: bool,
    background: terminal.color.RGB,
    background_opacity: f64,
    foreground: terminal.color.RGB,

    pub fn init(
        gpa: Allocator,
        config: *const configpkg.Config,
    ) !DerivedConfig {
        var arena = ArenaAllocator.init(gpa);
        errdefer arena.deinit();

        const cursor_invert = config.@"cursor-invert-fg-bg";

        return .{
            .cursor_color = if (!cursor_invert and config.@"cursor-color" != null)
                config.@"cursor-color".?.toTerminalRGB()
            else
                null,
            .cursor_invert = cursor_invert,
            .background = config.background.toTerminalRGB(),
            .background_opacity = @max(0, @min(1, config.@"background-opacity")),
            .foreground = config.foreground.toTerminalRGB(),
            .arena = arena,
        };
    }

    pub fn deinit(self: *DerivedConfig) void {
        self.arena.deinit();
    }
};

pub fn init(a: Allocator, options: renderer.Options) !Vulkan {
    return .{
        .a = a,
        .config = options.config,
        .foreground_color = options.config.foreground,
        .background_color = options.config.background,
        .cursor_color = options.config.cursor_color,
        .cursor_invert = options.config.cursor_invert,
        .draw_background = options.config.background,
        .surface_mailbox = options.surface_mailbox,
    };
}

pub fn deinit(self: *Vulkan) void {
    if (self.gpu_state) |*state| {
        self.waitForAllFrames() catch |err| switch (err) {
            error.Timeout => {
                log.err("frame timeout while deinitializing", .{});
                std.time.sleep(frame_timeout);
            },
            // We usually cannot really recover from any other error here,
            // but lets try to continue deinitializing anyway...
            else => {},
        };

        // Waiting on the fences above does not actually guarantee that the swapchain has finished
        // rendering, so explicitly wait for the queue to be finalized before destroying the
        // (potentially in use) frame's resources.
        state.graphics.dev.queueWaitIdle(state.graphics.present_queue.handle) catch {};

        for (&state.frames) |*frame| {
            frame.deinit(state.graphics);
        }

        state.swapchain.deinit(state.graphics);
        state.graphics.deinit();
    }
    self.* = undefined;
}

/// Returns the hints that we want for this
pub fn glfwWindowHints(config: *const configpkg.Config) glfw.Window.Hints {
    return .{
        .client_api = .no_api,
        .cocoa_graphics_switching = builtin.os.tag == .macos,
        .cocoa_retina_framebuffer = true,
        .transparent_framebuffer = config.@"background-opacity" < 1,
    };
}

pub fn surfaceInit(surface: *apprt.Surface) !void {
    _ = surface;

    // We don't do anything else here because we want to set everything
    // else up during actual initialization.
}

pub fn finalizeSurfaceInit(self: *Vulkan, surface: *apprt.Surface) !void {
    var graphics = try Graphics.init(self.a, surface);
    errdefer graphics.deinit();

    var swapchain = try Swapchain.init(graphics, self.a, .{
        .vsync = true,
        .desired_extent = .{
            // TODO: Intitial extent? Or should we defer swapchain initialization?
            .width = 0,
            .height = 0,
        },
        .swap_image_usage = .{
            .color_attachment_bit = true,
        },
    });
    errdefer swapchain.deinit(graphics);

    var frames: [frames_in_flight]Frame = undefined;
    var n_successfully_created: usize = 0;

    errdefer {
        for (frames[0..n_successfully_created]) |*frame| {
            frame.deinit(graphics);
        }
    }

    for (&frames) |*frame| {
        frame.* = try Frame.init(graphics);
        n_successfully_created += 1;
    }

    self.gpu_state = .{
        .graphics = graphics,
        .swapchain = swapchain,
        .frames = frames,
    };
}

pub fn displayUnrealized(self: *Vulkan) void {
    _ = self;
    // TODO
}

pub fn displayRealize(self: *Vulkan) !void {
    _ = self;
    // TODO
}

pub fn threadEnter(self: *const Vulkan, surface: *apprt.Surface) !void {
    _ = self;
    _ = surface;

    // Vulkan requires no per-thread state.
}

pub fn threadExit(self: *const Vulkan) void {
    _ = self;

    // Vulkan requires no per-thread state.
}

pub fn hasAnimations(self: *const Vulkan) bool {
    _ = self;
    return false; // TODO: Custom shaders.
}

pub fn hasVsync(self: *const Vulkan) bool {
    _ = self;
    return false; // TODO
}

pub fn markDirty(self: *Vulkan) void {
    _ = self;
    // TODO
}

pub fn setFocus(self: *Vulkan, focus: bool) !void {
    _ = self;
    _ = focus;
    // TODO
}

pub fn setVisible(self: *Vulkan, visible: bool) void {
    _ = self;
    _ = visible;
    // TODO
}

pub fn setFontGrid(self: *Vulkan, grid: *font.SharedGrid) void {
    _ = self;
    _ = grid;
    // TODO
}

pub fn updateFrame(
    self: *Vulkan,
    surface: *apprt.Surface,
    state: *renderer.State,
    cursor_blink_visible: bool,
) !void {
    _ = self;
    _ = surface;
    _ = state;
    _ = cursor_blink_visible;
}

pub fn rebuildCells(
    self: *Vulkan,
    rebuild: bool,
    screen: *terminal.Screen,
    screen_type: terminal.ScreenType,
    mouse: renderer.State.Mouse,
    preedit: ?renderer.State.Preedit,
    cursor_style_: ?renderer.CursorStyle,
    color_palette: *const terminal.color.Palette,
) !void {
    _ = self;
    _ = rebuild;
    _ = screen;
    _ = screen_type;
    _ = mouse;
    _ = preedit;
    _ = cursor_style_;
    _ = color_palette;
    // TODO
}

pub fn changeConfig(self: *Vulkan, config: *DerivedConfig) !void {
    _ = self;
    _ = config;
    // TODO
}

pub fn setScreenSize(
    self: *Vulkan,
    size: renderer.Size,
) !void {
    _ = self;
    _ = size;
    // TODO
}

pub fn drawFrame(self: *Vulkan, surface: *apprt.Surface) !void {
    _ = surface;
    const state = if (self.gpu_state) |*state| state else return;
    const dev = state.graphics.dev;

    const frame = &state.frames[state.frame_nr % frames_in_flight];
    try frame.wait(state.graphics);

    const cmd_buf = frame.cmd_buf;

    const present_state = state.swapchain.acquireNextImage(state.graphics, frame.image_acquired) catch |err| switch (err) {
        error.OutOfDateKHR => .suboptimal,
        else => |other| return other,
    };

    if (present_state == .suboptimal) {
        // TODO: Check deferred size?
        log.warn("TODO: Resize swapchain", .{});
    }

    try dev.resetCommandPool(frame.cmd_pool, .{});
    try dev.beginCommandBuffer(cmd_buf, &.{
        .flags = .{ .one_time_submit_bit = true },
    });

    {
        // Dynamic rendering does not automatically transition the render pass into the present state.
        // We have to do that manually...
        const image_mbar: vk.ImageMemoryBarrier = .{
            .src_access_mask = .{},
            .dst_access_mask = .{ .color_attachment_write_bit = true },
            // At this point, the image may actually either be in truly undefined layout,
            // or in present_src_khr layout. We don't care about the contents, so
            // just use undefined here.
            .old_layout = .undefined,
            .new_layout = .color_attachment_optimal,
            .src_queue_family_index = state.graphics.present_queue.family,
            .dst_queue_family_index = state.graphics.graphics_queue.family,
            .image = state.swapchain.images[state.swapchain.image_index],
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        };
        dev.cmdPipelineBarrier(
            cmd_buf,
            .{ .top_of_pipe_bit = true },
            .{ .color_attachment_output_bit = true },
            .{},
            0,
            null,
            0,
            null,
            1,
            @ptrCast(&image_mbar),
        );
    }

    const color_attachment: vk.RenderingAttachmentInfo = .{
        .image_view = state.swapchain.image_views[state.swapchain.image_index],
        .image_layout = .color_attachment_optimal,
        .resolve_mode = .{},
        .resolve_image_layout = .undefined, // Not used.
        .load_op = .clear,
        .store_op = .store,
        .clear_value = .{ .color = .{
            .float_32 = .{
                @floatCast(@as(f32, @floatFromInt(self.draw_background.r)) / 255 * self.config.background_opacity),
                @floatCast(@as(f32, @floatFromInt(self.draw_background.g)) / 255 * self.config.background_opacity),
                @floatCast(@as(f32, @floatFromInt(self.draw_background.b)) / 255 * self.config.background_opacity),
                @floatCast(self.config.background_opacity),
            },
        } },
    };

    dev.cmdBeginRendering(cmd_buf, &.{
        .render_area = .{
            .offset = .{ .x = 0, .y = 0 },
            .extent = state.swapchain.extent,
        },
        .layer_count = 1,
        .view_mask = 0,
        .color_attachment_count = 1,
        .p_color_attachments = @ptrCast(&color_attachment),
    });

    dev.cmdEndRendering(cmd_buf);

    {
        // Dynamic rendering does not automatically transition the render pass into the present state.
        // We have to do that manually...
        const image_mbar: vk.ImageMemoryBarrier = .{
            .src_access_mask = .{ .color_attachment_write_bit = true },
            .dst_access_mask = .{},
            .old_layout = .color_attachment_optimal,
            .new_layout = .present_src_khr,
            .src_queue_family_index = state.graphics.graphics_queue.family,
            .dst_queue_family_index = state.graphics.present_queue.family,
            .image = state.swapchain.images[state.swapchain.image_index],
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        };
        dev.cmdPipelineBarrier(
            cmd_buf,
            .{ .color_attachment_output_bit = true },
            .{ .bottom_of_pipe_bit = true },
            .{},
            0,
            null,
            0,
            null,
            1,
            @ptrCast(&image_mbar),
        );
    }

    try dev.endCommandBuffer(cmd_buf);

    const submit_info: vk.SubmitInfo = .{
        .wait_semaphore_count = 1,
        .p_wait_semaphores = @ptrCast(&frame.image_acquired),
        .p_wait_dst_stage_mask = &.{.{ .bottom_of_pipe_bit = true }},
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&cmd_buf),
        .signal_semaphore_count = 1,
        .p_signal_semaphores = @ptrCast(&frame.render_finished),
    };
    try dev.queueSubmit(state.graphics.graphics_queue.handle, 1, @ptrCast(&submit_info), frame.frame_fence);

    try state.swapchain.present(state.graphics, &.{frame.render_finished});

    state.frame_nr += 1;
}

fn waitForAllFrames(self: *Vulkan) !void {
    const state = if (self.gpu_state) |*state| state else return;
    for (state.frames) |frame| try frame.wait(state.graphics);
}

const Frame = struct {
    image_acquired: vk.Semaphore,
    render_finished: vk.Semaphore,
    frame_fence: vk.Fence,
    cmd_pool: vk.CommandPool,
    cmd_buf: vk.CommandBuffer,

    fn init(graphics: Graphics) !Frame {
        const image_acquired = try graphics.dev.createSemaphore(&.{}, null);
        errdefer graphics.dev.destroySemaphore(image_acquired, null);

        const render_finished = try graphics.dev.createSemaphore(&.{}, null);
        errdefer graphics.dev.destroySemaphore(render_finished, null);

        const frame_fence = try graphics.dev.createFence(&.{ .flags = .{ .signaled_bit = true } }, null);
        errdefer graphics.dev.destroyFence(frame_fence, null);

        const cmd_pool = try graphics.dev.createCommandPool(&.{
            .flags = .{},
            .queue_family_index = graphics.graphics_queue.family,
        }, null);
        errdefer graphics.dev.destroyCommandPool(cmd_pool, null);

        var cmd_buf: vk.CommandBuffer = undefined;
        try graphics.dev.allocateCommandBuffers(&.{
            .command_pool = cmd_pool,
            .level = .primary,
            .command_buffer_count = 1,
        }, @ptrCast(&cmd_buf));

        return .{
            .image_acquired = image_acquired,
            .render_finished = render_finished,
            .frame_fence = frame_fence,
            .cmd_pool = cmd_pool,
            .cmd_buf = cmd_buf,
        };
    }

    fn deinit(self: *Frame, graphics: Graphics) void {
        // Destroying a command pool will also free its associated command buffers.
        graphics.dev.destroyCommandPool(self.cmd_pool, null);
        graphics.dev.destroyFence(self.frame_fence, null);
        graphics.dev.destroySemaphore(self.render_finished, null);
        graphics.dev.destroySemaphore(self.image_acquired, null);
        self.* = undefined;
    }

    fn wait(self: Frame, graphics: Graphics) !void {
        const result = try graphics.dev.waitForFences(1, @ptrCast(&self.frame_fence), vk.TRUE, frame_timeout);
        if (result == .timeout) {
            return error.Timeout;
        }

        try graphics.dev.resetFences(1, @ptrCast(&self.frame_fence));
    }
};
