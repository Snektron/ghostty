//! This file deals with the Vulkan swapchain. It is mostly based on the one
//! from Zig-Showdown.
const Swapchain = @This();

const std = @import("std");
const log = std.log.scoped(.vulkan);
const Allocator = std.mem.Allocator;
const vk = @import("vulkan");
const Graphics = @import("Graphics.zig");

const acquire_timeout = 1 * std.time.ns_per_s;

/// This structure models a set of options that need to be considered when (re) creating
/// a swapchain.
pub const CreateInfo = struct {
    /// Whether vsync needs to be enabled. Currently, either mailbox or immediate is
    /// preferred, and vsync is used as fallback option. Setting this field to true
    /// overrides the present mode to .fifo_khr, which is vsync.
    vsync: bool = false,

    /// The desired extent of the swap images. Note that the actual size can differ in
    /// odd cases, for example in race conditions where the window size changes after
    /// (re)init has already been called.
    desired_extent: vk.Extent2D,

    /// The intended usage of the swap images. This is in most cases just .color_attachment_bit.
    swap_image_usage: vk.ImageUsageFlags,
};

handle: vk.SwapchainKHR,
surface_format: vk.SurfaceFormatKHR,
extent: vk.Extent2D,

a: Allocator,
images: []vk.Image,
image_views: []vk.ImageView,
image_index: u32,

/// Attempt to initialize a new swapchain.
pub fn init(
    graphics: Graphics,
    a: Allocator,
    create_info: CreateInfo,
) !Swapchain {
    var self = Swapchain{
        .handle = .null_handle,
        .surface_format = undefined,
        .extent = undefined,
        .a = a,
        .images = try a.alloc(vk.Image, 0),
        .image_views = try a.alloc(vk.ImageView, 0),
        .image_index = undefined,
    };

    self.reinit(graphics, create_info) catch |err| {
        self.deinit(graphics);
        return err;
    };

    return self;
}

/// Recreate the swapchain, where the current swapchain handle is recycled. Note that
/// if this function fails, it can still be called again. Furthermore, deinit needs to
/// be called to deinitialize the swapchain regardless whether this function fails.
pub fn reinit(
    self: *Swapchain,
    graphics: Graphics,
    create_info: CreateInfo,
) !void {
    self.surface_format = try findSurfaceFormat(graphics, create_info, self.a);

    const present_mode = if (create_info.vsync)
        // FIFO is always supported.
        .fifo_khr
    else
        try findPresentMode(graphics, self.a);

    const caps = try graphics.instance.getPhysicalDeviceSurfaceCapabilitiesKHR(graphics.pdev, graphics.surface);
    self.extent = findActualExtent(caps, create_info.desired_extent);

    log.info("creating swapchain with extent {}x{}", .{ self.extent.width, self.extent.height });

    if (!caps.supported_usage_flags.contains(create_info.swap_image_usage)) {
        return error.UnsupportedSwapImageUsage;
    }

    var image_count = caps.min_image_count + 1;
    if (caps.max_image_count > 0) {
        image_count = @min(image_count, caps.max_image_count);
    }

    var queue_families = std.BoundedArray(u32, 2).init(0) catch unreachable;
    queue_families.appendAssumeCapacity(graphics.graphics_queue.family);
    if (graphics.graphics_queue.family != graphics.present_queue.family) {
        queue_families.appendAssumeCapacity(graphics.present_queue.family);
    }

    const old_handle = self.handle;
    self.handle = try graphics.dev.createSwapchainKHR(&.{
        .flags = .{},
        .surface = graphics.surface,
        .min_image_count = image_count,
        .image_format = self.surface_format.format,
        .image_color_space = self.surface_format.color_space,
        .image_extent = self.extent,
        .image_array_layers = 1,
        .image_usage = create_info.swap_image_usage,
        .image_sharing_mode = if (queue_families.len > 1) .concurrent else .exclusive,
        .queue_family_index_count = @intCast(queue_families.buffer.len),
        .p_queue_family_indices = &queue_families.buffer,
        .pre_transform = caps.current_transform,
        .composite_alpha = .{ .opaque_bit_khr = true },
        .present_mode = present_mode,
        .clipped = vk.TRUE,
        .old_swapchain = self.handle,
    }, null);

    // TODO: Destroy the handle *after* acquiring the first frame, the give the
    // presentation engine the opportunity to finish presenting to the old frames.
    // It's technically valid to nuke the swapchain at any point, but it should
    // be a little more efficient.
    graphics.dev.destroySwapchainKHR(old_handle, null);

    self.images = try graphics.dev.getSwapchainImagesAllocKHR(self.handle, self.a);

    try self.createImageViews(graphics);

    self.image_index = undefined;
}

pub fn deinit(self: *Swapchain, graphics: Graphics) void {
    for (self.image_views) |view| {
        graphics.dev.destroyImageView(view, null);
    }

    self.a.free(self.image_views);
    self.a.free(self.images);
    graphics.dev.destroySwapchainKHR(self.handle, null);
    self.* = undefined;
}

pub const PresentState = enum {
    optimal,
    suboptimal,
};

/// Acquire the next image to render to. The resulting image index can be found in `self.image_index`.
/// When the image is ready to be presented to, `image_acquired` will be signalled. This function returns the
/// swapchain state: If the swapchain should be recreated (because the window was moved to a monitor with a different)
/// pixel layout, for example), .suboptimal is returned. When this function returns error.OutOfDateKHR (or .suboptimal)
/// `self.reinit` should be called.
pub fn acquireNextImage(self: *Swapchain, graphics: Graphics, image_acquired: vk.Semaphore) !PresentState {
    const result = try graphics.dev.acquireNextImage(self.handle, acquire_timeout, image_acquired, .null_handle);
    self.image_index = result.image_index;

    return switch (result.result) {
        .success => PresentState.optimal,
        .suboptimal_khr => PresentState.suboptimal,
        .not_ready => unreachable, // Only reachable if timeout is zero
        .timeout => return error.AcquireTimeout,
    };
}

/// Schedule the current swap image (self.image_index) for presentation. `wait_semaphores` is a list of
/// semaphores to wait on before presentation.
pub fn present(self: *Swapchain, graphics: Graphics, wait_semaphores: []const vk.Semaphore) !void {
    _ = try graphics.dev.queuePresentKHR(graphics.present_queue.handle, .{
        .wait_semaphore_count = @intCast(wait_semaphores.len),
        .p_wait_semaphores = wait_semaphores.ptr,
        .swapchain_count = 1,
        .p_swapchains = &self.handle,
        .p_image_indices = &self.image_index,
        .p_results = null,
    });
}

/// Create an image view for every swapchain image, and store them internally.
fn createImageViews(self: *Swapchain, graphics: Graphics) !void {
    for (self.image_views) |view| {
        graphics.dev.destroyImageView(view, null);
    }

    self.image_views = try self.a.realloc(self.image_views, self.images.len);

    // Make sure to destroy successfully created image views when one fails to be created.
    var n_successfully_created: usize = 0;
    errdefer {
        for (self.image_views[0..n_successfully_created]) |view| {
            graphics.dev.destroyImageView(view, null);
        }
    }

    for (self.image_views, 0..) |*view, i| {
        view.* = try graphics.dev.createImageView(&.{
            .flags = .{},
            .image = self.images[i],
            .view_type = .@"2d",
            .format = self.surface_format.format,
            .components = .{ .r = .identity, .g = .identity, .b = .identity, .a = .identity },
            .subresource_range = .{
                .aspect_mask = .{ .color_bit = true },
                .base_mip_level = 0,
                .level_count = 1,
                .base_array_layer = 0,
                .layer_count = 1,
            },
        }, null);
        n_successfully_created = i;
    }
}

/// Query for a surface format that satisfies the requirements in `create_info`.
fn findSurfaceFormat(graphics: Graphics, create_info: CreateInfo, a: Allocator) !vk.SurfaceFormatKHR {
    const surface_formats = try graphics.instance.getPhysicalDeviceSurfaceFormatsAllocKHR(
        graphics.pdev,
        graphics.surface,
        a,
    );
    defer a.free(surface_formats);

    // We need to check whether the intended usage of the swapchain is supported by the surface format.
    // .color_attachment_bit is apparently always supported.

    // According to https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#VUID-VkImageViewCreateInfo-usage-02274
    // transfer_src_bit and transfer_dst_bit are probably right but who knows.
    const required_format_features = vk.FormatFeatureFlags{
        .sampled_image_bit = create_info.swap_image_usage.sampled_bit,
        .storage_image_bit = create_info.swap_image_usage.storage_bit,
        .color_attachment_bit = create_info.swap_image_usage.color_attachment_bit,
        .depth_stencil_attachment_bit = create_info.swap_image_usage.depth_stencil_attachment_bit,
        .transfer_src_bit = create_info.swap_image_usage.transfer_src_bit,
        .transfer_dst_bit = create_info.swap_image_usage.transfer_dst_bit,
    };

    const preferred = vk.SurfaceFormatKHR{
        .format = .b8g8r8a8_srgb, // TODO: Maybe change for 10-bit monitors? Setting? What about the existing srgb setting?
        .color_space = .srgb_nonlinear_khr,
    };
    var surface_format: ?vk.SurfaceFormatKHR = null;

    for (surface_formats) |sfmt| {
        const fprops = graphics.instance.getPhysicalDeviceFormatProperties(graphics.pdev, sfmt.format);
        // According to the spec, swapchain images are always created with optimal tiling.
        const tiling_features = fprops.optimal_tiling_features;
        if (!tiling_features.contains(required_format_features)) {
            continue;
        }

        if (create_info.swap_image_usage.input_attachment_bit and !(tiling_features.color_attachment_bit or tiling_features.depth_stencil_attachment_bit)) {
            continue;
        }

        if (std.meta.eql(sfmt, preferred)) {
            return preferred;
        } else if (surface_format == null) {
            surface_format = sfmt;
        }
    }

    return surface_format orelse error.NoSupportedSurfaceFormat;
}

/// Find a present mode. Mailbox and immediate mode are preferred, and fifo is used as a
/// fallback option.
fn findPresentMode(graphics: Graphics, a: Allocator) !vk.PresentModeKHR {
    const present_modes = try graphics.instance.getPhysicalDeviceSurfacePresentModesAllocKHR(
        graphics.pdev,
        graphics.surface,
        a,
    );
    defer a.free(present_modes);

    const preferred = [_]vk.PresentModeKHR{
        .mailbox_khr,
        .immediate_khr,
    };

    for (preferred) |mode| {
        if (std.mem.indexOfScalar(vk.PresentModeKHR, present_modes, mode) != null) {
            return mode;
        }
    }

    return .fifo_khr;
}

/// Find the actual extent of the swapchain. Note that it may differ from the desired `extent`,
/// for example due to race conditions.
fn findActualExtent(caps: vk.SurfaceCapabilitiesKHR, extent: vk.Extent2D) vk.Extent2D {
    if (caps.current_extent.width != 0xFFFF_FFFF) {
        return caps.current_extent;
    } else {
        return .{
            .width = std.math.clamp(extent.width, caps.min_image_extent.width, caps.max_image_extent.width),
            .height = std.math.clamp(extent.height, caps.min_image_extent.height, caps.max_image_extent.height),
        };
    }
}
