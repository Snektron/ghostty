//! This file contains the common Vulkan rendering state & logic.
pub const Graphics = @This();

const std = @import("std");
const log = std.log.scoped(.vulkan);
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const vk = @import("vulkan");
const glfw = @import("glfw");
const glslang = @import("glslang");
const apprt = @import("../../apprt.zig");
const build_config = @import("../../build_config.zig");
const Vulkan = @import("../Vulkan.zig");

const required_device_extensions = [_][*:0]const u8{vk.extensions.khr_swapchain.name};

const required_dynamic_rendering_features: vk.PhysicalDeviceDynamicRenderingFeatures = .{
    .dynamic_rendering = vk.TRUE,
};

const apis = &.{
    vk.features.version_1_0,
    vk.features.version_1_1,
    vk.features.version_1_3,
    vk.extensions.khr_surface,
    vk.extensions.khr_swapchain,
};

const BaseDispatch = vk.BaseWrapper(apis);
const Instance = vk.InstanceProxy(apis);
const Device = vk.DeviceProxy(apis);

const DestructionQueue = std.ArrayListUnmanaged(DestructionQueueItem);
const DestructionQueueItem = union(enum) {
    swapchain: vk.SwapchainKHR,
    buffer: vk.Buffer,
    memory: vk.DeviceMemory,
    image_View: vk.ImageView,
};

alloc: Allocator,

vkb: BaseDispatch,
instance: Instance,

surface: vk.SurfaceKHR,

pdev: vk.PhysicalDevice,
props: vk.PhysicalDeviceProperties,
mem_props: vk.PhysicalDeviceMemoryProperties,
dev: Device,

graphics_queue: Queue,
present_queue: Queue,

copy_cmd_pool: vk.CommandPool,
copy_cmd_buf: vk.CommandBuffer,
copy_fence: vk.Fence,

destruction_queues: [Vulkan.frames_in_flight]DestructionQueue = .{.{}} ** Vulkan.frames_in_flight,

frame_nr: usize = 0,

fn getInstanceProcAddrWrapper(instance: vk.Instance, proc_name: [*:0]const u8) vk.PfnVoidFunction {
    return @ptrCast(glfw.getInstanceProcAddress(@ptrFromInt(@intFromEnum(instance)), proc_name));
}

fn addLayerIfSupported(
    enabled_layers: *std.ArrayList([*:0]const u8),
    available_layers: []const vk.LayerProperties,
    layer: [:0]const u8,
) !void {
    for (available_layers) |props| {
        const layer_name = std.mem.sliceTo(&props.layer_name, 0);
        if (std.mem.eql(u8, layer_name, layer)) {
            try enabled_layers.append(layer.ptr);
            log.debug("enabling layer '{s}'", .{layer});
            return;
        }
    }

    log.warn("could not enable layer '{s}'", .{layer});
}

pub fn init(alloc: Allocator, surface: *apprt.Surface) !Graphics {
    const vkb = try BaseDispatch.load(getInstanceProcAddrWrapper);

    const instance_exts = switch (apprt.runtime) {
        apprt.glfw => blk: {
            if (!glfw.vulkanSupported()) {
                log.err("GLFW does not support Vulkan. Is the vulkan loader available and in the library path?", .{});
                return error.VulkanUnsupported;
            }

            break :blk glfw.getRequiredInstanceExtensions() orelse {
                log.err("failed to query GLFW required instance extensions", .{});
                return error.VulkanUnsupported;
            };
        },
        else => @compileError("unsupported app runtime for Vulkan"),
    };

    var instance_layers = std.ArrayList([*:0]const u8).init(alloc);
    defer instance_layers.deinit();

    if (builtin.mode == .Debug) {
        const available_layers = try vkb.enumerateInstanceLayerPropertiesAlloc(alloc);
        defer alloc.free(available_layers);

        try addLayerIfSupported(&instance_layers, available_layers, "VK_LAYER_KHRONOS_validation");
    }

    const version = vk.makeApiVersion(0, build_config.version.major, build_config.version.minor, build_config.version.patch);

    const vk_inst = try vkb.createInstance(&.{
        .p_application_info = &.{
            .p_application_name = "Ghostty",
            .application_version = version,
            .p_engine_name = "Ghostty Vulkan Renderer",
            .engine_version = version,
            // We require Vulkan 1.3 for dynamic rendering.
            .api_version = vk.API_VERSION_1_3,
        },
        .enabled_extension_count = @intCast(instance_exts.len),
        .pp_enabled_extension_names = instance_exts.ptr,
        .enabled_layer_count = @intCast(instance_layers.items.len),
        .pp_enabled_layer_names = instance_layers.items.ptr,
    }, null);

    const vki = try alloc.create(Instance.Wrapper);
    errdefer alloc.destroy(vki);
    vki.* = try Instance.Wrapper.load(vk_inst, vkb.dispatch.vkGetInstanceProcAddr);
    const instance = Instance.init(vk_inst, vki);
    errdefer instance.destroyInstance(null);

    const vk_surface = switch (apprt.runtime) {
        apprt.glfw => blk: {
            var vk_surface: vk.SurfaceKHR = undefined;
            if (glfw.createWindowSurface(instance.handle, surface.window, null, &vk_surface) != @intFromEnum(vk.Result.success)) {
                log.err("Failed to initialize Vulkan surface from GLFW window", .{});
                return error.SurfaceInitFailed;
            }
            break :blk vk_surface;
        },
        else => unreachable,
    };

    const candidate = try DeviceCandidate.pick(instance, vk_surface, alloc);

    log.info("initializing Vulkan device '{s}'", .{std.mem.sliceTo(&candidate.props.device_name, 0)});
    const vk_dev = try candidate.initDevice(instance);
    const vkd = try alloc.create(Device.Wrapper);
    errdefer alloc.destroy(vkd);
    vkd.* = try Device.Wrapper.load(vk_dev, instance.wrapper.dispatch.vkGetDeviceProcAddr);
    const dev = Device.init(vk_dev, vkd);
    errdefer dev.destroyDevice(null);

    const graphics_queue = Queue.init(dev, candidate.queues.graphics_family, candidate.queues.graphics_index);
    const present_queue = Queue.init(dev, candidate.queues.present_family, candidate.queues.present_index);
    const mem_props = instance.getPhysicalDeviceMemoryProperties(candidate.pdev);

    const copy_cmd_pool = try dev.createCommandPool(&.{
        .flags = .{},
        .queue_family_index = graphics_queue.family,
    }, null);
    errdefer dev.destroyCommandPool(copy_cmd_pool, null);

    var copy_cmd_buf: vk.CommandBuffer = undefined;
    try dev.allocateCommandBuffers(&.{
        .command_pool = copy_cmd_pool,
        .level = .primary,
        .command_buffer_count = 1,
    }, @ptrCast(&copy_cmd_buf));

    const copy_fence = try dev.createFence(&.{}, null);
    errdefer dev.destroyFence(copy_fence, null);

    return .{
        .alloc = alloc,
        .vkb = vkb,
        .instance = instance,
        .surface = vk_surface,
        .pdev = candidate.pdev,
        .props = candidate.props,
        .mem_props = mem_props,
        .dev = dev,
        .graphics_queue = graphics_queue,
        .present_queue = present_queue,
        .copy_cmd_pool = copy_cmd_pool,
        .copy_cmd_buf = copy_cmd_buf,
        .copy_fence = copy_fence,
    };
}

pub fn deinit(self: *Graphics) void {
    for (&self.destruction_queues) |*queue| {
        self.processDestructionQueue(queue);
        queue.deinit(self.alloc);
    }

    self.dev.destroyFence(self.copy_fence, null);
    // Also destroys the copy_cmd_buf.
    self.dev.destroyCommandPool(self.copy_cmd_pool, null);

    self.dev.destroyDevice(null);
    self.alloc.destroy(self.dev.wrapper);

    self.instance.destroySurfaceKHR(self.surface, null);

    self.instance.destroyInstance(null);
    self.alloc.destroy(self.instance.wrapper);

    self.* = undefined;
}

/// Should be called after waiting for the current frame.
pub fn beginFrame(self: *Graphics) void {
    // Go through the pending deletions buffer and process all those items.
    const queue = &self.destruction_queues[self.frameIndex()];
    self.processDestructionQueue(queue);
}

/// Should be called at the end of a frame.
pub fn endFrame(self: *Graphics) void {
    self.frame_nr += 1;
}

pub fn frameIndex(self: Graphics) usize {
    return self.frame_nr % Vulkan.frames_in_flight;
}

fn processDestructionQueue(self: *Graphics, queue: *DestructionQueue) void {
    const dev = self.dev;
    for (queue.items) |item| {
        switch (item) {
            .swapchain => |swapchain| dev.destroySwapchainKHR(swapchain, null),
            .buffer => |buffer| dev.destroyBuffer(buffer, null),
            .memory => |memory| dev.freeMemory(memory, null),
            .image_view => |image_view| dev.destroyImageView(image_view, null),
        }
    }
    queue.items.len = 0;
}

pub fn destroyDeferred(self: *Graphics, item: DestructionQueueItem) !void {
    try self.destruction_queues[self.frameIndex()].append(self.alloc, item);
}

/// Mostly taken from shadertoy.zig's spirvFromGlsl
pub fn compileShader(
    graphics: Graphics,
    src: [:0]const u8,
    stage: enum { frag, vert },
) !vk.ShaderModule {
    const c = glslang.c;
    const glsl_stage: c_uint = switch (stage) {
        .vert => c.GLSLANG_STAGE_VERTEX,
        .frag => c.GLSLANG_STAGE_FRAGMENT,
    };
    const input: c.glslang_input_t = .{
        .language = c.GLSLANG_SOURCE_GLSL,
        .stage = glsl_stage,
        .client = c.GLSLANG_CLIENT_VULKAN,
        .client_version = c.GLSLANG_TARGET_VULKAN_1_3,
        .target_language = c.GLSLANG_TARGET_SPV,
        .target_language_version = c.GLSLANG_TARGET_SPV_1_5,
        .code = src.ptr,
        .default_version = 100,
        .default_profile = c.GLSLANG_NO_PROFILE,
        .force_default_version_and_profile = 0,
        .forward_compatible = 0,
        .messages = c.GLSLANG_MSG_DEFAULT_BIT,
        .resource = c.glslang_default_resource(),
    };

    const shader = try glslang.Shader.create(&input);
    defer shader.delete();

    shader.preprocess(&input) catch |err| {
        log.err("failed to preprocess shader:", .{});
        log.err("shader info: {s}", .{try shader.getInfoLog()});
        log.err("shader debug info: {s}", .{try shader.getDebugInfoLog()});
        return err;
    };
    shader.parse(&input) catch |err| {
        log.err("failed to parse shader:", .{});
        log.err("shader info: {s}", .{try shader.getInfoLog()});
        log.err("shader debug info: {s}", .{try shader.getDebugInfoLog()});
        return err;
    };

    const program = try glslang.Program.create();
    defer program.delete();
    program.addShader(shader);
    program.link(c.GLSLANG_MSG_SPV_RULES_BIT | c.GLSLANG_MSG_VULKAN_RULES_BIT) catch |err| {
        log.err("failed to link program:", .{});
        log.err("program info: {s}", .{try program.getInfoLog()});
        log.err("program debug info: {s}", .{try program.getDebugInfoLog()});
        return err;
    };
    program.spirvGenerate(glsl_stage);

    const spv = try graphics.alloc.alloc(u32, program.spirvGetSize());
    defer graphics.alloc.free(spv);

    program.spirvGet(spv);

    return try graphics.dev.createShaderModule(&.{
        .flags = .{},
        .code_size = spv.len * @sizeOf(u32),
        .p_code = spv.ptr,
    }, null);
}

pub fn findMemoryTypeIndex(self: Graphics, memory_type_bits: u32, flags: vk.MemoryPropertyFlags) !u32 {
    for (self.mem_props.memory_types[0..self.mem_props.memory_type_count], 0..) |mem_type, i| {
        if (memory_type_bits & (@as(u32, 1) << @truncate(i)) != 0 and mem_type.property_flags.contains(flags)) {
            return @truncate(i);
        }
    }

    return error.NoSuitableMemoryType;
}

pub fn allocate(self: Graphics, requirements: vk.MemoryRequirements, flags: vk.MemoryPropertyFlags) !vk.DeviceMemory {
    return try self.dev.allocateMemory(&.{
        .allocation_size = requirements.size,
        .memory_type_index = try self.findMemoryTypeIndex(requirements.memory_type_bits, flags),
    }, null);
}

pub fn copyBuffer(
    self: Graphics,
    dst: vk.Buffer,
    src: vk.Buffer,
    size: usize,
) !void {
    const dev = self.dev;

    try dev.resetCommandPool(self.copy_cmd_pool, .{});
    try dev.beginCommandBuffer(self.copy_cmd_buf, &.{
        .flags = .{ .one_time_submit_bit = true },
    });

    const region: vk.BufferCopy = .{
        .src_offset = 0,
        .dst_offset = 0,
        .size = size,
    };
    dev.cmdCopyBuffer(self.copy_cmd_buf, src, dst, 1, @ptrCast(&region));

    try dev.endCommandBuffer(self.copy_cmd_buf);

    const submit_info: vk.SubmitInfo = .{
        .command_buffer_count = 1,
        .p_command_buffers = @ptrCast(&self.copy_cmd_buf),
        .p_wait_dst_stage_mask = &.{.{}},
    };

    try dev.queueSubmit(self.graphics_queue.handle, 1, @ptrCast(&submit_info), self.copy_fence);

    const result = try dev.waitForFences(1, @ptrCast(&self.copy_fence), vk.TRUE, 1 * std.time.ns_per_s);
    if (result == .timeout) {
        return error.Timeout;
    }
    try dev.resetFences(1, @ptrCast(&self.copy_fence));
}

pub const Queue = struct {
    handle: vk.Queue,
    family: u32,
    index: u32,

    fn init(device: Device, family: u32, index: u32) Queue {
        return .{
            .handle = device.getDeviceQueue(family, index),
            .family = family,
            .index = index,
        };
    }
};

const DeviceCandidate = struct {
    const QueueAllocation = struct {
        graphics_family: u32,
        graphics_index: u32,
        present_family: u32,
        present_index: u32,
    };

    pdev: vk.PhysicalDevice,
    props: vk.PhysicalDeviceProperties,
    queues: QueueAllocation,

    fn pick(
        instance: Instance,
        surface: vk.SurfaceKHR,
        alloc: Allocator,
    ) !DeviceCandidate {
        const pdevs = try instance.enumeratePhysicalDevicesAlloc(alloc);
        defer alloc.free(pdevs);

        var best_score: usize = 0;
        var best_candidate: ?DeviceCandidate = null;

        for (pdevs) |pdev| {
            const candidate = try checkSuitable(instance, pdev, surface, alloc) orelse {
                continue;
            };

            var score: usize = 0;

            // TODO: Currently, device selection prefers a discrete GPU over an integrated
            // GPU. This may not always be wanted, however: Picking the dedicate GPU over
            // an integrated GPU on a laptop may drain the battery faster, which would
            // be a waste if the extra rendering power is not required. Should we always
            // pick an integrated GPU over a dedicated GPU here, or should we turn this
            // into a configuration option?

            switch (candidate.props.device_type) {
                .discrete_gpu => score += 1000,
                .virtual_gpu => score += 750,
                .integrated_gpu => score += 500,
                else => score += 250,
            }

            if (best_candidate == null or score > best_score) {
                best_score = score;
                best_candidate = candidate;
            }
        }

        return best_candidate orelse error.NoSuitableDevice;
    }

    fn initDevice(candidate: DeviceCandidate, instance: Instance) !vk.Device {
        const same_family = candidate.queues.graphics_family == candidate.queues.present_family;
        const same_index = candidate.queues.graphics_index == candidate.queues.present_family;

        const priority = [_]f32{1};
        const qci = [_]vk.DeviceQueueCreateInfo{
            .{
                .queue_family_index = candidate.queues.graphics_family,
                // If the families are the same but the indices are different, we are creating
                // 2 separate queues in the same family.
                .queue_count = if (same_family and !same_index) 2 else 1,
                .p_queue_priorities = &priority,
            },
            .{
                .queue_family_index = candidate.queues.present_family,
                .queue_count = 1,
                .p_queue_priorities = &priority,
            },
        };

        const queue_count: u32 = if (same_family) 1 else 2;

        return try instance.createDevice(candidate.pdev, &.{
            .p_next = @ptrCast(&required_dynamic_rendering_features),
            .queue_create_info_count = queue_count,
            .p_queue_create_infos = &qci,
            .enabled_extension_count = required_device_extensions.len,
            .pp_enabled_extension_names = @ptrCast(&required_device_extensions),
        }, null);
    }

    fn checkSuitable(
        instance: Instance,
        pdev: vk.PhysicalDevice,
        surface: vk.SurfaceKHR,
        alloc: Allocator,
    ) !?DeviceCandidate {
        if (!try checkExtensionSupport(instance, pdev, alloc)) {
            return null;
        }

        if (!try checkFeatureSupport(instance, pdev)) {
            return null;
        }

        if (!try checkSurfaceSupport(instance, pdev, surface)) {
            return null;
        }

        if (try allocateQueues(instance, pdev, surface, alloc)) |allocation| {
            const props = instance.getPhysicalDeviceProperties(pdev);
            return .{
                .pdev = pdev,
                .props = props,
                .queues = allocation,
            };
        }

        return null;
    }

    fn allocateQueues(
        instance: Instance,
        pdev: vk.PhysicalDevice,
        surface: vk.SurfaceKHR,
        alloc: Allocator,
    ) !?QueueAllocation {
        const families = try instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(pdev, alloc);
        defer alloc.free(families);

        var maybe_graphics_family: ?u32 = null;
        var maybe_present_family: ?u32 = null;

        for (families, 0..) |properties, i| {
            const family: u32 = @intCast(i);

            if (maybe_graphics_family == null and properties.queue_flags.graphics_bit) {
                maybe_graphics_family = family;
            }

            if (maybe_present_family == null and (try instance.getPhysicalDeviceSurfaceSupportKHR(pdev, family, surface)) == vk.TRUE) {
                maybe_present_family = family;
            }
        }

        const graphics_family = maybe_graphics_family orelse return null;
        const present_family = maybe_present_family orelse return null;

        const present_index: u32 = if (graphics_family == present_family and
            families[graphics_family].queue_count >= 2)
            1
        else
            0;

        return .{
            .graphics_family = graphics_family,
            .graphics_index = 0,
            .present_family = present_family,
            .present_index = present_index,
        };
    }

    fn checkSurfaceSupport(instance: Instance, pdev: vk.PhysicalDevice, surface: vk.SurfaceKHR) !bool {
        var format_count: u32 = undefined;
        _ = try instance.getPhysicalDeviceSurfaceFormatsKHR(pdev, surface, &format_count, null);

        var present_mode_count: u32 = undefined;
        _ = try instance.getPhysicalDeviceSurfacePresentModesKHR(pdev, surface, &present_mode_count, null);

        return format_count > 0 and present_mode_count > 0;
    }

    fn checkExtensionSupport(
        instance: Instance,
        pdev: vk.PhysicalDevice,
        alloc: Allocator,
    ) !bool {
        const propsv = try instance.enumerateDeviceExtensionPropertiesAlloc(pdev, null, alloc);
        defer alloc.free(propsv);

        for (required_device_extensions) |ext| {
            for (propsv) |props| {
                if (std.mem.eql(u8, std.mem.span(ext), std.mem.sliceTo(&props.extension_name, 0))) {
                    break;
                }
            } else {
                return false;
            }
        }

        return true;
    }

    fn checkFeatureSupport(
        instance: Instance,
        pdev: vk.PhysicalDevice,
    ) !bool {
        var dynamic_rendering_features: vk.PhysicalDeviceDynamicRenderingFeatures = .{};
        var features2: vk.PhysicalDeviceFeatures2 = .{
            .p_next = @ptrCast(&dynamic_rendering_features),
            .features = .{},
        };
        instance.getPhysicalDeviceFeatures2(pdev, &features2);

        inline for (std.meta.fields(vk.PhysicalDeviceDynamicRenderingFeatures)) |field| {
            if (field.type == vk.Bool32 and
                @field(required_dynamic_rendering_features, field.name) == vk.TRUE and
                @field(dynamic_rendering_features, field.name) == vk.FALSE)
            {
                return false;
            }
        }

        return true;
    }
};
