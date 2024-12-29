//! This file contains the common Vulkan rendering state & logic.
pub const Graphics = @This();

const std = @import("std");
const log = std.log.scoped(.vulkan);
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");
const vk = @import("vulkan");
const glfw = @import("glfw");
const apprt = @import("../../apprt.zig");
const build_config = @import("../../build_config.zig");

const required_device_extensions = [_][*:0]const u8{vk.extensions.khr_swapchain.name};

const apis = &.{
    vk.features.version_1_0,
    vk.extensions.khr_surface,
    vk.extensions.khr_swapchain,
};

const BaseDispatch = vk.BaseWrapper(apis);
const Instance = vk.InstanceProxy(apis);
const Device = vk.DeviceProxy(apis);

a: Allocator,

vkb: BaseDispatch,
instance: Instance,

surface: vk.SurfaceKHR,

pdev: vk.PhysicalDevice,
props: vk.PhysicalDeviceProperties,
mem_props: vk.PhysicalDeviceMemoryProperties,
dev: Device,

graphics_queue: Queue,
present_queue: Queue,

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

pub fn init(a: Allocator, surface: *apprt.Surface) !Graphics {
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

    var instance_layers = std.ArrayList([*:0]const u8).init(a);
    defer instance_layers.deinit();

    if (builtin.mode == .Debug) {
        const available_layers = try vkb.enumerateInstanceLayerPropertiesAlloc(a);
        defer a.free(available_layers);

        try addLayerIfSupported(&instance_layers, available_layers, "VK_LAYER_KHRONOS_validation");
    }

    const version = vk.makeApiVersion(0, build_config.version.major, build_config.version.minor, build_config.version.patch);

    const vk_inst = try vkb.createInstance(&.{
        .p_application_info = &.{
            .p_application_name = "Ghostty",
            .application_version = version,
            .p_engine_name = "Ghostty Vulkan Renderer",
            .engine_version = version,
            .api_version = vk.API_VERSION_1_0,
        },
        .enabled_extension_count = @intCast(instance_exts.len),
        .pp_enabled_extension_names = instance_exts.ptr,
        .enabled_layer_count = @intCast(instance_layers.items.len),
        .pp_enabled_layer_names = instance_layers.items.ptr,
    }, null);

    const vki = try a.create(Instance.Wrapper);
    errdefer a.destroy(vki);
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

    const candidate = try DeviceCandidate.pick(instance, vk_surface, a);

    log.info("initializing Vulkan device '{s}'", .{std.mem.sliceTo(&candidate.props.device_name, 0)});
    const vk_dev = try candidate.initDevice(instance);
    const vkd = try a.create(Device.Wrapper);
    errdefer a.destroy(vkd);
    vkd.* = try Device.Wrapper.load(vk_dev, instance.wrapper.dispatch.vkGetDeviceProcAddr);
    const dev = Device.init(vk_dev, vkd);
    errdefer dev.destroyDevice(null);

    const graphics_queue = Queue.init(dev, candidate.queues.graphics_family, candidate.queues.graphics_index);
    const present_queue = Queue.init(dev, candidate.queues.present_family, candidate.queues.present_index);
    const mem_props = instance.getPhysicalDeviceMemoryProperties(candidate.pdev);

    return .{
        .a = a,
        .vkb = vkb,
        .instance = instance,
        .surface = vk_surface,
        .pdev = candidate.pdev,
        .props = candidate.props,
        .mem_props = mem_props,
        .dev = dev,
        .graphics_queue = graphics_queue,
        .present_queue = present_queue,
    };
}

pub fn deinit(self: *Graphics) void {
    self.dev.destroyDevice(null);
    self.a.destroy(self.dev.wrapper);

    self.instance.destroySurfaceKHR(self.surface, null);

    self.instance.destroyInstance(null);
    self.a.destroy(self.instance.wrapper);

    self.* = undefined;
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
        a: Allocator,
    ) !DeviceCandidate {
        const pdevs = try instance.enumeratePhysicalDevicesAlloc(a);
        defer a.free(pdevs);

        var best_score: usize = 0;
        var best_candidate: ?DeviceCandidate = null;

        for (pdevs) |pdev| {
            const candidate = try checkSuitable(instance, pdev, surface, a) orelse {
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
        a: Allocator,
    ) !?DeviceCandidate {
        if (!try checkExtensionSupport(instance, pdev, a)) {
            return null;
        }

        if (!try checkSurfaceSupport(instance, pdev, surface)) {
            return null;
        }

        if (try allocateQueues(instance, pdev, surface, a)) |allocation| {
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
        a: Allocator,
    ) !?QueueAllocation {
        const families = try instance.getPhysicalDeviceQueueFamilyPropertiesAlloc(pdev, a);
        defer a.free(families);

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
        a: Allocator,
    ) !bool {
        const propsv = try instance.enumerateDeviceExtensionPropertiesAlloc(pdev, null, a);
        defer a.free(propsv);

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
};
