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

fn getInstanceProcAddrWrapper(instance: vk.Instance, proc_name: [*:0]const u8) vk.PfnVoidFunction {
    return @ptrCast(glfw.getInstanceProcAddress(@ptrFromInt(@intFromEnum(instance)), proc_name));
}

pub fn init(a: Allocator, surface: *apprt.Surface) !Graphics {
    _ = surface;

    log.info("initializing Vulkan graphics", .{});

    const vkb = try BaseDispatch.load(getInstanceProcAddrWrapper);

    const instance_exts = switch (apprt.runtime) {
        apprt.glfw => exts: {
            if (!glfw.vulkanSupported()) {
                log.err("GLFW does not support Vulkan. Is the vulkan loader available and in the library path?", .{});
                return error.VulkanUnsupported;
            }

            break :exts glfw.getRequiredInstanceExtensions() orelse {
                log.err("failed to query GLFW required instance extensions", .{});
                return error.VulkanUnsupported;
            };
        },
        else => @compileError("unsupported app runtime for Vulkan"),
    };

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
    }, null);

    const vki = try a.create(Instance.Wrapper);
    errdefer a.destroy(vki);
    vki.* = try Instance.Wrapper.load(vk_inst, vkb.dispatch.vkGetInstanceProcAddr);
    const instance = Instance.init(vk_inst, vki);
    errdefer instance.destroyInstance(null);

    return .{
        .a = a,
        .vkb = vkb,
        .instance = instance,
    };
}

pub fn deinit(self: *Graphics) void {
    log.info("deinitializing Vulkan graphics", .{});
    self.instance.destroyInstance(null);
    self.a.destroy(self.instance.wrapper);

    self.* = undefined;
}
