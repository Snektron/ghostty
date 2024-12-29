//! Rendering implementation for Vulkan.
pub const Vulkan = @This();

const std = @import("std");
const assert = std.debug.assert;
const log = std.log.scoped(.vulkan);
const Allocator = std.mem.Allocator;
const ArenaAllocator = std.heap.ArenaAllocator;

const builtin = @import("builtin");
const glfw = @import("glfw");
const configpkg = @import("../config.zig");
const apprt = @import("../apprt.zig");
const font = @import("../font/main.zig");
const renderer = @import("../renderer.zig");
const terminal = @import("../terminal/main.zig");

alloc: Allocator,

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

/// The mailbox for communicating with the window.
surface_mailbox: apprt.surface.Mailbox,

pub const DerivedConfig = struct {
    arena: ArenaAllocator,

    cursor_color: ?terminal.color.RGB,
    cursor_invert: bool,
    background: terminal.color.RGB,
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
            .foreground = config.foreground.toTerminalRGB(),
            .arena = arena,
        };
    }

    pub fn deinit(self: *DerivedConfig) void {
        self.arena.deinit();
    }
};

pub fn init(alloc: Allocator, options: renderer.Options) !Vulkan {
    return .{
        .alloc = alloc,
        .config = options.config,
        .foreground_color = options.config.foreground,
        .background_color = options.config.background,
        .cursor_color = options.config.cursor_color,
        .cursor_invert = options.config.cursor_invert,
        .surface_mailbox = options.surface_mailbox,
    };
}

pub fn deinit(self: *Vulkan) void {
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
    unreachable; // TODO
}

pub fn finalizeSurfaceInit(self: *const Vulkan, surface: *apprt.Surface) !void {
    _ = self;
    _ = surface;
    unreachable; // TODO
}

pub fn displayUnrealized(self: *Vulkan) void {
    _ = self;
    unreachable; // TODO
}

pub fn displayRealize(self: *Vulkan) !void {
    _ = self;
    unreachable; // TODO
}

pub fn threadEnter(self: *const Vulkan, surface: *apprt.Surface) !void {
    _ = self;
    _ = surface;
    unreachable; // TODO
}

pub fn threadExit(self: *const Vulkan) void {
    _ = self;
    unreachable; // TODO
}

pub fn hasAnimations(self: *const Vulkan) bool {
    _ = self;
    unreachable; // TODO
}

pub fn hasVsync(self: *const Vulkan) bool {
    _ = self;
    unreachable; // TODO
}

pub fn markDirty(self: *Vulkan) void {
    _ = self;
    unreachable; // TODO
}

pub fn setFocus(self: *Vulkan, focus: bool) !void {
    _ = self;
    _ = focus;
    unreachable; // TODO
}

pub fn setVisible(self: *Vulkan, visible: bool) void {
    _ = self;
    _ = visible;
    unreachable; // TODO
}

pub fn setFontGrid(self: *Vulkan, grid: *font.SharedGrid) void {
    _ = self;
    _ = grid;
    unreachable; // TODO
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
    unreachable; // TODO
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
    unreachable; // TODO
}

pub fn changeConfig(self: *Vulkan, config: *DerivedConfig) !void {
    _ = self;
    _ = config;
    unreachable; // TODO
}

pub fn setScreenSize(
    self: *Vulkan,
    size: renderer.Size,
) !void {
    _ = self;
    _ = size;
    unreachable; // TODO
}

pub fn drawFrame(self: *Vulkan, surface: *apprt.Surface) !void {
    _ = self;
    _ = surface;
    unreachable; // TODO
}
