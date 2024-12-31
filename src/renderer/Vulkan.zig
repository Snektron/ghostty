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

const link = @import("link.zig");
const configpkg = @import("../config.zig");
const apprt = @import("../apprt.zig");
const font = @import("../font/main.zig");
const renderer = @import("../renderer.zig");
const terminal = @import("../terminal/main.zig");
const isCovering = @import("cell.zig").isCovering;
const fgMode = @import("cell.zig").fgMode;

const Graphics = @import("vulkan/Graphics.zig");
const Swapchain = @import("vulkan/Swapchain.zig");
const CellPipeline = @import("vulkan/CellPipeline.zig");
const GpuBuffer = @import("vulkan/GpuBuffer.zig");

pub const frame_timeout = 1 * std.time.ns_per_s;
pub const frames_in_flight = 3;

alloc: Allocator,

config: DerivedConfig,

/// Current font metrics defining our grid.
grid_metrics: font.face.Metrics,

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

/// The size of everything.
size: renderer.Size,

/// True if the window is focused
focused: bool,

/// The current set of cells to render. Each set of cells goes into
/// a separate shader call.
cells_bg: std.ArrayListUnmanaged(CellPipeline.Cell) = .{},
cells: std.ArrayListUnmanaged(CellPipeline.Cell) = .{},

/// The last viewport that we based our rebuild off of. If this changes,
/// then we do a full rebuild of the cells. The pointer values in this pin
/// are NOT SAFE to read because they may be modified, freed, etc from the
/// termio thread. We treat the pointers as integers for comparison only.
cells_viewport: ?terminal.Pin = null,

/// The font structures.
font_grid: *font.SharedGrid,
font_shaper: font.Shaper,
font_shaper_cache: font.ShaperCache,

/// Whether we're doing padding extension for vertical sides.
padding_extend_top: bool = true,
padding_extend_bottom: bool = true,

/// Vulkan GPU state.
/// Initialization of the Vulkan GPU state is deferred until `finalizeSurfaceInit`,
/// as we require the surface in order to pick the right rendering device.
gpu_state: ?GPUState = null,

const GPUState = struct {
    graphics: Graphics,
    swapchain: Swapchain,
    frames: [frames_in_flight]Frame,
    recreate_swapchain: bool = false,
    cell_pipeline: CellPipeline,

    cells_bg: GpuBuffer,
    cells_bg_written: usize = 0,

    cells: GpuBuffer,
    cells_written: usize = 0,
};

pub const DerivedConfig = struct {
    arena: ArenaAllocator,

    font_thicken: bool,
    font_features: std.ArrayListUnmanaged([:0]const u8),
    cursor_color: ?terminal.color.RGB,
    cursor_invert: bool,
    cursor_text: ?terminal.color.RGB,
    cursor_opacity: f64,
    background: terminal.color.RGB,
    background_opacity: f64,
    foreground: terminal.color.RGB,
    selection_background: ?terminal.color.RGB,
    selection_foreground: ?terminal.color.RGB,
    invert_selection_fg_bg: bool,
    bold_is_bright: bool,
    padding_color: configpkg.WindowPaddingColor,
    links: link.Set,

    pub fn init(
        gpa: Allocator,
        config: *const configpkg.Config,
    ) !DerivedConfig {
        var arena = ArenaAllocator.init(gpa);
        errdefer arena.deinit();
        const alloc = arena.allocator();

        // Copy our font features
        const font_features = try config.@"font-feature".clone(alloc);

        const cursor_invert = config.@"cursor-invert-fg-bg";

        // Our link configs
        const links = try link.Set.fromConfig(
            alloc,
            config.link.links.items,
        );

        return .{
            .font_thicken = config.@"font-thicken",
            .font_features = font_features.list,
            .cursor_color = if (!cursor_invert and config.@"cursor-color" != null)
                config.@"cursor-color".?.toTerminalRGB()
            else
                null,
            .cursor_invert = cursor_invert,
            .cursor_text = if (config.@"cursor-text") |txt|
                txt.toTerminalRGB()
            else
                null,
            .cursor_opacity = @max(0, @min(1, config.@"cursor-opacity")),
            .background = config.background.toTerminalRGB(),
            .background_opacity = @max(0, @min(1, config.@"background-opacity")),
            .foreground = config.foreground.toTerminalRGB(),
            .padding_color = config.@"window-padding-color",
            .invert_selection_fg_bg = config.@"selection-invert-fg-bg",
            .bold_is_bright = config.@"bold-is-bright",
            .selection_background = if (config.@"selection-background") |bg|
                bg.toTerminalRGB()
            else
                null,

            .selection_foreground = if (config.@"selection-foreground") |bg|
                bg.toTerminalRGB()
            else
                null,
            .links = links,
            .arena = arena,
        };
    }

    pub fn deinit(self: *DerivedConfig) void {
        const alloc = self.arena.allocator();
        self.links.deinit(alloc);
        self.arena.deinit();
    }
};

pub fn init(alloc: Allocator, options: renderer.Options) !Vulkan {
    // Create the initial font shaper
    var shaper = try font.Shaper.init(alloc, .{
        .features = options.config.font_features.items,
    });
    errdefer shaper.deinit();

    // For the remainder of the setup we lock our font grid data because
    // we're reading it.
    const grid = options.font_grid;
    grid.lock.lockShared();
    defer grid.lock.unlockShared();

    return .{
        .alloc = alloc,
        .config = options.config,
        .foreground_color = options.config.foreground,
        .background_color = options.config.background,
        .cursor_color = options.config.cursor_color,
        .cursor_invert = options.config.cursor_invert,
        .draw_background = options.config.background,
        .surface_mailbox = options.surface_mailbox,
        .size = options.size,
        .focused = true,
        .grid_metrics = grid.metrics,
        .font_grid = grid,
        .font_shaper = shaper,
        .font_shaper_cache = font.ShaperCache.init(),
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

        state.cells_bg.deinit(state.graphics);
        state.cells.deinit(state.graphics);

        state.cell_pipeline.deinit(state.graphics);

        for (&state.frames) |*frame| {
            frame.deinit(state.graphics);
        }

        state.swapchain.deinit(&state.graphics);
        state.graphics.deinit();
    }

    self.font_shaper.deinit();
    self.font_shaper_cache.deinit(self.alloc);

    self.cells.deinit(self.alloc);
    self.cells_bg.deinit(self.alloc);

    self.config.deinit();

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
    // up when both the renderer and the surface are available.
}

pub fn finalizeSurfaceInit(self: *Vulkan, surface: *apprt.Surface) !void {
    var graphics = try Graphics.init(self.alloc, surface);
    errdefer graphics.deinit();

    var swapchain = try Swapchain.init(&graphics, self.alloc, .{
        .vsync = true,
        .desired_extent = .{
            .width = self.size.screen.width,
            .height = self.size.screen.height,
        },
        .swap_image_usage = .{
            .color_attachment_bit = true,
        },
    });
    errdefer swapchain.deinit(&graphics);

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

    var cell_pipeline = try CellPipeline.init(graphics, swapchain);
    errdefer cell_pipeline.deinit(graphics);

    var cells_bg = try GpuBuffer.init(
        graphics,
        16384,
        .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .{ .device_local_bit = true },
    );
    errdefer cells_bg.deinit(graphics);

    var cells = try GpuBuffer.init(
        graphics,
        16384,
        .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
        .{ .device_local_bit = true },
    );
    errdefer cells.deinit(graphics);

    self.gpu_state = .{
        .graphics = graphics,
        .swapchain = swapchain,
        .frames = frames,
        .cell_pipeline = cell_pipeline,
        .cells_bg = cells_bg,
        .cells = cells,
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
    // Vsync is managed by Vulkan, but the application still needs to
    // manually submit frames.
    return false;
}

pub fn markDirty(self: *Vulkan) void {
    _ = self;
    // TODO
}

pub fn setFocus(self: *Vulkan, focus: bool) !void {
    self.focused = focus;
}

pub fn setVisible(self: *Vulkan, visible: bool) void {
    _ = self;
    _ = visible;
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
    _ = surface;

    // Data we extract out of the critical area.
    const Critical = struct {
        full_rebuild: bool,
        gl_bg: terminal.color.RGB,
        screen: terminal.Screen,
        screen_type: terminal.ScreenType,
        mouse: renderer.State.Mouse,
        preedit: ?renderer.State.Preedit,
        cursor_style: ?renderer.CursorStyle,
        color_palette: terminal.color.Palette,
    };

    // Update all our data as tightly as possible within the mutex.
    var critical: Critical = critical: {
        const grid_size = self.size.grid();

        state.mutex.lock();
        defer state.mutex.unlock();

        // If we're in a synchronized output state, we pause all rendering.
        if (state.terminal.modes.get(.synchronized_output)) {
            log.debug("synchronized output started, skipping render", .{});
            return;
        }

        // Swap bg/fg if the terminal is reversed
        const bg = self.background_color;
        const fg = self.foreground_color;
        defer {
            self.background_color = bg;
            self.foreground_color = fg;
        }
        if (state.terminal.modes.get(.reverse_colors)) {
            self.background_color = fg;
            self.foreground_color = bg;
        }

        // If our terminal screen size doesn't match our expected renderer
        // size then we skip a frame. This can happen if the terminal state
        // is resized between when the renderer mailbox is drained and when
        // the state mutex is acquired inside this function.
        //
        // For some reason this doesn't seem to cause any significant issues
        // with flickering while resizing. '\_('-')_/'
        if (grid_size.rows != state.terminal.rows or
            grid_size.columns != state.terminal.cols)
        {
            return;
        }

        // Get the viewport pin so that we can compare it to the current.
        const viewport_pin = state.terminal.screen.pages.pin(.{ .viewport = .{} }).?;

        // We used to share terminal state, but we've since learned through
        // analysis that it is faster to copy the terminal state than to
        // hold the lock wile rebuilding GPU cells.
        var screen_copy = try state.terminal.screen.clone(
            self.alloc,
            .{ .viewport = .{} },
            null,
        );
        errdefer screen_copy.deinit();

        // Whether to draw our cursor or not.
        const cursor_style = if (state.terminal.flags.password_input)
            .lock
        else
            renderer.cursorStyle(
                state,
                self.focused,
                cursor_blink_visible,
            );

        // Get our preedit state
        const preedit: ?renderer.State.Preedit = preedit: {
            if (cursor_style == null) break :preedit null;
            const p = state.preedit orelse break :preedit null;
            break :preedit try p.clone(self.alloc);
        };
        errdefer if (preedit) |p| p.deinit(self.alloc);

        // TODO
        // // If we have Kitty graphics data, we enter a SLOW SLOW SLOW path.
        // // We only do this if the Kitty image state is dirty meaning only if
        // // it changes.
        // //
        // // If we have any virtual references, we must also rebuild our
        // // kitty state on every frame because any cell change can move
        // // an image.
        // if (state.terminal.screen.kitty_images.dirty or
        //     self.image_virtual)
        // {
        //     // // prepKittyGraphics touches self.images which is also used
        //     // // in drawFrame so if we're drawing on a separate thread we need
        //     // // to lock this.
        //     // if (single_threaded_draw) self.draw_mutex.lock();
        //     // defer if (single_threaded_draw) self.draw_mutex.unlock();
        //     // try self.prepKittyGraphics(state.terminal);
        // }

        // If we have any terminal dirty flags set then we need to rebuild
        // the entire screen. This can be optimized in the future.
        const full_rebuild: bool = rebuild: {
            {
                const Int = @typeInfo(terminal.Terminal.Dirty).Struct.backing_integer.?;
                const v: Int = @bitCast(state.terminal.flags.dirty);
                if (v > 0) break :rebuild true;
            }
            {
                const Int = @typeInfo(terminal.Screen.Dirty).Struct.backing_integer.?;
                const v: Int = @bitCast(state.terminal.screen.dirty);
                if (v > 0) break :rebuild true;
            }

            // If our viewport changed then we need to rebuild the entire
            // screen because it means we scrolled. If we have no previous
            // viewport then we must rebuild.
            const prev_viewport = self.cells_viewport orelse break :rebuild true;
            if (!prev_viewport.eql(viewport_pin)) break :rebuild true;

            break :rebuild false;
        };

        // Reset the dirty flags in the terminal and screen. We assume
        // that our rebuild will be successful since so we optimize for
        // success and reset while we hold the lock. This is much easier
        // than coordinating row by row or as changes are persisted.
        state.terminal.flags.dirty = .{};
        state.terminal.screen.dirty = .{};
        {
            var it = state.terminal.screen.pages.pageIterator(
                .right_down,
                .{ .screen = .{} },
                null,
            );
            while (it.next()) |chunk| {
                var dirty_set = chunk.node.data.dirtyBitSet();
                dirty_set.unsetAll();
            }
        }

        // Update our viewport pin for dirty tracking
        self.cells_viewport = viewport_pin;

        break :critical .{
            .full_rebuild = full_rebuild,
            .gl_bg = self.background_color,
            .screen = screen_copy,
            .screen_type = state.terminal.active_screen,
            .mouse = state.mouse,
            .preedit = preedit,
            .cursor_style = cursor_style,
            .color_palette = state.terminal.color_palette.colors,
        };
    };
    defer {
        critical.screen.deinit();
        if (critical.preedit) |p| p.deinit(self.alloc);
    }

    // Grab our draw mutex if we have it and update our data
    {
        // Set our draw data
        self.draw_background = critical.gl_bg;

        // Build our GPU cells
        try self.rebuildCells(
            critical.full_rebuild,
            &critical.screen,
            critical.screen_type,
            critical.mouse,
            critical.preedit,
            critical.cursor_style,
            &critical.color_palette,
        );

        // Notify our shaper we're done for the frame. For some shapers like
        // CoreText this triggers off-thread cleanup logic.
        self.font_shaper.endFrame();
    }
}

/// rebuildCells rebuilds all the GPU cells from our CPU state. This is a
/// slow operation but ensures that the GPU state exactly matches the CPU state.
/// In steady-state operation, we use some GPU tricks to send down stale data
/// that is ignored. This accumulates more memory; rebuildCells clears it.
///
/// Note this doesn't have to typically be manually called. Internally,
/// the renderer will do this when it needs more memory space.
fn rebuildCells(
    self: *Vulkan,
    rebuild: bool,
    screen: *terminal.Screen,
    screen_type: terminal.ScreenType,
    mouse: renderer.State.Mouse,
    preedit: ?renderer.State.Preedit,
    cursor_style_: ?renderer.CursorStyle,
    color_palette: *const terminal.color.Palette,
) !void {
    _ = screen_type;

    // Bg cells at most will need space for the visible screen size
    self.cells_bg.clearRetainingCapacity();
    self.cells.clearRetainingCapacity();

    // Create an arena for all our temporary allocations while rebuilding
    var arena = ArenaAllocator.init(self.alloc);
    defer arena.deinit();
    const arena_alloc = arena.allocator();

    // We've written no data to the GPU, refresh it all
    if (self.gpu_state) |*state| {
        state.cells_written = 0;
        state.cells_bg_written = 0;
    }

    // Create our match set for the links.
    var link_match_set: link.MatchSet = if (mouse.point) |mouse_pt| try self.config.links.matchSet(
        arena_alloc,
        screen,
        mouse_pt,
        mouse.mods,
    ) else .{};

    // Determine our x/y range for preedit. We don't want to render anything
    // here because we will render the preedit separately.
    const preedit_range: ?struct {
        y: terminal.size.CellCountInt,
        x: [2]terminal.size.CellCountInt,
        cp_offset: usize,
    } = if (preedit) |preedit_v| preedit: {
        const range = preedit_v.range(screen.cursor.x, screen.pages.cols - 1);
        break :preedit .{
            .y = screen.cursor.y,
            .x = .{ range.start, range.end },
            .cp_offset = range.cp_offset,
        };
    } else null;

    // These are all the foreground cells underneath the cursor.
    //
    // We keep track of these so that we can invert the colors and move them
    // in front of the block cursor so that the character remains visible.
    //
    // We init with a capacity of 4 to account for decorations such
    // as underline and strikethrough, as well as combining chars.
    var cursor_cells = try std.ArrayListUnmanaged(CellPipeline.Cell).initCapacity(arena_alloc, 4);
    defer cursor_cells.deinit(arena_alloc);

    if (rebuild) {
        switch (self.config.padding_color) {
            .background => {},

            .extend, .@"extend-always" => {
                self.padding_extend_top = true;
                self.padding_extend_bottom = true;
            },
        }
    }

    // Build each cell
    var row_it = screen.pages.rowIterator(.left_up, .{ .viewport = .{} }, null);
    var y: terminal.size.CellCountInt = screen.pages.rows;
    while (row_it.next()) |row| {
        y -= 1;

        // True if we want to do font shaping around the cursor. We want to
        // do font shaping as long as the cursor is enabled.
        const shape_cursor = screen.viewportIsBottom() and
            y == screen.cursor.y;

        // If this is the row with our cursor, then we may have to modify
        // the cell with the cursor.
        const start_i: usize = self.cells.items.len;
        defer if (shape_cursor and cursor_style_ == .block) {
            const x = screen.cursor.x;
            const wide = row.cells(.all)[x].wide;
            const min_x = switch (wide) {
                .narrow, .spacer_head, .wide => x,
                .spacer_tail => x -| 1,
            };
            const max_x = switch (wide) {
                .narrow, .spacer_head, .spacer_tail => x,
                .wide => x +| 1,
            };
            for (self.cells.items[start_i..]) |cell| {
                if (cell.grid_col < min_x or cell.grid_col > max_x) continue;
                if (cell.mode.isFg()) {
                    cursor_cells.append(arena_alloc, cell) catch {
                        // We silently ignore if this fails because
                        // worst case scenario some combining glyphs
                        // aren't visible under the cursor '\_('-')_/'
                    };
                }
            }
        };

        // We need to get this row's selection if there is one for proper
        // run splitting.
        const row_selection = sel: {
            const sel = screen.selection orelse break :sel null;
            const pin = screen.pages.pin(.{ .viewport = .{ .y = y } }) orelse
                break :sel null;
            break :sel sel.containedRow(screen, pin) orelse null;
        };

        // On primary screen, we still apply vertical padding extension
        // under certain conditions we feel are safe. This helps make some
        // scenarios look better while avoiding scenarios we know do NOT look
        // good.
        switch (self.config.padding_color) {
            // These already have the correct values set above.
            .background, .@"extend-always" => {},

            // Apply heuristics for padding extension.
            .extend => if (y == 0) {
                self.padding_extend_top = !row.neverExtendBg(
                    color_palette,
                    self.background_color,
                );
            } else if (y == self.size.grid().rows - 1) {
                self.padding_extend_bottom = !row.neverExtendBg(
                    color_palette,
                    self.background_color,
                );
            },
        }

        // Iterator of runs for shaping.
        var run_iter = self.font_shaper.runIterator(
            self.font_grid,
            screen,
            row,
            row_selection,
            if (shape_cursor) screen.cursor.x else null,
        );
        var shaper_run: ?font.shape.TextRun = try run_iter.next(self.alloc);
        var shaper_cells: ?[]const font.shape.Cell = null;
        var shaper_cells_i: usize = 0;

        const row_cells = row.cells(.all);

        for (row_cells, 0..) |*cell, x| {
            // If this cell falls within our preedit range then we
            // skip this because preedits are setup separately.
            if (preedit_range) |range| preedit: {
                // We're not on the preedit line, no actions necessary.
                if (range.y != y) break :preedit;
                // We're before the preedit range, no actions necessary.
                if (x < range.x[0]) break :preedit;
                // We're in the preedit range, skip this cell.
                if (x <= range.x[1]) continue;
                // After exiting the preedit range we need to catch
                // the run position up because of the missed cells.
                // In all other cases, no action is necessary.
                if (x != range.x[1] + 1) break :preedit;

                // Step the run iterator until we find a run that ends
                // after the current cell, which will be the soonest run
                // that might contain glyphs for our cell.
                while (shaper_run) |run| {
                    if (run.offset + run.cells > x) break;
                    shaper_run = try run_iter.next(self.alloc);
                    shaper_cells = null;
                    shaper_cells_i = 0;
                }

                const run = shaper_run orelse break :preedit;

                // If we haven't shaped this run, do so now.
                shaper_cells = shaper_cells orelse
                    // Try to read the cells from the shaping cache if we can.
                    self.font_shaper_cache.get(run) orelse
                    cache: {
                    // Otherwise we have to shape them.
                    const cells = try self.font_shaper.shape(run);

                    // Try to cache them. If caching fails for any reason we
                    // continue because it is just a performance optimization,
                    // not a correctness issue.
                    self.font_shaper_cache.put(
                        self.alloc,
                        run,
                        cells,
                    ) catch |err| {
                        log.warn(
                            "error caching font shaping results err={}",
                            .{err},
                        );
                    };

                    // The cells we get from direct shaping are always owned
                    // by the shaper and valid until the next shaping call so
                    // we can safely use them.
                    break :cache cells;
                };

                // Advance our index until we reach or pass
                // our current x position in the shaper cells.
                while (shaper_cells.?[shaper_cells_i].x < x) {
                    shaper_cells_i += 1;
                }
            }

            const wide = cell.wide;

            const style = row.style(cell);

            const cell_pin: terminal.Pin = cell: {
                var copy = row;
                copy.x = @intCast(x);
                break :cell copy;
            };

            // True if this cell is selected
            const selected: bool = if (screen.selection) |sel|
                sel.contains(screen, .{
                    .node = row.node,
                    .y = row.y,
                    .x = @intCast(
                        // Spacer tails should show the selection
                        // state of the wide cell they belong to.
                        if (wide == .spacer_tail)
                            x -| 1
                        else
                            x,
                    ),
                })
            else
                false;

            const bg_style = style.bg(cell, color_palette);
            const fg_style = style.fg(color_palette, self.config.bold_is_bright) orelse self.foreground_color;

            // The final background color for the cell.
            const bg = bg: {
                if (selected) {
                    break :bg if (self.config.invert_selection_fg_bg)
                        if (style.flags.inverse)
                            // Cell is selected with invert selection fg/bg
                            // enabled, and the cell has the inverse style
                            // flag, so they cancel out and we get the normal
                            // bg color.
                            bg_style
                        else
                            // If it doesn't have the inverse style
                            // flag then we use the fg color instead.
                            fg_style
                    else
                        // If we don't have invert selection fg/bg set then we
                        // just use the selection background if set, otherwise
                        // the default fg color.
                        break :bg self.config.selection_background orelse self.foreground_color;
                }

                // Not selected
                break :bg if (style.flags.inverse != isCovering(cell.codepoint()))
                    // Two cases cause us to invert (use the fg color as the bg)
                    // - The "inverse" style flag.
                    // - A "covering" glyph; we use fg for bg in that case to
                    //   help make sure that padding extension works correctly.
                    // If one of these is true (but not the other)
                    // then we use the fg style color for the bg.
                    fg_style
                else
                    // Otherwise they cancel out.
                    bg_style;
            };

            const fg = fg: {
                if (selected and !self.config.invert_selection_fg_bg) {
                    // If we don't have invert selection fg/bg set
                    // then we just use the selection foreground if
                    // set, otherwise the default bg color.
                    break :fg self.config.selection_foreground orelse self.background_color;
                }

                // Whether we need to use the bg color as our fg color:
                // - Cell is inverted and not selected
                // - Cell is selected and not inverted
                //    Note: if selected then invert sel fg / bg must be
                //    false since we separately handle it if true above.
                break :fg if (style.flags.inverse != selected)
                    bg_style orelse self.background_color
                else
                    fg_style;
            };

            // Foreground alpha for this cell.
            const alpha: u8 = if (style.flags.faint) 175 else 255;

            // If the cell has a background color, set it.
            const bg_color: [4]u8 = if (bg) |rgb| bg: {
                // Determine our background alpha. If we have transparency configured
                // then this is dynamic depending on some situations. This is all
                // in an attempt to make transparency look the best for various
                // situations. See inline comments.
                const bg_alpha: u8 = bg_alpha: {
                    const default: u8 = 255;

                    if (self.config.background_opacity >= 1) break :bg_alpha default;

                    // If we're selected, we do not apply background opacity
                    if (selected) break :bg_alpha default;

                    // If we're reversed, do not apply background opacity
                    if (style.flags.inverse) break :bg_alpha default;

                    // If we have a background and its not the default background
                    // then we apply background opacity
                    if (style.bg(cell, color_palette) != null and !rgb.eql(self.background_color)) {
                        break :bg_alpha default;
                    }

                    // We apply background opacity.
                    var bg_alpha: f64 = @floatFromInt(default);
                    bg_alpha *= self.config.background_opacity;
                    bg_alpha = @ceil(bg_alpha);
                    break :bg_alpha @intFromFloat(bg_alpha);
                };

                try self.cells_bg.append(self.alloc, .{
                    .mode = .bg,
                    .grid_col = @intCast(x),
                    .grid_row = @intCast(y),
                    .grid_width = cell.gridWidth(),
                    .glyph_x = 0,
                    .glyph_y = 0,
                    .glyph_width = 0,
                    .glyph_height = 0,
                    .glyph_offset_x = 0,
                    .glyph_offset_y = 0,
                    .r = rgb.r,
                    .g = rgb.g,
                    .b = rgb.b,
                    .a = bg_alpha,
                    .bg_r = 0,
                    .bg_g = 0,
                    .bg_b = 0,
                    .bg_a = 0,
                });

                break :bg .{
                    rgb.r, rgb.g, rgb.b, bg_alpha,
                };
            } else .{
                self.draw_background.r,
                self.draw_background.g,
                self.draw_background.b,
                @intFromFloat(@max(0, @min(255, @round(self.config.background_opacity * 255)))),
            };

            // If the invisible flag is set on this cell then we
            // don't need to render any foreground elements, so
            // we just skip all glyphs with this x coordinate.
            //
            // NOTE: This behavior matches xterm. Some other terminal
            // emulators, e.g. Alacritty, still render text decorations
            // and only make the text itself invisible. The decision
            // has been made here to match xterm's behavior for this.
            if (style.flags.invisible) {
                continue;
            }

            // Give links a single underline, unless they already have
            // an underline, in which case use a double underline to
            // distinguish them.
            const underline: terminal.Attribute.Underline = if (link_match_set.contains(screen, cell_pin))
                if (style.flags.underline == .single)
                    .double
                else
                    .single
            else
                style.flags.underline;

            // We draw underlines first so that they layer underneath text.
            // This improves readability when a colored underline is used
            // which intersects parts of the text (descenders).
            if (underline != .none) self.addUnderline(
                @intCast(x),
                @intCast(y),
                underline,
                style.underlineColor(color_palette) orelse fg,
                alpha,
                bg_color,
            ) catch |err| {
                log.warn(
                    "error adding underline to cell, will be invalid x={} y={}, err={}",
                    .{ x, y, err },
                );
            };

            if (style.flags.overline) self.addOverline(
                @intCast(x),
                @intCast(y),
                fg,
                alpha,
                bg_color,
            ) catch |err| {
                log.warn(
                    "error adding overline to cell, will be invalid x={} y={}, err={}",
                    .{ x, y, err },
                );
            };

            // If we're at or past the end of our shaper run then
            // we need to get the next run from the run iterator.
            if (shaper_cells != null and shaper_cells_i >= shaper_cells.?.len) {
                shaper_run = try run_iter.next(self.alloc);
                shaper_cells = null;
                shaper_cells_i = 0;
            }

            if (shaper_run) |run| glyphs: {
                // If we haven't shaped this run yet, do so.
                shaper_cells = shaper_cells orelse
                    // Try to read the cells from the shaping cache if we can.
                    self.font_shaper_cache.get(run) orelse
                    cache: {
                    // Otherwise we have to shape them.
                    const cells = try self.font_shaper.shape(run);

                    // Try to cache them. If caching fails for any reason we
                    // continue because it is just a performance optimization,
                    // not a correctness issue.
                    self.font_shaper_cache.put(
                        self.alloc,
                        run,
                        cells,
                    ) catch |err| {
                        log.warn(
                            "error caching font shaping results err={}",
                            .{err},
                        );
                    };

                    // The cells we get from direct shaping are always owned
                    // by the shaper and valid until the next shaping call so
                    // we can safely use them.
                    break :cache cells;
                };

                const cells = shaper_cells orelse break :glyphs;

                // If there are no shaper cells for this run, ignore it.
                // This can occur for runs of empty cells, and is fine.
                if (cells.len == 0) break :glyphs;

                // If we encounter a shaper cell to the left of the current
                // cell then we have some problems. This logic relies on x
                // position monotonically increasing.
                assert(cells[shaper_cells_i].x >= x);

                // NOTE: An assumption is made here that a single cell will never
                // be present in more than one shaper run. If that assumption is
                // violated, this logic breaks.

                while (shaper_cells_i < cells.len and cells[shaper_cells_i].x == x) : ({
                    shaper_cells_i += 1;
                }) {
                    self.addGlyph(
                        @intCast(x),
                        @intCast(y),
                        cell_pin,
                        cells[shaper_cells_i],
                        shaper_run.?,
                        fg,
                        alpha,
                        bg_color,
                    ) catch |err| {
                        log.warn(
                            "error adding glyph to cell, will be invalid x={} y={}, err={}",
                            .{ x, y, err },
                        );
                    };
                }
            }

            // Finally, draw a strikethrough if necessary.
            if (style.flags.strikethrough) self.addStrikethrough(
                @intCast(x),
                @intCast(y),
                fg,
                alpha,
                bg_color,
            ) catch |err| {
                log.warn(
                    "error adding strikethrough to cell, will be invalid x={} y={}, err={}",
                    .{ x, y, err },
                );
            };
        }
    }

    // Add the cursor at the end so that it overlays everything. If we have
    // a cursor cell then we invert the colors on that and add it in so
    // that we can always see it.
    if (cursor_style_) |cursor_style| cursor_style: {
        // If we have a preedit, we try to render the preedit text on top
        // of the cursor.
        if (preedit) |preedit_v| {
            const range = preedit_range.?;
            var x = range.x[0];
            for (preedit_v.codepoints[range.cp_offset..]) |cp| {
                self.addPreeditCell(cp, x, range.y) catch |err| {
                    log.warn("error building preedit cell, will be invalid x={} y={}, err={}", .{
                        x,
                        range.y,
                        err,
                    });
                };

                x += if (cp.wide) 2 else 1;
            }

            // Preedit hides the cursor
            break :cursor_style;
        }

        const cursor_color = self.cursor_color orelse color: {
            if (self.cursor_invert) {
                const sty = screen.cursor.page_pin.style(screen.cursor.page_cell);
                break :color sty.fg(color_palette, self.config.bold_is_bright) orelse self.foreground_color;
            } else {
                break :color self.foreground_color;
            }
        };

        _ = try self.addCursor(screen, cursor_style, cursor_color);
        for (cursor_cells.items) |*cell| {
            if (cell.mode.isFg() and cell.mode != .fg_color) {
                const cell_color = if (self.cursor_invert) blk: {
                    const sty = screen.cursor.page_pin.style(screen.cursor.page_cell);
                    break :blk sty.bg(screen.cursor.page_cell, color_palette) orelse self.background_color;
                } else if (self.config.cursor_text) |txt|
                    txt
                else
                    self.background_color;

                cell.r = cell_color.r;
                cell.g = cell_color.g;
                cell.b = cell_color.b;
                cell.a = 255;
            }
            try self.cells.append(self.alloc, cell.*);
        }
    }

    // Some debug mode safety checks
    if (std.debug.runtime_safety) {
        for (self.cells_bg.items) |cell| assert(cell.mode == .bg);
        for (self.cells.items) |cell| assert(cell.mode != .bg);
    }
}

fn addPreeditCell(
    self: *Vulkan,
    cp: renderer.State.Preedit.Codepoint,
    x: usize,
    y: usize,
) !void {
    // Preedit is rendered inverted
    const bg = self.foreground_color;
    const fg = self.background_color;

    // Render the glyph for our preedit text
    const render_ = self.font_grid.renderCodepoint(
        self.alloc,
        @intCast(cp.codepoint),
        .regular,
        .text,
        .{ .grid_metrics = self.grid_metrics },
    ) catch |err| {
        log.warn("error rendering preedit glyph err={}", .{err});
        return;
    };
    const render = render_ orelse {
        log.warn("failed to find font for preedit codepoint={X}", .{cp.codepoint});
        return;
    };

    // Add our opaque background cell
    try self.cells_bg.append(self.alloc, .{
        .mode = .bg,
        .grid_col = @intCast(x),
        .grid_row = @intCast(y),
        .grid_width = if (cp.wide) 2 else 1,
        .glyph_x = 0,
        .glyph_y = 0,
        .glyph_width = 0,
        .glyph_height = 0,
        .glyph_offset_x = 0,
        .glyph_offset_y = 0,
        .r = bg.r,
        .g = bg.g,
        .b = bg.b,
        .a = 255,
        .bg_r = 0,
        .bg_g = 0,
        .bg_b = 0,
        .bg_a = 0,
    });

    // Add our text
    try self.cells.append(self.alloc, .{
        .mode = .fg,
        .grid_col = @intCast(x),
        .grid_row = @intCast(y),
        .grid_width = if (cp.wide) 2 else 1,
        .glyph_x = render.glyph.atlas_x,
        .glyph_y = render.glyph.atlas_y,
        .glyph_width = render.glyph.width,
        .glyph_height = render.glyph.height,
        .glyph_offset_x = render.glyph.offset_x,
        .glyph_offset_y = render.glyph.offset_y,
        .r = fg.r,
        .g = fg.g,
        .b = fg.b,
        .a = 255,
        .bg_r = bg.r,
        .bg_g = bg.g,
        .bg_b = bg.b,
        .bg_a = 255,
    });
}

fn addCursor(
    self: *Vulkan,
    screen: *terminal.Screen,
    cursor_style: renderer.CursorStyle,
    cursor_color: terminal.color.RGB,
) !?*const CellPipeline.Cell {
    // Add the cursor. We render the cursor over the wide character if
    // we're on the wide character tail.
    const wide, const x = cell: {
        // The cursor goes over the screen cursor position.
        const cell = screen.cursor.page_cell;
        if (cell.wide != .spacer_tail or screen.cursor.x == 0)
            break :cell .{ cell.wide == .wide, screen.cursor.x };

        // If we're part of a wide character, we move the cursor back to
        // the actual character.
        const prev_cell = screen.cursorCellLeft(1);
        break :cell .{ prev_cell.wide == .wide, screen.cursor.x - 1 };
    };

    const alpha: u8 = if (!self.focused) 255 else alpha: {
        const alpha = 255 * self.config.cursor_opacity;
        break :alpha @intFromFloat(@ceil(alpha));
    };

    const render = switch (cursor_style) {
        .block,
        .block_hollow,
        .bar,
        .underline,
        => render: {
            const sprite: font.Sprite = switch (cursor_style) {
                .block => .cursor_rect,
                .block_hollow => .cursor_hollow_rect,
                .bar => .cursor_bar,
                .underline => .underline,
                .lock => unreachable,
            };

            break :render self.font_grid.renderGlyph(
                self.alloc,
                font.sprite_index,
                @intFromEnum(sprite),
                .{
                    .cell_width = if (wide) 2 else 1,
                    .grid_metrics = self.grid_metrics,
                },
            ) catch |err| {
                log.warn("error rendering cursor glyph err={}", .{err});
                return null;
            };
        },

        .lock => self.font_grid.renderCodepoint(
            self.alloc,
            0xF023, // lock symbol
            .regular,
            .text,
            .{
                .cell_width = if (wide) 2 else 1,
                .grid_metrics = self.grid_metrics,
            },
        ) catch |err| {
            log.warn("error rendering cursor glyph err={}", .{err});
            return null;
        } orelse {
            // This should never happen because we embed nerd
            // fonts so we just log and return instead of fallback.
            log.warn("failed to find lock symbol for cursor codepoint=0xF023", .{});
            return null;
        },
    };

    try self.cells.append(self.alloc, .{
        .mode = .fg,
        .grid_col = @intCast(x),
        .grid_row = @intCast(screen.cursor.y),
        .grid_width = if (wide) 2 else 1,
        .r = cursor_color.r,
        .g = cursor_color.g,
        .b = cursor_color.b,
        .a = alpha,
        .bg_r = 0,
        .bg_g = 0,
        .bg_b = 0,
        .bg_a = 0,
        .glyph_x = render.glyph.atlas_x,
        .glyph_y = render.glyph.atlas_y,
        .glyph_width = render.glyph.width,
        .glyph_height = render.glyph.height,
        .glyph_offset_x = render.glyph.offset_x,
        .glyph_offset_y = render.glyph.offset_y,
    });

    return &self.cells.items[self.cells.items.len - 1];
}

/// Add an underline decoration to the specified cell
fn addUnderline(
    self: *Vulkan,
    x: terminal.size.CellCountInt,
    y: terminal.size.CellCountInt,
    style: terminal.Attribute.Underline,
    color: terminal.color.RGB,
    alpha: u8,
    bg: [4]u8,
) !void {
    const sprite: font.Sprite = switch (style) {
        .none => unreachable,
        .single => .underline,
        .double => .underline_double,
        .dotted => .underline_dotted,
        .dashed => .underline_dashed,
        .curly => .underline_curly,
    };

    const render = try self.font_grid.renderGlyph(
        self.alloc,
        font.sprite_index,
        @intFromEnum(sprite),
        .{
            .cell_width = 1,
            .grid_metrics = self.grid_metrics,
        },
    );

    try self.cells.append(self.alloc, .{
        .mode = .fg,
        .grid_col = @intCast(x),
        .grid_row = @intCast(y),
        .grid_width = 1,
        .glyph_x = render.glyph.atlas_x,
        .glyph_y = render.glyph.atlas_y,
        .glyph_width = render.glyph.width,
        .glyph_height = render.glyph.height,
        .glyph_offset_x = render.glyph.offset_x,
        .glyph_offset_y = render.glyph.offset_y,
        .r = color.r,
        .g = color.g,
        .b = color.b,
        .a = alpha,
        .bg_r = bg[0],
        .bg_g = bg[1],
        .bg_b = bg[2],
        .bg_a = bg[3],
    });
}

/// Add an overline decoration to the specified cell
fn addOverline(
    self: *Vulkan,
    x: terminal.size.CellCountInt,
    y: terminal.size.CellCountInt,
    color: terminal.color.RGB,
    alpha: u8,
    bg: [4]u8,
) !void {
    const render = try self.font_grid.renderGlyph(
        self.alloc,
        font.sprite_index,
        @intFromEnum(font.Sprite.overline),
        .{
            .cell_width = 1,
            .grid_metrics = self.grid_metrics,
        },
    );

    try self.cells.append(self.alloc, .{
        .mode = .fg,
        .grid_col = @intCast(x),
        .grid_row = @intCast(y),
        .grid_width = 1,
        .glyph_x = render.glyph.atlas_x,
        .glyph_y = render.glyph.atlas_y,
        .glyph_width = render.glyph.width,
        .glyph_height = render.glyph.height,
        .glyph_offset_x = render.glyph.offset_x,
        .glyph_offset_y = render.glyph.offset_y,
        .r = color.r,
        .g = color.g,
        .b = color.b,
        .a = alpha,
        .bg_r = bg[0],
        .bg_g = bg[1],
        .bg_b = bg[2],
        .bg_a = bg[3],
    });
}

/// Add a strikethrough decoration to the specified cell
fn addStrikethrough(
    self: *Vulkan,
    x: terminal.size.CellCountInt,
    y: terminal.size.CellCountInt,
    color: terminal.color.RGB,
    alpha: u8,
    bg: [4]u8,
) !void {
    const render = try self.font_grid.renderGlyph(
        self.alloc,
        font.sprite_index,
        @intFromEnum(font.Sprite.strikethrough),
        .{
            .cell_width = 1,
            .grid_metrics = self.grid_metrics,
        },
    );

    try self.cells.append(self.alloc, .{
        .mode = .fg,
        .grid_col = @intCast(x),
        .grid_row = @intCast(y),
        .grid_width = 1,
        .glyph_x = render.glyph.atlas_x,
        .glyph_y = render.glyph.atlas_y,
        .glyph_width = render.glyph.width,
        .glyph_height = render.glyph.height,
        .glyph_offset_x = render.glyph.offset_x,
        .glyph_offset_y = render.glyph.offset_y,
        .r = color.r,
        .g = color.g,
        .b = color.b,
        .a = alpha,
        .bg_r = bg[0],
        .bg_g = bg[1],
        .bg_b = bg[2],
        .bg_a = bg[3],
    });
}

// Add a glyph to the specified cell.
fn addGlyph(
    self: *Vulkan,
    x: terminal.size.CellCountInt,
    y: terminal.size.CellCountInt,
    cell_pin: terminal.Pin,
    shaper_cell: font.shape.Cell,
    shaper_run: font.shape.TextRun,
    color: terminal.color.RGB,
    alpha: u8,
    bg: [4]u8,
) !void {
    const rac = cell_pin.rowAndCell();
    const cell = rac.cell;

    // Render
    const render = try self.font_grid.renderGlyph(
        self.alloc,
        shaper_run.font_index,
        shaper_cell.glyph_index,
        .{
            .grid_metrics = self.grid_metrics,
            .thicken = self.config.font_thicken,
        },
    );

    // If the glyph is 0 width or height, it will be invisible
    // when drawn, so don't bother adding it to the buffer.
    if (render.glyph.width == 0 or render.glyph.height == 0) {
        return;
    }

    // If we're rendering a color font, we use the color atlas
    const mode: CellPipeline.CellMode = switch (try fgMode(
        render.presentation,
        cell_pin,
    )) {
        .normal => .fg,
        .color => .fg_color,
        .constrained => .fg_constrained,
        .powerline => .fg_powerline,
    };

    try self.cells.append(self.alloc, .{
        .mode = mode,
        .grid_col = @intCast(x),
        .grid_row = @intCast(y),
        .grid_width = cell.gridWidth(),
        .glyph_x = render.glyph.atlas_x,
        .glyph_y = render.glyph.atlas_y,
        .glyph_width = render.glyph.width,
        .glyph_height = render.glyph.height,
        .glyph_offset_x = render.glyph.offset_x + shaper_cell.x_offset,
        .glyph_offset_y = render.glyph.offset_y + shaper_cell.y_offset,
        .r = color.r,
        .g = color.g,
        .b = color.b,
        .a = alpha,
        .bg_r = bg[0],
        .bg_g = bg[1],
        .bg_b = bg[2],
        .bg_a = bg[3],
    });
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
    self.size = size;
    if (self.gpu_state) |*state| state.recreate_swapchain = true;
}

fn uploadCells(
    graphics: Graphics,
    gpu_cells: *GpuBuffer,
    cells_written: *usize,
    cells: std.ArrayListUnmanaged(CellPipeline.Cell),
) !void {
    // TODO: This function is not correct currently. There
    // is only a single buffer, and it is written to while it may still be in use!
    // We don't need to create extra buffers, though, we can simply upload
    // the contents asynchronously during normal rendering operations.
    // If doing that, mind staging buffers and stuff. For now, it probably
    // works ish...

    if (cells_written.* < cells.items.len) {
        // Reallocate if necessary.
        if (gpu_cells.size < cells.capacity) {
            log.info("reallocating GPU buffer old={} new={}", .{
                gpu_cells.size,
                cells.capacity,
            });

            gpu_cells.deinit(graphics);
            gpu_cells.* = try GpuBuffer.init(
                graphics,
                cells.capacity,
                .{ .transfer_dst_bit = true, .vertex_buffer_bit = true },
                .{ .device_local_bit = true },
            );
        }

        try gpu_cells.uploadWithStagingBuffer(
            graphics,
            std.mem.sliceAsBytes(cells.items),
        );

        cells_written.* = cells.items.len;
    }
}

pub fn drawFrame(self: *Vulkan, surface: *apprt.Surface) !void {
    _ = surface;
    const state = if (self.gpu_state) |*state| state else return;
    const dev = state.graphics.dev;

    try uploadCells(state.graphics, &state.cells_bg, &state.cells_bg_written, self.cells_bg);
    try uploadCells(state.graphics, &state.cells, &state.cells_written, self.cells);

    const frame = &state.frames[state.graphics.frameIndex()];
    try frame.wait(state.graphics);

    state.graphics.beginFrame();

    const cmd_buf = frame.cmd_buf;

    // TODO: The current recreating logic causes black flickering when actually doing it
    // I'm not really sure why that happens, but its probably related to recreating the
    // swapchain the next frame, when it becomes suboptimal. It also looks like there is
    // currently some delay in ghostty when changing resolution, which doesn't help.
    while (true) {
        if (!state.recreate_swapchain) {
            const present_state = state.swapchain.acquireNextImage(&state.graphics, frame.image_acquired) catch |err| switch (err) {
                error.OutOfDateKHR => null,
                else => |other| return other,
            };

            if (present_state == null) {
                // The swapchain is out of date; we need to recreate it directly.
            } else if (present_state == .suboptimal) {
                // We need to recreate the swapchain, but we've already signaled the frame's image_acquired semaphore.
                // We can now either drop the frame, or schedule swapchain recreation for the next frame.
                state.recreate_swapchain = true;
                break;
            } else {
                break;
            }
        }

        log.debug("resizing {}x{} => {}x{}", .{
            state.swapchain.extent.width,
            state.swapchain.extent.height,
            self.size.screen.width,
            self.size.screen.height,
        });

        try state.swapchain.reinit(&state.graphics, .{
            .vsync = true,
            .desired_extent = .{
                .width = self.size.screen.width,
                .height = self.size.screen.height,
            },
            .swap_image_usage = .{
                .color_attachment_bit = true,
            },
        });

        // TODO: Do we need to update the CellPipeline's color attachment format? Can this realistically change?

        state.recreate_swapchain = false;

        // Try acquiring the frame again now that we've recreated the swapchain.
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

    try state.swapchain.present(&state.graphics, &.{frame.render_finished});

    state.graphics.endFrame();
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
