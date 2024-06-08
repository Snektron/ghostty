const std = @import("std");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

/// Returns the path to the cgroup for the given pid.
pub fn current(alloc: Allocator, pid: std.os.linux.pid_t) !?[]const u8 {
    var buf: [std.fs.MAX_PATH_BYTES]u8 = undefined;

    // Read our cgroup by opening /proc/<pid>/cgroup and reading the first
    // line. The first line will look something like this:
    // 0::/user.slice/user-1000.slice/session-1.scope
    // The cgroup path is the third field.
    const path = try std.fmt.bufPrint(&buf, "/proc/{}/cgroup", .{pid});
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    // Read it all into memory -- we don't expect this file to ever be that large.
    var buf_reader = std.io.bufferedReader(file.reader());
    const contents = try buf_reader.reader().readAllAlloc(
        alloc,
        1 * 1024 * 1024, // 1MB
    );
    defer alloc.free(contents);

    // Find the last ':'
    const idx = std.mem.lastIndexOfScalar(u8, contents, ':') orelse return null;
    const result = std.mem.trimRight(u8, contents[idx + 1 ..], " \r\n");
    return try alloc.dupe(u8, result);
}

/// Create a new cgroup. This will not move any process into it unless move is
/// set. If move is set, the given pid will be moved into the created cgroup.
pub fn create(
    cgroup: []const u8,
    child: []const u8,
    move: ?std.os.linux.pid_t,
) !void {
    var buf: [std.fs.MAX_PATH_BYTES]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "/sys/fs/cgroup{s}/{s}", .{ cgroup, child });
    try std.fs.cwd().makePath(path);

    // If we have a PID to move into the cgroup immediately, do it.
    if (move) |pid| {
        const pid_path = try std.fmt.bufPrint(
            &buf,
            "/sys/fs/cgroup{s}/{s}/cgroup.procs",
            .{ cgroup, child },
        );
        const file = try std.fs.cwd().openFile(pid_path, .{ .mode = .write_only });
        defer file.close();
        try file.writer().print("{}", .{pid});
    }
}

/// Move the given PID into the given cgroup.
pub fn moveInto(
    cgroup: []const u8,
    pid: std.os.linux.pid_t,
) !void {
    var buf: [std.fs.MAX_PATH_BYTES]u8 = undefined;
    const path = try std.fmt.bufPrint(&buf, "/sys/fs/cgroup{s}/cgroup.procs", .{cgroup});
    const file = try std.fs.cwd().openFile(path, .{ .mode = .write_only });
    defer file.close();
    try file.writer().print("{}", .{pid});
}

/// Returns all available cgroup controllers for the given cgroup.
/// The cgroup should have a '/'-prefix.
///
/// The returned list of is the raw space-separated list of
/// controllers from the /sys/fs directory. This avoids some extra
/// work since creating an iterator over this is easy and much cheaper
/// than allocating a bunch of copies for an array.
pub fn controllers(alloc: Allocator, cgroup: []const u8) ![]const u8 {
    assert(cgroup[0] == '/');
    var buf: [std.fs.MAX_PATH_BYTES]u8 = undefined;

    // Read the available controllers. These will be space separated.
    const path = try std.fmt.bufPrint(
        &buf,
        "/sys/fs/cgroup{s}/cgroup.controllers",
        .{cgroup},
    );
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    // Read it all into memory -- we don't expect this file to ever
    // be that large.
    var buf_reader = std.io.bufferedReader(file.reader());
    const contents = try buf_reader.reader().readAllAlloc(
        alloc,
        1 * 1024 * 1024, // 1MB
    );
    defer alloc.free(contents);

    // Return our raw list of controllers
    const result = std.mem.trimRight(u8, contents, " \r\n");
    return try alloc.dupe(u8, result);
}

/// Configure the set of controllers in the cgroup. The "v" should
/// be in a valid format for "cgroup.subtree_control"
pub fn configureControllers(
    cgroup: []const u8,
    v: []const u8,
) !void {
    assert(cgroup[0] == '/');
    var buf: [std.fs.MAX_PATH_BYTES]u8 = undefined;

    // Read the available controllers. These will be space separated.
    const path = try std.fmt.bufPrint(
        &buf,
        "/sys/fs/cgroup{s}/cgroup.subtree_control",
        .{cgroup},
    );
    const file = try std.fs.cwd().openFile(path, .{ .mode = .write_only });
    defer file.close();

    // Write
    try file.writer().writeAll(v);
}

pub const MemoryLimit = union(enum) {
    /// memory.high
    high: usize,
};

/// Configure the memory limit for the given cgroup. Use the various
/// fields in MemoryLimit to configure a specific type of limit.
pub fn configureMemoryLimit(cgroup: []const u8, limit: MemoryLimit) !void {
    assert(cgroup[0] == '/');

    const filename, const size = switch (limit) {
        .high => |v| .{ "memory.high", v },
    };

    // Open our file
    var buf: [std.fs.MAX_PATH_BYTES]u8 = undefined;
    const path = try std.fmt.bufPrint(
        &buf,
        "/sys/fs/cgroup{s}/{s}",
        .{ cgroup, filename },
    );
    const file = try std.fs.cwd().openFile(path, .{ .mode = .write_only });
    defer file.close();

    // Write our limit in bytes
    try file.writer().print("{}", .{size});
}