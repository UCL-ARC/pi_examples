const std = @import("std");

// Implementation in zig
pub fn main() !void {

	// Is this really all necessary?
	const stdout_file = std.io.getStdOut().writer();
	var bw = std.io.bufferedWriter(stdout_file);
	const stdout = bw.writer();

	var gpa = std.heap.GeneralPurposeAllocator(.{}){};
	const allocator = gpa.allocator();
	defer _ = gpa.deinit();

	var args = try std.process.argsAlloc(allocator);
	defer std.process.argsFree(allocator, args);
	// "definitely an improvement" 
	

	var n_slices: usize = 1000000000;

	if (args.len > 1) {
		n_slices = try std.fmt.parseInt(usize, args[1], 10);
	}

	try stdout.print("Estimating π using:\n  {d} slices.\n", .{n_slices});

	const start = try std.time.Instant.now();

	var s: f64 = 0.0;
	var step: f64 = 1.0 / @as(f64, @floatFromInt(n_slices));
	var x: f64 = 0.0;

	for (1..n_slices) |i| {
		x = (@as(f64, @floatFromInt(i)) - 0.5) * step;
		s = s + 4.0 / (1.0 + x * x);
	}

	const p: f64 = s * step;
	const finish = try std.time.Instant.now();
	const elapsed: f64 = @as(f64, @floatFromInt(finish.since(start)))/1000000000.0;

	try stdout.print("Estimated value of π: {d}\n", .{p});	
	try stdout.print("Elapsed time {d} seconds.\n", .{elapsed});	

	try bw.flush(); // don't forget to flush!
}

