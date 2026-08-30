#!/usr/bin/env julia

# Make sure we have all the CUDA gubbins set up.
using Pkg
Pkg.activate(".")
Pkg.instantiate()
using CUDA

# Actual CUDA kernel.
function pichunk!(result, step, work_per_thread)
    rank = (blockIdx().x - 1) * (blockDim().x) + (threadIdx().x - 1)
    lower = rank * work_per_thread
    upper = lower + work_per_thread - 1
    for i in lower:upper
        x = (i + 0.5) * step
        result[rank+1] += 4.0 / (1.0 + x * x)
    end
    return
end

# Sort out CUDA thread setup.
function _picalc(blocks, threads_per_block, numsteps)
    step = 1 / numsteps
    tot_threads = blocks * threads_per_block
    work_per_thread = numsteps ÷ tot_threads
    result = CUDA.zeros(Float64, tot_threads)
    @cuda blocks=(blocks) threads=(threads_per_block) pichunk!(result, step, work_per_thread)
    synchronize()
    s = sum(result) * step
    return s
end

# Run the benchmark.
function picalc(blocks, threads_per_block, numsteps)

    println("Calculating PI using:")
    println("  ", numsteps, " slices")
    println("  ", threads_per_block, " CUDA threads(s)")

    start = time()
    mypi = _picalc(blocks, threads_per_block, numsteps)
    elapsed = time() - start

    println("Obtained value of PI: ", mypi)
    println("Time taken: ", round(elapsed, digits=3), " seconds")
end

# Parse arguments.
numsteps = if length(ARGS) > 0
  parse(Int, ARGS[1])
else
    1099511627776
end

threads = if length(ARGS) > 1
  parse(Int, ARGS[2])
else
  512
end

# Warm things up
print("  Warming up...")
warms = time()
let
    blocks = 256
    threads_per_block = 256
    numsteps = blocks * threads_per_block * 2
    _picalc(blocks, threads_per_block, numsteps)
    nothing
end
warmt = time() - warms
println("done. [", round(warmt, digits=3), "s]\n")

# Run the full example
blocks = 4096
picalc(blocks, threads, numsteps)
