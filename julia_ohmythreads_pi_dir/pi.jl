#!/usr/bin/env julia

using Pkg
Pkg.activate(@__DIR__)
Pkg.resolve()
Pkg.instantiate()

using Base.Threads: nthreads
using OhMyThreads: tmapreduce

function _picalc(numsteps)
    slice = 1 / numsteps

    return tmapreduce(+, 1:numsteps; ntasks=nthreads()) do i
        4.0 / (1.0 + ((i - 0.5) * slice) ^ 2)
    end * slice
end

function picalc(numsteps)

    println("Calculating PI using:")
    println("  ", numsteps, " slices")
    println("  ", nthreads(), " thread(s)")

    start = time()
    mypi = _picalc(numsteps)
    elapsed = time() - start

    println("Obtained value of PI: ", mypi)
    println("Time taken: ", round(elapsed; digits=3), " seconds")

end

numsteps = if length(ARGS) > 0
    parse(Int, ARGS[1])
else
    1_000_000_000
end

# Warm up kernel
print("  Warming up...")
warms = time()
_picalc(10)
warmt = time() - warms
println("done. [", round(warmt; digits=3), "s]\n")

# Run the full example
picalc(numsteps)
