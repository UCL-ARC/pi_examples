import kotlin.time.measureTimedValue

fun calc_pi(n_slices: Long): Double {
	val step: Double = 1.0/n_slices.toDouble()
	var psum: Double = 0.0
	var x: Double
	
	for (i in 1..n_slices) {
		x = (i.toDouble() - 0.5) * step
		psum = psum + 4.0/(1.0 + (x * x))
	}
	
	return psum * step
}

fun main(args: Array<String>) {
	var n_slices: Long = 1000000000
	
	if (args.size > 0) {
		n_slices = args[0].toLong()
	}
	
	print("Esitmating π using:\n  $n_slices slices\n")

	val (p, elapsed) = measureTimedValue {
		calc_pi(n_slices)
	}	
	print("Estimated value of π: $p\n")
	print("Time taken: $elapsed\n")
}
