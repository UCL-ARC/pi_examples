import jax
from jax import pmap
import jax.numpy
import time
import sys

@pmap
def y(x):
	return 4.0/(1.0 + (x**2))

def estimate_pi(slices, ipus, prerun):
	print(f"Estimating Pi with:\n  {slices} slices\n  {ipus} IPU(s)\n")
	t1 = time.time()
	x = jax.numpy.linspace(0, 1.0, slices, dtype=jax.numpy.float32).reshape((ipus,-1))
	if prerun:
		p_temp = y(x)
		t2 = time.time()
	p = jax.numpy.sum(y(x))/n
	t3 = time.time()

	print(f"Estimated value of Pi: {p}")
	if prerun:
		print(f"Time taken in prerun: {t2 - t1} seconds.\nTime taken in computation: {t3 - t2} seconds.\n")
	else:
		print(f"Time taken: {t3 - t1} seconds.\n")

if __name__ == "__main__":
	n = 1600000
	if (len(sys.argv) > 1):
		n = int(sys.argv[1])

	#nd = len(jax.devices())
	nd = 1
	if (len(sys.argv) > 2):
		nd = int(sys.argv[2])

	pr = False
	if (len(sys.argv) > 3):
		if sys.argv[3].lower() == "yes":
			pr = True

	estimate_pi(n, nd, pr)
	


