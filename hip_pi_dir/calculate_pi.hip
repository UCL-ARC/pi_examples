#include <iostream>
#include <cstdlib>
#include <limits>

#include <hip/hip_runtime.h>

#include "timer.hpp"

using namespace std;

__global__ void calcpi(int threads, long n, double *results) {
   int rank = threadIdx.x;
   results[rank] = 0.0;
   double step = 1.0/n;
   double x = 0.0;

   long lower = rank * n/threads;
   long upper = (rank + 1) * n/threads;

   for (long i = lower; i < upper; i++) {
      x    = (i + 0.5) * step;
      results[rank] += 4.0 / (1.0 + x*x);
   }
}

int main( int argc, char **argv ) {
   long num_steps = 1000000000;
   double result;
   int threads = 1000; // threads needs to dived num_steps!

   cout.precision(numeric_limits<double>::digits10+2);
   
   if (argc > 1) {
      num_steps = atol(argv[1]);
   }
   if (argc > 2) {
      threads = atol(argv[2]);
   }

   double step, pi;
   Timer timer;
   
   cout << "Calculating PI using:" << endl <<
           "  " << num_steps << " slices" << endl <<
           "  " << threads << " HIP threads" << endl;
   
   timer.start();
   
   double *sum, *d_sum;
   size_t size = threads*sizeof(double);
   step = 1.0 / num_steps;
   sum = (double*)malloc(size);

   hipMalloc((void**)&d_sum, size);
   calcpi<<<1,threads>>>(threads, num_steps, d_sum);
   hipMemcpy(sum, d_sum, size, hipMemcpyDeviceToHost);
   hipFree(d_sum);

   result = 0.0;

   for (int i=0; i<threads; i++) {
      result +=sum[i];
   }
   pi = result * step;

   timer.stop();

   cout << "Obtained value for PI: " << pi << endl <<
           "Time taken: " << timer.duration() << " seconds" << endl;

   return 0;
}

