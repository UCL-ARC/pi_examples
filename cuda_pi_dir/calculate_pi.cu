#include <iostream>
#include <cstdlib>
#include <limits>

#include <cuda.h>

#include "timer.hpp"

using namespace std;

__global__ void calcpi(double step, long work_per_thread, double *results) {
   int rank = blockIdx.x * blockDim.x + threadIdx.x;
   results[rank] = 0.0;
   double x = 0.0;

   long lower = rank * work_per_thread;
   long upper = lower + work_per_thread;

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
           "  " << threads << " CUDA threads" << endl;

   double *sum, *d_sum;
   size_t size = threads*sizeof(double);
   step = 1.0 / num_steps;
   long work_per_thread = num_steps / threads;
   sum = (double*)malloc(size);

   cudaMalloc((void**)&d_sum, size);
   int threadsPerBlock = 128;
   int numBlocks = (threads + threadsPerBlock - 1) / threadsPerBlock;
   timer.start();
   calcpi<<<numBlocks, threadsPerBlock>>>(step, work_per_thread, d_sum);
   cudaDeviceSynchronize();
   timer.stop();
   cudaMemcpy(sum, d_sum, size, cudaMemcpyDeviceToHost);
   cudaFree(d_sum);

   result = 0.0;

   for (int i=0; i<threads; i++) {
      result +=sum[i];
   }
   pi = result * step;


   cout << "Obtained value for PI: " << pi << endl <<
           "Time taken: " << timer.duration() << " seconds" << endl;

   return 0;
}
