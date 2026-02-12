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
   double result;
   int blocks = 4096;
   int threads_per_block = 512;
   int tot_threads = blocks * threads_per_block;
   long num_steps = 1099511627776; // must be a multiple of tot_threads

   cout.precision(numeric_limits<double>::digits10+2);

   if (argc > 1) {
      num_steps = atol(argv[1]);
   }
   if (argc > 2) {
      threads_per_block = atol(argv[2]);
   }

   double step, pi;
   Timer timer;

   cout << "Calculating PI using:" << endl <<
           "  " << num_steps << " slices" << endl <<
           "  " << threads_per_block << " CUDA threads" << endl;

   double *sum, *d_sum;
   size_t size = tot_threads*sizeof(double);
   step = 1.0 / num_steps;
   long work_per_thread = num_steps / tot_threads;
   sum = (double*)malloc(size);

   timer.start();
   cudaMalloc((void**)&d_sum, size);
   calcpi<<<blocks, threads_per_block>>>(step, work_per_thread, d_sum);
   cudaDeviceSynchronize();
   timer.stop();
   cudaMemcpy(sum, d_sum, size, cudaMemcpyDeviceToHost);
   cudaFree(d_sum);

   result = 0.0;

   for (int i=0; i<tot_threads; i++) {
      result +=sum[i];
   }
   pi = result * step;


   cout << "Obtained value for PI: " << pi << endl <<
           "Time taken: " << timer.duration() << " seconds" << endl;

   return 0;
}
