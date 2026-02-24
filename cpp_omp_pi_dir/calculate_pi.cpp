#include <iostream>
#include <cstdlib>
#include <limits>
#include <omp.h>
#include <chrono>

int main( int argc, char **argv ) {
   long i, num_steps = 1000000000;
   int thread_count = omp_get_max_threads();

   std::cout.precision(std::numeric_limits<double>::digits10+2);
   
   if (argc > 1) {
      num_steps = atol(argv[1]);
   }

   double step, x, sum, pi;
   
   std::cout << "Calculating PI using:" << std::endl <<
           "  " << num_steps << " slices" << std::endl <<
           "  1 process" << std::endl <<
           "  " << thread_count << " threads" << std::endl;
   
   auto start = std::chrono::high_resolution_clock::now();

   sum = 0.0;
   step = 1.0 / num_steps;

#pragma omp parallel for private(x) reduction(+:sum)
   for (i=0;i<num_steps;i++) {
      x    = (i + 0.5) * step;
      sum += 4.0 / (1.0 + x*x);
   }

   pi = sum * step;

   auto stop = std::chrono::high_resolution_clock::now();
   auto duration_s = std::chrono::duration<double>(stop - start).count();

   std::cout << "Obtained value for PI: " << pi << std::endl <<
           "Time taken: " << duration_s << " seconds" << std::endl;

   return 0;
}

