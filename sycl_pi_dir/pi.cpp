#include <iostream>
#include <cstdlib>
#include <limits>
#include <chrono>
#include <sycl/sycl.hpp>

//using namespace sycl;
//using namespace std;
//using namespace std::chrono;

int main(int argc, char **argv) {
   long i, num_steps = 1000000000;

   sycl::property_list q_p{sycl::property::queue::in_order()};
   sycl::queue q{q_p};

   std::cout.precision(std::numeric_limits<double>::digits10+2);
   
   if (argc > 1) {
      num_steps = atol(argv[1]);
   }

   double step, sum, pi;
   
   std::cout << "Calculating PI using:" << std::endl <<
           "  " << num_steps << " slices" << std::endl <<
           "  1 process" << std::endl <<
           "  SYCL: " << q.get_device().get_info<sycl::info::device::name>() << std::endl;
   
   auto start = std::chrono::high_resolution_clock::now();

   double* s = sycl::malloc_shared<double>(1, q);
   s[0] = 0.0;
   step = 1.0 / num_steps;

   q.submit([&](sycl::handler &cgh) {

      auto sr = sycl::reduction(s, sycl::plus<double>());

      cgh.parallel_for (sycl::range<1>(num_steps), sr, [=] (sycl::id<1> i, auto& slocal) {
         double x    = ((double)i + 0.5) * step;
         slocal +=  4.0 / (1.0 + x*x);
      });
   }).wait();

   sum = s[0];
   sycl::free(s, q);
   pi = sum * step;

   auto stop = std::chrono::high_resolution_clock::now();
   auto duration_s = std::chrono::duration<double>(stop - start).count();

   std::cout << "Obtained value for PI: " << pi << std::endl <<
           "Time taken: " << duration_s << " seconds" << std::endl;

   return 0;

}
