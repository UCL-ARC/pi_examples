#include <time.h>
#include <sys/time.h>

class Timer {
  public:
    double resolution() {
      return ((double)clock()) / CLOCKS_PER_SEC;
    }

    void start() {
      gettimeofday(&start_time, NULL);
    }

    void stop() {
      gettimeofday(&stop_time, NULL);
    }

    double duration() {
      return (double)(stop_time.tv_sec  - start_time.tv_sec) +
             (double)((stop_time.tv_usec - start_time.tv_usec) / 1000000.0);
    }

  private:
    struct timeval start_time, stop_time;
};

