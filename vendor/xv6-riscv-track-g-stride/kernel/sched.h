#ifndef XV6_SCHED_H
#define XV6_SCHED_H

#define SCHED_MIN_WEIGHT 1
#define SCHED_MAX_WEIGHT 100
#define SCHED_DEFAULT_WEIGHT 10
#define SCHED_STRIDE_BASE 10000000ULL

struct schedstats {
  int weight;
  uint64 stride;
  uint64 pass;
  uint64 dispatches;
  uint64 run_ticks;
};

#endif
