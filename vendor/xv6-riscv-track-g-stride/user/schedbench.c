#include "kernel/types.h"
#ifdef STRIDE_SCHED
#include "kernel/sched.h"
#endif
#include "user/user.h"

static volatile uint64 sink;

struct result {
  int role;
  int weight;
  int elapsed;
  int response;
  uint64 ticks;
  uint64 dispatches;
  uint64 checksum;
};

static uint64
burn(uint64 rounds, uint64 seed)
{
  uint64 value = seed;
  for (uint64 i = 0; i < rounds; i++)
    value = value * 1664525 + 1013904223 + i;
  sink = value;
  return value;
}

static void
configure(int weight)
{
#ifdef STRIDE_SCHED
  if (setweight(weight) < 0)
    exit(2);
#else
  (void)weight;
#endif
}

static void
send_result(int fd, int role, int weight, int started, uint64 checksum,
            int response)
{
  struct result result;
  result.role = role;
  result.weight = weight;
  result.elapsed = uptime() - started;
  result.response = response;
  result.checksum = checksum;
#ifdef STRIDE_SCHED
  struct schedstats stats;
  if (getschedstats(&stats) < 0)
    exit(3);
  result.ticks = stats.run_ticks;
  result.dispatches = stats.dispatches;
#else
  result.ticks = (uint64)-1;
  result.dispatches = (uint64)-1;
#endif
  if (write(fd, &result, sizeof(result)) != sizeof(result))
    exit(4);
}

static void
collect(char *workload, int fd, int count)
{
  struct result results[3];
  for (int i = 0; i < count; i++) {
    int received = 0;
    while (received < sizeof(struct result)) {
      int n = read(fd, ((char *)&results[i]) + received,
                   sizeof(struct result) - received);
      if (n <= 0)
        exit(5);
      received += n;
    }
  }
  for (int i = 0; i < count; i++)
    for (int j = i + 1; j < count; j++)
      if (results[j].role < results[i].role) {
        struct result temporary = results[i];
        results[i] = results[j];
        results[j] = temporary;
      }
  for (int i = 0; i < count; i++)
    printf("BENCH_RESULT workload=%s role=%d weight=%d elapsed=%d response=%d ticks=%ld dispatches=%ld checksum=%ld\n",
           workload, results[i].role, results[i].weight, results[i].elapsed,
           results[i].response, results[i].ticks, results[i].dispatches,
           results[i].checksum);
}

static void
cpu_workload(void)
{
  int channel[2];
  int weights[] = {1, 3, 6};
  pipe(channel);
  for (int role = 0; role < 3; role++) {
    int pid = fork();
    if (pid == 0) {
      close(channel[0]);
      configure(weights[role]);
      int start = uptime();
      send_result(channel[1], role, weights[role], start,
                  burn(200000000, role + 17), -1);
      close(channel[1]);
      exit(0);
    }
    if (pid < 0)
      exit(1);
  }
  close(channel[1]);
  collect("cpu", channel[0], 3);
  close(channel[0]);
  for (int i = 0; i < 3; i++)
    wait(0);
}

static void
mixed_workload(void)
{
  int channel[2];
  pipe(channel);
  int pid = fork();
  if (pid == 0) {
    close(channel[0]);
    configure(1);
    int start = uptime();
    send_result(channel[1], 0, 1, start, burn(200000000, 41), -1);
    close(channel[1]);
    exit(0);
  }
  if (pid < 0)
    exit(1);
  pid = fork();
  if (pid == 0) {
    close(channel[0]);
    configure(8);
    int start = uptime();
    uint64 checksum = 0;
    for (int i = 0; i < 5; i++) {
      checksum ^= burn(5000000, 70 + i);
      pause(1);
    }
    send_result(channel[1], 1, 8, start, checksum, -1);
    close(channel[1]);
    exit(0);
  }
  if (pid < 0)
    exit(1);
  close(channel[1]);
  collect("mixed", channel[0], 2);
  close(channel[0]);
  wait(0);
  wait(0);
}

static void
latency_workload(void)
{
  int channel[2];
  pipe(channel);
  for (int role = 0; role < 2; role++) {
    int pid = fork();
    if (pid == 0) {
      close(channel[0]);
      configure(1);
      int start = uptime();
      send_result(channel[1], role, 1, start,
                  burn(300000000, 90 + role), -1);
      close(channel[1]);
      exit(0);
    }
    if (pid < 0)
      exit(1);
  }
  pause(1);
  int released = uptime();
  int pid = fork();
  if (pid == 0) {
    close(channel[0]);
    configure(10);
    int first = uptime();
    send_result(channel[1], 2, 10, released, burn(10000000, 99),
                first - released);
    close(channel[1]);
    exit(0);
  }
  if (pid < 0)
    exit(1);
  close(channel[1]);
  collect("latency", channel[0], 3);
  close(channel[0]);
  for (int i = 0; i < 3; i++)
    wait(0);
}

int
main(int argc, char **argv)
{
  if (argc != 2) {
    printf("usage: schedbench cpu|mixed|latency\n");
    exit(1);
  }
  printf("BENCH_BEGIN workload=%s\n", argv[1]);
  if (strcmp(argv[1], "cpu") == 0)
    cpu_workload();
  else if (strcmp(argv[1], "mixed") == 0)
    mixed_workload();
  else if (strcmp(argv[1], "latency") == 0)
    latency_workload();
  else
    exit(1);
  printf("BENCH_END workload=%s\n", argv[1]);
  exit(0);
}
