
COUNTERS="cycles,instructions,cache-references,cache-misses,branches,branch-misses,page-faults,context-switches,cpu-migrations,cpu-clock,mem-loads,mem-stores,sw_prefetch_access.t1_t2,power/energy-cores/"

RUNS=5000

sudo perf stat -e $COUNTERS -r $RUNS ./x86.elf

sudo perf stat -e $COUNTERS -r $RUNS ./x86.opt.elf

hyperfine -i --warmup 100 -N --runs $RUNS './x86.elf' './x86.opt.elf'