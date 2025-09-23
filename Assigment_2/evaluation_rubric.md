## Evaluation rubric

| Evaluation metric | Max Credit | Calculation |
|---|---:|---|
| Achieved Occupancy | 1 | Achieved Occupancy >= 70% |
| Memory Throughput | 1 | Memory Throughput >= 80% |
| | | |
| Performance Option 1 | | |
| Million elements per second (meps) | 14 | if meps > 900 → min((meps/1000)*14, 14) |
| | | |
| Performance Option 2 | | |
| Kernel time (ms) | 10 | min((80/kernelTime)*10, 10) |
| Memory Transfer Time D2H + H2D | 4 | min((30/memTime)*4, 4) |
