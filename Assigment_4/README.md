# CS8803 Project-4: Extending the pipeline with compute cores
This assignment extends the project 3 codebase by introducing a compute core to simulate compute operation latencies.
Unlike project 3, which focused on load-store instructions, this project includes execution of non-memory operations such as add and multiply.

## Introduction
### Compute Core
In this stage, all instructions - both non-memory and memory - are executed. The compute core is responsible for determining 
the latency of non-memory operations, such as add and multiply, and accessing caches for memory operations. 
It plays a crucial role in simulating the overall performance of GPU programs by accounting for compute operation latencies.

### Tensor Core
NVIDIA Tensor Cores serve as dedicated accelerators for matrix multiplication operations, particularly in artificial intelligence computations. 
They offer significantly higher throughput compared to CUDA Cores, making them essential for accelerating AI applications. 
Tensor Cores excel in performing matrix multiplications efficiently, contributing to improved performance in tasks dominated by GEMM operations.

### Differences
While the compute core handles one-dimensional point-by-point calculations, the tensor core is optimized for two-dimensional tile-by-tile calculations. 
This difference in computation approach reflects their respective strengths in processing different types of operations efficiently.

### Traces
For this project, we will utilize two traces: gemm_float and gemm_half.
- gemm_float is intended for the compute core.
- gemm_half is intended for the tensor core.
It's crucial to note the disparities in the number of instructions and opcodes between the two traces. NVbit is employed for trace generation. 
gemm_half incorporates HMMA instructions, indicating tensor (float16) operations, while gemm_float predominantly employs FFMA instructions. 
In gemm_half traces, HMMA opcodes are in the form "HMMA.1688.F16," signifying the use of Float16 operands. 
However, it's worth mentioning that macsim does not utilize the flags that follow the opcode.

## Assignment Tasks
### Task-1: Implement Naive Compute Core
To begin with, you'll need to implement a basic compute core functionality within the `core.cpp` file. The `run_a_cycle()` function, 
called each cycle, will now handle compute instructions instead of disregarding them as in project 3. Here's what your code should accomplish:

1. **Identify Compute Instructions:** Check whether an instruction is a compute instruction and retrieve its latency.
2. **Buffer Compute Instructions:** If the instruction is indeed a compute instruction, add it to the `c_exec_buffer` data structure (per core) along with its completion cycle timestamp.
4. **Handle Buffer Fullness:** If the buffer reaches its capacity, then stall the warp.
5. **Execute Instructions from Buffer:** Periodically, check whether the current cycle timestamp exceeds the completion timestamp of any instructions in the buffer. If so, remove these instructions from the buffer.

> Hint: Look for "// TODO: Task 1" comments!

### Task-2: Extend to Tensor Core
In this task, we aim to extend our code to simulate a tensor core-like behavior.
This involves increasing the execution width, allowing for parallel execution of multiple compute instructions per core.
Additionally, we'll adjust the latencies for tensor instructions to reflect their characteristics.

Below is the configurations to be simulated:

| Latency (cycles) | Execution Width |
|------------------|---------------------|
| 64               | 8                   |

The latency above is for tensor instructions only, other compute instructions continue to the take the default value of 1.
For report purposes, you can explore varying the tensor operator's latency (32-128) and execution bandwidth (2-16).
The reduced number of instructions in tensor operations allows for adjustments in latency or limits on the number of operations. Analyzing these trade-offs will provide valuable insights.

It's worth noting that the trace generation has already accounted for tensor operations, resulting in fewer instructions (observe the `NUM_INSTRS_RETIRED` statistic for both traces).

> Hint: Look for "// TODO: Task 2" comments!

> Hint: Tensor instruction opcodes start with "H*".

Moreover, given the extended duration of compute instructions, it's crucial to address register dependencies to prevent hazards.
Therefore, the RR scheduling scheme should be revised to consider register dependencies within a warp when determining the scheduling of warps.
- While macsim can handle multiple destination registers for an instruction, the provided traces feature only one destination register, simplifying dependency checks.
- Iterate through warps to schedule one without dependencies in a round-robin manner.
- For simplicity, warps with empty trace buffers, which are refilled after scheduling, can be disregarded for dependency checks.

> from macsim documentation (useful during implementation):
> uint8_t m_num_read_regs: number of source registers
> uint16_t m_src[MAX_GPU_SRC_NUM] : source register IDs

### Task-3: Report
In this task, you'll write a report containing the following things:
- For Task-1 and Task-2: 
  - A short explanation of your implementation.
  - A comparison of Task-1 (gemm_float) and Task-2 (gemm_half) performances.
- (must contain):
  - plots depicting varying tensor latency and execution width for Task-2. The number of data points and plots is left to your discretion.
  - Discussion on how various warp scheduling strategies impact tensor core, supported by either insights derived from implementation or analytical reasoning.
- (Optional) any interesting findings encountered during the analysis process.

> Note: Keep your report within 2 pages (not hard-limit), and submit a PDF file.

> Report grades are typically assessed on a pass/fail basis, where submission guarantees full credit. Additionally,
> Reports may contribute bonus points to compensate for any potential loss in grades for Task 1 and Task 2.

## Grading
- The assignment is worth **15 points** and is divided into 3 components, Task-1, Task-2 and a report.
- **0 points** if you don't submit anything by the deadline.
- **Task-1 is worth 6  points**, **Task-2 is worth 6   points**, and the **report is worth 3  points**.
- Rubric for Task-1 and Task-2:
  - **+50% points**: if you submit a code that compiles and runs without any errors/warnings but does **NOT** match
    the stats within the acceptable tolerance.
  - **+40% points**: if your stats match the reference stats within acceptable tolerance **(+-5%)**.
  - **+10% points**: if your stats exactly match the reference stats.

> For Task-1 We will use the *NUM_CYCLES* values from the logs to award points. 

> For Task-2 We will use the *NUM_STALL_CYCLES* values from the logs to award points. 

## Submission Guidelines
- You must submit all the deliverables on GradeScope (Which can be accessed through the Canvas menu).
- **Deliverables**
  - A tar file containing all the code. Use the `make submit` command to generate the tar file.
  - A PDF report.

---
## Macsim Simulator
This simulator is a stripped-down version of the original [Macsim](https://sites.gatech.edu/hparch/software/#macsim) simulator 
developed by [HpArch Lab](https://sites.gatech.edu/hparch/) at GT. It is a trace-based simulator where traces for various 
benchmarks are captured using [NVBit](https://research.nvidia.com/publication/2019-10_nvbit-dynamic-binary-instrumentation-framework-nvidia-gpus). 
The traces contain information about warps and the instructions they execute.

### Macsim Architecture
Macsim simulator consists of a trace reader (`trace.h`), several GPU cores (`core.cpp`), L1/L2 caches (`cache.cpp`) 
and a basic fixed latency memory (`ram.cpp`). Each core has a local L1 cache and all cores share an L2 cache. When we 
invoke macsim with a `kernel_config.txt` which contains the trace metadata, macsim retrieves information about how many 
kernels need to be executed and trace file for each warp. During simulation, macsim launches the kernels in a trace 
sequentially. In each kernel, macsim sets up the cores and caches. 

A cycle in macsim is simulated in `macsim::run_a_cycle()` method which performs the following actions:
- Incrementing the global cycle counter (`macsim::m_cycle`).
- Call `core::run_a_cycle()` for each core,
- Check if any memory responses have been returned from the memory subsystem and send them to the corresponding core. 

Each core mainly consists of an L1 cache, a dispatch queue (aka active warps pool, `core::c_dispatched_warps`), a 
suspended queue (`core::c_suspended_warps`), and a warp scheduler (`core.cpp::schedule_warps`). Since we are only concerned 
about the scheduling of warps and performance of different warp scheduling algorithms in terms of parameters like number 
of stall cycles, cache hits, etc. We only simulate load-store instructions and do not simulate any compute operations 
(like add, multiply etc.).


## Prerequisites
- Linux-based OS/WSL with python3 and GNU C/C++ compiler is required.
- **Using PACE-ICE cluster:**
  - **Using PACE-ICE cluster:**
    - We recommend using the following settings while allocating a machine on PACE cluster (Interactive Shell/Desktop):
      - Node Type: CPU (first avail)
      - Cores Per Node: 2
      - Memory Per Core: 2 GB
  - Traces are located at `/storage/ice-shared/cs8803o21/macsim_traces`. The simulator will automatically pick them up.
- **Using your own machine**:
  - Traces will be automatically downloaded to the local `macsim_traces` directory.
  - Macsim needs zlib to uncompress trace files. zlib and correspoding headers can be installed using `$ sudo apt install zlib1g zlib1g-dev`.


> Note: Traces are ~150MB in size and may take a couple of minutes to download and uncompress.


You can either clone the repository or download a zip from GitHub.
```bash
$ git clone <url>           # Clone the repo
$ cd Macsim_cs8803      
```

Plotting graphs requires `matplotlib` package, we've provided a bash script that you can source to setup a local Python 
virtual environment.

```bash
$ source sourceme           # source the sourceme script to setup environment
```

## Build Instructions
```bash
$ make -j`nproc`            # Build Macsim
# OR
$ make DEBUG=1 -j`nproc`    # Build Macsim for debugging with GDB
```

Try `make help` to see what else the makefile can do!

## Running benchmarks and plotting results
To run a single benchmark trace use the following command:
```bash
$ ./macsim -g <GPU config> -t <trace_path/kernel_config.txt>
```
- GPU configs are XML files that define parameters such as the number of cores, scheduling algorithm to use, etc. These 
  are located under `xmls` directory. We've provided 2 XML configs which are identical except the tensor latency and execution width variables 
- Traces are located under `/storage/ice-shared/cs8803o21/macsim_traces` on the PACE cluster. If you are using a local
  machine, these will be downloaded to the `macsim_traces` directory.

For your convenience, we've provided Make targets to run all benchmarks and plot the results.
```bash
$ make run_task1            # Run the simulator for task-1
$ make run_task2            # Run the simulator for task-2
```
The 1st and 2nd commands will run macsim for all the benchmarks and will generate log files in the `log` directory.

## Collaboration/Plagiarism Policy
* Feel free to use Ed for doubts/discussions, but **DO NOT** share your code snippets or discuss any implementation details.
* You are not allowed to publicly post your solutions online. (such as on GitHub)
* All submitted code will be checked for plagiarism, any violators will receive a 0.

## Additional Information
### Using GDB
GDB is a powerful tool if you want to resolve issues with your code or get a better understanding of the control flow, 
especially while working with a new codebase. To use GDB, follow these steps:

```bash
$ make DEBUG=1                            # Compile the project with debug flags
$ gdb --args ./macsim <macsim arguments>  # invoke gdb
```

## FAQ 
### Can we modify beyond the TODO session? 
Yes, you can modify other parts, but please make sure your final outputs are not changed.

### Overview of Changes from Project 3 Codebase:
1. Runtime support added for `tensor_latency` and `execution_width` variables, with updates to `xmls` and `exec` folder files to pass values to Macsim.
2. Introduced new functions in `trace.h` to check for compute opcode and determine latency.
3. Modified `core.cpp` and `core.h` to buffer executing compute instructions and clear them upon completion timestamps. Revised RR scheduling logic to consider register dependency.

