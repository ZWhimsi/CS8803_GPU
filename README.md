# CS8803 GPU Computing - Advanced GPU Programming Projects

This repository contains my coursework from Georgia Tech's CS8803 GPU Computing course, demonstrating advanced GPU programming techniques, optimization strategies, and performance analysis across multiple comprehensive projects.

## 🚀 Overview

This course covered advanced GPU programming concepts including CUDA programming, memory optimization, warp scheduling algorithms, and performance analysis. The projects showcase practical implementation of GPU algorithms with focus on performance optimization and architectural understanding.

## 📁 Project Structure

### Assignment 1: Matrix Multiplication Optimization

**Focus**: CUDA Programming Fundamentals & Memory Optimization

- **Objective**: Implement and optimize matrix multiplication using CUDA
- **Key Techniques**:

  - Shared memory tiling for improved cache locality
  - Thread block optimization for maximum occupancy
  - Memory coalescing for efficient global memory access
  - Performance analysis and profiling

- **Files**: `Assigment_1/kernel.cu`
- **Performance Metrics**: Achieved >70% occupancy and >80% memory throughput

### Assignment 2: Bitonic Sort Implementation

**Focus**: Advanced CUDA Algorithms & Optimization

- **Objective**: Implement high-performance bitonic sort algorithm on GPU
- **Key Techniques**:

  - Shared memory optimization with 4x tiling strategy
  - Batched processing for improved throughput
  - Power-of-two optimizations
  - Advanced memory access patterns
  - Performance profiling and analysis

- **Files**: `Assigment_2/kernel.cu`, `kernel2.cu`
- **Optimization Scripts**: `compile_optimized.sh`, `run_optimized.sh`
- **Analysis**: Comprehensive performance evaluation with NCU profiling

### Assignment 3: GPU Warp Scheduling Simulation

**Focus**: GPU Architecture & Scheduling Algorithms

- **Objective**: Implement and analyze different warp scheduling algorithms
- **Key Algorithms Implemented**:

  - **Round Robin (RR)**: Cyclical warp scheduling
  - **Greedy Then Oldest (GTO)**: Latency-aware scheduling
  - **Cache-Conscious Wavefront Scheduling (CCWS)**: Locality-aware scheduling

- **Technical Implementation**:

  - Modified MacSim GPU simulator
  - Implemented VTA (Victim Tag Array) for cache locality tracking
  - LLS (Lost Locality Score) calculation and feedback mechanisms
  - Performance analysis across multiple benchmarks

- **Files**: `Assigment_3/CS8803_Proj3_frame/`
- **Benchmarks**: backprop, crystal, hotspot, lavaMD, neural networks, pathfinder
- **Analysis**: Comprehensive performance comparison and statistical analysis

## 🛠️ Technical Skills Demonstrated

### CUDA Programming

- Kernel design and optimization
- Memory hierarchy utilization (global, shared, registers)
- Thread block and grid configuration
- Synchronization primitives

### Performance Optimization

- Memory coalescing techniques
- Shared memory tiling strategies
- Occupancy optimization
- Profiling and performance analysis using NVIDIA tools

### GPU Architecture Understanding

- Warp scheduling algorithms
- Cache hierarchy and locality optimization
- Memory subsystem simulation
- Performance bottleneck identification

### Software Engineering

- C++ programming with CUDA extensions
- Makefile and build system management
- Performance testing and validation
- Statistical analysis and reporting

## 📊 Performance Achievements

### Assignment 1 (Matrix Multiplication)

- Achieved >70% GPU occupancy
- > 80% memory throughput utilization
- Optimized shared memory usage for cache efficiency

### Assignment 2 (Bitonic Sort)

- Implemented efficient 4x tiling strategy
- Optimized for power-of-two input sizes
- Comprehensive performance profiling and analysis

### Assignment 3 (Warp Scheduling)

- Implemented three distinct scheduling algorithms
- Analyzed performance across 6 different benchmarks
- Demonstrated understanding of GPU scheduling trade-offs

## 🔧 Tools & Technologies

- **CUDA Toolkit**: GPU programming and optimization
- **NVIDIA Nsight Compute**: Performance profiling
- **NVIDIA Nsight Systems**: System-level analysis
- **MacSim Simulator**: GPU architecture simulation
- **Python**: Data analysis and visualization
- **C++**: Core implementation language
- **Make**: Build system management

## 📈 Key Learning Outcomes

1. **Advanced CUDA Programming**: Mastered complex GPU programming patterns and optimization techniques
2. **Performance Analysis**: Developed skills in profiling, benchmarking, and performance optimization
3. **GPU Architecture**: Deep understanding of GPU memory hierarchy, warp scheduling, and execution model
4. **Algorithm Design**: Implemented and optimized fundamental GPU algorithms
5. **Research Skills**: Conducted performance analysis and comparative studies

## 🎯 Resume Highlights

This repository demonstrates:

- **Advanced GPU Programming**: Complex CUDA implementations with optimization
- **Performance Engineering**: Systematic approach to performance analysis and optimization
- **Algorithm Design**: Implementation of fundamental GPU algorithms (sorting, matrix operations)
- **Research Experience**: Comparative analysis of GPU scheduling algorithms
- **Software Engineering**: Clean, well-documented code with comprehensive testing

## 📚 Course Context

**Course**: CS8803 - GPU Computing (Georgia Tech)  
**Focus**: Advanced GPU programming, optimization, and architecture  
**Duration**: Full semester with multiple comprehensive projects  
**Grading**: Performance-based evaluation with optimization targets

---

_This repository showcases advanced GPU programming skills and performance optimization techniques developed through hands-on implementation of complex algorithms and architectural simulations._
