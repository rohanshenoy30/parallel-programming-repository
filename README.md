# Parallel Programming Lab Manual

A comprehensive collection of parallel programming implementations using **MPI**, **CUDA**, and **OpenCL** frameworks for distributed computing and GPU acceleration.

## 📋 Overview

This repository contains laboratory work for a parallel programming course, covering:
- **Distributed Computing** using Message Passing Interface (MPI) - Labs 1-4
- **GPU Computing** using CUDA - Labs 5-10  
- **Alternative GPU Framework** using OpenCL - Kernel examples
- **Exam Preparation** - Midsemester practical problems

## 📁 Project Structure

```
ppl_lab/
├── lab1/          # MPI Fundamentals
├── lab2/          # MPI Inter-process Communication
├── lab3/          # MPI Collective Operations & Scheduling
├── lab4/          # MPI Advanced Operations
├── lab5/          # CUDA Basics & Vector Operations
├── lab6/          # CUDA Advanced Kernels
├── lab7/          # CUDA Optimization Techniques
├── lab8/          # CUDA Memory Operations
├── lab9/          # CUDA Complex Algorithms
├── lab10/         # CUDA Advanced Topics
├── midsem/        # Midsemester Exam Problems
├── opencl/        # OpenCL Kernel Implementations
└── README.md
```

## 🔬 Lab Descriptions

### Labs 1-4: MPI (Distributed Computing)

**Technologies:** C with MPI library

| Lab | Topics | Key Exercises |
|-----|--------|---------------|
| **Lab 1** | MPI Basics, Process Management | Power calculation, Process communication |
| **Lab 2** | Point-to-point Communication, Message Passing | Word toggling, String manipulation across processes |
| **Lab 3** | Collective Operations, Synchronization | Factorial computation, Prefix sums, Scatter/Gather |
| **Lab 4** | Advanced Operations, Load Balancing | Parallel algorithms, Error handling |

- **aq* files**: Additional question implementations
- **q* files**: Main question assignments

### Labs 5-10: CUDA (GPU Computing)

**Technologies:** CUDA C/C++, NVIDIA GPU

| Lab | Topics | Key Exercises |
|-----|--------|---------------|
| **Lab 5** | CUDA Basics, Memory Management | Vector addition, Matrix operations |
| **Lab 6** | Thread Blocks, Grid Organization | 2D kernels, Tiling strategies |
| **Lab 7** | Optimization, Warp Efficiency | Performance tuning, Memory coalescing |
| **Lab 8** | Advanced Memory Operations | Shared memory, Global memory optimization |
| **Lab 9** | Complex Algorithms | Sorting, Parallel reductions, Graph algorithms |
| **Lab 10** | Advanced Topics | Atomic operations, Dynamic parallelism |

### Midsem: Exam Practice Problems

Practical CUDA problems including:
- Palindrome checking with parallel processing
- String operations on GPU
- Complex data structure processing
- Performance evaluation

### OpenCL: Alternative GPU Framework

Cross-platform GPU programming implementations:
- Vector Addition
- Scalar Multiplication  
- Matrix Operations
- Platform-agnostic kernel code

## 🛠 Technologies & Tools

| Technology | Purpose |
|-----------|---------|
| **MPI (OpenMPI/MPICH)** | Distributed memory parallel computing |
| **CUDA Toolkit** | NVIDIA GPU programming |
| **OpenCL** | Cross-platform GPU/accelerator computing |
| **C/C++** | Core programming language |
| **GCC/NVCC** | Compilation |

## 📋 Prerequisites

### For MPI (Labs 1-4):
```bash
# macOS
brew install open-mpi

# Linux (Ubuntu/Debian)
sudo apt-get install libopenmpi-dev openmpi-bin

# Linux (Fedora/RHEL)
sudo dnf install openmpi-devel
```

### For CUDA (Labs 5-10):
- NVIDIA GPU with CUDA Compute Capability 3.0 or higher
- CUDA Toolkit 10.0 or later
- NVIDIA cuDNN (optional, for advanced examples)

### For OpenCL:
- OpenCL headers and libraries
- Compatible GPU drivers

## 🚀 Compilation & Execution

### MPI Programs (Labs 1-4):
```bash
# Compile
mpicc lab1/q1.c -o lab1_q1 -lm

# Run with N processes
mpirun -np 4 ./lab1_q1
```

### CUDA Programs (Labs 5-10):
```bash
# Compile
nvcc lab5/q1.cu -o lab5_q1

# Run
./lab5_q1
```

### OpenCL Programs:
```bash
# Compile (system dependent)
gcc opencl/vectoraddition.cl -o vec_add -lOpenCL

# Run
./vec_add
```

## 📚 File Naming Convention

- **q1.c / q1.cu**: Main question implementation
- **aq1.c / aq1.cu**: Additional/alternative question approach
- **.c**: C source files (typically MPI)
- **.cu**: CUDA source files
- **.cl**: OpenCL kernel files

## 💡 Key Concepts Covered

### MPI Concepts:
- Process creation and management
- Point-to-point communication (Send/Recv)
- Collective operations (Broadcast, Scatter, Gather, AllReduce)
- Synchronization and deadlock avoidance
- Error handling

### CUDA Concepts:
- Thread hierarchy (blocks, threads, grids)
- Memory hierarchy (global, shared, local, constant)
- Kernel execution and synchronization
- Data transfer between host and device
- Performance optimization techniques
- Atomic operations and memory coalescing

### OpenCL Concepts:
- Platform and device management
- Kernel compilation and execution
- Work-item organization
- Platform-agnostic GPU programming

## 📝 Notes

- Each lab builds upon previous concepts
- Multiple solution approaches are provided (q* and aq* files)
- Some labs include timing and performance analysis
- Midsem folder contains exam-style problems with varying difficulty
- All code includes comments for educational clarity

## 🔗 Related Concepts

This lab course covers essential parallel programming paradigms:
1. **Distributed Memory Computing**: Multiple computers/processes sharing data via messages (MPI)
2. **Shared Memory Computing**: Multiple processors sharing memory (not explicitly covered here)
3. **GPU Acceleration**: Massively parallel computation on graphics processors (CUDA/OpenCL)

## 📖 Recommended Reading Order

1. Start with **Lab 1** for MPI basics
2. Progress through **Labs 2-4** for intermediate and advanced MPI
3. Move to **Lab 5** for CUDA fundamentals
4. Continue through **Labs 6-10** for CUDA mastery
5. Review **Midsem** problems for comprehensive understanding
6. Explore **OpenCL** for alternative GPU programming approach

---

**Course**: Parallel and Distributed Computing Lab  
**Institution**: MIT  
**Semester**: 6th Semester  
**Last Updated**: 2026
