# CUDA High-Performance Computing Template

A production-ready CUDA computing template for GPU-accelerated scientific computing, featuring parallel algorithms, machine learning, and high-performance simulations for 2025.

## 🚀 Features

- **CUDA C/C++** - GPU programming with CUDA toolkit
- **cuBLAS** - GPU-accelerated linear algebra
- **cuDNN** - Deep neural network primitives
- **Thrust** - Parallel algorithms library
- **cuFFT** - Fast Fourier Transform on GPU
- **cuRAND** - Random number generation
- **NCCL** - Multi-GPU communication
- **CUDA Graphs** - Optimized kernel execution
- **Memory Management** - Unified memory and streams
- **Performance Profiling** - Nsight tools integration
- **Multi-GPU Support** - Distributed computing
- **Python Integration** - CuPy, Numba CUDA

## 📋 Prerequisites

- NVIDIA GPU with CUDA Compute Capability 6.0+
- CUDA Toolkit 12.0+
- GCC/G++ 9.0+
- CMake 3.18+
- Python 3.9+ (for Python bindings)

## 🛠️ Quick Start

### 1. Create New CUDA Project

```bash
git clone <this-repo> my-cuda-app
cd my-cuda-app
```

### 2. Environment Setup

```bash
# Verify CUDA installation
nvcc --version
nvidia-smi

# Set CUDA environment variables
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### 3. Build Project

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### 4. Run Examples

```bash
# Matrix multiplication
./bin/matrix_multiply

# Vector addition
./bin/vector_add

# Monte Carlo simulation
./bin/monte_carlo

# Neural network training
./bin/neural_network
```

## 📁 Project Structure

```
├── src/                        # Source code
│   ├── cuda/                   # CUDA kernels
│   │   ├── matrix_ops.cu       # Matrix operations
│   │   ├── vector_ops.cu       # Vector operations
│   │   ├── neural_network.cu   # Neural network kernels
│   │   ├── monte_carlo.cu      # Monte Carlo simulation
│   │   └── fft.cu              # Fast Fourier Transform
│   ├── cpp/                    # C++ host code
│   │   ├── main.cpp            # Main application
│   │   ├── matrix_multiply.cpp # Matrix multiplication
│   │   ├── vector_add.cpp      # Vector addition
│   │   └── neural_network.cpp  # Neural network
│   ├── python/                 # Python bindings
│   │   ├── cuda_ops.py         # CUDA operations wrapper
│   │   ├── neural_network.py   # Neural network implementation
│   │   └── benchmarks.py       # Performance benchmarks
│   └── utils/                  # Utility functions
│       ├── memory_manager.cpp  # Memory management
│       ├── profiler.cpp        # Performance profiling
│       └── error_checking.cpp  # CUDA error checking
├── include/                    # Header files
│   ├── cuda_ops.h
│   ├── neural_network.h
│   └── utils.h
├── tests/                      # Test files
├── benchmarks/                 # Performance benchmarks
├── docs/                       # Documentation
└── CMakeLists.txt              # Build configuration
```

## 🔧 Available Scripts

```bash
# Building
make build                      # Build all targets
make clean                      # Clean build files
make test                       # Run tests
make benchmark                  # Run benchmarks

# CUDA Development
make matrix_multiply           # Build matrix multiplication
make vector_add                # Build vector addition
make neural_network            # Build neural network
make monte_carlo               # Build Monte Carlo simulation

# Python Development
python -m pytest tests/        # Run Python tests
python benchmarks/benchmark.py # Run Python benchmarks
```

## ⚡ CUDA Kernel Examples

### Matrix Multiplication

```cuda
// src/cuda/matrix_ops.cu
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

__global__ void matrix_multiply_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matrix_multiply_shared_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    // Shared memory for tile-based multiplication
    extern __shared__ float shared_mem[];
    
    float* As = shared_mem;
    float* Bs = shared_mem + 16 * 16;
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    int row = by * 16 + ty;
    int col = bx * 16 + tx;
    
    float sum = 0.0f;
    
    // Tile-based multiplication
    for (int tile = 0; tile < (K + 15) / 16; tile++) {
        // Load tiles into shared memory
        if (row < M && (tile * 16 + tx) < K) {
            As[ty * 16 + tx] = A[row * K + tile * 16 + tx];
        } else {
            As[ty * 16 + tx] = 0.0f;
        }
        
        if ((tile * 16 + ty) < K && col < N) {
            Bs[ty * 16 + tx] = B[(tile * 16 + ty) * N + col];
        } else {
            Bs[ty * 16 + tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < 16; k++) {
            sum += As[ty * 16 + k] * Bs[k * 16 + tx];
        }
        
        __syncthreads();
    }
    
    // Store result
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Host function to launch kernel
void matrix_multiply_cuda(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    // Allocate device memory
    float *d_A, *d_B, *d_C;
    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);
    
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);
    
    // Copy data to device
    cudaMemcpy(d_A, A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size_B, cudaMemcpyHostToDevice);
    
    // Configure kernel launch
    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);
    
    // Launch kernel
    matrix_multiply_shared_kernel<<<gridSize, blockSize, 2 * 16 * 16 * sizeof(float)>>>(
        d_A, d_B, d_C, M, N, K
    );
    
    // Copy result back to host
    cudaMemcpy(C, d_C, size_C, cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
```

### Neural Network Kernels

```cuda
// src/cuda/neural_network.cu
#include <cuda_runtime.h>
#include <cudnn.h>
#include <curand.h>

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

__global__ void relu_backward_kernel(
    const float* grad_output, const float* input, float* grad_input, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_input[idx] = (input[idx] > 0.0f) ? grad_output[idx] : 0.0f;
    }
}

__global__ void softmax_kernel(float* data, int batch_size, int num_classes) {
    int batch_idx = blockIdx.x;
    int class_idx = threadIdx.x;
    
    if (batch_idx < batch_size && class_idx < num_classes) {
        int offset = batch_idx * num_classes;
        
        // Find maximum for numerical stability
        __shared__ float max_val;
        if (class_idx == 0) {
            max_val = data[offset];
            for (int i = 1; i < num_classes; i++) {
                max_val = fmaxf(max_val, data[offset + i]);
            }
        }
        __syncthreads();
        
        // Compute exponentials
        float exp_val = expf(data[offset + class_idx] - max_val);
        data[offset + class_idx] = exp_val;
        __syncthreads();
        
        // Compute sum
        __shared__ float sum_val;
        if (class_idx == 0) {
            sum_val = 0.0f;
            for (int i = 0; i < num_classes; i++) {
                sum_val += data[offset + i];
            }
        }
        __syncthreads();
        
        // Normalize
        data[offset + class_idx] /= sum_val;
    }
}

__global__ void dropout_kernel(
    float* data, const float* mask, float scale, int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= mask[idx] * scale;
    }
}

// Convolution forward pass
__global__ void conv2d_forward_kernel(
    const float* input, const float* weights, float* output,
    int batch_size, int input_height, int input_width, int input_channels,
    int output_height, int output_width, int output_channels,
    int kernel_size, int stride, int padding
) {
    int batch_idx = blockIdx.z;
    int out_channel = blockIdx.y;
    int out_row = blockIdx.x;
    int out_col = threadIdx.x;
    
    if (batch_idx < batch_size && out_channel < output_channels && 
        out_row < output_height && out_col < output_width) {
        
        float sum = 0.0f;
        
        for (int in_channel = 0; in_channel < input_channels; in_channel++) {
            for (int k_row = 0; k_row < kernel_size; k_row++) {
                for (int k_col = 0; k_col < kernel_size; k_col++) {
                    int in_row = out_row * stride + k_row - padding;
                    int in_col = out_col * stride + k_col - padding;
                    
                    if (in_row >= 0 && in_row < input_height && 
                        in_col >= 0 && in_col < input_width) {
                        
                        int input_idx = batch_idx * input_channels * input_height * input_width +
                                      in_channel * input_height * input_width +
                                      in_row * input_width + in_col;
                        
                        int weight_idx = out_channel * input_channels * kernel_size * kernel_size +
                                       in_channel * kernel_size * kernel_size +
                                       k_row * kernel_size + k_col;
                        
                        sum += input[input_idx] * weights[weight_idx];
                    }
                }
            }
        }
        
        int output_idx = batch_idx * output_channels * output_height * output_width +
                        out_channel * output_height * output_width +
                        out_row * output_width + out_col;
        
        output[output_idx] = sum;
    }
}
```

### Monte Carlo Simulation

```cuda
// src/cuda/monte_carlo.cu
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

__global__ void monte_carlo_pi_kernel(
    curandState* states, float* results, int num_samples
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_samples) {
        curandState local_state = states[idx];
        
        float x = curand_uniform(&local_state);
        float y = curand_uniform(&local_state);
        
        float distance = x * x + y * y;
        results[idx] = (distance <= 1.0f) ? 1.0f : 0.0f;
        
        states[idx] = local_state;
    }
}

__global__ void monte_carlo_option_pricing_kernel(
    curandState* states, float* results, int num_paths,
    float S0, float K, float r, float sigma, float T
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < num_paths) {
        curandState local_state = states[idx];
        
        // Generate random normal variable using Box-Muller transform
        float u1 = curand_uniform(&local_state);
        float u2 = curand_uniform(&local_state);
        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
        
        // Calculate stock price at maturity
        float ST = S0 * expf((r - 0.5f * sigma * sigma) * T + sigma * sqrtf(T) * z);
        
        // Calculate option payoff (call option)
        float payoff = fmaxf(ST - K, 0.0f);
        
        // Discount to present value
        results[idx] = payoff * expf(-r * T);
        
        states[idx] = local_state;
    }
}

// Host function for Monte Carlo Pi estimation
float monte_carlo_pi_cuda(int num_samples) {
    // Allocate device memory
    curandState* d_states;
    float* d_results;
    
    cudaMalloc(&d_states, num_samples * sizeof(curandState));
    cudaMalloc(&d_results, num_samples * sizeof(float));
    
    // Initialize random states
    curandSetup<<<(num_samples + 255) / 256, 256>>>(
        d_states, num_samples, time(NULL)
    );
    
    // Launch Monte Carlo kernel
    monte_carlo_pi_kernel<<<(num_samples + 255) / 256, 256>>>(
        d_states, d_results, num_samples
    );
    
    // Copy results back to host
    float* h_results = (float*)malloc(num_samples * sizeof(float));
    cudaMemcpy(h_results, d_results, num_samples * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate Pi estimate
    float sum = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        sum += h_results[i];
    }
    float pi_estimate = 4.0f * sum / num_samples;
    
    // Cleanup
    free(h_results);
    cudaFree(d_states);
    cudaFree(d_results);
    
    return pi_estimate;
}
```

## 🐍 Python Integration with CuPy

```python
# src/python/cuda_ops.py
import cupy as cp
import numpy as np
from typing import Tuple, Optional

class CUDAMatrixOps:
    """GPU-accelerated matrix operations using CuPy."""
    
    def __init__(self, device_id: int = 0):
        """Initialize CUDA operations."""
        cp.cuda.Device(device_id).use()
        self.device_id = device_id
    
    def matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Matrix multiplication on GPU."""
        # Convert to CuPy arrays
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        
        # Perform multiplication
        C_gpu = cp.dot(A_gpu, B_gpu)
        
        # Convert back to NumPy
        return cp.asnumpy(C_gpu)
    
    def batch_matrix_multiply(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Batch matrix multiplication."""
        A_gpu = cp.asarray(A)
        B_gpu = cp.asarray(B)
        
        # Use einsum for batch operations
        C_gpu = cp.einsum('bij,bjk->bik', A_gpu, B_gpu)
        
        return cp.asnumpy(C_gpu)
    
    def svd(self, A: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Singular Value Decomposition on GPU."""
        A_gpu = cp.asarray(A)
        U_gpu, s_gpu, Vt_gpu = cp.linalg.svd(A_gpu)
        
        return (
            cp.asnumpy(U_gpu),
            cp.asnumpy(s_gpu),
            cp.asnumpy(Vt_gpu)
        )
    
    def fft(self, x: np.ndarray) -> np.ndarray:
        """Fast Fourier Transform on GPU."""
        x_gpu = cp.asarray(x)
        X_gpu = cp.fft.fft(x_gpu)
        return cp.asnumpy(X_gpu)
    
    def convolution2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """2D convolution on GPU."""
        image_gpu = cp.asarray(image)
        kernel_gpu = cp.asarray(kernel)
        
        # Use CuPy's convolution
        result_gpu = cp.convolve2d(image_gpu, kernel_gpu, mode='same')
        
        return cp.asnumpy(result_gpu)

class CUDANeuralNetwork:
    """GPU-accelerated neural network using CuPy."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """Initialize neural network."""
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize weights
        self.W1 = cp.random.randn(input_size, hidden_size) * 0.1
        self.b1 = cp.zeros((1, hidden_size))
        self.W2 = cp.random.randn(hidden_size, output_size) * 0.1
        self.b2 = cp.zeros((1, output_size))
    
    def relu(self, x: cp.ndarray) -> cp.ndarray:
        """ReLU activation function."""
        return cp.maximum(0, x)
    
    def relu_derivative(self, x: cp.ndarray) -> cp.ndarray:
        """ReLU derivative."""
        return (x > 0).astype(cp.float32)
    
    def softmax(self, x: cp.ndarray) -> cp.ndarray:
        """Softmax activation function."""
        exp_x = cp.exp(x - cp.max(x, axis=1, keepdims=True))
        return exp_x / cp.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X: np.ndarray) -> np.ndarray:
        """Forward pass."""
        X_gpu = cp.asarray(X)
        
        # Hidden layer
        z1 = cp.dot(X_gpu, self.W1) + self.b1
        a1 = self.relu(z1)
        
        # Output layer
        z2 = cp.dot(a1, self.W2) + self.b2
        a2 = self.softmax(z2)
        
        return cp.asnumpy(a2)
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> dict:
        """Backward pass."""
        X_gpu = cp.asarray(X)
        y_gpu = cp.asarray(y)
        output_gpu = cp.asarray(output)
        
        m = X_gpu.shape[0]
        
        # Output layer gradients
        dz2 = output_gpu - y_gpu
        dW2 = (1/m) * cp.dot(a1.T, dz2)
        db2 = (1/m) * cp.sum(dz2, axis=0, keepdims=True)
        
        # Hidden layer gradients
        da1 = cp.dot(dz2, self.W2.T)
        dz1 = da1 * self.relu_derivative(z1)
        dW1 = (1/m) * cp.dot(X_gpu.T, dz1)
        db1 = (1/m) * cp.sum(dz1, axis=0, keepdims=True)
        
        return {
            'dW1': cp.asnumpy(dW1),
            'db1': cp.asnumpy(db1),
            'dW2': cp.asnumpy(dW2),
            'db2': cp.asnumpy(db2)
        }
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 1000, lr: float = 0.01):
        """Train the neural network."""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Calculate loss
            loss = -cp.mean(cp.sum(y * cp.log(output + 1e-8), axis=1))
            
            # Backward pass
            gradients = self.backward(X, y, output)
            
            # Update weights
            self.W1 -= lr * gradients['dW1']
            self.b1 -= lr * gradients['db1']
            self.W2 -= lr * gradients['dW2']
            self.b2 -= lr * gradients['db2']
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
```

## 🚀 Performance Optimization

### Memory Management

```cpp
// src/utils/memory_manager.cpp
#include <cuda_runtime.h>
#include <memory>

class CUDAMemoryManager {
private:
    std::vector<void*> allocated_ptrs;
    
public:
    template<typename T>
    T* allocate(size_t count) {
        T* ptr;
        cudaMalloc(&ptr, count * sizeof(T));
        allocated_ptrs.push_back(ptr);
        return ptr;
    }
    
    template<typename T>
    void copy_to_device(T* d_ptr, const T* h_ptr, size_t count) {
        cudaMemcpy(d_ptr, h_ptr, count * sizeof(T), cudaMemcpyHostToDevice);
    }
    
    template<typename T>
    void copy_to_host(T* h_ptr, const T* d_ptr, size_t count) {
        cudaMemcpy(h_ptr, d_ptr, count * sizeof(T), cudaMemcpyDeviceToHost);
    }
    
    void free_all() {
        for (void* ptr : allocated_ptrs) {
            cudaFree(ptr);
        }
        allocated_ptrs.clear();
    }
    
    ~CUDAMemoryManager() {
        free_all();
    }
};

// Unified memory management
class UnifiedMemoryManager {
public:
    template<typename T>
    T* allocate_unified(size_t count) {
        T* ptr;
        cudaMallocManaged(&ptr, count * sizeof(T));
        return ptr;
    }
    
    template<typename T>
    void prefetch_to_gpu(T* ptr, size_t count) {
        cudaMemPrefetchAsync(ptr, count * sizeof(T), 0);
    }
    
    template<typename T>
    void prefetch_to_cpu(T* ptr, size_t count) {
        cudaMemPrefetchAsync(ptr, count * sizeof(T), cudaCpuDeviceId);
    }
};
```

### CUDA Streams and Events

```cpp
// src/utils/stream_manager.cpp
#include <cuda_runtime.h>
#include <vector>

class CUDAStreamManager {
private:
    std::vector<cudaStream_t> streams;
    std::vector<cudaEvent_t> events;
    
public:
    CUDAStreamManager(int num_streams = 4) {
        streams.resize(num_streams);
        events.resize(num_streams);
        
        for (int i = 0; i < num_streams; i++) {
            cudaStreamCreate(&streams[i]);
            cudaEventCreate(&events[i]);
        }
    }
    
    cudaStream_t get_stream(int index) {
        return streams[index % streams.size()];
    }
    
    void synchronize_stream(int index) {
        cudaStreamSynchronize(streams[index % streams.size()]);
    }
    
    void synchronize_all() {
        for (auto& stream : streams) {
            cudaStreamSynchronize(stream);
        }
    }
    
    void record_event(int index) {
        cudaEventRecord(events[index % events.size()], streams[index % streams.size()]);
    }
    
    float get_elapsed_time(int start_index, int end_index) {
        float elapsed;
        cudaEventElapsedTime(&elapsed, events[start_index], events[end_index]);
        return elapsed;
    }
    
    ~CUDAStreamManager() {
        for (auto& stream : streams) {
            cudaStreamDestroy(stream);
        }
        for (auto& event : events) {
            cudaEventDestroy(event);
        }
    }
};
```

## 📊 Performance Benchmarking

```python
# src/python/benchmarks.py
import time
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from typing import List, Tuple

class CUDABenchmark:
    """CUDA performance benchmarking suite."""
    
    def __init__(self):
        self.results = {}
    
    def benchmark_matrix_multiply(self, sizes: List[int]) -> dict:
        """Benchmark matrix multiplication performance."""
        results = {'sizes': sizes, 'cpu_times': [], 'gpu_times': [], 'speedups': []}
        
        for size in sizes:
            # Generate random matrices
            A = np.random.randn(size, size).astype(np.float32)
            B = np.random.randn(size, size).astype(np.float32)
            
            # CPU benchmark
            start_time = time.time()
            C_cpu = np.dot(A, B)
            cpu_time = time.time() - start_time
            
            # GPU benchmark
            A_gpu = cp.asarray(A)
            B_gpu = cp.asarray(B)
            
            cp.cuda.Stream.null.synchronize()
            start_time = time.time()
            C_gpu = cp.dot(A_gpu, B_gpu)
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time
            
            results['cpu_times'].append(cpu_time)
            results['gpu_times'].append(gpu_time)
            results['speedups'].append(speedup)
            
            print(f"Size {size}x{size}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Speedup={speedup:.2f}x")
        
        return results
    
    def benchmark_neural_network(self, batch_sizes: List[int], input_size: int = 784, hidden_size: int = 128, output_size: int = 10) -> dict:
        """Benchmark neural network training performance."""
        results = {'batch_sizes': batch_sizes, 'cpu_times': [], 'gpu_times': [], 'speedups': []}
        
        for batch_size in batch_sizes:
            # Generate random data
            X = np.random.randn(batch_size, input_size).astype(np.float32)
            y = np.random.randn(batch_size, output_size).astype(np.float32)
            
            # CPU benchmark (simplified)
            start_time = time.time()
            # Simulate CPU computation
            for _ in range(100):
                _ = np.dot(X, np.random.randn(input_size, hidden_size))
            cpu_time = time.time() - start_time
            
            # GPU benchmark
            X_gpu = cp.asarray(X)
            y_gpu = cp.asarray(y)
            
            cp.cuda.Stream.null.synchronize()
            start_time = time.time()
            for _ in range(100):
                _ = cp.dot(X_gpu, cp.random.randn(input_size, hidden_size))
            cp.cuda.Stream.null.synchronize()
            gpu_time = time.time() - start_time
            
            speedup = cpu_time / gpu_time
            
            results['cpu_times'].append(cpu_time)
            results['gpu_times'].append(gpu_time)
            results['speedups'].append(speedup)
            
            print(f"Batch size {batch_size}: CPU={cpu_time:.4f}s, GPU={gpu_time:.4f}s, Speedup={speedup:.2f}x")
        
        return results
    
    def plot_results(self, results: dict, title: str, xlabel: str, ylabel: str):
        """Plot benchmark results."""
        plt.figure(figsize=(10, 6))
        
        if 'sizes' in results:
            x = results['sizes']
        else:
            x = results['batch_sizes']
        
        plt.subplot(1, 2, 1)
        plt.plot(x, results['cpu_times'], 'b-o', label='CPU')
        plt.plot(x, results['gpu_times'], 'r-o', label='GPU')
        plt.xlabel(xlabel)
        plt.ylabel('Time (seconds)')
        plt.title(f'{title} - Execution Time')
        plt.legend()
        plt.grid(True)
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        plt.plot(x, results['speedups'], 'g-o', label='Speedup')
        plt.xlabel(xlabel)
        plt.ylabel('Speedup (x)')
        plt.title(f'{title} - Speedup')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    benchmark = CUDABenchmark()
    
    # Matrix multiplication benchmark
    sizes = [256, 512, 1024, 2048, 4096]
    matrix_results = benchmark.benchmark_matrix_multiply(sizes)
    benchmark.plot_results(matrix_results, 'Matrix Multiplication', 'Matrix Size', 'Time (seconds)')
    
    # Neural network benchmark
    batch_sizes = [32, 64, 128, 256, 512]
    nn_results = benchmark.benchmark_neural_network(batch_sizes)
    benchmark.plot_results(nn_results, 'Neural Network Training', 'Batch Size', 'Time (seconds)')
```

## 📚 Learning Resources

- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CuPy Documentation](https://cupy.dev/)
- [Numba CUDA Documentation](https://numba.readthedocs.io/en/stable/cuda/index.html)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

## 🔗 Upstream Source

- **Repository**: [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)
- **CuPy**: [cupy/cupy](https://github.com/cupy/cupy)
- **Documentation**: [developer.nvidia.com/cuda](https://developer.nvidia.com/cuda-zone)
- **License**: Apache-2.0
