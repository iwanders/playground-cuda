#include <iostream>
#include <cmath>
#include <cstdint>
#include <unistd.h>

__global__
void add(std::size_t n, float *x, float *y) {
  //  int index = threadIdx.x;
  //  int stride = blockDim.x;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  //  printf("Stride %d, index: %d\n", stride, index);
  for (int i = index; i < n; i += stride) {
    y[i] = x[i] + y[i];
    //  printf("i %d  y[i]: %f\n", i, y[i]);
  }
}

int main(int argc, char* argv[]) {

  constexpr std::size_t N = 1<<24;
  std::cout << "N: " << N << std::endl;

  float *x = nullptr;
  float *y = nullptr;

  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (std::size_t i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  //  sleep(5);

  // Run kernel on 1M elements on the CPU
  // This threads doesn't need to be compile time constant!
  //  int threads = 256;
  int blockSize = 256;
  int numBlocks = ((N + blockSize - 1) / blockSize);
  //  int blockSize = 1;
  //  int numBlocks = 1;
  add<<<numBlocks, blockSize>>>(N, x, y);

  cudaDeviceSynchronize();

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (std::size_t i = 0; i < N; i++){
    const auto difference = fabs(y[i]-3.0f);
    if (difference > 0.01) {
      std::cout << "i: " << i << " has: " << difference << std::endl;
    }
    //  printf("i %lu  y[i]: %f\n", i, y[i]);
    maxError = fmax(maxError,  difference);
  }
  std::cout << "Max error: " << maxError << std::endl;
  // Max error is 1.

  std::cout << std::endl;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0); // 0-th device
  std::cout << "name: " << deviceProp.name << std::endl;
  std::cout << "multiProcessorCount: " << deviceProp.multiProcessorCount << std::endl;
  std::cout << "memoryClockRate: " << deviceProp.memoryClockRate << std::endl;
  std::cout << "maxThreadsPerMultiProcessor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
  std::cout << "maxThreadsPerBlock: " << deviceProp.maxThreadsPerBlock << std::endl;
  std::cout << "maxBlocksPerMultiProcessor: " << deviceProp.maxBlocksPerMultiProcessor << std::endl;
  std::cout << std::endl;
  // Free memory
  cudaFree(x);
  cudaFree(y);
  return 0;
}