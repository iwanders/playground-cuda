#include <iostream>
#include <cmath>
#include <cstdint>

__global__
void add(std::size_t n, float *x, float *y)
{
  for (std::size_t i = 0; i < n; i++) {
    y[i] = x[i] + y[i];
  }
}

int main(int argc, char* argv[]) {
  constexpr std::size_t N = 1<<20;

  float *x = nullptr;
  float *y = nullptr;

  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  for (std::size_t i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  add<<<1, 1>>>(N, x, y);

  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (std::size_t i = 0; i < N; i++){
    maxError = std::max(maxError, std::fabs(y[i]-3.0f));
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