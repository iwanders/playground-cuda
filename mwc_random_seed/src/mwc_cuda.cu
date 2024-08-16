#include <cstdint>
#include <array>
#include <iostream>
#include <iomanip>
#include <unistd.h>

extern "C" {

__device__ inline void advance(const std::uint64_t& factor, std::uint32_t& carry, std::uint32_t& value) {
    std::uint64_t value_u64 = static_cast<std::uint64_t>(value);
    std::uint64_t carry_u64 = static_cast<std::uint64_t>(carry);
    std::uint64_t new_value = factor * value_u64 + carry_u64;
    carry = new_value >> 32;
    value = new_value;
}

__global__ void mwc_store_output_kernel(
  __restrict__ std::uint32_t factor_in,
  std::size_t count_in,
  __restrict__ std::uint32_t* carry_init,
  __restrict__ std::uint32_t* value_init,
  __restrict__ std::uint32_t** out,
  std::size_t advances) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count_in) {
    return;
  }

  std::uint32_t factor = factor_in;
  std::uint32_t carry = carry_init[i];
  std::uint32_t value = value_init[i];
  
  for (std::size_t c = 0; c < advances; c++) {
    advance(factor, carry, value);
    out[i][c] = value;
  }
}



std::string hp(std::uint32_t v){
  std::stringstream ss;
  ss << std::hex<< std::setfill('0') << std::setw(8) << v;
  return ss.str();
}
}

#ifdef MAIN

void test_generation() {
  std::size_t N = 1ul<<16;
  std::size_t advances = 500;

  std::uint32_t* carry_init;
  std::uint32_t* value_init;
  cudaMallocManaged(&carry_init, N * sizeof(std::uint32_t));
  cudaMallocManaged(&value_init, N * sizeof(std::uint32_t));
  std::uint32_t** value_out;
  cudaMallocManaged(&value_out, N * sizeof(std::uint32_t*));
  for (std::size_t o=0; o < N; o++){
    cudaMallocManaged(&value_out[o], advances * sizeof(std::uint32_t));
  }
  sleep(5);

  // initialize x and y arrays on the host
  for (std::size_t i = 0; i < N; i++) {
    carry_init[i] = 333 * 2;
    value_init[i] = 1;
  }

  int blockSize = 256;
  int numBlocks = ((N + blockSize - 1) / blockSize) + 1;
  mwc_store_output_kernel<<<numBlocks, blockSize>>>(1791398085, N, carry_init, value_init, value_out, advances);

  cudaDeviceSynchronize();

  std::array<std::uint32_t, 5> expected{0x6AC6935F, 0x2F2ED81B, 0x280687C4, 0xB6AAB839, 0xBFC793C3};
  for (std::size_t o = 0; o < N; o++){
    for (std::size_t i = 0; i < expected.size(); i++) {
      if (expected[i] != value_out[o][i]) {
        std::cout <<  "Test failed, at o=" << o << " i: " << i << " value: " << hp(value_out[o][i]) << " expected: " << hp(expected[i])  << std::endl;
      }
    }
  }

  // Free memory
  cudaFree(carry_init);
  cudaFree(value_init);
  for (std::size_t o=0; o < N; o++){
    cudaMallocManaged(&value_out[o], advances * sizeof(std::uint32_t));
  }
  cudaFree(value_out);
}

int main(int argc, char* argv[]) {
  test_generation();
  return 0;
}

#endif
