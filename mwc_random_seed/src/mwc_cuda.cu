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


#define EXPECTED_COUNT_MAX 16
__global__ void mwc_find_seed_kernel(
  __restrict__ std::uint32_t factor_in,
  __restrict__ std::size_t init_addition,
  __restrict__ std::size_t init_limit,
  __restrict__ std::uint32_t carry_init,
  __restrict__ std::uint32_t advance_limit,
  __restrict__ std::uint32_t modulo,
  __restrict__ std::uint32_t* expected,
  __restrict__ std::size_t expected_count  
) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) + init_addition;
  if (i > init_limit) {
    return;
  }
  std::uint64_t factor = factor_in;
  std::uint16_t calc[EXPECTED_COUNT_MAX * 2];
  std::uint32_t carry = carry_init;
  std::uint32_t value = i;

  std::uint32_t offsets[2] = {0, 0};
  for (std::size_t a = 0; a < advance_limit; a++) {
    advance(factor, carry, value);
    std::uint32_t inner_carry = carry_init;
    std::uint32_t inner_value = value;
    advance(factor, inner_carry, inner_value);
    const auto oddeven = a % 2;
    calc[oddeven * EXPECTED_COUNT_MAX + offsets[oddeven]] = inner_value % modulo;
    offsets[oddeven] = (offsets[oddeven] + 1) % EXPECTED_COUNT_MAX;
    // Do the compare.
    printf("i: %d, a: %lu, oddeven: %lu\n", i, a, oddeven);
    for (std::size_t c = 0; c < expected_count; c++) {

      const auto expect = expected[c];
      const auto value_offset = (offsets[oddeven] + c) % EXPECTED_COUNT_MAX;
      const auto value_obtained = calc[oddeven * EXPECTED_COUNT_MAX + value_offset];
      printf("%d, ", value_obtained);
      if (expect != value_obtained) {
        break;
      }
      if (c == (expected_count- 1)) {
        printf("Found it at %d, %lu\n", i, a);
        return;
      }
    }
    printf("\n");
  }
}

std::string hp(std::uint32_t v){
  std::stringstream ss;
  ss << std::hex<< std::setfill('0') << std::setw(8) << v;
  return ss.str();
}
}

#define MAIN 1 // for syntax highlighting.
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


void test_mwc_find_seed() {
  const std::size_t seed_limit = 5;
  const std::size_t advance_limit = 500;
  const std::size_t factor = 1791398085;

  const auto l = 150;
  const auto h = 500;
  const auto modulo = h - l;
  std::array<std::uint32_t, 7> expected_values {201 - l, 484 - l, 188 - l, 496 - l, 432 - l, 347 - l, 356 - l};


  std::uint32_t* expected;
  cudaMallocManaged(&expected, sizeof(std::uint32_t) * expected_values.size());
  for (std::size_t i = 0; i < expected_values.size(); i++){
    expected[i] = expected_values[i];
  }

  const auto carry_init = 333 * 2;

  //  const std::size_t N = 512;
  const auto N = seed_limit;
  int blockSize = 1;
  //  int numBlocks = ((N +  blockSize - 1) / blockSize) + 1;
  int numBlocks = 1;
  mwc_find_seed_kernel<<<numBlocks, blockSize>>>(factor, 3, seed_limit, carry_init, advance_limit, modulo, expected, expected_values.size());

  cudaDeviceSynchronize();


  // Free memory
  cudaFree(expected);

}
int main(int argc, char* argv[]) {
  //  test_generation();
  test_mwc_find_seed();
  return 0;
}

#endif
