#include <cstdint>
#include <array>
#include <iostream>
#include <iomanip>
#include <unistd.h>

extern "C" {


/// Single advance for the multiply with carry random number generator.
__device__ inline void advance(const std::uint64_t& factor, std::uint32_t& carry, std::uint32_t& value) {
    std::uint64_t value_u64 = static_cast<std::uint64_t>(value);
    std::uint64_t carry_u64 = static_cast<std::uint64_t>(carry);
    std::uint64_t new_value = factor * value_u64 + carry_u64;
    carry = new_value >> 32;
    value = new_value;
}

/// Function to produce and store sequences of generated numbers.
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


/// Maximum number of expected values, this needs to be a constant an array of this size is allocated.
#define EXPECTED_COUNT_MAX 16


/// Function that searches for an expected sequence.
///
/// This searches all seeds up to init_limit, advances up to advance_limit for each seed. The
/// generated values are modulo'd with the provided value and matched against the expected values.
/// If the consecutive sequence of generated and modulo'd numbers match against the expected numbers
/// it gets added to output_matches. If output_matches is fully populated, the search is stopped.
__global__ void mwc_find_seed_kernel(
  __restrict__ std::uint32_t factor_in,
  __restrict__ std::size_t init_addition,
  __restrict__ std::size_t init_limit,
  __restrict__ std::uint32_t carry_init,
  __restrict__ std::uint32_t advance_limit,
  __restrict__ std::uint32_t modulo,
  __restrict__ std::uint32_t* expected,
  __restrict__ std::size_t expected_count,
  __restrict__ std::uint32_t* output_matches,
  __restrict__ std::size_t output_in,
  __restrict__ std::uint32_t* output_found // handled atomically.
) {
  int i = (blockIdx.x * blockDim.x + threadIdx.x) + init_addition;
  if (i > init_limit) {
    return;
  }

  {
    const std::uint32_t current_index = atomicAdd(output_found, 0);
    if (current_index >= output_in) {
      return;
    }
  }

  std::uint64_t factor = factor_in;
  std::uint16_t calc[EXPECTED_COUNT_MAX * 2];
  std::uint32_t carry = carry_init;
  //  std::uint32_t value = static_cast<std::uint32_t>(i) + static_cast<std::uint32_t>(1u<<31);
  std::uint32_t value = i;
  std::uint32_t init_value = i;
  //  advance(factor, carry, value);

  if (i % (2 << 24) == 0) {
    printf("i: %u\n", value);
  }

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
    //  printf("i: %d, a: %lu, oddeven: %lu\n", i, a, oddeven);
    for (std::size_t c = 0; c < expected_count; c++) {

      const auto expect = expected[c];
      const auto value_offset = (offsets[oddeven] + c) % EXPECTED_COUNT_MAX;
      const auto value_obtained = calc[oddeven * EXPECTED_COUNT_MAX + value_offset];
      //  printf("%d, ", value_obtained);
      if (expect != value_obtained) {
        break;
      }
      if (c == (expected_count- 1)) {
        const auto index = atomicAdd(output_found, 1);
        if (index >= output_in) {
          return;
        }
        output_matches[index] = init_value;
      }
      //  if (c > 3) {
        //  printf("Found it at %d, %lu, c: %lu\n", i, a, c);
      //  }
    }
    //  printf("\n");
    if ((a == advance_limit - 1) && (i % (2<<16) == 0)) {
      //  printf("Reached limit at %lu, for seed %d, value %x\n", a, i, value);
    }
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
  std::size_t advances = 10000;

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
  const std::size_t seed_limit = 1u<<24;
  //  const std::size_t advance_limit = 500;
  const std::size_t advance_limit = 500;
  const std::size_t factor = 1791398085;

  const auto l = 150;
  const auto h = 500;
  const auto modulo = h - l;

  // All roughly ~400 in
  //  std::array<std::uint32_t, 7> expected_values {201 - l, 484 - l, 188 - l, 496 - l, 432 - l, 347 - l, 356 - l}; // seed 3, known, works
  //  std::array<std::uint32_t, 7> expected_values {374 - l, 488 - l, 441 - l, 332 - l, 417 - l, 254 - l, 294 - l}; // seed 4, known, works
  //  std::array<std::uint32_t, 7> expected_values {286 - l, 301 - l, 485 - l, 271 - l, 443 - l, 449 - l, 281 - l}; // seed 1337, known, works
  std::array<std::uint32_t, 7> expected_values {418 - l, 363 - l, 274 - l, 348 - l, 162 - l, 219 - l, 282 - l}; // seed 65536, known, works
  //  std::array<std::uint32_t, 7> expected_values {386 - l, 201 - l, 311 - l, 164 - l, 185 - l, 251 - l, 264 - l}; // seed 33554432, known, works
  //  std::array<std::uint32_t, 7> expected_values {263 - l, 375 - l, 393 - l, 269 - l, 422 - l, 418 - l, 328 - l}; // seed 536870912, known, works
  //  std::array<std::uint32_t, 7> expected_values {494 - l, 228 - l, 341 - l, 478 - l, 310 - l, 498 - l, 281 - l}; // seed 1073872896, known, works

  //  std::array<std::uint32_t, 7> expected_values {207 - l, 272 - l, 297 - l, 413 - l, 207 - l, 235 - l, 268 - l}; // seed 2181038080, overflow!, known, 402, at 2147483647 = 0x7fffffff
  //  std::array<std::uint32_t, 7> expected_values {394 - l, 306 - l, 448 - l, 203 - l, 449 - l, 389 - l, 408 - l}; // offline, random seed; calculated 845361015 @ 408
  //  std::array<std::uint32_t, 7> expected_values {284 - l, 296 - l, 393 - l, 248 - l, 434 - l, 162 - l, 291 - l}; // offline, random seed; calculated 1702494920,  @ 398







  // Stash is full...
  // All have pickup from ground.
  //  std::array<std::uint32_t, 7> expected_values {364 - l, 480 - l, 317 - l, 210 - l, 368 - l, 224 - l, 303 - l}; // new, late in advance, unknown
  //  std::array<std::uint32_t, 7> expected_values {427 - l, 439 - l, 173 - l, 356 - l, 170 - l, 355 - l, 382 - l}; // new, 'early' in advance, unknown
  //  std::array<std::uint32_t, 6> expected_values {460 - l, 284 - l, 367 - l, 326 - l, 256 - l, 230 - l}; // new, 'early' in advance, unknown, 
  // not from ground
  //  std::array<std::uint32_t, 7> expected_values {399 - l, 468 - l, 313 - l, 235 - l, 377 - l, 362 - l, 247 - l}; // new, 'early' in advance, unknown, 


  std::uint32_t* expected;
  cudaMallocManaged(&expected, sizeof(std::uint32_t) * expected_values.size());
  for (std::size_t i = 0; i < expected_values.size(); i++){
    expected[i] = expected_values[i];
  }


  //  __restrict__ std::uint32_t factor_in,
  //  __restrict__ std::size_t init_addition,
  //  __restrict__ std::size_t init_limit,
  //  __restrict__ std::uint32_t carry_init,
  //  __restrict__ std::uint32_t advance_limit,
  //  __restrict__ std::uint32_t modulo,
  //  __restrict__ std::uint32_t* expected,
  //  __restrict__ std::size_t expected_count,
  //  __restrict__ std::uint32_t* output_matches,
  //  __restrict__ std::size_t output_count,
  //  __restrict__ std::uint32_t* output_found, // handled atomically.
  std::uint32_t* output_matches;
  std::size_t output_count = 10;
  cudaMallocManaged(&output_matches, sizeof(std::uint32_t) * output_count);
  std::uint32_t* output_found;
  cudaMallocManaged(&output_found, sizeof(std::uint32_t));

  const auto carry_init = 333 * 2;

  //  const std::size_t N = 512;
  const auto N = seed_limit;
  int blockSize = 256;
  int numBlocks = ((N +  blockSize - 1) / blockSize) + 1;
  //  int numBlocks = 1;
  mwc_find_seed_kernel<<<numBlocks, blockSize>>>(factor, 0, seed_limit, carry_init, advance_limit, modulo, expected, expected_values.size(), output_matches, output_count, output_found);


  // Avoid spinloop by sleeping this thread until woken by the OS.
  // https://forums.developer.nvidia.com/t/cpu-spins-while-waiting-for-gpu-to-finish-computation/241672/5
  // https://forums.developer.nvidia.com/t/best-practices-for-cudadevicescheduleblockingsync-usage-pattern-on-linux/180741/2
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  cudaDeviceSynchronize();

  printf("Found outputs: %u\n", *output_found);
  for (int i =0; i < *output_found; i++) {
    printf(" at: 0x%08x (%d)\n", output_matches[i], output_matches[i]);

  }
  // Free memory
  cudaFree(expected);

}
int main(int argc, char* argv[]) {
  //  test_generation();
  test_mwc_find_seed();
  return 0;
}

#endif
