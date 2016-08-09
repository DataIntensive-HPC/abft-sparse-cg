#include "ecc.h"
#include <cuda.h>

#define PARITY_METHOD_0 0 //slower than __builtin_parity
#define PARITY_METHOD_1 1 //slightly than __builtin_parity
#define PARITY_METHOD_2 2 //around the same as __builtin_parity, maybe sligtly faster
#define PARITY_METHOD_3 3
#define PARITY_METHOD_4 4
#define PARITY_METHOD_5 5
#define __PARITY_METHOD PARITY_METHOD_2

#if __PARITY_METHOD == PARITY_METHOD_4
uint8_t PARITY_TABLE[256] =
{
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,

  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,

  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,

  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
  0, 1, 1, 0, 1, 0, 0, 1,
  0, 1, 1, 0, 1, 0, 0, 1,
  1, 0, 0, 1, 0, 1, 1, 0,
};
#endif

__device__ inline uint8_t cu_calc_parity(uint32_t x)
{
#if __PARITY_METHOD == PARITY_METHOD_0
   uint32_t y;
   y = x ^ (x >> 1);
   y = y ^ (y >> 2);
   y = y ^ (y >> 4);
   y = y ^ (y >> 8);
   y = y ^ (y >>16);
   return y & 1;
#elif __PARITY_METHOD == PARITY_METHOD_1
    x ^= x >> 1;
    x ^= x >> 2;
    x = (x & 0x11111111U) * 0x11111111U;
    return (x >> 28) & 1;
#elif __PARITY_METHOD == PARITY_METHOD_2
  return __popc(x) & 1;
#elif __PARITY_METHOD == PARITY_METHOD_3
  return __builtin_parity(x);
#elif __PARITY_METHOD == PARITY_METHOD_4
  uint8_t * p = (uint8_t *) &x;
  return PARITY_TABLE[p[0] ^ p[1] ^ p[2] ^ p[3]];
#elif __PARITY_METHOD == PARITY_METHOD_5
  x ^= x >> 16;
  x ^= x >> 8;
  x ^= x >> 4;
  x &= 0xf;
  return (0x6996 >> x) & 1;
#endif
}

__device__ inline uint32_t cu_is_power_of_2(uint32_t x)
{
  return ((x != 0) && !(x & (x - 1)));
}

// This function will generate/check the 7 parity bits for the given matrix
// element, with the parity bits stored in the high order bits of the column
// index.
//
// This will return a 32-bit integer where the high 7 bits are the generated
// parity bits.
//
// To check a matrix element for errors, simply use this function again, and
// the returned value will be the error 'syndrome' which will be non-zero if
// an error occured.
__device__ inline uint32_t cu_ecc_compute_col8(uint32_t col, uint32_t val[2])
{
  uint32_t result = 0;

  uint32_t p;

  p = (val[0] & ECC7_P1_0) ^ (val[1] & ECC7_P1_1) ^ (col & ECC7_P1_2);
  result |= cu_calc_parity(p) << 31U;

  p = (val[0] & ECC7_P2_0) ^ (val[1] & ECC7_P2_1) ^ (col & ECC7_P2_2);
  result |= cu_calc_parity(p) << 30U;

  p = (val[0] & ECC7_P3_0) ^ (val[1] & ECC7_P3_1) ^ (col & ECC7_P3_2);
  result |= cu_calc_parity(p) << 29U;

  p = (val[0] & ECC7_P4_0) ^ (val[1] & ECC7_P4_1) ^ (col & ECC7_P4_2);
  result |= cu_calc_parity(p) << 28U;

  p = (val[0] & ECC7_P5_0) ^ (val[1] & ECC7_P5_1) ^ (col & ECC7_P5_2);
  result |= cu_calc_parity(p) << 27U;

  p = (val[0] & ECC7_P6_0) ^ (val[1] & ECC7_P6_1) ^ (col & ECC7_P6_2);
  result |= cu_calc_parity(p) << 26U;

  p = (val[0] & ECC7_P7_0) ^ (val[1] & ECC7_P7_1) ^ (col & ECC7_P7_2);
  result |= cu_calc_parity(p) << 25U;

  return result;
}

// Compute the overall parity of a 96-bit matrix element
__device__ inline uint32_t cu_ecc_compute_overall_parity(const uint32_t col, const uint32_t val[2])
{
  return cu_calc_parity(col ^ val[0] ^ val[1]);
}

// This function will use the error 'syndrome' generated from a 7-bit parity
// check to determine the index of the bit that has been flipped
__device__ inline uint32_t cu_ecc_get_flipped_bit_col8(const uint32_t syndrome)
{
  // Compute position of flipped bit
  uint32_t hamm_bit = 0;
  for (int p = 1; p <= 7; p++)
  {
    hamm_bit += (syndrome >> (32U-p)) & 0x1U ? 0x1U<<(p-1) : 0;
  }

  // Map to actual data bit position
  uint32_t data_bit = hamm_bit - (32-__clz(hamm_bit)) - 1;
  if(cu_is_power_of_2(hamm_bit))
    data_bit = __clz(hamm_bit) + 64;

  return data_bit;
}

#define CU_CORRECT_BIT(bit, col, val)\
if(1){\
  if     (bit < 32) val[0] ^= 0x1U << bit;\
  else if(bit < 64) val[1] ^= 0x1U << (bit - 32);\
  else              col    ^= 0x1U << (bit - 64);\
} else
