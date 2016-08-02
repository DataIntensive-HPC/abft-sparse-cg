#define ECC7_P1_0 0x56AAAD5B
#define ECC7_P1_1 0xAB555555
#define ECC7_P1_2 0x80AAAAAA

#define ECC7_P2_0 0x9B33366D
#define ECC7_P2_1 0xCD999999
#define ECC7_P2_2 0x40CCCCCC

#define ECC7_P3_0 0xE3C3C78E
#define ECC7_P3_1 0xF1E1E1E1
#define ECC7_P3_2 0x20F0F0F0

#define ECC7_P4_0 0x03FC07F0
#define ECC7_P4_1 0x01FE01FE
#define ECC7_P4_2 0x10FF00FF

#define ECC7_P5_0 0x03FFF800
#define ECC7_P5_1 0x01FFFE00
#define ECC7_P5_2 0x08FFFF00

#define ECC7_P6_0 0xFC000000
#define ECC7_P6_1 0x01FFFFFF
#define ECC7_P6_2 0x04000000

#define ECC7_P7_0 0x00000000
#define ECC7_P7_1 0xFE000000
#define ECC7_P7_2 0x02FFFFFF

typedef struct
{
  double value;
  uint column;
} __attribute__((packed)) csr_element;

#define PARITY_METHOD_0 0 //slower than __builtin_parity
#define PARITY_METHOD_1 1 //slightly than __builtin_parity
#define PARITY_METHOD_2 2 //around the same as __builtin_parity, maybe sligtly faster
#define PARITY_METHOD_3 3
#define PARITY_METHOD_4 4
#define PARITY_METHOD_5 5
#define __PARITY_METHOD PARITY_METHOD_5

__constant uchar PARITY_TABLE[256] =
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

inline uchar calc_parity(uint x)
{
#if __PARITY_METHOD == PARITY_METHOD_0
   uint y;
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
  return popcount(x) & 1;
#elif __PARITY_METHOD == PARITY_METHOD_3
  return __builtin_parity(x);
#elif __PARITY_METHOD == PARITY_METHOD_4
  uchar * p = (uchar *) &x;
  return PARITY_TABLE[p[0] ^ p[1] ^ p[2] ^ p[3]];
#elif __PARITY_METHOD == PARITY_METHOD_5
  x ^= x >> 16;
  x ^= x >> 8;
  x ^= x >> 4;
  x &= 0xf;
  return (0x6996 >> x) & 1;
#endif
}

inline uint is_power_of_2(uint x)
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
inline uint ecc_compute_col8(csr_element colval)
{
  uint *data = (uint*)&colval;

  uint result = 0;

  uint p;

  p = (data[0] & ECC7_P1_0) ^ (data[1] & ECC7_P1_1) ^ (data[2] & ECC7_P1_2);
  result |= calc_parity(p) << 31U;

  p = (data[0] & ECC7_P2_0) ^ (data[1] & ECC7_P2_1) ^ (data[2] & ECC7_P2_2);
  result |= calc_parity(p) << 30U;

  p = (data[0] & ECC7_P3_0) ^ (data[1] & ECC7_P3_1) ^ (data[2] & ECC7_P3_2);
  result |= calc_parity(p) << 29U;

  p = (data[0] & ECC7_P4_0) ^ (data[1] & ECC7_P4_1) ^ (data[2] & ECC7_P4_2);
  result |= calc_parity(p) << 28U;

  p = (data[0] & ECC7_P5_0) ^ (data[1] & ECC7_P5_1) ^ (data[2] & ECC7_P5_2);
  result |= calc_parity(p) << 27U;

  p = (data[0] & ECC7_P6_0) ^ (data[1] & ECC7_P6_1) ^ (data[2] & ECC7_P6_2);
  result |= calc_parity(p) << 26U;

  p = (data[0] & ECC7_P7_0) ^ (data[1] & ECC7_P7_1) ^ (data[2] & ECC7_P7_2);
  result |= calc_parity(p) << 25U;

  return result;
}

// Compute the overall parity of a 96-bit matrix element
inline uint ecc_compute_overall_parity(csr_element colval)
{
  uint *data = (uint*)&colval;
  return calc_parity(data[0] ^ data[1] ^ data[2]);
}

// This function will use the error 'syndrome' generated from a 7-bit parity
// check to determine the index of the bit that has been flipped
inline uint ecc_get_flipped_bit_col8(uint syndrome)
{
  // Compute position of flipped bit
  uint hamm_bit = 0;
  for (int p = 1; p <= 7; p++)
  {
    hamm_bit += (syndrome >> (32U-p)) & 0x1U ? 0x1U<<(p-1) : 0;
  }

  // Map to actual data bit position
  uint data_bit = hamm_bit - (32-clz(hamm_bit)) - 1;
  if(is_power_of_2(hamm_bit))
    data_bit = clz(hamm_bit) + 64;

  return data_bit;
}
