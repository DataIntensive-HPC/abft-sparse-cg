#include "crc.h"

uint32_t crc32c_software_simple(uint32_t crc, const uint8_t * data, size_t num_bytes)
{
  while (num_bytes--)
  {
    crc = (crc >> 8) ^ crc32c_table[0][(crc & 0xFF) ^ *data++];
  }
  return crc;
}

//This function does 32bit at a time vs the simple version with only 8 bits at a time
uint32_t crc32c_software_split(uint32_t crc, const uint32_t * data, size_t num_bytes)
{
  // process eight bytes at once
  while (num_bytes >= 8)
  {
    uint32_t one = *data++ ^ crc;
    uint32_t two = *data++;
    crc = crc32c_table[7][ one      & 0xFF] ^ crc32c_table[6][(one>> 8) & 0xFF] ^
          crc32c_table[5][(one>>16) & 0xFF] ^ crc32c_table[4][ one>>24        ] ^
          crc32c_table[3][ two      & 0xFF] ^ crc32c_table[2][(two>> 8) & 0xFF] ^
          crc32c_table[1][(two>>16) & 0xFF] ^ crc32c_table[0][ two>>24        ];
    num_bytes -= 8;
  }
  return crc32c_software_simple(crc, (uint8_t*)data, num_bytes);
}

#define CALC_CRC32C(crc_macro, crc, type, data, num_bytes)                                 \
  do                                                                                       \
  {                                                                                        \
    for (; num_bytes >= sizeof(type); num_bytes -= sizeof(type), data += sizeof(type))     \
    {                                                                                      \
      crc_macro(crc, crc, *(type *)data);                                                  \
    }                                                                                      \
  } while(0)


uint32_t crc32c_chunk(uint32_t crc, const uint8_t * data, size_t num_bytes)
{
#if defined(SOFTWARE_CRC_SIMPLE)
  crc = crc32c_software_simple(crc, data, num_bytes);
#elif defined(SOFTWARE_CRC_SPLIT)
  crc = crc32c_software_split(crc, (uint32_t *)data, num_bytes);
#elif defined(INTEL_ASM)
  //use Intel assembly code to accelerate crc calculations
  crc = crc_pcl(data, num_bytes, crc);
#else
  //run hardware crc instructions using intrinsics
  //do as much as possible with each instruction
  CALC_CRC32C(CRC32CD, crc, uint64_t, data, num_bytes);
  CALC_CRC32C(CRC32CW, crc, uint32_t, data, num_bytes);
  CALC_CRC32C(CRC32CH, crc, uint16_t, data, num_bytes);
  CALC_CRC32C(CRC32CB, crc, uint8_t, data, num_bytes);
#endif
  return crc;
}

uint32_t generate_crc32c_bits(uint32_t * a_cols, double * a_non_zeros, uint32_t num_elements)
{
  uint32_t crc = 0xFFFFFFFF;
  //remove masks
  crc = crc32c_chunk(crc, (uint8_t*)a_cols, sizeof(uint32_t) * num_elements);
  crc = crc32c_chunk(crc, (uint8_t*)a_non_zeros, sizeof(double) * num_elements);

  return crc;
}

void printBits(size_t const size, void const * const ptr)
{
  unsigned char *b = (unsigned char*) ptr;
  unsigned char byte;
  int i, j;

  for (i=size-1;i>=0;i--)
  {
    for (j=7;j>=0;j--)
    {
        byte = (b[i] >> j) & 1;
        printf("%u", byte);
    }
  }
}

uint8_t check_correct_crc32c_bits(uint32_t * a_cols, double * a_non_zeros, uint32_t idx, uint32_t num_elements)
{
  uint32_t masks[4];
  //get the CRC and recalculate to check it's correct
  uint32_t prev_crc = 0;

  for(int i = 0; i < 4; i++)
  {
    prev_crc |= (a_cols[idx + i] & 0xFF000000)>>(8*i);
    masks[i] = a_cols[idx + i] & 0xFF000000;
    a_cols[idx + i] &= 0x00FFFFFF;
  }
  uint32_t current_crc = generate_crc32c_bits(&a_cols[idx], &a_non_zeros[idx], num_elements);
  uint8_t correct_crc = prev_crc == current_crc;

  //restore masks
  for(int i = 0; i < 4; i++)
  {
    a_cols[idx + i] += masks[i];
  }

  if(!correct_crc)
  {
    // for(uint32_t i = 0; i < num_elements; i++)
    // {
    //   printf("%u ", a_cols[idx+i]);
    //   printBits(sizeof(uint32_t), &a_cols[idx+i]);
    //   printf("\n");
    // }
    //try to correct one bit of CRC

    //first try to correct the data
    const uint32_t crc_xor = prev_crc ^ current_crc;
    const size_t num_bytes = num_elements * (sizeof(uint32_t) + sizeof(double));
    uint8_t * test_data = (uint8_t*) malloc(num_bytes);

    uint8_t found_bitflip = 0;

    uint32_t bit_index = 0;
    uint32_t element_index = idx;

    size_t row_bitflip_index;

    for(size_t i = 0; i < num_bytes * 8; i++)
    {
      for(size_t byte = 0; byte < num_bytes; byte++)
      {
        test_data[byte] = 0;
      }
      test_data[i/8] = 1 << (i%8);

      uint32_t crc = 0;
      crc = crc32c_chunk(crc, test_data, num_bytes);

      //found the bit flip
      if(crc == crc_xor)
      {
        row_bitflip_index = i;
        found_bitflip = 1;
        printf("Found bitlfip %zu\n", row_bitflip_index);

        if(row_bitflip_index < num_elements * (8 * sizeof(uint32_t)))
        {
          bit_index = 64 + row_bitflip_index % (8 * sizeof(uint32_t));
          element_index += row_bitflip_index / (8 * sizeof(uint32_t));
        }
        else
        {
          row_bitflip_index -= num_elements * (8 * sizeof(uint32_t));
          bit_index = row_bitflip_index % (8 * sizeof(double));
          element_index += row_bitflip_index / (8 * sizeof(double));
        }
      }
    }

    //if the bitflip was not found in the data
    if(!found_bitflip)
    {
      // the CRC might be corrupted
      // if there is one bit difference between stored CRC
      // and the calculated CRC then this was the error
      if(__builtin_popcount(crc_xor) == 1)
      {
        found_bitflip = 1;
        uint32_t crc_bit_diff_index = __builtin_ctz(crc_xor);
        bit_index = 88 + crc_bit_diff_index % 8;
        element_index += 3 - crc_bit_diff_index / 8;
        printf("crc_bit_diff_index %u bit index %u element_index %u\n", crc_bit_diff_index, bit_index, element_index);
      }
    }

    //if the bitflip was found then fixit
    if(found_bitflip)
    {
      printf("Bit flip found\n");
      if (bit_index < 64)
      {
        uint64_t temp;
        // printBits(sizeof(double), &a_non_zeros[element_index]);
        // printf("\n");
        memcpy(&temp, &a_non_zeros[element_index], sizeof(uint64_t));
        temp ^= 0x1ULL << bit_index;
        memcpy(&a_non_zeros[element_index], &temp, sizeof(uint64_t));
        // printBits(sizeof(double), &a_non_zeros[element_index]);
        // printf("\n");
      }
      else
      {
        // printBits(sizeof(uint32_t), &a_cols[element_index]);
        // printf("\n");
        uint32_t temp;
        memcpy(&temp, &a_cols[element_index], sizeof(uint32_t));
        temp ^= 0x1U << bit_index;
        memcpy(&a_cols[element_index], &temp, sizeof(uint32_t));
        // printBits(sizeof(uint32_t), &a_cols[element_index]);
        // printf("\n");
      }

      printf("[CRC32C] Bitflip occured at element index %u, bit index %u\n", element_index, bit_index);
      correct_crc = 1;
    }
    free(test_data);
    // for(uint32_t i = 0; i < num_elements; i++)
    // {
    //   printf("%u ", a_cols[idx+i]);
    //   printBits(sizeof(uint32_t), &a_cols[idx+i]);
    //   printf("\n");
    // }
  }

  return correct_crc;
}

void assign_crc32c_bits(uint32_t * a_cols, double * a_non_zeros, uint32_t idx, uint32_t num_elements)
{
  if(num_elements < 4)
  {
    printf("Row is too small! Has %u elements, should have at least 4.\n", num_elements);
    return;
  }
  //generate the CRC32C bits and put them in the right places
  if(   a_cols[idx    ] & 0xFF000000
     || a_cols[idx + 1] & 0xFF000000
     || a_cols[idx + 2] & 0xFF000000
     || a_cols[idx + 3] & 0xFF000000
     || a_cols[idx + 4] & 0xFF000000)
  {
    printf("Index too big to be stored correctly with CRC!\n");
    exit(1);
  }
  uint32_t crc = generate_crc32c_bits(&a_cols[idx], &a_non_zeros[idx], num_elements);
  a_cols[idx    ] += crc & 0xFF000000;
  a_cols[idx + 1] += (crc & 0x00FF0000) << 8;
  a_cols[idx + 2] += (crc & 0x0000FF00) << 16;
  a_cols[idx + 3] += (crc & 0x000000FF) << 24;
}
