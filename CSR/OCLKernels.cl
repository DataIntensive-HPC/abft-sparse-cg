#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || \
    defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || \
    defined(__Capeverde__) || defined(__Cayman__) || defined(__Barts__) || \
    defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || \
    defined(__Cedar__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || \
    defined(__ATI_RV710__) || defined(__Loveland__) || defined(__GPU__) || \
    defined(__Hawaii__) || defined(__Fiji__)
#define AMD
#endif

#ifndef AMD
#define PRINT
#endif

typedef struct
{
  double value;
  uint column;
} __attribute__((packed)) csr_element;

inline uint is_power_of_2(uint x)
{
  return ((x != 0) && !(x & (x - 1)));
}

#define PARITY_METHOD_0 0 //slower than __builtin_parity
#define PARITY_METHOD_1 1 //slightly than __builtin_parity
#define PARITY_METHOD_2 2 //around the same as __builtin_parity, maybe sligtly faster
#define PARITY_METHOD_3 3
#define __PARITY_METHOD PARITY_METHOD_2

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
#endif
}

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
  result |= calc_parity(p) << 31;

  p = (data[0] & ECC7_P2_0) ^ (data[1] & ECC7_P2_1) ^ (data[2] & ECC7_P2_2);
  result |= calc_parity(p) << 30;

  p = (data[0] & ECC7_P3_0) ^ (data[1] & ECC7_P3_1) ^ (data[2] & ECC7_P3_2);
  result |= calc_parity(p) << 29;

  p = (data[0] & ECC7_P4_0) ^ (data[1] & ECC7_P4_1) ^ (data[2] & ECC7_P4_2);
  result |= calc_parity(p) << 28;

  p = (data[0] & ECC7_P5_0) ^ (data[1] & ECC7_P5_1) ^ (data[2] & ECC7_P5_2);
  result |= calc_parity(p) << 27;

  p = (data[0] & ECC7_P6_0) ^ (data[1] & ECC7_P6_1) ^ (data[2] & ECC7_P6_2);
  result |= calc_parity(p) << 26;

  p = (data[0] & ECC7_P7_0) ^ (data[1] & ECC7_P7_1) ^ (data[2] & ECC7_P7_2);
  result |= calc_parity(p) << 25;

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
    if((syndrome >> (32-p)) & 0x1)
      hamm_bit += 0x1<<(p-1);
  }

  // Map to actual data bit position
  uint data_bit = hamm_bit - (32-clz(hamm_bit)) - 1;
  if(is_power_of_2(hamm_bit))
    data_bit = clz(hamm_bit) + 64;

  return data_bit;
}

__kernel void dot_product(
  const uint N, //vector size
  const uint items_per_work_item,
  const uint items_per_work_group,
  __local volatile double * partial_dot_product,
  __global const double * restrict a,
  __global const double * restrict b,
  __global double * restrict result)
{
  const uint local_id = get_local_id(0);
  const uint group_size = get_local_size(0);
  const uint group_id = get_group_id(0);

  double ret = 0.0;
  uint offset = group_id * items_per_work_group + local_id;
  for (uint i = 0; i < items_per_work_item; i++, offset += items_per_work_item)
  {
    ret += offset < N ? a[offset] * b[offset] : 0.0;
  }
  partial_dot_product[local_id] = ret;

  //do a reduction
  for(uint step = group_size >> 1; step > 1; step>>=1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(local_id < step)
    {
      partial_dot_product[local_id] += partial_dot_product[local_id + step];
    }
  }
  //store result in a global array
  barrier(CLK_LOCAL_MEM_FENCE);
  if(local_id == 0)
  {
    result[group_id] = partial_dot_product[0] + partial_dot_product[1];
  }
}

__kernel void calc_p(
  const uint N, //vector size
  const double beta,
  const uint items_per_work_item,
  const uint items_per_work_group,
  __global const double * restrict r,
  __global double * restrict p)
{
  const uint local_id = get_local_id(0);
  const uint group_id = get_group_id(0);

  uint offset = group_id * items_per_work_group + local_id;
  for (uint i = 0; i < items_per_work_item && offset < N; i++, offset += items_per_work_item)
  {
    p[offset] = r[offset] + beta * p[offset];
  }
}

__kernel void calc_xr(
  const uint N, //vector size
  const double alpha,
  const uint items_per_work_item,
  const uint items_per_work_group,
  __local volatile double * partial_result,
  __global const double * restrict p,
  __global const double * restrict w,
  __global double * restrict x,
  __global double * restrict r,
  __global double * restrict result)
{
  const uint local_id = get_local_id(0);
  const uint group_size = get_local_size(0);
  const uint group_id = get_group_id(0);

  double ret = 0.0;

  uint offset = group_id * items_per_work_group + local_id;
  for (uint i = 0; i < items_per_work_item && offset < N; i++, offset += items_per_work_item)
  {
    x[offset] += alpha * p[offset];
    r[offset] -= alpha * w[offset];

    ret += r[offset] * r[offset];
  }
  partial_result[local_id] = ret;
  //do a reduction
  for(uint step = group_size >> 1; step > 1; step>>=1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(local_id < step)
    {
      partial_result[local_id] += partial_result[local_id + step];
    }
  }
  //store result in a global array
  barrier(CLK_LOCAL_MEM_FENCE);
  if(local_id == 0)
  {
    result[group_id] = partial_result[0] + partial_result[1];
  }
}

__kernel void sum_vector(
  const uint N, //vector size
  const uint items_per_work_item,
  __local volatile double  * partial_result,
  __global const double * restrict buffer,
  __global double * restrict result)
{
  const uint local_id = get_local_id(0);
  const uint group_size = get_local_size(0);

  double ret = 0.0;

  uint offset = local_id;
  for (uint i = 0; i < items_per_work_item && offset < N; i++, offset += group_size)
  {
    ret += buffer[offset];
  }
  partial_result[local_id] = ret;
  //do a reduction
  for(uint step = group_size >> 1; step > 1; step>>=1)
  {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(local_id < step)
    {
      partial_result[local_id] += partial_result[local_id + step];
    }
  }
  //store result in a global array
  barrier(CLK_LOCAL_MEM_FENCE);
  if(local_id == 0)
  {
    result[0] = partial_result[0] + partial_result[1];
  }
}

__kernel void inject_bitflip_val(
  const uint bit, //vector size
  const uint index,
  __global double * values)
{
#ifdef PRINT
  printf("*** flipping bit %u of value at index %u ***\n", bit, index);
#endif
  values[index] = as_double(as_ulong(values[index]) ^ 0x1 << (bit % 32));
}

__kernel void inject_bitflip_col(
  const uint bit, //vector size
  const uint index,
  __global uint * values)
{
#ifdef PRINT
  printf("*** flipping bit %u of column at index %u ***\n", bit, index);
#endif
  values[index] ^= 0x1 << (bit % 32);
}

//CSR_SCALAR TECHNIQUE
__kernel void spmv_scalar(
  const uint N, //vector size
  __global const uint * restrict mat_rows,
  __global uint * restrict mat_cols,
  __global double * restrict mat_values,
  __global const double * restrict vec,
  __global double * restrict result,
  __global volatile uint * error_flag
#if defined(FT_CONSTRAINTS)
  ,const uint nnz
#endif
  )
{
  const uint global_id = get_global_id(0);
  if(global_id < N)
  {
    uint start = mat_rows[global_id];
    uint end   = mat_rows[global_id+1];

#if defined(FT_CONSTRAINTS)
    if(end > nnz)
    {
      if(atomic_add(error_flag, 0) == NO_ERROR)
      {
        atomic_xchg(error_flag, ERROR_CONSTRAINT_ROW_SIZE);
#ifdef PRINT
        printf("row size constraint violated for row %u\n", global_id);
#endif
        return;
      }
    }
    else if(end < start)
    {
      if(atomic_add(error_flag, 0) == NO_ERROR)
      {
        atomic_xchg(error_flag, ERROR_CONSTRAINT_ROW_ORDER);
#ifdef PRINT
        printf("row order constraint violated for row %u\n", global_id);
#endif
        return;
      }
    }
#endif

    // initialize local sum
    double tmp = 0;
    // accumulate local sums
    for(uint i = start; i < end; i++)
    {
      uint col = mat_cols[i];
#if defined(FT_CONSTRAINTS)
      if(col >= N)
      {
        if(atomic_add(error_flag, 0) == NO_ERROR)
        {
          atomic_xchg(error_flag, ERROR_CONSTRAINT_COL_SIZE);
#ifdef PRINT
          printf("column size constraint violated at index %u\n", i);
#endif
          break;
        }
      }
      else if(i < end-1 && mat_cols[i+1] <= col)
      {
        if(atomic_add(error_flag, 0) == NO_ERROR)
        {
          atomic_xchg(error_flag, ERROR_CONSTRAINT_COL_ORDER);
#ifdef PRINT
          printf("column order constraint violated at index %u\n", i);
#endif
          break;
        }
      }

#elif defined(FT_SED)
      csr_element element;
      element.value  = mat_values[i];
      element.column = col;
      // Check overall parity bit
      if(ecc_compute_overall_parity(element))
      {
        if(atomic_add(error_flag, 0) == NO_ERROR)
        {
          atomic_xchg(error_flag, ERROR_SED);
#ifdef PRINT
          printf("[ECC] error detected at index %u\n", i);
#endif
          break;
        }
      }
      // Mask out ECC from high order column bits
      element.column &= 0x00FFFFFF;
      col = element.column;

#elif defined(FT_SEC7)
      csr_element element;
      element.value  = mat_values[i];
      element.column = col;
      // Check ECC
      uint syndrome = ecc_compute_col8(element);
      if(syndrome)
      {
        // Unflip bit
        uint bit = ecc_get_flipped_bit_col8(syndrome);
        ((uint*)(&element))[bit/32] ^= 0x1 << (bit % 32);
        mat_cols[i] = element.column;
        mat_values[i] = element.value;
#ifdef PRINT
        printf("[ECC] corrected bit %u at index %u\n", bit, i);
#endif
      }

      // Mask out ECC from high order column bits
      element.column &= 0x00FFFFFF;
      col = element.column;

#elif defined(FT_SEC8)
      csr_element element;
      element.value  = mat_values[i];
      element.column = col;
      // Check overall parity bit
      if(ecc_compute_overall_parity(element))
      {
        // Compute error syndrome from hamming bits
        uint syndrome = ecc_compute_col8(element);
        if(syndrome)
        {
          // Unflip bit
          uint bit = ecc_get_flipped_bit_col8(syndrome);
          ((uint*)(&element))[bit/32] ^= 0x1 << (bit % 32);
#ifdef PRINT
          printf("[ECC] corrected bit %u at index %u\n", bit, i);
#endif
        }
        else
        {
          // Correct overall parity bit
          element.column ^= 0x1 << 24;
#ifdef PRINT
          printf("[ECC] corrected overall parity bit at index %u\n", i);
#endif
        }

        mat_cols[i] = element.column;
        mat_values[i] = element.value;
      }
      // Mask out ECC from high order column bits
      element.column &= 0x00FFFFFF;
      col = element.column;

#elif defined(FT_SECDED)
      csr_element element;
      element.value  = mat_values[i];
      element.column = col;
      // Check parity bits
      uint overall_parity = ecc_compute_overall_parity(element);
      uint syndrome = ecc_compute_col8(element);
      if(overall_parity)
      {
        if(syndrome)
        {
          // Unflip bit
          uint bit = ecc_get_flipped_bit_col8(syndrome);
          ((uint*)(&element))[bit/32] ^= 0x1 << (bit % 32);
#ifdef PRINT
          printf("[ECC] corrected bit %u at index %d\n", bit, i);
#endif
        }
        else
        {
          // Correct overall parity bit
          element.column ^= 0x1 << 24;
#ifdef PRINT
          printf("[ECC] corrected overall parity bit at index %d\n", i);
#endif
        }

        mat_cols[i] = element.column;
        mat_values[i] = element.value;
      }
      else
      {
        if(syndrome)
        {
          // Overall parity fine but error in syndrom
          // Must be double-bit error - cannot correct this
          if(atomic_add(error_flag, 0) == NO_ERROR)
          {
            atomic_xchg(error_flag, ERROR_SECDED);
#ifdef PRINT
            printf("[ECC] double-bit error detected\n");
#endif
            break;

          }
        }
      }
      // Mask out ECC from high order column bits
      element.column &= 0x00FFFFFF;
      col = element.column;
#endif
      tmp += mat_values[i] * vec[col];
    }
    result[global_id] = tmp;
  }
}

//CSR_VECTOR TECHNIQUE
//csr_spmv kernel extracted from bhSPARSE: https://github.com/bhSPARSE/bhSPARSE
//when an error in this kernel occurs we can't just return as there is a reduction at the end
__kernel void spmv_vector(
  const uint N,
  __global const uint * restrict mat_rows,
  __global uint * restrict mat_cols,
  __global double * restrict mat_values,
  __global const double * restrict vec,
  __global double * restrict result,
  __global volatile uint * error_flag,
  __local volatile double * restrict partial_result, //[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2]
  __local volatile uint * restrict ptrs, //[VECTORS_PER_BLOCK][2]
  const uint VECTORS_PER_BLOCK,
  const uint THREADS_PER_VECTOR
#if defined(FT_CONSTRAINTS)
  ,const uint nnz
#endif
  )
{
  const uint local_id   = get_local_id(0);

  const uint thread_lane = local_id & (THREADS_PER_VECTOR - 1); // thread index within the vector
  const uint vector_id   = get_global_id(0)   /  THREADS_PER_VECTOR; // global vector index
  const uint vector_lane = local_id /  THREADS_PER_VECTOR; // vector index within the block
  const uint num_vectors = VECTORS_PER_BLOCK * get_num_groups(0); // total number of active vectors
  uchar error_occured = NO_ERROR;
  for(uint row = vector_id; row < N && error_occured == NO_ERROR; row += num_vectors)
  {
    // use two threads to fetch mat_rows[row] and mat_rows[row+1]
    // this is considerably faster than the straightforward version
    if(thread_lane < 2)
    {
      ptrs[vector_lane * 2 + thread_lane] = mat_rows[row + thread_lane];
    }

    const uint start = ptrs[vector_lane * 2 + 0]; //same as: start = mat_rows[row];
    const uint end   = ptrs[vector_lane * 2 + 1]; //same as: end   = mat_rows[row+1];

#if defined(FT_CONSTRAINTS)
    if(end > nnz)
    {
      if(atomic_add(error_flag, 0) == NO_ERROR)
      {
        error_occured = ERROR_CONSTRAINT_ROW_SIZE;
        atomic_xchg(error_flag, error_occured);
#ifdef PRINT
        printf("row size constraint violated for row %u\n", row);
#endif
      }
    }
    else if(end < start)
    {
      if(atomic_add(error_flag, 0) == NO_ERROR)
      {
        error_occured = ERROR_CONSTRAINT_ROW_ORDER;
        atomic_xchg(error_flag, error_occured);
#ifdef PRINT
        printf("row order constraint violated for row %u\n", row);
#endif
      }
    }
#endif

    // initialize local sum
    double tmp = 0;
    // accumulate local sums
    for(uint i = start + thread_lane; i < end && error_occured == NO_ERROR; i += THREADS_PER_VECTOR)
    {
      uint col = mat_cols[i];
#if defined(FT_CONSTRAINTS)
      if(col >= N)
      {
        if(atomic_add(error_flag, 0) == NO_ERROR)
        {
          error_occured = ERROR_CONSTRAINT_COL_SIZE;
          atomic_xchg(error_flag, error_occured);
#ifdef PRINT
          printf("column size constraint violated at index %u\n", i);
#endif
          break;
        }
      }
      else if(i < end-1 && mat_cols[i+1] <= col)
      {
        if(atomic_add(error_flag, 0) == NO_ERROR)
        {
          error_occured = ERROR_CONSTRAINT_COL_ORDER;
          atomic_xchg(error_flag, error_occured);
#ifdef PRINT
          printf("column order constraint violated at index %u\n", i);
#endif
          break;
        }
      }
#elif defined(FT_SED)
      csr_element element;
      element.value  = mat_values[i];
      element.column = col;
      // Check overall parity bit
      if(ecc_compute_overall_parity(element))
      {
        if(atomic_add(error_flag, 0) == NO_ERROR)
        {
#ifdef PRINT
          printf("[ECC] error detected at index %u\n", i);
#endif
          error_occured = ERROR_SED;
          atomic_xchg(error_flag, error_occured);
          break;
        }
      }
      // Mask out ECC from high order column bits
      element.column &= 0x00FFFFFF;
      col = element.column;

#elif defined(FT_SEC7)
      csr_element element;
      element.value  = mat_values[i];
      element.column = col;
      // Check ECC
      uint syndrome = ecc_compute_col8(element);
      if(syndrome)
      {
        // Unflip bit
        uint bit = ecc_get_flipped_bit_col8(syndrome);
        ((uint*)(&element))[bit/32] ^= 0x1 << (bit % 32);
        mat_cols[i] = element.column;
        mat_values[i] = element.value;
#ifdef PRINT
        printf("[ECC] corrected bit %u at index %u\n", bit, i);
#endif
      }

      // Mask out ECC from high order column bits
      element.column &= 0x00FFFFFF;
      col = element.column;

#elif defined(FT_SEC8)
      csr_element element;
      element.value  = mat_values[i];
      element.column = col;
      // Check overall parity bit
      if(ecc_compute_overall_parity(element))
      {
        // Compute error syndrome from hamming bits
        uint syndrome = ecc_compute_col8(element);
        if(syndrome)
        {
          // Unflip bit
          uint bit = ecc_get_flipped_bit_col8(syndrome);
          ((uint*)(&element))[bit/32] ^= 0x1 << (bit % 32);
#ifdef PRINT
          printf("[ECC] corrected bit %u at index %u\n", bit, i);
#endif
        }
        else
        {
          // Correct overall parity bit
          element.column ^= 0x1 << 24;
#ifdef PRINT
          printf("[ECC] corrected overall parity bit at index %u\n", i);
#endif
        }

        mat_cols[i] = element.column;
        mat_values[i] = element.value;
      }
      // Mask out ECC from high order column bits
      element.column &= 0x00FFFFFF;
      col = element.column;

#elif defined(FT_SECDED)
      csr_element element;
      element.value  = mat_values[i];
      element.column = col;
      // Check parity bits
      uint overall_parity = ecc_compute_overall_parity(element);
      uint syndrome = ecc_compute_col8(element);
      if(overall_parity)
      {
        if(syndrome)
        {
          // Unflip bit
          uint bit = ecc_get_flipped_bit_col8(syndrome);
          ((uint*)(&element))[bit/32] ^= 0x1 << (bit % 32);
#ifdef PRINT
          printf("[ECC] corrected bit %u at index %d\n", bit, i);
#endif
        }
        else
        {
          // Correct overall parity bit
          element.column ^= 0x1 << 24;
#ifdef PRINT
          printf("[ECC] corrected overall parity bit at index %d\n", i);
#endif
        }

        mat_cols[i] = element.column;
        mat_values[i] = element.value;
      }
      else
      {
        if(syndrome)
        {
          // Overall parity fine but error in syndrom
          // Must be double-bit error - cannot correct this
          if(atomic_add(error_flag, 0) == NO_ERROR)
          {
            error_occured = ERROR_SECDED;
            atomic_xchg(error_flag, error_occured);
#ifdef PRINT
            printf("[ECC] double-bit error detected\n");
#endif
            break;

          }
        }
      }
      // Mask out ECC from high order column bits
      element.column &= 0x00FFFFFF;
      col = element.column;
#endif
      tmp += mat_values[i] * vec[col];
    }
    // store local sum in shared memory
    partial_result[local_id] = tmp;

    // reduce local sums to row tmp
    for(uint step = THREADS_PER_VECTOR >> 1; step > 0; step>>=1)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      partial_result[local_id] = tmp = tmp + partial_result[local_id + step];
    }

    // first thread writes the result
    barrier(CLK_LOCAL_MEM_FENCE);
    if(thread_lane == 0)
    {
      result[row] = partial_result[local_id];
    }
  }
}

