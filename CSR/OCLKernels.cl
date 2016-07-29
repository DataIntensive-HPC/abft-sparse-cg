#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#ifdef PRINTF_ARM_KERNEL
#pragma OPENCL EXTENSION cl_arm_printf : enable
#endif

typedef struct
{
  double value;
  uint column;
} __attribute__((packed)) csr_element;

#if defined(__WinterPark__) || defined(__BeaverCreek__) || defined(__Turks__) || \
    defined(__Caicos__) || defined(__Tahiti__) || defined(__Pitcairn__) || \
    defined(__Capeverde__) || defined(__Cayman__) || defined(__Barts__) || \
    defined(__Cypress__) || defined(__Juniper__) || defined(__Redwood__) || \
    defined(__Cedar__) || defined(__ATI_RV770__) || defined(__ATI_RV730__) || \
    defined(__ATI_RV710__) || defined(__Loveland__) || defined(__GPU__) || \
    defined(__Hawaii__) || defined(__Fiji__)
#define AMD
#endif

//macro for printf - don't do printf on platforms that struggle with them
#if (defined(__NV_CL_C_VERSION) || defined(PRINTF_ARM_KERNEL)) && !defined(AMD)
#define PRINTF_CL(MESSAGE, args...) { \
  printf(MESSAGE,##args); \
}
#else
#define PRINTF_CL(MESSAGE,args...) { \
}
#endif

//On the arm platform the atomic operations add a lot of overhead (around 150%)
//atomics are used to signal that an error has occured and to exit the kernel safely
//On arm we don't do this
//#define USE_ATOMICS_FOR_ERROR_FLAG

inline uchar update_error(__global uint * restrict error_flag, uint error)
{
#ifdef USE_ATOMICS_FOR_ERROR_FLAG
  atomic_xchg(error_flag, error);
#else
  error_flag[0] = error;
#endif
  return error;
}
inline uint is_power_of_2(uint x)
{
  return ((x != 0) && !(x & (x - 1)));
}

#define PARITY_METHOD_0 0 //slower than __builtin_parity
#define PARITY_METHOD_1 1 //slightly than __builtin_parity
#define PARITY_METHOD_2 2 //around the same as __builtin_parity, maybe sligtly faster
#define PARITY_METHOD_3 3
#define PARITY_METHOD_4 4
#define PARITY_METHOD_5 5
#define __PARITY_METHOD PARITY_METHOD_5

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
  const uchar ParityTable256[256] =
  {
#   define P2(n) n, n^1, n^1, n
#   define P4(n) P2(n), P2(n^1), P2(n^1), P2(n)
#   define P6(n) P4(n), P4(n^1), P4(n^1), P4(n)
    P6(0), P6(1), P6(1), P6(0)
  };
  uchar * p = (uchar *) &x;
  return ParityTable256[p[0] ^ p[1] ^ p[2] ^ p[3]];
#elif __PARITY_METHOD == PARITY_METHOD_5
  x ^= x >> 16;
  x ^= x >> 8;
  x ^= x >> 4;
  x &= 0xf;
  return (0x6996 >> x) & 1;
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

__kernel void dot_product(
  const uint N, //vector size
  __local double * restrict partial_dot_product,
  __global const double * restrict a,
  __global const double * restrict b,
  __global double * restrict result)
{
  const uint local_id = get_local_id(0);
  const uint group_size = get_local_size(0);
  const uint group_id = get_group_id(0);

  double ret = 0.0;
  double tmp;
  uint offset = group_id * DOT_PRODUCT_KERNEL_ITEMS_PER_WORK_GROUP + local_id;
  for (uint i = 0; i < DOT_PRODUCT_KERNEL_ITEMS_PER_WORK_ITEM; i++, offset += group_size)
  {
    uchar in_range = offset < N;
    uint local_offset = in_range ? offset : 0;
    tmp = a[local_offset] * b[local_offset];
    ret += in_range ? tmp : 0.0;
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
  __global const double * restrict r,
  __global double * restrict p)
{
  const uint local_id = get_local_id(0);
  const uint group_id = get_group_id(0);
  const uint group_size = get_local_size(0);

  uint offset = group_id * CALC_P_KERNEL_ITEMS_PER_WORK_GROUP + local_id;
  for (uint i = 0; i < CALC_P_KERNEL_ITEMS_PER_WORK_ITEM && offset < N; i++, offset += group_size)
  {
    p[offset] = fma(beta, p[offset], r[offset]);
  }
}

__kernel void calc_xr(
  const uint N, //vector size
  const double alpha,
  __local double * restrict partial_result,
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

  const uint offset = group_id * CALC_XR_KERNEL_ITEMS_PER_WORK_GROUP + local_id;

  uint j = offset;
  for (uint i = 0; i < CALC_XR_KERNEL_ITEMS_PER_WORK_ITEM && j < N; i++, j += group_size)
  {
    x[j] = fma(alpha, p[j], x[j]);
  }

  j = offset;
  for (uint i = 0; i < CALC_XR_KERNEL_ITEMS_PER_WORK_ITEM && j < N; i++, j += group_size)
  {
    r[j] = fma(-alpha, w[j], r[j]);
    ret = fma(r[j], r[j], ret);
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
  __local double  * restrict partial_result,
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
  __global double * restrict values)
{
  PRINTF_CL("*** flipping bit %u of value at index %u ***\n", bit, index);
  values[index] = as_double(as_ulong(values[index]) ^ 0x1U << (bit));
}

__kernel void inject_bitflip_col(
  const uint bit, //vector size
  const uint index,
  __global uint * restrict values)
{
  PRINTF_CL("*** flipping bit %u of column at index %u ***\n", bit, index);
  values[index] ^= 0x1U << (bit % 32);
}

//CSR_SCALAR TECHNIQUE
__kernel void spmv_scalar(
  const uint N, //vector size
  __global const uint * restrict mat_rows,
  __global uint * restrict mat_cols,
  __global double * restrict mat_values,
  __global const double * restrict vec,
  __global double * restrict result,
  __global uint * restrict error_flag
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
      PRINTF_CL("row size constraint violated for row %u\n", global_id);
      update_error(error_flag, ERROR_CONSTRAINT_ROW_SIZE);
      return;
    }
    else if(end < start)
    {
      PRINTF_CL("row order constraint violated for row %u\n", global_id);
      update_error(error_flag, ERROR_CONSTRAINT_ROW_ORDER);
      return;
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
        PRINTF_CL("column size constraint violated at index %u\n", i);
        update_error(error_flag, ERROR_CONSTRAINT_COL_SIZE);
        return;
      }
      else if(i < end-1 && mat_cols[i+1] <= col)
      {
        PRINTF_CL("column order constraint violated at index %u\n", i);
        update_error(error_flag, ERROR_CONSTRAINT_COL_ORDER);
        return;
      }

#elif defined(FT_SED)
      csr_element element;
      element.value  = mat_values[i];
      element.column = col;
      // Check overall parity bit
      if(ecc_compute_overall_parity(element))
      {
        PRINTF_CL("[ECC] error detected at index %u\n", i);
        update_error(error_flag, ERROR_SED);
        return;
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
        ((uint*)(&element))[bit/32] ^= 0x1U << (bit % 32);
        mat_cols[i] = element.column;
        mat_values[i] = element.value;
        PRINTF_CL("[ECC] corrected bit %u at index %u\n", bit, i);
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
          ((uint*)(&element))[bit/32] ^= 0x1U << (bit % 32);
          PRINTF_CL("[ECC] corrected bit %u at index %u\n", bit, i);
        }
        else
        {
          // Correct overall parity bit
          element.column ^= 0x1U << 24;
          PRINTF_CL("[ECC] corrected overall parity bit at index %u\n", i);
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
          ((uint*)(&element))[bit/32] ^= 0x1U << (bit % 32);
          PRINTF_CL("[ECC] corrected bit %u at index %d\n", bit, i);
        }
        else
        {
          // Correct overall parity bit
          element.column ^= 0x1U << 24;
          PRINTF_CL("[ECC] corrected overall parity bit at index %d\n", i);
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
          PRINTF_CL("[ECC] double-bit error detected\n");
          update_error(error_flag, ERROR_SECDED);
          return;
        }
      }
      // Mask out ECC from high order column bits
      element.column &= 0x00FFFFFF;
      col = element.column;
#endif
      tmp = fma(mat_values[i], vec[col], tmp);
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
  __global uint * restrict error_flag,
  __local double * restrict partial_result, //[VECTORS_PER_BLOCK * THREADS_PER_VECTOR + THREADS_PER_VECTOR / 2]
  const uint VECTORS_PER_BLOCK,
  const uint THREADS_PER_VECTOR
#if defined(FT_CONSTRAINTS)
  ,const uint nnz
#endif
  )
{
  const uint local_id   = get_local_id(0);

  const uint thread_lane = local_id % THREADS_PER_VECTOR; // thread index within the vector
  const uint vector_id   = get_global_id(0)   /  THREADS_PER_VECTOR; // global vector index
  const uint vector_lane = local_id /  THREADS_PER_VECTOR; // vector index within the block
  const uint num_vectors = VECTORS_PER_BLOCK * get_num_groups(0); // total number of active vectors
  uchar error_occured = NO_ERROR;
  for(uint row = vector_id; row < N && error_occured == NO_ERROR; row += num_vectors)
  {
    const uint start = mat_rows[row];
    const uint end   = mat_rows[row+1];

#if defined(FT_CONSTRAINTS)
    if(end > nnz)
    {
      PRINTF_CL("row size constraint violated for row %u\n", row);
      error_occured = update_error(error_flag, ERROR_CONSTRAINT_ROW_SIZE);
    }
    else if(end < start)
    {
      PRINTF_CL("row order constraint violated for row %u\n", row);
      error_occured = update_error(error_flag, ERROR_CONSTRAINT_ROW_ORDER);
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
        PRINTF_CL("column size constraint violated at index %u\n", i);
        error_occured = update_error(error_flag, ERROR_CONSTRAINT_COL_SIZE);
        break;
      }
      else if(i < end-1 && mat_cols[i+1] <= col)
      {
        PRINTF_CL("column order constraint violated at index %u\n", i);
        error_occured = update_error(error_flag, ERROR_CONSTRAINT_COL_ORDER);
        break;
      }
#elif defined(FT_SED)
      csr_element element;
      element.value  = mat_values[i];
      element.column = col;
      // Check overall parity bit
      if(ecc_compute_overall_parity(element))
      {
        PRINTF_CL("[ECC] error detected at index %u\n", i);
        error_occured = update_error(error_flag, ERROR_SED);
        break;
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
        ((uint*)(&element))[bit/32] ^= 0x1U << (bit % 32);
        mat_cols[i] = element.column;
        mat_values[i] = element.value;
        PRINTF_CL("[ECC] corrected bit %u at index %u\n", bit, i);
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
          ((uint*)(&element))[bit/32] ^= 0x1U << (bit % 32);
          PRINTF_CL("[ECC] corrected bit %u at index %u\n", bit, i);
        }
        else
        {
          // Correct overall parity bit
          element.column ^= 0x1U << 24;
          PRINTF_CL("[ECC] corrected overall parity bit at index %u\n", i);
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
          ((uint*)(&element))[bit/32] ^= 0x1U << (bit % 32);
          PRINTF_CL("[ECC] corrected bit %u at index %d\n", bit, i);
        }
        else
        {
          // Correct overall parity bit
          element.column ^= 0x1U << 24;
          PRINTF_CL("[ECC] corrected overall parity bit at index %d\n", i);
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
          PRINTF_CL("[ECC] double-bit error detected\n");
          error_occured = update_error(error_flag, ERROR_SECDED);
          break;
        }
      }
      // Mask out ECC from high order column bits
      element.column &= 0x00FFFFFF;
      col = element.column;
#endif
      tmp = fma(mat_values[i], vec[col], tmp);
    }
    // store local sum in shared memory
    partial_result[local_id] = tmp;

    // reduce local sums to row tmp
    for(uint step = THREADS_PER_VECTOR >> 1; step > 0; step>>=1)
    {
      barrier(CLK_LOCAL_MEM_FENCE);
      if(thread_lane < step)
      {
        partial_result[local_id] += partial_result[local_id + step];
      }
    }

    // first thread writes the result
    barrier(CLK_LOCAL_MEM_FENCE);
    if(thread_lane == 0)
    {
      result[row] = partial_result[local_id];
    }
  }
}

