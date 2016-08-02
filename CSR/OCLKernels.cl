#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#ifdef PRINTF_ARM_KERNEL
#pragma OPENCL EXTENSION cl_arm_printf : enable
#endif

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
#define PRINTF_CL(MESSAGE, args...) printf(MESSAGE,##args)
#else
#define PRINTF_CL(MESSAGE,args...)
#endif

//On the arm platform the atomic operations add a lot of overhead (around 150%)
//atomics are used to signal that an error has occured and to exit the kernel safely
//On arm we don't do this
//#define USE_ATOMICS_FOR_ERROR_FLAG

inline uchar update_error(__global uint * restrict error_flag_buffer, uint error)
{
#ifdef USE_ATOMICS_FOR_ERROR_FLAG
  atomic_xchg(error_flag_buffer, error);
#else
  error_flag_buffer[0] = error;
#endif
  return error;
}


//macro for checking correct row constraints
#define CONSTRAINTS_CHECK_ROW(row, start, end, nnz, error_flag, error_flag_buffer, exit_op)\
if(1){\
  if(end > nnz)\
  {\
    PRINTF_CL("row size constraint violated for row %u\n", row);\
    error_flag = update_error(error_flag_buffer, ERROR_CONSTRAINT_ROW_SIZE);\
    exit_op;\
  }\
  else if(end < start)\
  {\
    PRINTF_CL("row order constraint violated for row %u\n", row);\
    error_flag = update_error(error_flag_buffer, ERROR_CONSTRAINT_ROW_ORDER);\
    exit_op;\
  }\
} else

//macro for checking correct column constraints
#define CONSTRAINTS_CHECK_COL(i, col, next_col, N, nnz, error_flag, error_flag_buffer, exit_op)\
if(1){\
  if(col >= N)\
  {\
    PRINTF_CL("column size constraint violated at index %u\n", i);\
    error_flag = update_error(error_flag_buffer, ERROR_CONSTRAINT_COL_SIZE);\
    exit_op;\
  }\
  else if(i < end-1 && next_col <= col)\
  {\
    PRINTF_CL("column order constraint violated at index %u\n", i);\
    error_flag = update_error(error_flag_buffer, ERROR_CONSTRAINT_COL_ORDER);\
    exit_op;\
  }\
} else

//macro for SED
#define SED(col, val, i, error_flag, error_flag_buffer, exit_op)\
if(1){\
  csr_element element;\
  element.value  = val;\
  element.column = col;\
  /* Check overall parity bit*/\
  if(ecc_compute_overall_parity(element))\
  {\
    PRINTF_CL("[ECC] error detected at index %u\n", i);\
    error_flag = update_error(error_flag_buffer, ERROR_SED);\
    exit_op;\
  }\
  /* Mask out ECC from high order column bits */\
  element.column &= 0x00FFFFFF;\
  col = element.column;\
} else

//macro for SEC7
#define SEC7(col, mat_values, mat_cols, i)\
if(1){\
  csr_element element;\
  element.value  = mat_values[i];\
  element.column = col;\
  /* Check ECC */\
  uint syndrome = ecc_compute_col8(element);\
  if(syndrome)\
  {\
    /* Unflip bit */\
    uint bit = ecc_get_flipped_bit_col8(syndrome);\
    ((uint*)(&element))[bit/32] ^= 0x1U << (bit % 32);\
    mat_cols[i] = element.column;\
    mat_values[i] = element.value;\
    PRINTF_CL("[ECC] corrected bit %u at index %u\n", bit, i);\
  }\
  /* Mask out ECC from high order column bits */\
  element.column &= 0x00FFFFFF;\
  col = element.column;\
} else

//macro for SEC8
#define SEC8(col, mat_values, mat_cols, i)\
if(1){\
  csr_element element;\
  element.value  = mat_values[i];\
  element.column = col;\
  /* Check overall parity bit */\
  if(ecc_compute_overall_parity(element))\
  {\
    /* Compute error syndrome from hamming bits */\
    uint syndrome = ecc_compute_col8(element);\
    if(syndrome)\
    {\
      /* Unflip bit */\
      uint bit = ecc_get_flipped_bit_col8(syndrome);\
      ((uint*)(&element))[bit/32] ^= 0x1U << (bit % 32);\
      PRINTF_CL("[ECC] corrected bit %u at index %u\n", bit, i);\
    }\
    else\
    {\
      /* Correct overall parity bit */\
      element.column ^= 0x1U << 24;\
      PRINTF_CL("[ECC] corrected overall parity bit at index %u\n", i);\
    }\
    mat_cols[i] = element.column;\
    mat_values[i] = element.value;\
  }\
  /* Mask out ECC from high order column bits */\
  element.column &= 0x00FFFFFF;\
  col = element.column;\
} else

//macro for SECDED
#define SECDED(col, mat_values, mat_cols, i, error_flag, error_flag_buffer, exit_op)\
if(1){\
  csr_element element;\
  element.value  = mat_values[i];\
  element.column = col;\
  /* Check parity bits */\
  uint overall_parity = ecc_compute_overall_parity(element);\
  uint syndrome = ecc_compute_col8(element);\
  if(overall_parity)\
  {\
    if(syndrome)\
    {\
      /* Unflip bit */\
      uint bit = ecc_get_flipped_bit_col8(syndrome);\
      ((uint*)(&element))[bit/32] ^= 0x1U << (bit % 32);\
      PRINTF_CL("[ECC] corrected bit %u at index %d\n", bit, i);\
    }\
    else\
    {\
      /* Correct overall parity bit */\
      element.column ^= 0x1U << 24;\
      PRINTF_CL("[ECC] corrected overall parity bit at index %d\n", i);\
    }\
    mat_cols[i] = element.column;\
    mat_values[i] = element.value;\
  }\
  else\
  {\
    if(syndrome)\
    {\
      /* Overall parity fine but error in syndrom */\
      /*  Must be double-bit error - cannot correct this */\
      PRINTF_CL("[ECC] double-bit error detected\n");\
      error_flag = update_error(error_flag_buffer, ERROR_SECDED);\
      exit_op;\
    }\
  }\
  /* Mask out ECC from high order column bits */\
  element.column &= 0x00FFFFFF;\
  col = element.column;\
} else

//Kernels
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
  __global uint * restrict error_flag_buffer
#if defined(FT_CONSTRAINTS)
  ,const uint nnz
#endif
  )
{
  const uint global_id = get_global_id(0);
  uchar error_flag = NO_ERROR;
  if(global_id < N)
  {
    uint start = mat_rows[global_id];
    uint end   = mat_rows[global_id+1];

#if defined(FT_CONSTRAINTS)
    CONSTRAINTS_CHECK_ROW(global_id, start, end, nnz, error_flag, error_flag_buffer, return);
#endif
    // initialize local sum
    double tmp = 0;
    // accumulate local sums
    for(uint i = start; i < end; i++)
    {
      uint col = mat_cols[i];
#if defined(FT_CONSTRAINTS)
      CONSTRAINTS_CHECK_COL(i, col, mat_cols[i+1], N, nnz, error_flag, error_flag_buffer, return);
#elif defined(FT_SED)
      SED(col, mat_values[i], i, error_flag, error_flag_buffer, return);
#elif defined(FT_SEC7)
      SEC7(col, mat_values, mat_cols, i);
#elif defined(FT_SEC8)
      SEC8(col, mat_values, mat_cols, i);
#elif defined(FT_SECDED)
      SECDED(col, mat_values, mat_cols, i, error_flag, error_flag_buffer, return);
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
  __global uint * restrict error_flag_buffer,
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
  uchar error_flag = NO_ERROR;
  for(uint row = vector_id; row < N; row += num_vectors)
  {
    const uint start = mat_rows[row];
    const uint end   = mat_rows[row+1];

#if defined(FT_CONSTRAINTS)
    CONSTRAINTS_CHECK_ROW(row, start, end, nnz, error_flag, error_flag_buffer, );
#endif
    // initialize local sum
    double tmp = 0;
    // accumulate local sums
    for(uint i = start + thread_lane; i < end && error_flag == NO_ERROR; i += THREADS_PER_VECTOR)
    {
      uint col = mat_cols[i];
#if defined(FT_CONSTRAINTS)
      CONSTRAINTS_CHECK_COL(i, col, mat_cols[i+1], N, nnz, error_flag, error_flag_buffer, break);
#elif defined(FT_SED)
      SED(col, mat_values[i], i, error_flag, error_flag_buffer, break);
#elif defined(FT_SEC7)
      SEC7(col, mat_values, mat_cols, i);
#elif defined(FT_SEC8)
      SEC8(col, mat_values, mat_cols, i);
#elif defined(FT_SECDED)
      SECDED(col, mat_values, mat_cols, i, error_flag, error_flag_buffer, break);
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

