#include "CUDAContext.h"
#include "CUDAecc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>


__device__ inline void cuda_terminate()
{
  __threadfence();
  asm("trap;");
}
//macro for checking correct row constraints
__device__ void constraints_check_row(uint32_t row, uint32_t start, uint32_t end, uint32_t nnz)
{
  if(end > nnz)
  {
    printf("row size constraint violated for row %u\n", row);
    cuda_terminate();
  }
  else if(end < start)
  {
    printf("row order constraint violated for row %u\n", row);
    cuda_terminate();
  }
}

//macro for checking correct column constraints
__device__ void constraints_check_col(uint32_t i, uint32_t end, uint32_t col, uint32_t next_col, uint32_t N, uint32_t nnz)
{
  if(col >= N)
  {
    printf("column size constraint violated at index %u\n", i);
    cuda_terminate();
  }
  else if(i < end-1 && next_col <= col)
  {
    printf("column order constraint violated at index %u\n", i);
    cuda_terminate();
  }
}

//macro for SED
__device__ void sed(uint32_t * col, double val, uint32_t i)
{
  uint32_t * b_val = (uint32_t*)&val;
  /* Check overall parity bit*/
  if(cu_ecc_compute_overall_parity(*col, b_val))
  {
    printf("[ECC] error detected at index %u\n", i);
    cuda_terminate();
  }
  /* Mask out ECC from high order column bits */
  *col &= 0x00FFFFFF;
}

//macro for SEC7
__device__ void sec7(uint32_t * col, double * val, double * mat_values, uint32_t * mat_cols, uint32_t i)
{
  uint32_t * b_val = (uint32_t*)val;
  /* Check ECC */
  uint32_t syndrome = cu_ecc_compute_col8(*col, b_val);
  if(syndrome)
  {
    /* Unflip bit */
    uint32_t bit = cu_ecc_get_flipped_bit_col8(syndrome);
    CU_CORRECT_BIT(bit, (*col), b_val);
    mat_cols[i] = *col;
    mat_values[i] = *val = *(double*)b_val;
    printf("[ECC] corrected bit %u at index %u\n", bit, i);
  }
  /* Mask out ECC from high order column bits */
  *col &= 0x00FFFFFF;
}

//macro for SEC8
__device__ void sec8(uint32_t * col, double * val, double * mat_values, uint32_t * mat_cols, uint32_t i)
{
  uint32_t * b_val = (uint32_t*)val;
  /* Check overall parity bit */
  if(cu_ecc_compute_overall_parity(*col, b_val))
  {
    /* Compute error syndrome from hamming bits, if syndrome 0, then fix parity*/
    uint32_t syndrome = cu_ecc_compute_col8(*col, b_val);
    uint32_t bit = !syndrome ? 88 : cu_ecc_get_flipped_bit_col8(syndrome);
    CU_CORRECT_BIT(bit, (*col), b_val);
    printf("[ECC] corrected bit %u at index %u\n", bit, i);
    mat_cols[i] = *col;
    mat_values[i] = *val = *(double*)b_val;
  }
  /* Mask out ECC from high order column bits */
  *col &= 0x00FFFFFF;
}

//macro for SECDED
__device__ void secded(uint32_t * col, double * val, double * mat_values, uint32_t * mat_cols, uint32_t i)
{
  uint32_t * b_val = (uint32_t*)val;
  /* Check parity bits */
  uint32_t overall_parity = cu_ecc_compute_overall_parity(*col, b_val);
  uint32_t syndrome = cu_ecc_compute_col8(*col, b_val);
  if(overall_parity)
  {
    /* if syndrome 0, then fix parity*/
    uint32_t bit = !syndrome ? 88 : cu_ecc_get_flipped_bit_col8(syndrome);
    CU_CORRECT_BIT(bit, (*col), b_val);
    printf("[ECC] corrected bit %u at index %u\n", bit, i);
    mat_cols[i] = *col;
    mat_values[i] = *val = *(double*)b_val;
  }
  else
  {
    if(syndrome)
    {
      /* Overall parity fine but error in syndrom */
      /* Must be double-bit error - cannot correct this */
      printf("[ECC] double-bit error detected\n");
      cuda_terminate();
    }
  }
  /* Mask out ECC from high order column bits */
  *col &= 0x00FFFFFF;
}

__global__ void inject_bitflip_val(
  const uint32_t bit, //vector size
  const uint32_t index,
  double * values)
{
  printf("*** flipping bit %u of value at index %u ***\n", bit, index);
	*((uint64_t*)values+index) ^= 0x1ULL << bit;
}

__global__ void inject_bitflip_col(
  const uint32_t bit, //vector size
  const uint32_t index,
  uint32_t * values)
{
  printf("*** flipping bit %u of column at index %u ***\n", bit, index);
  values[index] ^= 0x1U << bit;
}

template <uint32_t blockSize, uint32_t items_per_work_item, uint32_t items_per_work_group>
__global__ void dot_product_kernel(
  const unsigned int N,
  const double * __restrict__ a,
  const double * __restrict__ b,
  double * __restrict__ result)
{
  __shared__ double partial_result[blockSize];

  const uint32_t local_id = threadIdx.x;
  const uint32_t group_size = blockDim.x;
  const uint32_t group_id = blockIdx.x;

  double ret = 0.0;
  double tmp;
  uint32_t offset = group_id * items_per_work_group + local_id;
  for (uint32_t i = 0; i < items_per_work_item; i++, offset += group_size)
  {
    uint8_t in_range = offset < N;
    uint32_t local_offset = in_range ? offset : 0;
    tmp = a[local_offset] * b[local_offset];
    ret += in_range ? tmp : 0.0;
  }

  partial_result[local_id] = ret;

  //do a reduction
  for(uint32_t step = blockSize >> 1; step > 0; step>>=1)
  {
		__syncthreads();
    if(local_id < step)
    {
      partial_result[local_id] += partial_result[local_id + step];
    }
  }

	__syncthreads();
  if(local_id == 0)
  {
  	result[group_id] = partial_result[0];
  }
}

template <uint32_t items_per_work_item, uint32_t items_per_work_group>
__global__ void calc_p_kernel(
  const uint32_t N, //vector size
  const double beta,
  const double * __restrict__ r,
  double * __restrict__ p)
{
  const uint32_t local_id = threadIdx.x;
  const uint32_t group_size = blockDim.x;
  const uint32_t group_id = blockIdx.x;

  uint32_t offset = group_id * items_per_work_group + local_id;
  for (uint32_t i = 0; i < items_per_work_item && offset < N; i++, offset += group_size)
  {
    p[offset] = fma(beta, p[offset], r[offset]);
  }

}

template <uint32_t blockSize, uint32_t items_per_work_item, uint32_t items_per_work_group>
__global__ void calc_xr_kernel(
  const uint32_t N, //vector size
  const double alpha,
  const double * __restrict__ p,
  const double * __restrict__ w,
  double * __restrict__ x,
  double * __restrict__ r,
  double * __restrict__ result)
{
  __shared__ double partial_result[blockSize];

  const uint32_t local_id = threadIdx.x;
  const uint32_t group_size = blockDim.x;
  const uint32_t group_id = blockIdx.x;

  double ret = 0.0;

  const uint32_t offset = group_id * items_per_work_group + local_id;

  uint32_t j = offset;
  for (uint32_t i = 0; i < items_per_work_item && j < N; i++, j += group_size)
  {
    x[j] = fma(alpha, p[j], x[j]);
  }

  j = offset;
  for (uint32_t i = 0; i < items_per_work_item && j < N; i++, j += group_size)
  {
    r[j] = fma(-alpha, w[j], r[j]);
    ret = fma(r[j], r[j], ret);
  }
  partial_result[local_id] = ret;
  //do a reduction
  for(uint32_t step = group_size >> 1; step > 1; step>>=1)
  {
    __syncthreads();
    if(local_id < step)
    {
      partial_result[local_id] += partial_result[local_id + step];
    }
  }
  //store result in a global array
  __syncthreads();
  if(local_id == 0)
  {
    result[group_id] = partial_result[0] + partial_result[1];
  }
}


//CSR_SCALAR TECHNIQUE
template <FT_Type ftType>
__global__ void spmv_scalar_kernel(
  const uint32_t N, //vector size
  const uint32_t * __restrict__ mat_rows,
  uint32_t * __restrict__ mat_cols,
  double * __restrict__ mat_values,
  const double * __restrict__ vec,
  double * __restrict__ result,
  const uint32_t nnz)
{
  const uint32_t global_id = threadIdx.x + blockDim.x * blockIdx.x;
  if(global_id < N)
  {
    uint32_t start = mat_rows[global_id];
    uint32_t end   = mat_rows[global_id+1];

    if(ftType == CONSTRAINTS)
    {
      constraints_check_row(global_id, start, end, nnz);
    }

    // initialize local sum
    double tmp = 0;
    // accumulate local sums
    for(uint32_t i = start; i < end; i++)
    {
      uint32_t col = mat_cols[i];
      double val = mat_values[i];
      switch(ftType)
      {
      	case CONSTRAINTS:
      	  constraints_check_col(i, end, col, mat_cols[i+1], N, nnz);
        break; //CONSTRAINTS
				case SED:
          sed(&col, val, i);
		    break; //SED
				case SEC7:
		      sec7(&col, &val, mat_values, mat_cols, i);
        break; //SEC7
				case SEC8:
		      sec8(&col, &val, mat_values, mat_cols, i);
        break; //SEC8
				case SECDED:
		      secded(&col, &val, mat_values, mat_cols, i);
        break;
			}
      tmp = fma(val, vec[col], tmp);
    }
    result[global_id] = tmp;
  }
}


//CSR_VECTOR TECHNIQUE
//csr_spmv kernel extracted from bhSPARSE: https://github.com/bhSPARSE/bhSPARSE
template <FT_Type ftType>
__global__ void spmv_vector_kernel(
  const uint32_t N, //vector size
  const uint32_t * __restrict__ mat_rows,
  uint32_t * __restrict__ mat_cols,
  double * __restrict__ mat_values,
  const double * __restrict__ vec,
  double * __restrict__ result,
  const uint32_t nnz,
  const uint32_t VECTORS_PER_BLOCK,
  const uint32_t THREADS_PER_VECTOR
  )
{
  extern __shared__ double partial_result[];
  const uint32_t local_id   = threadIdx.x;

  const uint32_t thread_lane = local_id % THREADS_PER_VECTOR; // thread index within the vector
  const uint32_t vector_id   = (blockIdx.x *blockDim.x + threadIdx.x)/  THREADS_PER_VECTOR; // global vector index
  const uint32_t num_vectors = VECTORS_PER_BLOCK * gridDim.x; // total number of active vectors

  for(uint32_t row = vector_id; row < N; row += num_vectors)
  {
    const uint32_t start = mat_rows[row];
    const uint32_t end   = mat_rows[row+1];
    if(ftType == CONSTRAINTS)
    {
      constraints_check_row(row, start, end, nnz);
    }
    // initialize local sum
    double tmp = 0;
    // accumulate local sums
    for(uint32_t i = start + thread_lane; i < end; i += THREADS_PER_VECTOR)
    {
      uint32_t col = mat_cols[i];
      double val = mat_values[i];
      switch(ftType)
      {
        case CONSTRAINTS:
          constraints_check_col(i, end, col, mat_cols[i+1], N, nnz);
        break; //CONSTRAINTS
        case SED:
          sed(&col, val, i);
        break; //SED
        case SEC7:
          sec7(&col, &val, mat_values, mat_cols, i);
        break; //SEC7
        case SEC8:
          sec8(&col, &val, mat_values, mat_cols, i);
        break; //SEC8
        case SECDED:
          secded(&col, &val, mat_values, mat_cols, i);
        break;
      }
      tmp = fma(val, vec[col], tmp);
    }
    // store local sum in shared memory
    partial_result[local_id] = tmp;

    // reduce local sums to row tmp
    for(uint32_t step = THREADS_PER_VECTOR >> 1; step > 0; step>>=1)
    {
      __syncthreads();
      if(thread_lane < step)
      {
        partial_result[local_id] += partial_result[local_id + step];
      }
    }

    // first thread writes the result
    __syncthreads();
    if(thread_lane == 0)
    {
      result[row] = partial_result[local_id];
    }
  }
}

void CUDAContext::generate_ecc_bits(csr_element& element)
{
}

CUDAContext::CUDAContext(FT_Type type)
{
  ftType = type;

  k_inject_bitflip_val = new cuda_kernel;
  k_inject_bitflip_val->first_run = 1;

  k_inject_bitflip_col = new cuda_kernel;
  k_inject_bitflip_col->first_run = 1;

  k_dot_product = new cuda_kernel;
  k_dot_product->first_run = 1;

  k_calc_xr = new cuda_kernel;
  k_calc_xr->first_run = 1;

  k_calc_p = new cuda_kernel;
  k_calc_p->first_run = 1;

  k_spmv = new cuda_kernel;
  k_spmv->first_run = 1;
}

CUDAContext::~CUDAContext()
{
  delete[] h_dot_product_partial;
  delete[] h_calc_xr_partial;

  cudaCheck(cudaFree(d_dot_product_partial));
  cudaCheck(cudaFree(d_calc_xr_partial));
}

cg_matrix* CUDAContext::create_matrix(const uint32_t *columns,
                                     const uint32_t *rows,
                                     const double *values,
                                     int N, int nnz)
{
  cg_matrix* M = new cg_matrix;
  M->N      = N;
  M->nnz    = nnz;
  //allocate buffers on the device
  cudaCheck(cudaMalloc((void**)&M->cols, sizeof(uint32_t) * nnz));
  cudaCheck(cudaMalloc((void**)&M->rows, sizeof(uint32_t) * (N+1)));
  cudaCheck(cudaMalloc((void**)&M->values, sizeof(double) * nnz));

  //allocate temp memory which is then copied to the device
  uint32_t *h_cols   = new uint32_t[nnz];
  uint32_t *h_rows   = new uint32_t[N+1];
  double   *h_values = new double[nnz];

  uint32_t next_row = 0;
  for (int i = 0; i < nnz; i++)
  {
    csr_element element;
    element.column = columns[i];
    element.value  = values[i];

    generate_ecc_bits(element);

    h_cols[i]   = element.column;
    h_values[i] = element.value;

    while (next_row <= rows[i])
    {
      h_rows[next_row++] = i;
    }
  }
  h_rows[N] = nnz;
  cudaCheck(cudaMemcpy(M->cols, h_cols, sizeof(uint32_t) * nnz, cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(M->rows, h_rows, sizeof(uint32_t) * (N+1), cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(M->values, h_values, sizeof(double) * nnz, cudaMemcpyHostToDevice));

  //clean up temp buffers
  delete[] h_cols;
  delete[] h_rows;
  delete[] h_values;

  return M;
}

void CUDAContext::destroy_matrix(cg_matrix *mat)
{
  cudaCheck(cudaFree(mat->cols));
  cudaCheck(cudaFree(mat->rows));
  cudaCheck(cudaFree(mat->values));
  delete mat;
}

cg_vector* CUDAContext::create_vector(int N)
{
  cg_vector *result = new cg_vector;
  result->N = N;
  cudaCheck(cudaMalloc((void**)&result->data, sizeof(double) * N));
  return result;
}

void CUDAContext::destroy_vector(cg_vector *vec)
{
  cudaCheck(cudaFree(vec->data));
  delete vec;
}

double* CUDAContext::map_vector(cg_vector *v)
{
  double* h = new double[v->N];
  cudaCheck(cudaMemcpy(h, v->data, sizeof(double) * v->N, cudaMemcpyDeviceToHost));
  return h;
}

void CUDAContext::unmap_vector(cg_vector *v, double *h)
{
  cudaCheck(cudaMemcpy(v->data, h, sizeof(double) * v->N, cudaMemcpyHostToDevice));
  delete[] h;
}

void CUDAContext::copy_vector(cg_vector *dst, const cg_vector *src)
{
  cudaCheck(cudaMemcpy(dst->data, src->data, sizeof(double) * dst->N, cudaMemcpyDeviceToDevice));
}

double CUDAContext::dot(const cg_vector *a, const cg_vector *b)
{
  if(k_dot_product->first_run){
    CUDAContext::setup_cuda_kernel(k_dot_product, DOT_PRODUCT_KERNEL_ITEMS_PER_WORK_ITEM, DOT_PRODUCT_KERNEL_WG, a->N);
#if VECTOR_SUM_METHOD_USE == VECTOR_SUM_NO_PINNED
    cudaCheck(cudaMalloc((void**)&d_dot_product_partial, sizeof(double) * k_dot_product->ngroups));
    h_dot_product_partial = new double[k_dot_product->ngroups];
#elif VECTOR_SUM_METHOD_USE == VECTOR_SUM_PINNED
#endif
  }

  dot_product_kernel<DOT_PRODUCT_KERNEL_WG,
  									 DOT_PRODUCT_KERNEL_ITEMS_PER_WORK_ITEM,
  									 DOT_PRODUCT_KERNEL_WG*DOT_PRODUCT_KERNEL_ITEMS_PER_WORK_ITEM>
	  								<<<k_dot_product->ngroups,DOT_PRODUCT_KERNEL_WG>>>
	  								(a->N, a->data, b->data, d_dot_product_partial);

  return CUDAContext::sum_vector(d_dot_product_partial, h_dot_product_partial, k_dot_product->ngroups);
}

double CUDAContext::calc_xr(cg_vector *x, cg_vector *r,
                           const cg_vector *p, const cg_vector *w,
                           double alpha)
{
  if(k_calc_xr->first_run){
    CUDAContext::setup_cuda_kernel(k_calc_xr, CALC_XR_KERNEL_ITEMS_PER_WORK_ITEM, CALC_XR_KERNEL_WG, x->N);
#if VECTOR_SUM_METHOD_USE == VECTOR_SUM_NO_PINNED
    cudaCheck(cudaMalloc((void**)&d_calc_xr_partial, sizeof(double) * k_calc_xr->ngroups));
    h_calc_xr_partial = new double[k_calc_xr->ngroups];
#elif VECTOR_SUM_METHOD_USE == VECTOR_SUM_PINNED
#endif
  }

	calc_xr_kernel<CALC_XR_KERNEL_WG,
								 CALC_XR_KERNEL_ITEMS_PER_WORK_ITEM,
								 CALC_XR_KERNEL_WG*CALC_XR_KERNEL_ITEMS_PER_WORK_ITEM>
								<<<k_calc_xr->ngroups,CALC_XR_KERNEL_WG>>>
							  (x->N, alpha, p->data, w->data, x->data, r->data, d_calc_xr_partial);
  return CUDAContext::sum_vector(d_calc_xr_partial, h_calc_xr_partial, k_calc_xr->ngroups);
}

void CUDAContext::calc_p(cg_vector *p, const cg_vector *r, double beta)
{
  if(k_calc_p->first_run){
    CUDAContext::setup_cuda_kernel(k_calc_p, CALC_P_KERNEL_ITEMS_PER_WORK_ITEM, CALC_P_KERNEL_WG, p->N);
  }

	calc_p_kernel<CALC_P_KERNEL_ITEMS_PER_WORK_ITEM,
	              CALC_P_KERNEL_ITEMS_PER_WORK_ITEM*CALC_P_KERNEL_WG>
	             <<<k_calc_p->ngroups,CALC_P_KERNEL_WG>>>
	             (p->N, beta, r->data, p->data);
}

void CUDAContext::spmv(const cg_matrix *mat, const cg_vector *vec,
                      cg_vector *result)
{

  if(k_spmv->first_run){
    size_t total_work = mat->N;
#if SPMV_METHOD == SPMV_VECTOR

    const int nnz_per_row = mat->nnz / mat->N;

    if (nnz_per_row <=  2)
    {
      _SPMV_THREADS_PER_VECTOR = 2;
    }
    else if (nnz_per_row <=  4)
    {
      _SPMV_THREADS_PER_VECTOR = 4;
    }
    else if (nnz_per_row <=  8)
    {
      _SPMV_THREADS_PER_VECTOR = 8;
    }
    else if (nnz_per_row <= 16)
    {
      _SPMV_THREADS_PER_VECTOR = 16;
    }
    else if (nnz_per_row <= 32)
    {
      _SPMV_THREADS_PER_VECTOR = 32;
    }
    else
    {
      _SPMV_THREADS_PER_VECTOR = 64;
    }

    _SPMV_VECTORS_PER_BLOCK  = SPMV_KERNEL_WG / _SPMV_THREADS_PER_VECTOR;
    total_work = mat->nnz;
#endif
    CUDAContext::setup_cuda_kernel(k_spmv, SPMV_KERNEL_ITEMS_PER_WORK_ITEM, SPMV_KERNEL_WG, total_work);
  }
  switch(ftType)
  {
  	case NONE:

#if SPMV_METHOD == SPMV_SCALAR
  		spmv_scalar_kernel<NONE>
    		<<<k_spmv->ngroups,SPMV_KERNEL_WG>>>
    		(mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz);
#elif SPMV_METHOD == SPMV_VECTOR
      spmv_vector_kernel<NONE>
        <<<k_spmv->ngroups,SPMV_KERNEL_WG, sizeof(double) * (_SPMV_VECTORS_PER_BLOCK * _SPMV_THREADS_PER_VECTOR + _SPMV_THREADS_PER_VECTOR / 2)>>>
        (mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz, _SPMV_VECTORS_PER_BLOCK, _SPMV_THREADS_PER_VECTOR);
#endif

		break;
  	case CONSTRAINTS:

#if SPMV_METHOD == SPMV_SCALAR
  		spmv_scalar_kernel<CONSTRAINTS>
    		<<<k_spmv->ngroups,SPMV_KERNEL_WG>>>
    		(mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz);
#elif SPMV_METHOD == SPMV_VECTOR
      spmv_vector_kernel<CONSTRAINTS>
        <<<k_spmv->ngroups,SPMV_KERNEL_WG, sizeof(double) * (_SPMV_VECTORS_PER_BLOCK * _SPMV_THREADS_PER_VECTOR + _SPMV_THREADS_PER_VECTOR / 2)>>>
        (mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz, _SPMV_VECTORS_PER_BLOCK, _SPMV_THREADS_PER_VECTOR);
#endif

		break;
  	case SED:

#if SPMV_METHOD == SPMV_SCALAR
  		spmv_scalar_kernel<SED>
    		<<<k_spmv->ngroups,SPMV_KERNEL_WG>>>
    		(mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz);
#elif SPMV_METHOD == SPMV_VECTOR
      spmv_vector_kernel<SED>
        <<<k_spmv->ngroups,SPMV_KERNEL_WG, sizeof(double) * (_SPMV_VECTORS_PER_BLOCK * _SPMV_THREADS_PER_VECTOR + _SPMV_THREADS_PER_VECTOR / 2)>>>
        (mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz, _SPMV_VECTORS_PER_BLOCK, _SPMV_THREADS_PER_VECTOR);
#endif

		break;
  	case SEC7:

#if SPMV_METHOD == SPMV_SCALAR
  		spmv_scalar_kernel<SEC7>
    		<<<k_spmv->ngroups,SPMV_KERNEL_WG>>>
    		(mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz);
#elif SPMV_METHOD == SPMV_VECTOR
      spmv_vector_kernel<SEC7>
        <<<k_spmv->ngroups,SPMV_KERNEL_WG, sizeof(double) * (_SPMV_VECTORS_PER_BLOCK * _SPMV_THREADS_PER_VECTOR + _SPMV_THREADS_PER_VECTOR / 2)>>>
        (mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz, _SPMV_VECTORS_PER_BLOCK, _SPMV_THREADS_PER_VECTOR);
#endif

		break;
  	case SEC8:

#if SPMV_METHOD == SPMV_SCALAR
  		spmv_scalar_kernel<SEC8>
    		<<<k_spmv->ngroups,SPMV_KERNEL_WG>>>
    		(mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz);
#elif SPMV_METHOD == SPMV_VECTOR
      spmv_vector_kernel<SEC8>
        <<<k_spmv->ngroups,SPMV_KERNEL_WG, sizeof(double) * (_SPMV_VECTORS_PER_BLOCK * _SPMV_THREADS_PER_VECTOR + _SPMV_THREADS_PER_VECTOR / 2)>>>
        (mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz, _SPMV_VECTORS_PER_BLOCK, _SPMV_THREADS_PER_VECTOR);
#endif

		break;
  	case SECDED:

#if SPMV_METHOD == SPMV_SCALAR
  		spmv_scalar_kernel<SECDED>
    		<<<k_spmv->ngroups,SPMV_KERNEL_WG>>>
    		(mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz);
#elif SPMV_METHOD == SPMV_VECTOR
      spmv_vector_kernel<SECDED>
        <<<k_spmv->ngroups,SPMV_KERNEL_WG, sizeof(double) * (_SPMV_VECTORS_PER_BLOCK * _SPMV_THREADS_PER_VECTOR + _SPMV_THREADS_PER_VECTOR / 2)>>>
        (mat->N, mat->rows, mat->cols, mat->values, vec->data, result->data, mat->nnz, _SPMV_VECTORS_PER_BLOCK, _SPMV_THREADS_PER_VECTOR);
#endif

		break;
	}
}

double CUDAContext::sum_vector(double * d_buffer, double * h_buffer, const uint32_t N)
{
  //sum the vector in the kernel
	double result = 0;
#if VECTOR_SUM_METHOD_USE == VECTOR_SUM_NO_PINNED
	cudaCheck(cudaMemcpy(h_buffer, d_buffer, sizeof(double) * N, cudaMemcpyDeviceToHost));
#elif VECTOR_SUM_METHOD_USE == VECTOR_SUM_PINNED
#endif

  for(uint32_t i = 0; i < N; i++){
    result += h_buffer[i];
  }

#if VECTOR_SUM_METHOD_USE == VECTOR_SUM_PINNED
#endif
  return result;

}

void CUDAContext::setup_cuda_kernel(cuda_kernel* kernel, const size_t items_per_work_item, const size_t group_size, const size_t total_work)
{
  kernel->group_size = group_size;
  kernel->items_per_work_item = items_per_work_item;
  kernel->ngroups = ceil((float)total_work / (float)(group_size * items_per_work_item));
  kernel->global_size = group_size * kernel->ngroups;
  kernel->first_run = 0;
}

void CUDAContext::inject_bitflip(cg_matrix *mat, BitFlipKind kind, int num_flips)
{
  uint32_t index = rand() % mat->nnz;

  uint32_t start = 0;
  uint32_t end   = 96;

  if (kind == VALUE)
    end = 64;
  else if (kind == INDEX)
    start = 64;

  for (int i = 0; i < num_flips; i++)
  {
    uint32_t bit = (rand() % (end-start)) + start;
    if (bit < 64)
    {
			inject_bitflip_val<<<1,1>>>(bit, index, mat->values);
    }
    else
    {
      bit = bit - 64;
			inject_bitflip_col<<<1,1>>>(bit, index, mat->cols);
    }
  }
}

void CUDAContext_SED::generate_ecc_bits(csr_element& element)
{
  element.column |= ecc_compute_overall_parity(element) << 31;
}

void CUDAContext_SEC7::generate_ecc_bits(csr_element& element)
{
  element.column |= ecc_compute_col8(element);
}

void CUDAContext_SEC8::generate_ecc_bits(csr_element& element)
{
  element.column |= ecc_compute_col8(element);
  element.column |= ecc_compute_overall_parity(element) << 24;
}

void CUDAContext_SECDED::generate_ecc_bits(csr_element& element)
{
  element.column |= ecc_compute_col8(element);
  element.column |= ecc_compute_overall_parity(element) << 24;
}

namespace
{
  static CGContext::Register<CUDAContext> A("cuda", "none");
  static CGContext::Register<CUDAContext_Constraints> B("cuda", "constraints");
  static CGContext::Register<CUDAContext_SED> C("cuda", "sed");
  static CGContext::Register<CUDAContext_SEC7> D("cuda", "sec7");
  static CGContext::Register<CUDAContext_SEC8> E("cuda", "sec8");
  static CGContext::Register<CUDAContext_SECDED> F("cuda", "secded");
}