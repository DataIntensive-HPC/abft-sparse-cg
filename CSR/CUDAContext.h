#include "CGContext.h"
#include <cuda.h>

#include "ecc.h"

//different spmv methods
#define SPMV_SCALAR 0
#define SPMV_VECTOR 1

#ifndef SPMV_METHOD
#define SPMV_METHOD SPMV_SCALAR
#endif

//Group sizes for kernels
#ifndef DOT_PRODUCT_KERNEL_WG
#define DOT_PRODUCT_KERNEL_WG 256
#endif
#ifndef DOT_PRODUCT_KERNEL_ITEMS_PER_WORK_ITEM
#define DOT_PRODUCT_KERNEL_ITEMS_PER_WORK_ITEM 32
#endif


#ifndef CALC_XR_KERNEL_WG
#define CALC_XR_KERNEL_WG 256
#endif
#ifndef CALC_XR_KERNEL_ITEMS_PER_WORK_ITEM
#define CALC_XR_KERNEL_ITEMS_PER_WORK_ITEM 2
#endif

#ifndef CALC_P_KERNEL_WG
#define CALC_P_KERNEL_WG 256
#endif
#ifndef CALC_P_KERNEL_ITEMS_PER_WORK_ITEM
#define CALC_P_KERNEL_ITEMS_PER_WORK_ITEM 1
#endif

#ifndef SPMV_KERNEL_WG
#if SPMV_METHOD == SPMV_SCALAR
  #define SPMV_KERNEL_WG 32
#elif SPMV_METHOD == SPMV_VECTOR
  #define SPMV_KERNEL_WG 256
#endif
#endif

#ifndef SPMV_KERNEL_ITEMS_PER_WORK_ITEM
#if SPMV_METHOD == SPMV_SCALAR
  #define SPMV_KERNEL_ITEMS_PER_WORK_ITEM 1
#elif SPMV_METHOD == SPMV_VECTOR
  #define SPMV_KERNEL_ITEMS_PER_WORK_ITEM 1
#endif
#endif

#define VECTOR_SUM_NO_PINNED  0
#define VECTOR_SUM_PINNED     1
#define VECTOR_SUM_METHOD_USE VECTOR_SUM_NO_PINNED

struct cg_vector
{
  int N;
  double * data;
};

struct cg_matrix
{
  unsigned N;
  unsigned nnz;
  uint32_t *cols;
  uint32_t *rows;
  double   *values;
};

struct cuda_kernel
{
  uint8_t first_run;
  size_t ngroups;
  size_t group_size;
  size_t global_size;
  uint32_t items_per_work_item;
};

class CUDAContext : public CGContext
{

public:
  CUDAContext(FT_Type type);
  CUDAContext() : CUDAContext(NONE) {};
  virtual ~CUDAContext();

  virtual void generate_ecc_bits(csr_element& element);
  virtual cg_matrix* create_matrix(const uint32_t *columns,
                                   const uint32_t *rows,
                                   const double *values,
                                   int N, int nnz);
  virtual void destroy_matrix(cg_matrix *mat);

  virtual cg_vector* create_vector(int N);
  virtual void destroy_vector(cg_vector *vec);
  virtual double* map_vector(cg_vector *v);
  virtual void unmap_vector(cg_vector *v, double *h);
  virtual void copy_vector(cg_vector *dst, const cg_vector *src);

  virtual double dot(const cg_vector *a, const cg_vector *b);
  virtual double calc_xr(cg_vector *x, cg_vector *r,
                         const cg_vector *p, const cg_vector *w,
                         double alpha);
  virtual void calc_p(cg_vector *p, const cg_vector *r, double beta);

  virtual void spmv(const cg_matrix *mat, const cg_vector *vec,
                    cg_vector *result);

  virtual void inject_bitflip(cg_matrix *mat, BitFlipKind kind, int num_flips);
  double sum_vector(double * d_buffer, double * h_buffer, const uint32_t N);
  void setup_cuda_kernel(cuda_kernel* kernel, const size_t items_per_work_item, const size_t group_size, const size_t total_work);


  //kernels
  //bit fault injection
  cuda_kernel * k_inject_bitflip_val;
  cuda_kernel * k_inject_bitflip_col;

  cuda_kernel * k_dot_product;
  cuda_kernel * k_calc_xr;
  cuda_kernel * k_calc_p;
  cuda_kernel * k_spmv;

#if SPMV_METHOD == SPMV_VECTOR
  size_t _SPMV_THREADS_PER_VECTOR;
  size_t _SPMV_VECTORS_PER_BLOCK;
#endif

private:
  FT_Type ftType;

  //support buffers
  double * h_dot_product_partial;
  double * d_dot_product_partial;
  double * h_calc_xr_partial;
  double * d_calc_xr_partial;
};

class CUDAContext_Constraints : public CUDAContext
{
public:
  using CUDAContext::CUDAContext;
  CUDAContext_Constraints() : CUDAContext(CONSTRAINTS) {};
};

class CUDAContext_SED : public CUDAContext
{
  virtual void generate_ecc_bits(csr_element& element);

public:
  using CUDAContext::CUDAContext;
  CUDAContext_SED() : CUDAContext(SED) {};
};

class CUDAContext_SEC7 : public CUDAContext
{
  virtual void generate_ecc_bits(csr_element& element);

public:
  using CUDAContext::CUDAContext;
  CUDAContext_SEC7() : CUDAContext(SEC7) {};
};

class CUDAContext_SEC8 : public CUDAContext
{
  virtual void generate_ecc_bits(csr_element& element);

public:
  using CUDAContext::CUDAContext;
  CUDAContext_SEC8() : CUDAContext(SEC8) {};
};

class CUDAContext_SECDED : public CUDAContext
{
  virtual void generate_ecc_bits(csr_element& element);

public:
  using CUDAContext::CUDAContext;
  CUDAContext_SECDED() : CUDAContext(SECDED) {};
};

#define cudaCheck(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
