#include "CGContext.h"

#include "ecc.h"
#include "OCLUtility.h"
#include "OCL_FTErrors.h"

#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define OCL_DEVICE_ID 0

#define KERNELS_SOURCE "CSR/OCLKernels.cl"
#define OCL_FTERRORS_SOURCE "CSR/OCL_FTErrors.h"
#define OCL_ECC_SOURCE "CSR/OCLecc.h"
#define OPENCL_FLAGS ""

//kernel names
//bit fault injection
#define INJECT_BITFLIP_VAL_KERNEL "inject_bitflip_val"
#define INJECT_BITFLIP_COL_KERNEL "inject_bitflip_col"
//none
#define DOT_PRODUCT_KERNEL "dot_product"
#define CALC_P_KERNEL "calc_p"
#define CALC_XR_KERNEL "calc_xr"
#define SUM_VECTOR_KERNEL "sum_vector"

//different spmv methods
#define SPMV_SCALAR 0
#define SPMV_VECTOR 1

#ifndef SPMV_METHOD
#define SPMV_METHOD SPMV_SCALAR
#endif

#if SPMV_METHOD == SPMV_SCALAR
  #define SPMV_KERNEL "spmv_scalar"
#elif SPMV_METHOD == SPMV_VECTOR
  #define SPMV_KERNEL "spmv_vector"
#endif

//different ft techniques
#define SPMV_FT_NONE "FT_NONE"
#define SPMV_FT_CONSTRAINTS "FT_CONSTRAINTS"
#define SPMV_FT_SED "FT_SED"
#define SPMV_FT_SEC7 "FT_SEC7"
#define SPMV_FT_SEC8 "FT_SEC8"
#define SPMV_FT_SECDED "FT_SECDED"

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
  #define SPMV_KERNEL_WG 64
#elif SPMV_METHOD == SPMV_VECTOR
  #define SPMV_KERNEL_WG 32
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
#define VECTOR_SUM_METHOD_USE VECTOR_SUM_PINNED

#define ERROR_CHECK_NO_PINNED 0
#define ERROR_CHECK_PINNED    1
#define ERROR_CHECK_MEM_USE ERROR_CHECK_PINNED



struct cg_vector
{
  int N;
  cl_mem data;
};

struct cg_matrix
{
  unsigned N;
  unsigned nnz;
  cl_mem cols;
  cl_mem rows;
  cl_mem values;
};

class OCLContext : public CGContext
{
public:
  enum FT_Type { NONE, CONSTRAINTS, SED, SEC7, SEC8, SECDED };
  OCLContext(FT_Type type);
  OCLContext() : OCLContext(NONE) {};
  virtual ~OCLContext();

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
  double sum_vector(cl_mem d_buffer, double * h_buffer, const uint32_t N);
  void check_error();


  cl_device_id ocl_device;
  uint32_t ocl_max_compute_units;
  cl_context ocl_context;
  cl_command_queue ocl_queue;
  cl_program ocl_program;

  //kernels
  //bit fault injection
  ocl_kernel * k_inject_bitflip_val;
  ocl_kernel * k_inject_bitflip_col;

  ocl_kernel * k_dot_product;
  ocl_kernel * k_calc_xr;
  ocl_kernel * k_calc_p;
  ocl_kernel * k_spmv;

#if SPMV_METHOD == SPMV_VECTOR
  size_t _SPMV_THREADS_PER_VECTOR;
  size_t _SPMV_VECTORS_PER_BLOCK;
#endif

private:
  FT_Type ftType;

  //support buffers
  double * h_dot_product_partial;
  cl_mem   d_dot_product_partial;
  double * h_calc_xr_partial;
  cl_mem   d_calc_xr_partial;

  cl_mem    d_error_flag = NULL;
  cl_uint * h_error_flag;
  bool check_for_error = 0;

};

class OCLContext_Constraints : public OCLContext
{
public:
  using OCLContext::OCLContext;
  OCLContext_Constraints() : OCLContext(CONSTRAINTS) {};
};

class OCLContext_SED : public OCLContext
{
  virtual void generate_ecc_bits(csr_element& element);

public:
  using OCLContext::OCLContext;
  OCLContext_SED() : OCLContext(SED) {};
};

class OCLContext_SEC7 : public OCLContext
{
  virtual void generate_ecc_bits(csr_element& element);

public:
  using OCLContext::OCLContext;
  OCLContext_SEC7() : OCLContext(SEC7) {};
};

class OCLContext_SEC8 : public OCLContext
{
  virtual void generate_ecc_bits(csr_element& element);

public:
  using OCLContext::OCLContext;
  OCLContext_SEC8() : OCLContext(SEC8) {};
};

class OCLContext_SECDED : public OCLContext
{
  virtual void generate_ecc_bits(csr_element& element);

public:
  using OCLContext::OCLContext;
  OCLContext_SECDED() : OCLContext(SECDED) {};
};
