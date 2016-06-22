#include "CGContext.h"

#include "ecc.h"
#include "OCLUtility.h"

#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define OCL_DEVICE_ID 0

#define KERNELS_SOURCE "CSR/OCLKernels.cl"
#define OPENCL_FLAGS ""

//kernels
//bit fault injection
#define INJECT_BITFLIP_VAL_KERNEL "inject_bitflip_val"
#define INJECT_BITFLIP_COL_KERNEL "inject_bitflip_col"
//none
#define DOT_PRODUCT_KERNEL "dot_product"
#define CALC_P_KERNEL "calc_p"
#define CALC_XR_KERNEL "calc_xr"
#define SPMV_KERNEL "spmv"
#define SPMV_CONSTRAINTS_KERNEL "spmv_constraints"
#define SPMV_SED_KERNEL "spmv_sed"
#define SPMV_SEC7_KERNEL "spmv_sec7"
#define SPMV_SEC8_KERNEL "spmv_sec8"
#define SPMV_SECDED_KERNEL "spmv_secded"
#define SUM_VECTOR_KERNEL "sum_vector"


//Group sizes for kernels
#define DOT_PRODUCT_KERNEL_WG 16
#define DOT_PRODUCT_KERNEL_ITEMS 16

#define CALC_XR_KERNEL_WG 16
#define CALC_XR_KERNEL_ITEMS 16

#define CALC_P_KERNEL_WG 16
#define CALC_P_KERNEL_ITEMS 16

#define SPMV_KERNEL_WG 16
#define SPMV_KERNEL_ITEMS 1

#define SPMV_CONSTRAINTS_KERNEL_WG 16
#define SPMV_CONSTRAINTS_KERNEL_ITEMS 1

#define SPMV_SED_KERNEL_WG 16
#define SPMV_SED_KERNEL_ITEMS 1

#define SPMV_SEC7_KERNEL_WG 16
#define SPMV_SEC7_KERNEL_ITEMS 1

#define SPMV_SEC8_KERNEL_WG 16
#define SPMV_SEC8_KERNEL_ITEMS 1

#define SPMV_SECDED_KERNEL_WG 16
#define SPMV_SECDED_KERNEL_ITEMS 1

#define VECTOR_SUM_SIMPLE 1
#define VECTOR_SUM_PINNED 2
#define VECTOR_SUM_METHOD_USE VECTOR_SUM_SIMPLE



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
private:
  cl_device_id ocl_device;
  uint32_t ocl_max_compute_units;

  //kernels
  //bit fault injection
  ocl_kernel * k_inject_bitflip_val;
  ocl_kernel * k_inject_bitflip_col;
  //none
  ocl_kernel * k_dot_product;
  ocl_kernel * k_calc_xr;
  ocl_kernel * k_calc_p;

  //support buffers
  double * h_dot_product_partial;
  cl_mem d_dot_product_partial;
  double * h_calc_xr_partial;
  cl_mem d_calc_xr_partial;

#if VECTOR_SUM_METHOD_USE == VECTOR_SUM_PINNED
  ocl_kernel * k_sum_vector;
  double * h_pinned_return;
  cl_mem d_pinned_return;
#endif

public:
  OCLContext();
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
  double sum_vector(cl_mem buffer, const uint32_t N);

  ocl_kernel * k_spmv;
  cl_context ocl_context;
  cl_command_queue ocl_queue;
  cl_program ocl_program;

};

class OCLContext_Constraints : public OCLContext
{
  using OCLContext::OCLContext;
  virtual void spmv(const cg_matrix *mat, const cg_vector *vec,
                    cg_vector *result);
public:
  OCLContext_Constraints();
};

class OCLContext_SED : public OCLContext
{
  using OCLContext::OCLContext;
  virtual void generate_ecc_bits(csr_element& element);
  virtual void spmv(const cg_matrix *mat, const cg_vector *vec,
                    cg_vector *result);
public:
  OCLContext_SED();
};

class OCLContext_SEC7 : public OCLContext
{
  using OCLContext::OCLContext;
  virtual void generate_ecc_bits(csr_element& element);
  virtual void spmv(const cg_matrix *mat, const cg_vector *vec,
                    cg_vector *result);
public:
  OCLContext_SEC7();
};

class OCLContext_SEC8 : public OCLContext
{
  using OCLContext::OCLContext;
  virtual void generate_ecc_bits(csr_element& element);
  virtual void spmv(const cg_matrix *mat, const cg_vector *vec,
                    cg_vector *result);
public:
  OCLContext_SEC8();
};

class OCLContext_SECDED : public OCLContext
{
  using OCLContext::OCLContext;
  virtual void generate_ecc_bits(csr_element& element);
  virtual void spmv(const cg_matrix *mat, const cg_vector *vec,
                    cg_vector *result);
public:
  OCLContext_SECDED();
};
