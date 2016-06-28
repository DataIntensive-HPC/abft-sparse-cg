#include "OCLContext.h"

OCLContext::OCLContext(FT_Type type)
{
  ftType = type;
  //get device, set up context etc
  ocl_device  = OCLUtils::get_opencl_device(OCL_DEVICE_ID);
  ocl_context = OCLUtils::get_opencl_context(ocl_device);
  ocl_queue = OCLUtils::get_opencl_command_queue(ocl_context, ocl_device);
  ocl_program = OCLUtils::get_opencl_program_from_file(ocl_context, KERNELS_SOURCE);

  cl_int err = clGetDeviceInfo(ocl_device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(uint32_t), &ocl_max_compute_units, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d getting max compute units", err);
  //build program
  std::string defines = " -D ";
  switch(type){
    case NONE:
      defines += SPMV_FT_NONE;
    break;
    case CONSTRAINTS:
      defines += SPMV_FT_CONSTRAINTS;
    break;
    case SED:
      defines += SPMV_FT_SED;
    break;
    case SEC7:
      defines += SPMV_FT_SEC7;
    break;
    case SEC8:
      defines += SPMV_FT_SEC8;
    break;
    case SECDED:
      defines += SPMV_FT_SECDED;
    break;
  }

  if(!OCLUtils::build_opencl_program(ocl_program, ocl_device, ocl_context, OPENCL_FLAGS + defines))
  {
    DIE("Failed to build the program at source %s with flags \"%s\".", KERNELS_SOURCE, OPENCL_FLAGS);
  }
  //set up kernels
  k_dot_product = OCLUtils::get_opencl_kernel(ocl_program, DOT_PRODUCT_KERNEL);
  k_spmv = OCLUtils::get_opencl_kernel(ocl_program, SPMV_KERNEL);
  k_calc_p = OCLUtils::get_opencl_kernel(ocl_program, CALC_P_KERNEL);
  k_calc_xr = OCLUtils::get_opencl_kernel(ocl_program, CALC_XR_KERNEL);
  k_inject_bitflip_val = OCLUtils::get_opencl_kernel(ocl_program, INJECT_BITFLIP_VAL_KERNEL);
  k_inject_bitflip_col = OCLUtils::get_opencl_kernel(ocl_program, INJECT_BITFLIP_COL_KERNEL);

#if VECTOR_SUM_METHOD_USE == VECTOR_SUM_PINNED
  k_sum_vector = OCLUtils::get_opencl_kernel(ocl_program, SUM_VECTOR_KERNEL);
  //set up a pinned buffer for returning a value
  d_pinned_return = clCreateBuffer(ocl_context, CL_MEM_ALLOC_HOST_PTR, sizeof(cl_double), NULL, &err);
  if (CL_SUCCESS != err) DIE("OpenCL error %d creating d_pinned_return", err);
#endif
}

OCLContext::~OCLContext()
{
  delete[] h_dot_product_partial;
  delete[] h_calc_xr_partial;
#if VECTOR_SUM_METHOD_USE == VECTOR_SUM_PINNED
  clReleaseKernel(k_sum_vector->kernel);
  clReleaseMemObject(d_pinned_return);
#endif
  clReleaseKernel(k_dot_product->kernel);
  clReleaseKernel(k_spmv->kernel);
  clReleaseKernel(k_calc_p->kernel);
  clReleaseKernel(k_calc_xr->kernel);
  clReleaseKernel(k_inject_bitflip_val->kernel);
  clReleaseKernel(k_inject_bitflip_col->kernel);
  clReleaseMemObject(d_dot_product_partial);
  clReleaseMemObject(d_calc_xr_partial);
  clReleaseProgram(ocl_program);
  clReleaseCommandQueue(ocl_queue);
  clReleaseContext(ocl_context);
}


void OCLContext::generate_ecc_bits(csr_element& element)
{
}

double OCLContext::sum_vector(cl_mem buffer, const uint32_t N)
{

  //sum the vector in the kernel
  cl_int err;
  double result = 0;

#if VECTOR_SUM_METHOD_USE == VECTOR_SUM_SIMPLE
  double * h = new double[N];
  err = clEnqueueReadBuffer(ocl_queue, buffer, CL_TRUE, 0, sizeof(double) * N, h, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d whilst mapping a vector", err);
  for(uint32_t i = 0; i < N; i++){
    result += h[i];
  }

#elif VECTOR_SUM_METHOD_USE == VECTOR_SUM_PINNED
  size_t items_per_work_item = ceil((float)N/(float)ocl_max_compute_units);
  size_t group_size = ocl_max_compute_units;

  err  = clSetKernelArg(k_sum_vector->kernel, 0, sizeof(uint32_t), &N);
  err |= clSetKernelArg(k_sum_vector->kernel, 1, sizeof(uint32_t), &items_per_work_item);
  err |= clSetKernelArg(k_sum_vector->kernel, 2, sizeof(cl_double) * group_size, NULL);
  err |= clSetKernelArg(k_sum_vector->kernel, 3, sizeof(cl_mem), &buffer);
  err |= clSetKernelArg(k_sum_vector->kernel, 4, sizeof(cl_mem), &d_pinned_return);

  clFinish(ocl_queue);

  err |= clEnqueueNDRangeKernel(ocl_queue, k_sum_vector->kernel, 1, NULL, &group_size, &group_size, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d enquing kernel collision", err);

  clFinish(ocl_queue);
  //return single value using pinned memory
  h_pinned_return = (double *) clEnqueueMapBuffer(ocl_queue, d_pinned_return, CL_TRUE, CL_MAP_READ, 0, sizeof(cl_double), 0, NULL, NULL, &err);
  if (CL_SUCCESS != err) DIE("OpenCL error %d whilst mapping pinned memory", err);
  result = h_pinned_return[0];
  err = clEnqueueUnmapMemObject(ocl_queue, d_pinned_return, h_pinned_return, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d whilst unmapping pinnned memory", err);
#endif
  return result;

}

cg_matrix* OCLContext::create_matrix(const uint32_t *columns,
                                     const uint32_t *rows,
                                     const double *values,
                                     int N, int nnz)
{
  cl_int err;
  cg_matrix* M = new cg_matrix;
  M->N      = N;
  M->nnz    = nnz;
  //allocate buffers on the device
  M->cols   = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, sizeof(uint32_t) * nnz, NULL, &err);
  if (CL_SUCCESS != err) DIE("OpenCL error %d creating buffer cols", err);
  M->rows   = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, sizeof(uint32_t) * (N+1), NULL, &err);
  if (CL_SUCCESS != err) DIE("OpenCL error %d creating buffer rows", err);
  M->values = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, sizeof(double) * nnz, NULL, &err);
  if (CL_SUCCESS != err) DIE("OpenCL error %d creating buffer values", err);

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
  err = clEnqueueWriteBuffer(ocl_queue, M->cols, CL_TRUE, 0, sizeof(uint32_t) * nnz, h_cols, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d writing to buffer cols", err);
  err = clEnqueueWriteBuffer(ocl_queue, M->rows, CL_TRUE, 0, sizeof(uint32_t) * (N+1), h_rows, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d writing to buffer rows", err);
  err = clEnqueueWriteBuffer(ocl_queue, M->values, CL_TRUE, 0, sizeof(double) * nnz, h_values, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d writing to buffer values", err);

  //clean up temp buffers
  delete[] h_cols;
  delete[] h_rows;
  delete[] h_values;

  return M;
}

void OCLContext::destroy_matrix(cg_matrix *mat)
{
  clReleaseMemObject(mat->cols);
  clReleaseMemObject(mat->rows);
  clReleaseMemObject(mat->values);
  delete mat;
}

cg_vector* OCLContext::create_vector(int N)
{
  cl_int err;
  cg_vector *result = new cg_vector;
  result->N    = N;
  result->data = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, sizeof(double) * N, NULL, &err);
  if (CL_SUCCESS != err) DIE("OpenCL error %d creating buffer values", err);
  return result;
}

void OCLContext::destroy_vector(cg_vector *vec)
{
  clReleaseMemObject(vec->data);
  delete vec;
}

double* OCLContext::map_vector(cg_vector *v)
{
  double* h = new double[v->N];
  cl_int err;
  err = clEnqueueReadBuffer(ocl_queue, v->data, CL_TRUE, 0, sizeof(double) * v->N, h, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d whilst mapping a vector", err);
  return h;
}

void OCLContext::unmap_vector(cg_vector *v, double *h)
{
  cl_int err;
  err = clEnqueueWriteBuffer(ocl_queue, v->data, CL_TRUE, 0, sizeof(double) * v->N, h, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d unmapping a buffer", err);
  delete[] h;
}

void OCLContext::copy_vector(cg_vector *dst, const cg_vector *src)
{
  cl_int err = clEnqueueCopyBuffer(ocl_queue, src->data, dst->data, 0, 0, sizeof(double) * dst->N, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d copying buffers", err);
}

double OCLContext::dot(const cg_vector *a, const cg_vector *b)
{
  cl_int err;


  if(k_dot_product->first_run){
    OCLUtils::setup_opencl_kernel(k_dot_product, DOT_PRODUCT_KERNEL_ITEMS, DOT_PRODUCT_KERNEL_WG, a->N);
    d_dot_product_partial = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, sizeof(double) * k_dot_product->ngroups, NULL, &err);
    h_dot_product_partial = new double[k_dot_product->ngroups];
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating d_dot_product_partial", err);
  }
  uint32_t items_per_work_group = k_dot_product->items_per_work_item * k_dot_product->group_size;
  err  = clSetKernelArg(k_dot_product->kernel, 0, sizeof(uint32_t), &a->N);
  err |= clSetKernelArg(k_dot_product->kernel, 1, sizeof(uint32_t), &k_dot_product->items_per_work_item);
  err |= clSetKernelArg(k_dot_product->kernel, 2, sizeof(uint32_t), &items_per_work_group);
  err |= clSetKernelArg(k_dot_product->kernel, 3, sizeof(cl_double)*k_dot_product->group_size, NULL);
  err |= clSetKernelArg(k_dot_product->kernel, 4, sizeof(cl_mem), &a->data);
  err |= clSetKernelArg(k_dot_product->kernel, 5, sizeof(cl_mem), &b->data);
  err |= clSetKernelArg(k_dot_product->kernel, 6, sizeof(cl_mem), &d_dot_product_partial);

  clFinish(ocl_queue);

  err |= clEnqueueNDRangeKernel(ocl_queue, k_dot_product->kernel, 1, NULL, &k_dot_product->global_size, &k_dot_product->group_size, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d enquing kernel dot_product", err);
  return OCLContext::sum_vector(d_dot_product_partial, k_dot_product->ngroups);
}

double OCLContext::calc_xr(cg_vector *x, cg_vector *r,
                           const cg_vector *p, const cg_vector *w,
                           double alpha)
{
  cl_int err;

  if(k_calc_xr->first_run){
    OCLUtils::setup_opencl_kernel(k_calc_xr, CALC_XR_KERNEL_ITEMS, CALC_XR_KERNEL_WG, x->N);
    d_calc_xr_partial = clCreateBuffer(ocl_context, CL_MEM_READ_WRITE, sizeof(double) * k_calc_xr->ngroups, NULL, &err);
    h_calc_xr_partial = new double[k_calc_xr->ngroups];
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating d_calc_xr_partial", err);
  }

  uint32_t items_per_work_group = k_calc_xr->items_per_work_item * k_calc_xr->group_size;
  err  = clSetKernelArg(k_calc_xr->kernel, 0, sizeof(uint32_t), &x->N);
  err  = clSetKernelArg(k_calc_xr->kernel, 1, sizeof(cl_double), &alpha);
  err |= clSetKernelArg(k_calc_xr->kernel, 2, sizeof(uint32_t), &k_calc_xr->items_per_work_item);
  err |= clSetKernelArg(k_calc_xr->kernel, 3, sizeof(uint32_t), &items_per_work_group);
  err |= clSetKernelArg(k_calc_xr->kernel, 4, sizeof(cl_double)*k_calc_xr->group_size, NULL);
  err |= clSetKernelArg(k_calc_xr->kernel, 5, sizeof(cl_mem), &p->data);
  err |= clSetKernelArg(k_calc_xr->kernel, 6, sizeof(cl_mem), &w->data);
  err |= clSetKernelArg(k_calc_xr->kernel, 7, sizeof(cl_mem), &x->data);
  err |= clSetKernelArg(k_calc_xr->kernel, 8, sizeof(cl_mem), &r->data);
  err |= clSetKernelArg(k_calc_xr->kernel, 9, sizeof(cl_mem), &d_calc_xr_partial);

  clFinish(ocl_queue);

  err |= clEnqueueNDRangeKernel(ocl_queue, k_calc_xr->kernel, 1, NULL, &k_calc_xr->global_size, &k_calc_xr->group_size, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d enquing kernel calc_xr", err);
  return OCLContext::sum_vector(d_calc_xr_partial, k_calc_xr->ngroups);
}

void OCLContext::calc_p(cg_vector *p, const cg_vector *r, double beta)
{
  cl_int err;

  if(k_calc_p->first_run){
    OCLUtils::setup_opencl_kernel(k_calc_p, CALC_P_KERNEL_ITEMS, CALC_P_KERNEL_WG, p->N);
  }

  uint32_t items_per_work_group = k_calc_p->items_per_work_item * k_calc_p->group_size;
  err  = clSetKernelArg(k_calc_p->kernel, 0, sizeof(uint32_t), &p->N);
  err |= clSetKernelArg(k_calc_p->kernel, 1, sizeof(cl_double), &beta);
  err |= clSetKernelArg(k_calc_p->kernel, 2, sizeof(uint32_t), &k_calc_p->items_per_work_item);
  err |= clSetKernelArg(k_calc_p->kernel, 3, sizeof(uint32_t), &items_per_work_group);
  err |= clSetKernelArg(k_calc_p->kernel, 4, sizeof(cl_mem), &r->data);
  err |= clSetKernelArg(k_calc_p->kernel, 5, sizeof(cl_mem), &p->data);

  clFinish(ocl_queue);

  err |= clEnqueueNDRangeKernel(ocl_queue, k_calc_p->kernel, 1, NULL, &k_calc_p->global_size, &k_calc_p->group_size, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d enquing kernel calc_p", err);
  
}

void OCLContext::spmv(const cg_matrix *mat, const cg_vector *vec,
                      cg_vector *result)
{
  cl_int err;

  if(k_spmv->first_run){
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
#endif
    OCLUtils::setup_opencl_kernel(k_spmv, SPMV_KERNEL_ITEMS, SPMV_KERNEL_WG, mat->N);
  }
  uint32_t lastKernelArg = 5;
  err  = clSetKernelArg(k_spmv->kernel, 0, sizeof(uint32_t), &mat->N);
  err |= clSetKernelArg(k_spmv->kernel, 1, sizeof(cl_mem), &mat->rows);
  err |= clSetKernelArg(k_spmv->kernel, 2, sizeof(cl_mem), &mat->cols);
  err |= clSetKernelArg(k_spmv->kernel, 3, sizeof(cl_mem), &mat->values);
  err |= clSetKernelArg(k_spmv->kernel, 4, sizeof(cl_mem), &vec->data);
  err |= clSetKernelArg(k_spmv->kernel, 5, sizeof(cl_mem), &result->data);
#if SPMV_METHOD == SPMV_VECTOR
  err |= clSetKernelArg(k_spmv->kernel, 6, sizeof(cl_double) * (_SPMV_VECTORS_PER_BLOCK * _SPMV_THREADS_PER_VECTOR + _SPMV_THREADS_PER_VECTOR / 2), NULL);
  err |= clSetKernelArg(k_spmv->kernel, 7, sizeof(cl_uint) * (_SPMV_VECTORS_PER_BLOCK * 2), NULL);
  err |= clSetKernelArg(k_spmv->kernel, 8, sizeof(cl_uint), (void*)&_SPMV_VECTORS_PER_BLOCK);
  err |= clSetKernelArg(k_spmv->kernel, 9, sizeof(cl_uint), (void*)&_SPMV_THREADS_PER_VECTOR);
  lastKernelArg = 9;
#endif
  if(ftType == CONSTRAINTS){
    err |= clSetKernelArg(k_spmv->kernel, lastKernelArg+1, sizeof(uint32_t), &mat->nnz);
  }

  clFinish(ocl_queue);

  err |= clEnqueueNDRangeKernel(ocl_queue, k_spmv->kernel, 1, NULL, &k_spmv->global_size, &k_spmv->group_size, 0, NULL, NULL);
  if (CL_SUCCESS != err) DIE("OpenCL error %d enquing kernel spmv", err);
  
}

void OCLContext::inject_bitflip(cg_matrix *mat, BitFlipKind kind, int num_flips)
{
  cl_int err;

  uint32_t index = rand() % mat->nnz;

  uint32_t start = 0;
  uint32_t end   = 96;

  const size_t one = 1;

  if (kind == VALUE)
    end = 64;
  else if (kind == INDEX)
    start = 64;

  for (int i = 0; i < num_flips; i++)
  {
    uint32_t bit = (rand() % (end-start)) + start;
    if (bit < 64)
    {
      err  = clSetKernelArg(k_inject_bitflip_val->kernel, 0, sizeof(uint32_t), &bit);
      err |= clSetKernelArg(k_inject_bitflip_val->kernel, 1, sizeof(uint32_t), &index);
      err |= clSetKernelArg(k_inject_bitflip_val->kernel, 2, sizeof(cl_mem), &mat->values);
      if (CL_SUCCESS != err) DIE("OpenCL error %d enquing kernel; inject_bitflip_val", err);

      clFinish(ocl_queue);

      err |= clEnqueueNDRangeKernel(ocl_queue, k_inject_bitflip_val->kernel, 1, NULL, &one, &one, 0, NULL, NULL);
      if (CL_SUCCESS != err) DIE("OpenCL error %d enquing kernel inject_bitflip_val", err);
    }
    else
    {
      bit = bit - 64;
      err  = clSetKernelArg(k_inject_bitflip_col->kernel, 0, sizeof(uint32_t), &bit);
      err |= clSetKernelArg(k_inject_bitflip_col->kernel, 1, sizeof(uint32_t), &index);
      err |= clSetKernelArg(k_inject_bitflip_col->kernel, 2, sizeof(cl_mem), &mat->cols);
      if (CL_SUCCESS != err) DIE("OpenCL error %d enquing kernel; inject_bitflip_col", err);
      
      clFinish(ocl_queue);

      err |= clEnqueueNDRangeKernel(ocl_queue, k_inject_bitflip_col->kernel, 1, NULL, &one, &one, 0, NULL, NULL);
      if (CL_SUCCESS != err) DIE("OpenCL error %d enquing kernel inject_bitflip_col", err);
    }
  }
  clFinish(ocl_queue);
}

void OCLContext_SED::generate_ecc_bits(csr_element& element)
{
  element.column |= ecc_compute_overall_parity(element) << 31;
}

void OCLContext_SEC7::generate_ecc_bits(csr_element& element)
{
  element.column |= ecc_compute_col8(element);
}

void OCLContext_SEC8::generate_ecc_bits(csr_element& element)
{
  element.column |= ecc_compute_col8(element);
  element.column |= ecc_compute_overall_parity(element) << 24;
}

void OCLContext_SECDED::generate_ecc_bits(csr_element& element)
{
  element.column |= ecc_compute_col8(element);
  element.column |= ecc_compute_overall_parity(element) << 24;
}

namespace
{
  static CGContext::Register<OCLContext> A("ocl", "none");
  static CGContext::Register<OCLContext_Constraints> B("ocl", "constraints");
  static CGContext::Register<OCLContext_SED> C("ocl", "sed");
  static CGContext::Register<OCLContext_SEC7> D("ocl", "sec7");
  static CGContext::Register<OCLContext_SEC8> E("ocl", "sec8");
  static CGContext::Register<OCLContext_SECDED> F("ocl", "secded");
}
