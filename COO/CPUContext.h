#include "CGContext.h"

#include "ecc.h"

struct cg_vector
{
  int N;
  double *data;
};

struct cg_matrix
{
  unsigned N;
  unsigned nnz;
  coo_element *elements;
};

class CPUContext : public CGContext
{
  virtual void generate_ecc_bits(coo_element& element);
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
};

class CPUContext_Constraints : public CPUContext
{
  virtual void spmv(const cg_matrix *mat, const cg_vector *vec,
                    cg_vector *result);
};

class CPUContext_SED : public CPUContext
{
  virtual void generate_ecc_bits(coo_element& element);
  virtual void spmv(const cg_matrix *mat, const cg_vector *vec,
                    cg_vector *result);
};

class CPUContext_SEC7 : public CPUContext
{
  virtual void generate_ecc_bits(coo_element& element);
  virtual void spmv(const cg_matrix *mat, const cg_vector *vec,
                    cg_vector *result);
};

class CPUContext_SEC8 : public CPUContext
{
  virtual void generate_ecc_bits(coo_element& element);
  virtual void spmv(const cg_matrix *mat, const cg_vector *vec,
                    cg_vector *result);
};

class CPUContext_SECDED : public CPUContext
{
  virtual void generate_ecc_bits(coo_element& element);
  virtual void spmv(const cg_matrix *mat, const cg_vector *vec,
                    cg_vector *result);
};
