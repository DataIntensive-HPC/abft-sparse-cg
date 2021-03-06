#include <cstdint>
#include <list>

// Opaque types
struct cg_matrix;
struct cg_vector;

class CGContext
{
public:
  enum BitFlipKind {ANY, VALUE, INDEX};

  virtual ~CGContext(){};

  virtual cg_matrix* create_matrix(const uint32_t *columns,
                                   const uint32_t *rows,
                                   const double *values,
                                   int N, int nnz) = 0;
  virtual void       destroy_matrix(cg_matrix *mat) = 0;

  virtual cg_vector* create_vector(int N) = 0;
  virtual void       destroy_vector(cg_vector *vec) = 0;
  virtual double*    map_vector(cg_vector *v) = 0;
  virtual void       unmap_vector(cg_vector *v, double *h) = 0;
  virtual void       copy_vector(cg_vector *dst, const cg_vector *src) = 0;

  virtual double     dot(const cg_vector *a, const cg_vector *b) = 0;
  virtual double     calc_xr(cg_vector *x, cg_vector *r,
                             const cg_vector *p, const cg_vector *w,
                             double alpha) = 0;
  virtual void       calc_p(cg_vector *p, const cg_vector *r, double beta) = 0;
  virtual void       spmv(const cg_matrix *mat, const cg_vector *vec,
                          cg_vector *result) = 0;

  virtual void       inject_bitflip(cg_matrix *mat,
                                    BitFlipKind kind, int num_flips) = 0;

  static CGContext* create(const char *impl, const char *mode);
  static void       list_contexts();

private:
  template<class ContextClass>
  static CGContext *call_context_constructor() { return new ContextClass(); }

  struct Entry
  {
    const char *target;
    const char *mode;
    CGContext* (*constructor)();
  };

  static std::list<Entry> context_list;

public:
  template<class T>
  class Register
  {
  public:
    Register(const char *target, const char *mode)
    {
      context_list.push_back({target, mode, call_context_constructor<T>});
    }
  };

protected:
  CGContext(){};
};
