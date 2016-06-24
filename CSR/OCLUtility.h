#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <stdarg.h>

#include <cmath>


struct ocl_kernel
{
	uint8_t first_run;
  size_t ngroups;
  size_t group_size;
  size_t global_size;
  uint32_t items_per_work_item;
  cl_kernel kernel;
};


#define DIE(...) exit_with_error(__LINE__, __FILE__, __VA_ARGS__)
void exit_with_error(int line, const char* filename, const char* format, ...)
__attribute__ ((format (printf, 3, 4)));

namespace OCLUtils
{
	cl_device_id get_opencl_device(cl_uint device_id);
	cl_context get_opencl_context(cl_device_id device);
	cl_command_queue get_opencl_command_queue(cl_context context, cl_device_id device);
	cl_program get_opencl_program_from_file(cl_context context, const std::string kernel_file_name);
	uint8_t build_opencl_program(cl_program program, const cl_device_id device, const cl_context context, const std::string build_flags);
	ocl_kernel * get_opencl_kernel(cl_program program, const std::string kernel_name);
	void setup_opencl_kernel(ocl_kernel* kernel, const size_t items_per_work_item, const size_t group_size, const size_t total_work);
}