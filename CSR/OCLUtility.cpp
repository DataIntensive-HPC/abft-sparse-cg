#include "OCLUtility.h"

void exit_with_error(int line, const char* filename, const char* format, ...)
{
    va_list arglist;

    fprintf(stderr, "Fatal error at line %d in %s: ", line, filename);

    va_start(arglist, format);
    vfprintf(stderr, format, arglist);
    va_end(arglist);

    fprintf(stderr, "\n");

    exit(EXIT_FAILURE);
}

namespace OCLUtils
{

  void get_opencl_platforms(cl_platform_id ** platforms, cl_uint * num_platforms)
  {
    cl_int err;

    err = clGetPlatformIDs(0, NULL, num_platforms);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting number of platforms", err);

    *platforms = (cl_platform_id *) calloc(*num_platforms, sizeof(cl_platform_id));
    err = clGetPlatformIDs(*num_platforms, *platforms, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting platforms", err);
  }

  void get_platform_devices(cl_platform_id platform, cl_device_id ** devices, cl_uint * num_devices)
  {
    cl_int err;

    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, num_devices);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting return size of platform parameter", err);

    *devices = (cl_device_id *) calloc(*num_devices, sizeof(cl_device_id));
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, *num_devices, *devices, NULL);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting platform info", err);
  }



  char * get_device_info(cl_device_info param_name, cl_device_id device)
  {
    cl_int err;

    char * return_string = NULL;
    size_t return_size;

    err = clGetDeviceInfo(device, param_name, 0, NULL, &return_size);
    if (CL_SUCCESS != err) DIE("OpenCL error %d getting return size of device parameter", err);

    switch (param_name)
    {
    case CL_DEVICE_TYPE:
        return_string = (char *) calloc(50, sizeof(char));
        cl_device_type device_type;
        err = clGetDeviceInfo(device, param_name, return_size, &device_type, NULL);

        switch (device_type)
        {
        case CL_DEVICE_TYPE_GPU:
            strcat(return_string, "GPU");
            break;
        case CL_DEVICE_TYPE_CPU:
            strcat(return_string, "CPU");
            break;
        case CL_DEVICE_TYPE_ACCELERATOR:
            strcat(return_string, "ACCELERATOR");
            break;
        default:
            strcat(return_string, "DEFAULT");
            break;
        }
        break;
    case CL_DEVICE_NAME:
        return_string = (char *) calloc(return_size, sizeof(char));
        err = clGetDeviceInfo(device, param_name, return_size, return_string, NULL);
        break;
    default:
        DIE("Other device_info types not implemented\n");
    }

    if (CL_SUCCESS != err) DIE("OpenCL error %d getting device parameter", err);

    return return_string;
  }

  void print_device_info(cl_device_id device, cl_uint device_id)
  {
    char * device_name = get_device_info(CL_DEVICE_NAME, device);
    char * device_type = get_device_info(CL_DEVICE_TYPE, device);

    fprintf(stdout, " Device %u: %s (%s)\n", device_id, device_name, device_type);

    free(device_name);
    free(device_type);
  }



  cl_device_id get_opencl_device(cl_uint device_id)
  {
    cl_platform_id * platforms = NULL;
    cl_uint num_platforms;
    get_opencl_platforms(&platforms, &num_platforms);

    cl_device_id * devices = NULL;
    cl_uint total_devices = 0;

    for (cl_uint i = 0; i < num_platforms; i++)
    {
      cl_uint num_platform_devices = 0;
      cl_device_id * platform_devices = NULL;

      get_platform_devices(platforms[i], &platform_devices, &num_platform_devices);

      devices = (cl_device_id *) realloc(devices, sizeof(cl_device_id)*(total_devices + num_platform_devices));
      memcpy(&devices[total_devices], platform_devices, num_platform_devices*sizeof(cl_device_id));

      total_devices += num_platform_devices;

      free(platform_devices);
    }
    if(device_id >= total_devices)
    {
        DIE("Asked for device %d but there were only %u available!\n", device_id, total_devices);
    }
    cl_device_id target_device = devices[device_id];
    // fprintf(stdout, "Got %u OpenCL devices\n", total_devices);
    // for(cl_uint i = 0; i < total_devices; i++){
    //   print_device_info(devices[i], i);
    // }
    printf("Using device:\n");
    print_device_info(devices[device_id], device_id);
    free(devices);
    free(platforms);
    return target_device;
  }

  /* Define a printf callback function for arm. */
  void printf_callback( const char *buffer, size_t len, size_t complete, void *user_data )
  {
      printf( "%.*s", len, buffer );
  }

  cl_context get_opencl_context(cl_device_id device)
  {
    cl_int err;
#ifdef PRINTF_ARM_KERNEL
    //cl_context with a printf_callback and user specified buffer size for arm
    cl_platform_id platform;
    clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform, NULL);
    cl_context_properties properties[] =
    {
        CL_PRINTF_CALLBACK_ARM,   (cl_context_properties) printf_callback,
        CL_PRINTF_BUFFERSIZE_ARM, (cl_context_properties) 0x100000,
        CL_CONTEXT_PLATFORM,      (cl_context_properties) platform,
        0
    };
    cl_context context = clCreateContext(properties, 1, &device, NULL, NULL, &err);
#else
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
#endif
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating context", err);
    return context;
  }

  cl_command_queue get_opencl_command_queue(cl_context context, cl_device_id device)
  {
    cl_int err;
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating command queue", err);
    return command_queue;
  }


  char * get_source(const char * file_name){
    FILE * source_fp;
    source_fp = fopen(file_name, "r");

    if (NULL == source_fp)
    {
      printf("Unable to open kernel file %s", file_name);
      exit(EXIT_FAILURE);
    }

    size_t source_size;
    fseek(source_fp, 0, SEEK_END);
    source_size = ftell(source_fp);
    char * source = (char *) calloc(source_size + 1, sizeof(char));
    fseek(source_fp, 0, SEEK_SET);
    size_t bytes_read = fread(source, 1, source_size, source_fp);

    if (bytes_read != source_size)
    {
      printf("Expected to read %lu bytes from kernel file, actually read %lu bytes", source_size, bytes_read);
      exit(EXIT_FAILURE);
    }
    fclose(source_fp);
    source[source_size] = '\0';
    return source;
  }

  cl_program get_opencl_program_from_file(cl_context context, const std::vector<std::string> source_file_names)
  {
    cl_int err;
    char ** sources = new char*[source_file_names.size()];
    for(uint32_t i = 0; i < source_file_names.size(); i++){
      sources[i] = get_source(source_file_names[i].c_str());
    }
    cl_program program = clCreateProgramWithSource(context, source_file_names.size(), (const char**)sources, NULL, &err);

    for(uint32_t i = 0; i < source_file_names.size(); i++){
      free(sources[i]);
    }
    free(sources);
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating program", err);

    return program;
  }

  uint8_t build_opencl_program(cl_program program, const cl_device_id device, const cl_context context, const std::string build_flags)
  {
    cl_int err;
    err = clBuildProgram(program, 1, &device, build_flags.c_str(), NULL, NULL);
    if (err != CL_SUCCESS)
    {
      cl_int build_err;
      size_t log_size;

      build_err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
      if (CL_SUCCESS != build_err) DIE("OpenCL error %d getting size of build log", build_err);

      char * build_log = (char *) calloc(log_size + 1, sizeof(char));
      build_err = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, build_log, NULL);
      if (CL_SUCCESS != build_err) DIE("OpenCL error %d getting build log", build_err);

      printf("OpenCL program build log:\n%s\n", build_log);
      free(build_log);
      return 0;
    }
    return 1;
  }

  ocl_kernel * get_opencl_kernel(cl_program program, const std::string kernel_name)
  {
    cl_int err;
    ocl_kernel * result = new ocl_kernel;
    result->kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (CL_SUCCESS != err) DIE("OpenCL error %d creating %s kernel", err, kernel_name.c_str());
    result->first_run = 1;
    return result;
  }

  void setup_opencl_kernel(ocl_kernel* kernel, const size_t items_per_work_item, const size_t group_size, const size_t total_work)
  {
    kernel->group_size = group_size;
    kernel->items_per_work_item = items_per_work_item;
    kernel->ngroups = ceil((float)total_work / (float)(group_size * items_per_work_item));
    kernel->global_size = group_size * kernel->ngroups;
    kernel->first_run = 0;
  }

}