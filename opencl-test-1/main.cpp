//
//  main.cpp
//  opencl-test-1
//
//  Created by taisuke on 2015/11/07.
//  license: public domain
//  reference: http://www.amazon.co.jp/dp/4798026085


#include <stdio.h>

#include <OpenCL/opencl.h>

#include <mach/mach_time.h> // for stopwatch


static void checkError(cl_int status);

static void printPlatformInfo(const cl_platform_id platform_id);
static int printPlatforms();
static int firstOpenCL();

#define MAX_DEVICES 10
#define NUM_ELEMENTS 10000

static float in1[NUM_ELEMENTS];
static float in2[NUM_ELEMENTS];
static float out[NUM_ELEMENTS];

int min(int n, int m) {
    return n < m ? n : m;
}
int main(int argc, const char * argv[]) {
    printPlatforms();
    firstOpenCL();
    return 0;
}

static void checkError(cl_int status) {
    if (status != CL_SUCCESS) {
        fprintf(stderr, "OpenCL err: %d\n", status);
        exit(0);
    }
}
static int printPlatforms() {
    cl_platform_id platforms[10];
    cl_uint num_platforms;
    cl_int status;
    
    status = clGetPlatformIDs(sizeof(platforms) / sizeof(*platforms), platforms, &num_platforms);
    if (status != CL_SUCCESS) {
        fprintf(stderr, "failed %d\n", status);
        return 1;
    }
    printf("num of platforms: %d\n", num_platforms);
    for (int i = 0; i < num_platforms; i++) {
        printPlatformInfo(platforms[i]);
    }
    return 0;
}

static void printPlatformInfo(const cl_platform_id platform_id) {
    char buffer[1024];
    cl_int status;
    
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_PROFILE, sizeof(buffer) - 1, buffer, NULL);
    printf("%s\n", buffer);
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(buffer) - 1, buffer, NULL);
    printf("%s\n", buffer);
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(buffer) - 1, buffer, NULL);
    printf("%s\n", buffer);
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, sizeof(buffer) - 1, buffer, NULL);
    printf("%s\n", buffer);
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_EXTENSIONS, sizeof(buffer) - 1, buffer, NULL);
    printf("%s\n", buffer);
    
    cl_device_id devices[MAX_DEVICES];
    cl_uint ndevices;
    status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &ndevices);
    checkError(status);
    printf("OpenCL ndevices: %zu\n", ndevices); // MacBookAir == 2
    for (int i = 0; i < ndevices; i++) {
        size_t size;
        cl_device_type type;
        status = clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof(cl_device_type), &type, &size);
        printf("device %d\ntype: %zu (CPU:%d, GPU:%d)\n", i, type, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU);
        
        cl_uint ncu;
        status = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &ncu, &size);
        printf("num of compute units: %d\n", ncu); // MacBookAir CPU:4 GPU:40
    }
}



// first OpenCL


static int firstOpenCL() {
    cl_int status;
    
    cl_context context = clCreateContextFromType(NULL, CL_DEVICE_TYPE_GPU, NULL, NULL, &status);
    
    cl_device_id devices[MAX_DEVICES];
    size_t ndevices;
    status = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(devices), devices, &ndevices);
    checkError(status);
    printf("OpenCL ndevices: %zu\n", ndevices); // MacBookAir == 8
    if (ndevices == 0)
        return 0;
    
    cl_command_queue queue = clCreateCommandQueue(context, devices[0], 0, &status);
    
    static const char* src[] = {
        "__kernel void addVector(\
        __global const float *in1,\n\
        __global const float *in2,\n\
        __global float *out) {\n\
        int index = get_global_id(0);\n\
        out[index] = in1[index] + in2[index];\n\
        }\n"
    };
    
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&src, NULL, &status);
    
    status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);
    checkError(status);
    //    clUnloadCompiler();
    
    cl_kernel kernel = clCreateKernel(program, "addVector", &status);
    
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        in1[i] = (float)i * 100.0f;
        in2[i] = (float)i / 100.0f;
        out[i] = 0.0f;
    }
    
    cl_mem memIn1 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * NUM_ELEMENTS, in1, &status);
    cl_mem memIn2 = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * NUM_ELEMENTS, in2, &status);
    cl_mem memOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(cl_float) * NUM_ELEMENTS, out, &status);
    
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&memIn1);
    checkError(status);
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&memIn2);
    checkError(status);
    status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&memOut);
    checkError(status);
    
    size_t globalSize[] = { NUM_ELEMENTS };
    status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, globalSize, 0, 0, NULL, NULL);
    checkError(status);
    
    uint64_t t = mach_absolute_time();
    
    //    usleep(1000000); // = 1sec, wait microseconds -> 1,003,741,061nsec
#define USE_OPENCL 0
#if USE_OPENCL
    status = clEnqueueReadBuffer(queue, memOut, CL_TRUE, 0, sizeof(cl_float) * NUM_ELEMENTS, out, 0, NULL, NULL);
    checkError(status);
#else
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        out[i] = in1[i] + in2[i];
    }
#endif
    uint64_t dt0 = mach_absolute_time() - t;
    mach_timebase_info_data_t machinfo;
    mach_timebase_info(&machinfo);
    uint64_t dt = dt0  * machinfo.numer / machinfo.denom; // dt0 * 1 / 1  // https://developer.apple.com/library/mac/qa/qa1398/_index.html
    printf("time %dnano sec %d %d\n", (int)*(uint64_t*)&dt, (int)machinfo.numer, (int)machinfo.denom);
    
    // check
    printf("in1, in2, out\n");
    for (int i = 0; i < min(NUM_ELEMENTS, 5); i++) {
        printf("%f %f %f (%f)\n", in1[i], in2[i], out[i], in1[i] + in2[i]);
    }
    
    clReleaseMemObject(memOut);
    clReleaseMemObject(memIn1);
    clReleaseMemObject(memIn2);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    return 0;
}
