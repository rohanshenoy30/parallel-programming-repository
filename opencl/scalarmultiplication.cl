#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define N 1024

const char* kernelSource =
"__kernel void scalar_mult(__global int* A, int k) {\n"
"    int id = get_global_id(0);\n"
"    A[id] = A[id] * k;\n"
"}\n";

int main() {

    int A[N];
    int k = 5;

    for(int i=0;i<N;i++) A[i] = i;

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    cl_program program;
    cl_kernel kernel;

    cl_mem bufferA;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    bufferA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(int)*N, NULL, NULL);

    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, sizeof(int)*N, A, 0, NULL, NULL);

    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    kernel = clCreateKernel(program, "scalar_mult", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(int), &k);

    size_t globalSize = N;

    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, bufferA, CL_TRUE, 0, sizeof(int)*N, A, 0, NULL, NULL);

    for(int i=0;i<10;i++)
        printf("%d ", A[i]);

    return 0;
}