#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define N 1024

// OpenCL Kernel as string
const char* kernelSource =
"__kernel void vec_add(__global int* A, __global int* B, __global int* C) {\n"
"    int id = get_global_id(0);\n"
"    C[id] = A[id] + B[id];\n"
"}\n";

int main() {

    int A[N], B[N], C[N];

    for(int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i;
    }


    cl_mem bufferA, bufferB, bufferC;

    // STEP 1: Platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // STEP 2: Device
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // STEP 3: Context
    cl_context context;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // STEP 4: Command Queue
    cl_command_queue queue;
    queue = clCreateCommandQueue(context, device, 0, NULL);

    // STEP 5: Buffers
    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*N, NULL, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*N, NULL, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*N, NULL, NULL);

    // STEP 6: Copy data
    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, sizeof(int)*N, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, sizeof(int)*N, B, 0, NULL, NULL);

    // STEP 7: Program
    cl_program program;
    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // STEP 8: Kernel
    cl_kernel kernel;
    kernel = clCreateKernel(program, "vec_add", NULL);

    // STEP 9: Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    // STEP 10-11: Execute
    size_t globalSize = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // STEP 12: Read result
    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(int)*N, C, 0, NULL, NULL);

    // Print result
    for(int i = 0; i < 10; i++)
        printf("%d + %d = %d\n", A[i], B[i], C[i]);

    // STEP 13: Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}