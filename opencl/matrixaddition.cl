#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define N 4

const char* kernelSource =
"__kernel void mat_add(__global int* A, __global int* B, __global int* C) {\n"
"    int row = get_global_id(0);\n"
"    int col = get_global_id(1);\n"
"    int idx = row * 4 + col;\n"
"    C[idx] = A[idx] + B[idx];\n"
"}\n";

int main() {

    int A[N][N], B[N][N], C[N][N];

    for(int i=0;i<N;i++)
        for(int j=0;j<N;j++) {
            A[i][j] = i;
            B[i][j] = j;
        }

    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;

    cl_program program;
    cl_kernel kernel;

    cl_mem bufferA, bufferB, bufferC;

    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    queue = clCreateCommandQueue(context, device, 0, NULL);

    bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*N*N, NULL, NULL);
    bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(int)*N*N, NULL, NULL);
    bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*N*N, NULL, NULL);

    clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, sizeof(int)*N*N, A, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, sizeof(int)*N*N, B, 0, NULL, NULL);

    program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    kernel = clCreateKernel(program, "mat_add", NULL);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    size_t globalSize[2] = {N, N};

    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, sizeof(int)*N*N, C, 0, NULL, NULL);

    printf("Result:\n");
    for(int i=0;i<N;i++) {
        for(int j=0;j<N;j++)
            printf("%d ", C[i][j]);
        printf("\n");
    }

    return 0;
}