#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <CL/cl.h>

#define MAX_SIZE 1024

// Kernel
const char* kernelSource =
"__kernel void reverse_string(__global char* input, __global char* output, int len) {\n"
"    int id = get_global_id(0);\n"
"    if (id < len) {\n"
"        output[id] = input[len - 1 - id];\n"
"    }\n"
"}\n";

int main() {

    char input[] = "HELLO";
    int len = strlen(input);

    char output[MAX_SIZE];

    cl_mem bufferIn, bufferOut;

    // STEP 1: Platform
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    // STEP 2: Device
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // STEP 3: Context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // STEP 4: Command Queue (with profiling)
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    // STEP 5: Buffers
    bufferIn = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(char)*len, NULL, NULL);
    bufferOut = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(char)*len, NULL, NULL);

    // STEP 6: Copy input
    clEnqueueWriteBuffer(queue, bufferIn, CL_TRUE, 0, sizeof(char)*len, input, 0, NULL, NULL);

    // STEP 7: Program
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    // STEP 8: Kernel
    cl_kernel kernel = clCreateKernel(program, "reverse_string", NULL);

    // STEP 9: Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferIn);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferOut);
    clSetKernelArg(kernel, 2, sizeof(int), &len);

    // STEP 10: Execute
    size_t globalSize = len;

    cl_event event;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, &event);

    clWaitForEvents(1, &event);

    // STEP 11: Measure execution time
    cl_ulong start, end;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);

    double time_ms = (end - start) * 1e-6;

    // STEP 12: Read output
    clEnqueueReadBuffer(queue, bufferOut, CL_TRUE, 0, sizeof(char)*len, output, 0, NULL, NULL);

    output[len] = '\0';

    // Output
    printf("Input String  : %s\n", input);
    printf("Reversed String: %s\n", output);
    printf("Kernel Execution Time: %lf ms\n", time_ms);

    // STEP 13: Cleanup
    clReleaseMemObject(bufferIn);
    clReleaseMemObject(bufferOut);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}