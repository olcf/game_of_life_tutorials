#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
#include <sys/stat.h>
 
#define SRAND_VALUE 1985
#define LOCAL_SIZE 16
 
int main(int argc, char* argv[])
{
    int i,j,iter;
    int *h_grid;
    cl_mem d_grid;
    cl_mem d_newGrid;
 
    // Linear game grid dimension
    int dim = 1024;
    // Number of game iterations
    int maxIter = 1<<10;
 
    // Size, in bytes, of each vector
    size_t bytes = sizeof(int)*(dim+2)*(dim+2);
 
    // Allocate host Grid used for initial setup and read back from device
    h_grid = (int*)malloc(bytes);
 
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    //Kernels
    cl_kernel k_gol, k_ghostRows, k_ghostCols;
 
    // Assign initial population randomly
    srand(SRAND_VALUE);
    for(i = 1; i<=dim; i++) {
        for(j = 1; j<=dim; j++) {
            h_grid[i*(dim+2)+j] = rand() % 2;
        }
    }
 
   cl_int err;
 
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    if (err != CL_SUCCESS) {
      printf( "Error: Failed to find a platform\n");
      return EXIT_FAILURE;
    }
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    if (err != CL_SUCCESS) {
        printf("Error: Failed to create a device group\n");
        return EXIT_FAILURE;
    }
 
    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    if (!context) {
      printf("Error: Failed to create a compute context\n");
      return EXIT_FAILURE;
    }
 
    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    if (!queue) {
      printf("Error: Failed to create a command commands\n");
      return EXIT_FAILURE;
    }
 
    // Create the compute program from the kernel source file
    char *fileName = "GOL-kernels.cl";
    FILE *fh = fopen(fileName, "r");
    if(!fh) {
        printf("Error: Failed to open file\n");
        return 0;
    }
    struct stat statbuf;
    stat(fileName, &statbuf);
    char *kernelSource = (char *) malloc(statbuf.st_size + 1);
    fread(kernelSource, statbuf.st_size, 1, fh);
    kernelSource[statbuf.st_size] = '\0';
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
    if (!program) {
      printf("Error: Failed to create compute program\n");
      return EXIT_FAILURE;
    }
 
    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
      printf("Error: Failed to build program executable %d\n", err);
      return EXIT_FAILURE;
    }
 
    // Create the GOL kernel in the program we wish to run
    k_gol = clCreateKernel(program, "GOL", &err);
    if (!k_gol || err != CL_SUCCESS) {
      printf("Error: Failed to create GOL kernel\n");
      return EXIT_FAILURE;
    }
 
    // Create the ghostRows kernel in the program we wish to run
    k_ghostRows = clCreateKernel(program, "ghostRows", &err);
    if (!k_ghostRows || err != CL_SUCCESS) {
      printf("Error: Failed to create ghostRows kernel\n");
      return EXIT_FAILURE;
    }
 
    // Create the ghostCols kernel in the program we wish to run
    k_ghostCols = clCreateKernel(program, "ghostCols", &err);
    if (!k_ghostCols || err != CL_SUCCESS) {
      printf("Error: Failed to create ghostCols kernel\n");
      return EXIT_FAILURE;
    }
 
    // Create the input and output arrays in device memory for our calculation
    d_grid = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
    d_newGrid = clCreateBuffer(context, CL_MEM_READ_WRITE, bytes, NULL, NULL);
    if (!d_grid || !d_newGrid) {
      printf("Error: Failed to allocate device memory\n");
      return EXIT_FAILURE;
    }
 
    // Write our data set into the input array in device memory
    err = clEnqueueWriteBuffer(queue, d_grid, CL_TRUE, 0,
                                   bytes, h_grid, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
      printf("Error: Failed to write to source array\n");
      return EXIT_FAILURE;
    }
 
    // Set the arguments to GOL  kernel
    err  = clSetKernelArg(k_gol, 0, sizeof(int), &dim);
    err |= clSetKernelArg(k_gol, 1, sizeof(cl_mem), &d_grid);
    err |= clSetKernelArg(k_gol, 2, sizeof(cl_mem), &d_newGrid);
    if (err != CL_SUCCESS) {
      printf("Error: Failed to set kernel arguments\n");
      return EXIT_FAILURE;
    }
 
    // Set the arguments to ghostRows kernel
    err  = clSetKernelArg(k_ghostRows, 0, sizeof(int), &dim);
    err |= clSetKernelArg(k_ghostRows, 1, sizeof(cl_mem), &d_grid);
    if (err != CL_SUCCESS) {
      printf("Error: Failed to set kernel arguments\n");
      return EXIT_FAILURE;
    }
 
    // Set the arguments to ghostCols kernel
    err  = clSetKernelArg(k_ghostCols, 0, sizeof(int), &dim);
    err |= clSetKernelArg(k_ghostCols, 1, sizeof(cl_mem), &d_grid);
    if (err != CL_SUCCESS) {
      printf("Error: Failed to set kernel arguments\n");
      return EXIT_FAILURE;
    }
 
    // Set kernel local and global sizes
    size_t cpyRowsGlobalSize, cpyColsGlobalSize, cpyLocalSize;
    cpyLocalSize = LOCAL_SIZE;
    // Number of total work items - localSize must be devisor
    cpyRowsGlobalSize = (size_t)ceil(dim/(float)cpyLocalSize)*cpyLocalSize;
    cpyColsGlobalSize = (size_t)ceil((dim+2)/(float)cpyLocalSize)*cpyLocalSize;
 
    size_t GolLocalSize[2] = {LOCAL_SIZE, LOCAL_SIZE};
    size_t linGlobal = (size_t)ceil(ceil(dim/(float)(LOCAL_SIZE-2))*(LOCAL_SIZE-2)/LOCAL_SIZE)*LOCAL_SIZE;
    size_t GolGlobalSize[2] = {linGlobal, linGlobal};
 
    // Main game loop
    for (iter = 0; iter<maxIter; iter++) {
        err = clEnqueueNDRangeKernel(queue, k_ghostRows, 1, NULL, &cpyRowsGlobalSize, &cpyLocalSize,
                                                              0, NULL, NULL);
        err |= clEnqueueNDRangeKernel(queue, k_ghostCols, 1, NULL, &cpyColsGlobalSize, &cpyLocalSize,
                                                              0, NULL, NULL);
        err |= clEnqueueNDRangeKernel(queue, k_gol, 2, NULL, GolGlobalSize, GolLocalSize,
                                                              0, NULL, NULL);
 
        if(iter%2 == 1) {
            err |= clSetKernelArg(k_ghostRows, 1, sizeof(cl_mem), &d_grid);
            err |= clSetKernelArg(k_ghostCols, 1, sizeof(cl_mem), &d_grid);
            err |= clSetKernelArg(k_gol, 1, sizeof(cl_mem), &d_grid);
            err |= clSetKernelArg(k_gol, 2, sizeof(cl_mem), &d_newGrid);
        } else {
            err |= clSetKernelArg(k_ghostRows, 1, sizeof(cl_mem), &d_newGrid);
            err |= clSetKernelArg(k_ghostCols, 1, sizeof(cl_mem), &d_newGrid);
            err |= clSetKernelArg(k_gol, 1, sizeof(cl_mem), &d_newGrid);
            err |= clSetKernelArg(k_gol, 2, sizeof(cl_mem), &d_grid);
        }
    }// End main game loop
 
    if (err != CL_SUCCESS) {
       printf("Error: Failed to launch kernels%d\n",err);
       return EXIT_FAILURE;
    }
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    err =  clEnqueueReadBuffer(queue, d_grid, CL_TRUE, 0,
                        bytes, h_grid, 0, NULL, NULL );
    if (err != CL_SUCCESS) {
      printf("Error: Failed to read output array\n");
      return EXIT_FAILURE;;
    }
 
    // Sum up alive cells and print results
    int total = 0;
    for (i = 1; i<=dim; i++) {
        for (j = 1; j<=dim; j++) {
            total += h_grid[i*(dim+2)+j];
        }
    }
    printf("Total Alive: %d\n", total);
 
    // Release memory
    free(h_grid);
 
    return 0;
}
