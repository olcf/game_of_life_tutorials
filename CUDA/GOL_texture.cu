#include <stdio.h>
#include <stdlib.h>
 
#define SRAND_VALUE 1985
 
texture<int,2> gridTex;
 
__global__ void GOL(int dim, int *newGrid)
{
    int iy = blockDim.y * blockIdx.y + threadIdx.y;
    int ix = blockDim.x * blockIdx.x + threadIdx.x;
    int id = iy * dim + ix;
 
    int numNeighbors;
 
    float iyTex = (iy + 0.5f)/dim;
    float ixTex = (ix + 0.5f)/dim;
    float oneTex = 1.0f/dim;
 
    if(iy < dim && ix < dim)
{
    //Get the number of neighbors for a given grid point
    numNeighbors = tex2D(gridTex, ixTex+oneTex, iyTex) //right/left
                 + tex2D(gridTex, ixTex-oneTex, iyTex)
                 + tex2D(gridTex, ixTex, iyTex+oneTex) //upper/lower
                 + tex2D(gridTex, ixTex, iyTex-oneTex)
                 + tex2D(gridTex, ixTex-oneTex, iyTex-oneTex) //diagonals
                 + tex2D(gridTex, ixTex-oneTex, iyTex+oneTex)
                 + tex2D(gridTex, ixTex+oneTex, iyTex-oneTex)
                 + tex2D(gridTex, ixTex+oneTex, iyTex+oneTex);

    int cell = tex2D(gridTex, ixTex, iyTex);

    //Here we have explicitly all of the game rules
    if (cell == 1 && numNeighbors < 2)
        newGrid[id] = 0;
    else if (cell == 1 && (numNeighbors == 2 || numNeighbors == 3))
        newGrid[id] = 1;
    else if (cell == 1 && numNeighbors > 3)
        newGrid[id] = 0;
    else if (cell == 0 && numNeighbors == 3)
         newGrid[id] = 1;
    else
       newGrid[id] = cell;
 
}
}
 
int main(int argc, char* argv[])
{
    int i,j,iter;
    int* h_grid; //Grid on host
    cudaArray* d_grid; //Grid on device
    int* d_newGrid; //Second grid used on device only
 
    int dim = 1024; //Linear dimension of our grid - not counting ghost cells
    int maxIter = 1<<10; //Number of game steps
 
    size_t bytes = sizeof(int)*dim*dim;
    //Allocate host Grid used for initial setup and read back from device
    h_grid = (int*)malloc(bytes);
 
    //Allocate device grids
    cudaMallocArray(&d_grid, &gridTex.channelDesc, dim, dim);
    cudaMalloc(&d_newGrid, bytes);
 
    //Assign initial population randomly
    srand(SRAND_VALUE);
    for(i = 0; i<dim; i++) {
        for(j = 0; j<dim; j++) {
            h_grid[i*dim+j] = rand() % 2;
        }
    }
 
    //Copy over initial game grid (Dim-1 threads)
    cudaMemcpyToArray (d_grid, 0, 0, h_grid, bytes, cudaMemcpyHostToDevice);
    cudaBindTextureToArray(gridTex, d_grid);
 
    gridTex.normalized = true;
    gridTex.addressMode[0] = cudaAddressModeWrap;
    gridTex.addressMode[1] = cudaAddressModeWrap;
 
    dim3 dimBlock(8,8);
    int linGrid = (int)ceil(dim/(float)dimBlock.x);
    dim3 dimGrid(linGrid,linGrid);
 
    //Main game loop
    for (iter = 0; iter<maxIter; iter++) {
        GOL<<<dimGrid,dimBlock>>>(dim, d_newGrid);
 
        //Swap our grids and iterate again
        cudaMemcpyToArray (d_grid, 0, 0, d_newGrid, bytes, cudaMemcpyDeviceToDevice);
    }//iter loop
 
    //Copy back results and sum
    cudaMemcpy(h_grid, d_newGrid, bytes, cudaMemcpyDeviceToHost);
 
    //Sum up alive cells and print results
    int total = 0;
    for (i = 0; i<dim; i++) {
        for (j = 0; j<dim; j++) {
            total += h_grid[i*dim+j];
        }
    }
    printf("Total Alive: %d\n", total);
 
    cudaFree(d_grid);
    cudaFree(d_newGrid);
    free(h_grid);
 
    return 0;
}
