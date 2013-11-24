__kernel void ghostRows(const int dim,
                        __global *grid)
{
    // We want id to range from 1 to dim
    int id = get_global_id(0) + 1;
 
    if (id <= dim)
    {
        grid[(dim+2)*(dim+1)+id] = grid[(dim+2)+id]; //Copy first real row to bottom ghost row
        grid[id] = grid[(dim+2)*dim + id]; //Copy last real row to top ghost row
    }
}
 
__kernel void ghostCols(const int dim,
                        __global *grid)
{
    // We want id to range from 0 to dim+1
    int id = get_global_id(0);
 
    if (id <= dim+1)
    {
        grid[id*(dim+2)+dim+1] = grid[id*(dim+2)+1]; //Copy first real column to right most ghost column
        grid[id*(dim+2)] = grid[id*(dim+2) + dim]; //Copy last real column to left most ghost column
    }
}
 
__kernel void GOL(const int dim,
                  __global int *grid,
                  __global int *newGrid)
{
    int ix = (get_local_size(0)-2) * get_group_id(0) + get_local_id(0);
    int iy = (get_local_size(1)-2) * get_group_id(1) + get_local_id(1);
    int id = iy * (dim+2) + ix;
 
    int i = get_local_id(0);
    int j = get_local_id(1);
    int numNeighbors;
 
    // Declare the local memory on a per work group level
    __local int s_grid[BLOCK_SIZE_y][BLOCK_SIZE_x];
 
    // Copy cells into local memory
    if (ix <= dim+1 && iy <= dim+1)
        s_grid[i][j] = grid[id];
 
    //Sync all work items in work group
    barrier(CLK_LOCAL_MEM_FENCE);
 
    if (iy <= dim && ix <= dim) {
        if(i != 0 && i !=blockDim.y-1 && j != 0 && j !=blockDim.x-1) {
 
            // Get the number of neighbors for a given grid point
            numNeighbors = s_grid[i+1][j] + s_grid[i-1][j] //upper lower
                         + s_grid[i][j+1] + s_grid[i][j-1] //right left
                         + s_grid[i+1][j+1] + s_grid[i-1][j-1] //diagonals
                         + s_grid[i-1][j+1] + s_grid[i+1][j-1];
 
             int cell = s_grid[i][j];
 
             // Here we have explicitly all of the game rules
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
}
