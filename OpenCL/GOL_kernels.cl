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
    int ix = get_global_id(0) + 1;
    int iy = get_global_id(1) + 1;
    int id = iy * (dim+2) + ix;
 
    int numNeighbors;
 
    if (iy <= dim && ix <= dim) {
 
    // Get the number of neighbors for a given grid point
    numNeighbors = grid[id+(dim+2)] + grid[id-(dim+2)] //upper lower
                 + grid[id+1] + grid[id-1]             //right left
                 + grid[id+(dim+3)] + grid[id-(dim+3)] //diagonals
                 + grid[id-(dim+1)] + grid[id+(dim+1)];
 
    int cell = grid[id];
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
