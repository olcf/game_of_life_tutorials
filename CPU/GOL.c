#include <stdio.h>
#include <stdlib.h>
 
#define SRAND_VALUE 1985
 
// Add up all neighbors
int getNeighbors(int** grid, int i, int j)
{
    int numNeighbors;
    numNeighbors = grid[i+1][j] + grid[i-1][j]     //upper lower
                 + grid[i][j+1] + grid[i][j-1]     //right left
                 + grid[i+1][j+1] + grid[i-1][j-1] //diagonals
                 + grid[i-1][j+1] + grid[i+1][j-1];
 
    return numNeighbors;
}
 
int main(int argc, char* argv[])
{
    int i,j,iter;
    // Linear game grid dimension
    int dim = 1024;
    // Number of game iterations
    int maxIter = 1<<10;
 
    // Allocate square grid of (dim+2)^2 elements, 2 added for ghost cells
    int **grid = (int**) malloc( sizeof(int*) * (dim+2) );
    for(i = 0; i<dim+2; i++)
        grid[i] = (int*) malloc( sizeof(int*) * (dim+2) );
 
    // Allocate newGrid
    int **newGrid = (int**) malloc( sizeof(int*) * (dim+2) );
    for(i = 0; i<dim+2; i++)
        newGrid[i] = (int*) malloc( sizeof(int*) * (dim+2) );
 
    // Assign initial population randomly
    srand(SRAND_VALUE);
    for(i = 1; i<=dim; i++) {
        for(j = 1; j<=dim; j++) {
            grid[i][j] = rand() % 2;
        }
    }
 
    // Main game loop
    for (iter = 0; iter<maxIter; iter++) {
        // Left-Right columns
        for (i = 1; i<=dim; i++) {
            grid[i][0] = grid[i][dim]; // Copy first real column to right ghost column
            grid[i][dim+1] = grid[i][1]; // Copy last real column to left ghost column
        }
        // Top-Bottom rows
        for (j = 0; j<=dim+1; j++) {
            grid[0][j] = grid[dim][j]; // Copy first real row to bottom ghost row
            grid[dim+1][j] = grid[1][j]; // Copy last real row to top ghost row
        }
 
        // Now we loop over all cells and determine their fate
        for (i = 1; i<=dim; i++) {
            for (j = 1; j<=dim; j++) {
                // Get the number of neighbors for a given grid point
                int numNeighbors = getNeighbors(grid, i, j);
 
                // Here we have explicitly all of the game rules
                if (grid[i][j] == 1 && numNeighbors < 2)
                    newGrid[i][j] = 0;
                else if (grid[i][j] == 1 && (numNeighbors == 2 || numNeighbors == 3))
                    newGrid[i][j] = 1;
                else if (grid[i][j] == 1 && numNeighbors > 3)
                    newGrid[i][j] = 0;
                else if (grid[i][j] == 0 && numNeighbors == 3)
                    newGrid[i][j] = 1;
                else
                    newGrid[i][j] = grid[i][j];
            }
        }
 
        // Done with one step so we swap our grids and iterate again
        int **tmpGrid = grid;
        grid = newGrid;
        newGrid = tmpGrid;
    }// End main game loop
 
    // Sum up alive cells and print results
    int total = 0;
    for (i = 1; i<=dim; i++) {
        for (j = 1; j<=dim; j++) {
            total += grid[i][j];
        }
    }
    printf("Total Alive: %d\n", total);
 
    // Release memory
    free(grid);
    free(newGrid);
 
    return 0;
}
