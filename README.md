Game of Life
==============

The GOL is an example of cellular automaton that utilizes a 2 dimensional stencil. For each game iteration the integer value of each cell in the 2D game grid is determined by summing it’s 8 closest neighbors and then applying the game rules, with the initial game state randomly generated. Each cell has two states, alive or dead, represented as an integer 1 or 0. Periodic boundary conditions are enforced through the use of ghost cells. Cell updates are not propagated through until the end of each iteration, leaving the board static during calculations.

If we wish to play GOL with a 3×3 grid to compensate for the periodic boundary conditions we will need to actually hold a 5×5 grid in memory. This would be the area inside of the blue dashed line in the image below. The cells that are not part of the ‘visible’ 3×3 grid but part of the 5×5 grid we will refer to as ghost cells.

![alt tag](https://raw.github.com/olcf/game_of_life_tutorials/master/GOL-grid.png)
