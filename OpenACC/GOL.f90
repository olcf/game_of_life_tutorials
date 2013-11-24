program main
    implicit none
    integer :: i,j,iter,seed(8),numNeighbors,total
    real :: randm
    ! Linear game grid dimension
    integer :: dim = 1024
    ! Number of game iterations
    integer :: maxIter = 2**10
 
    ! Game grid pointers
    integer,dimension(:,:),allocatable :: grid, newGrid
  
    ! Allocate square grid of (dim+2)^2 elements, 2 added for ghost cells
    allocate(grid(dim+2,dim+2))
    allocate(newGrid(dim+2,dim+2))
 
    ! Assign initial population randomly
    seed = (/1985, 2011, 2012, 500, 24, 15, 99, 8/)
    call random_seed(PUT=seed)
    do j=1,dim
        do i=1,dim
            call random_number(randm)
            grid(i,j) = nint(randm)
        enddo
    enddo
 
    !$acc data copyin(grid), create(newGrid)
 
    ! Main game loop
    do iter=1,maxIter
 
    !$acc kernels
 
        ! Top-Bottom ghost rows
        !$acc loop independent
        do j=2,dim+1
            grid(1,j) = grid(dim+1,j) !Copy first game grid row to bottom ghost row
            grid(dim+2,j) = grid(2,j) !Copy first game grid row to top ghost row
        enddo
 
        ! Left-Right ghost columns
        !$acc loop independent
        do i=1,dim+2
            grid(i,1) = grid(i, dim+1) !Copy first game grid column to right ghost column
            grid(i,dim+2) = grid(i,2)  !Copy last game grid column to left ghost column
        enddo
 
        ! Now we loop over all cells and determine their fate
        do j=2,dim+1
            do i=2,dim+1
                ! Get the number of neighbors for a given grid point
                numNeighbors = grid(i,j+1) + grid(i,j-1)&     !left and right
                             + grid(i+1,j) + grid(i-1,j)&     !upper and lower
                             + grid(i+1,j+1) + grid(i-1,j-1)& !diagonals
                             + grid(i-1,j+1) + grid(i+1,j-1)
 
                ! Here we have explicitly all of the game rules
                if(grid(i,j) == 1 .AND. numNeighbors < 2) then
                    newGrid(i,j) = 0
                elseif(grid(i,j) == 1 .AND. (numNeighbors == 2 .OR. numNeighbors == 3)) then
                    newGrid(i,j) = 1
                elseif(grid(i,j) == 1 .AND. numNeighbors > 3) then
                    newGrid(i,j) = 0
                elseif(grid(i,j) == 0 .AND. numNeighbors == 3) then
                    newGrid(i,j) = 1
                else
                    newGrid(i,j) = grid(i,j)
                endif
            enddo
        enddo
 
        ! Can't switch pointers so we mannually have to copy array over
        do j=2,dim+1
            do i=2,dim+1
                grid(i,j) = newGrid(i,j)
            enddo
        enddo
 
    !$acc end kernels
    enddo!end main game loop  
 
    total = 0
    ! Sum up alive cells and print results

    !$acc parallel 
    !$acc loop reduction(+:total)
    do j=2,dim+1
	!$acc loop reduction(+:total)
        do i=2,dim+1
            total = total + grid(i,j)
        enddo
    enddo
    !$acc end parallel
 
    !$acc end data
 
    print *, "Total Alive", total
 
end program
