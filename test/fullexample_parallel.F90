! BEGINLICENSE
!
! This file is part of helPME, which is distributed under the BSD 3-clause license,
! as described in the LICENSE file in the top level directory of this project.
!
! Author: Andrew C. Simmonett
!
! ENDLICENSE

subroutine print_results_D(nAtoms, label, e, f, v)
    use iso_c_binding
    integer(c_size_t), intent(in) :: nAtoms
    character(len=*), intent(in) :: label
    real(c_double), intent(in) :: e, f(3,*), v(6)

    integer atom

    write(*,*) label
    write(*,'(A9,F16.10)') "Energy = ", e
    write(*,*) "Forces:"
    do atom = 1,nAtoms
        write(*,'(3F16.10)') f(:,atom)
    enddo
    write(*,*) "Virial:"
    write(*,'(6F16.10)') v
    write(*,*)

    return
end subroutine print_results_D

subroutine check_value_D(expected, found, tolerance, filename, lineno)
    use iso_c_binding
    character(len=*), intent(in) :: filename
    integer, intent(in) :: lineno
    real(c_double), intent(in) :: expected, found, tolerance

    if(ABS(expected - found) .gt. tolerance) then
        write(*,*) "Assertion failed on line ", lineno, " of file ", filename
        stop 1
    endif
    return
end subroutine check_value_D

subroutine check_matrix_D(nrows, ncols, expected, found, tolerance, filename, lineno)
    use iso_c_binding
    character(len=*), intent(in) :: filename
    integer, intent(in) :: lineno, nrows, ncols
    real(c_double), intent(in) :: expected(nrows,ncols), found(nrows,ncols), tolerance

    if(MAXVAL(ABS(expected - found)) .gt. tolerance) then
        write(*,*) "Assertion failed on line ", lineno, " of file ", filename
        stop 1
    endif
    return
end subroutine check_matrix_D


program testfortran
    use iso_c_binding
    use helpme
    use mpi
    implicit none

    type(c_ptr) :: pmeS, pmeP
    real(c_double) :: alpha
    integer(c_int) :: rPower, splineOrder, aDim, bDim, cDim, angMom
    integer(c_size_t) :: nAtoms
    real(c_double), allocatable, target :: coords(:,:), charges(:), forcesS(:,:), nodeForces(:,:), parallelForces(:,:)
    real(c_double) :: scaleFactor, energyS, nodeEnergy, parallelEnergy
    real(c_double), target :: virialS(6), parallelVirial(6), nodeVirial(6), tolerance
    character(len=20) :: tmp

    integer :: argc, error, numNodes, myRank, nx, ny, nz

    call MPI_Init(error)
    call MPI_Comm_size(MPI_COMM_WORLD, numNodes, error)
    call MPI_Comm_rank(MPI_COMM_WORLD, myRank, error)

    argc = command_argument_count()

    angMom = 0
    rPower = 1
    nAtoms = 6
    splineOrder = 5
    aDim = 32
    bDim = 32
    cDim = 32
    tolerance = 1d-8

    !!
    !! Instantiate double precision PME object
    !!
    allocate(coords(3,nAtoms), charges(nAtoms))
    allocate(forcesS(3,nAtoms), nodeForces(3,nAtoms), parallelForces(3,nAtoms))
    scaleFactor =  332.0716d0
    coords = reshape( [ 2.0d0, 2.0d0, 2.0d0, &
                        2.5d0, 2.0d0, 3.0d0, &
                        1.5d0, 2.0d0, 3.0d0, &
                        0.0d0, 0.0d0, 0.0d0, &
                        0.5d0, 0.0d0, 1.0d0, &
                       -0.5d0, 0.0d0, 1.0d0 ], [ 3, size(coords,2) ] )
    charges = [ -0.834d0, 0.417d0, 0.417d0, -0.834d0, 0.417d0, 0.417d0 ]
    alpha = 0.3d0

    forcesS = 0d0
    virialS = 0d0
    if(myRank .eq. 0) then
        ! Generate a serial benchmark first
        pmeS = helpme_createD()
        call helpme_setupD(pmeS, rPower, alpha, splineOrder, aDim, bDim, cDim, scaleFactor, 1)
        call helpme_set_lattice_vectorsD(pmeS, 20d0, 20d0, 20d0, 90d0, 90d0, 90d0, XAligned)
        energyS = helpme_compute_EFV_recD(pmeS, nAtoms, angMom, c_loc(charges), c_loc(coords),&
                                          c_loc(forcesS), c_loc(virialS))
        call print_results_D(nAtoms, "Serial Results:", energyS, forcesS, virialS)
        call helpme_destroyD(pmeS)
    endif

    ! Now the parallel version
    pmeP = helpme_createD()
    if(argc .eq. 3) then
        call get_command_argument(1, tmp)
        read(tmp, *) nx
        call get_command_argument(2, tmp)
        read(tmp, *) ny
        call get_command_argument(3, tmp)
        read(tmp, *) nz
        nodeForces = 0d0
        nodeVirial = 0d0
        call helpme_setup_parallelD(pmeP, rPower, alpha, splineOrder, aDim, bDim, cDim, scaleFactor, 1,&
                                    MPI_COMM_WORLD, ZYX, nx, ny, nz)
        call helpme_set_lattice_vectorsD(pmeP, 20d0, 20d0, 20d0, 90d0, 90d0, 90d0, XAligned)
        nodeEnergy = helpme_compute_EFV_recD(pmeP, nAtoms, angMom, c_loc(charges), c_loc(coords),&
                                             c_loc(nodeForces), c_loc(nodeVirial))
        call MPI_Reduce(nodeEnergy, parallelEnergy, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, error)
        call MPI_Reduce(nodeForces, parallelForces, 6 * 3, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, error)
        call MPI_Reduce(nodeVirial, parallelVirial, 6, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD, error)
        if(myRank .eq. 0) then
            call print_results_D(nAtoms, "Parallel Results:", parallelEnergy, parallelForces, parallelVirial)
            call check_value_D(parallelEnergy, energyS, tolerance,&
                              __FILE__, __LINE__)
            call check_matrix_D(3, 6, parallelForces, forcesS, tolerance,&
                              __FILE__, __LINE__)
            call check_matrix_D(6, 1, parallelVirial, virialS, tolerance,&
                              __FILE__, __LINE__)
        endif
    else
        write(*,*) "This test should be run with exactly 3 arguments describing the number of X,Y and Z nodes."
        stop 1
    endif
    call helpme_destroyD(pmeP)

    deallocate(coords, charges)
    deallocate(forcesS, nodeForces, parallelForces)
    call MPI_Finalize (error)
end program testfortran
