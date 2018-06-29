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


subroutine print_results_F(nAtoms, label, e, f, v)
    use iso_c_binding
    integer(c_size_t), intent(in) :: nAtoms
    character(len=*), intent(in) :: label
    real(c_float), intent(in) :: e, f(3,*), v(6)

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
end subroutine print_results_F

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

subroutine check_value_F(expected, found, tolerance, filename, lineno)
    use iso_c_binding
    character(len=*), intent(in) :: filename
    integer, intent(in) :: lineno
    real(c_float), intent(in) :: expected, found, tolerance

    if(ABS(expected - found) .gt. tolerance) then
        write(*,*) "Assertion failed on line ", lineno, " of file ", filename
        stop 1
    endif
    return
end subroutine check_value_F

subroutine check_matrix_F(nrows, ncols, expected, found, tolerance, filename, lineno)
    use iso_c_binding
    character(len=*), intent(in) :: filename
    integer, intent(in) :: lineno, nrows, ncols
    real(c_float), intent(in) :: expected(nrows,ncols), found(nrows,ncols), tolerance

    integer :: row, col

    if(MAXVAL(ABS(expected - found)) .gt. tolerance) then
        write(*,*) "Assertion failed on line ", lineno, " of file ", filename
        stop 1
    endif
    return
end subroutine check_matrix_F


program testfortran
    use iso_c_binding
    use helpme
    implicit none

    type(c_ptr) :: pmeD, pmeF
    real(c_float) :: alphaF
    real(c_double) :: alphaD
    integer(c_int) :: rPower, splineOrder, aDim, bDim, cDim, angMom
    integer(c_size_t) :: nAtoms, atom
    real(c_float), allocatable, target :: coordsF(:,:), chargesF(:), forcesF(:,:), potentialAndGradientF(:,:)
    real(c_double), allocatable, target :: coordsD(:,:), chargesD(:), forcesD(:,:), potentialAndGradientD(:,:)
    real(c_float), target :: scaleFactorF, energyF, virialF(6,1), toleranceF
    real(c_double), target :: scaleFactorD, energyD, virialD(6,1), toleranceD
    real(c_double) expectedEnergy, expectedForces(3,6), expectedVirial(6,1), expectedPotential(4,6)

    !
    ! Some reference values for testing purposes
    !
    toleranceD = 1d-8
    toleranceF = 1e-4
    expectedEnergy = 5.864957414d0;
    expectedForces = reshape( [-1.20630693d0, -1.49522843d0, 12.65589187d0,&
                                1.00695882d0,  0.88956328d0, -5.08428301d0,&
                                0.69297661d0,  1.09547848d0, -5.22771480d0,&
                               -2.28988057d0, -2.10832506d0, 10.18914165d0,&
                                0.81915340d0,  0.92013663d0, -6.43738026d0,&
                                0.97696467d0,  0.69833887d0, -6.09492437d0], [ 3, 6 ] )
    expectedVirial = reshape( [0.65613058d0,  0.49091167d0,   0.61109732d0,&
                               2.26906257d0,  2.31925449d0, -10.04901641d0], [ 6, 1 ] )
    expectedPotential = reshape( [ 1.18119329d0, -0.72320559d0, -0.89641992d0, 7.58746515d0,&
                                   7.69247982d0, -1.20738468d0, -1.06662264d0, 6.09626260d0,&
                                   8.73449635d0, -0.83090721d0, -1.31352336d0, 6.26824317d0,&
                                  -9.98483179d0, -1.37283008d0, -1.26398385d0, 6.10859811d0,&
                                  -3.50591589d0, -0.98219832d0, -1.10328133d0, 7.71868137d0,&
                                  -2.39904512d0, -1.17142047d0, -0.83733677d0, 7.30806279d0 ], [4, 6])
    angMom = 0
    rPower = 1
    nAtoms = 6
    splineOrder = 5
    aDim = 32
    bDim = 32
    cDim = 32

    !
    ! Instantiate double precision PME object
    !
    allocate(coordsD(3,nAtoms), chargesD(nAtoms), forcesD(3,nAtoms), potentialAndGradientD(4,nAtoms))
    scaleFactorD =  332.0716d0
    coordsD = reshape( [ 2.0d0, 2.0d0, 2.0d0, &
                         2.5d0, 2.0d0, 3.0d0, &
                         1.5d0, 2.0d0, 3.0d0, &
                         0.0d0, 0.0d0, 0.0d0, &
                         0.5d0, 0.0d0, 1.0d0, &
                        -0.5d0, 0.0d0, 1.0d0 ], [ 3, size(coordsD,2) ] )
    chargesD = [ -0.834d0, 0.417d0, 0.417d0, -0.834d0, 0.417d0, 0.417d0 ]

    alphaD = 0.3d0

    energyD = 0d0
    forcesD = 0d0
    virialD = 0d0
    potentialAndGradientD = 0d0

    pmeD = helpme_createD()
    call helpme_setupD(pmeD, rPower, alphaD, splineOrder, aDim, bDim, cDim, scaleFactorD, 1)
    call helpme_set_lattice_vectorsD(pmeD, 20d0, 20d0, 20d0, 90d0, 90d0, 90d0, XAligned)
    call print_results_D(nAtoms, "Before helpme_compute_E_recD", energyD, forcesD, virialD)
    energyD = helpme_compute_E_recD(pmeD, nAtoms, angMom, c_loc(chargesD), c_loc(coordsD))
    call print_results_D(nAtoms, "After helpme_compute_E_recD", energyD, forcesD, virialD)
    energyD = helpme_compute_EF_recD(pmeD, nAtoms, angMom, c_loc(chargesD), c_loc(coordsD), c_loc(forcesD))
    call print_results_D(nAtoms, "After helpme_compute_EF_recD", energyD, forcesD, virialD)
    energyD = helpme_compute_EFV_recD(pmeD, nAtoms, angMom, c_loc(chargesD), c_loc(coordsD), c_loc(forcesD), c_loc(virialD))
    call print_results_D(nAtoms, "After helpme_compute_EFV_recD", energyD, forcesD, virialD)
    call helpme_compute_P_recD(pmeD, nAtoms, 0, c_loc(chargesD), c_loc(coordsD), nAtoms,&
                               c_loc(coordsD), 1, c_loc(potentialAndGradientD));
    write(*,*) "Potential and its gradient:"
    do atom = 1,nAtoms
        write(*,'(4F16.10)') potentialAndGradientD(:,atom)
    enddo
    write(*,*)

    call helpme_destroyD(pmeD)

    call check_value_D(expectedEnergy, energyD, toleranceD,&
                      __FILE__, __LINE__)
    call check_matrix_D(3, 6, expectedForces, forcesD, toleranceD,&
                      __FILE__, __LINE__)
    call check_matrix_D(6, 1, expectedVirial, virialD, toleranceD,&
                      __FILE__, __LINE__)
    call check_matrix_D(4, 6, expectedPotential, potentialAndGradientD, toleranceD,&
                      __FILE__, __LINE__)


    !
    ! Instantiate single precision PME object
    !
    allocate(coordsF(3,nAtoms), chargesF(nAtoms), forcesF(3,nAtoms), potentialAndGradientF(4,nAtoms))
    scaleFactorF =  332.0716
    coordsF = real(coordsD, c_float)
    chargesF = real(chargesD, c_float)
    alphaF = real(alphaD)

    energyF = 0.0
    forcesF = 0.0
    virialF = 0.0
    potentialAndGradientF = 0.0

    pmeF = helpme_createF()
    call helpme_setupF(pmeF, rPower, alphaF, splineOrder, aDim, bDim, cDim, scaleFactorF, 1)
    call helpme_set_lattice_vectorsF(pmeF, 20.0, 20.0, 20.0, 90.0, 90.0, 90.0, XAligned)
    call print_results_F(nAtoms, "Before helpme_compute_E_recF", energyF, forcesF, virialF)
    energyF = helpme_compute_E_recF(pmeF, nAtoms, angMom, c_loc(chargesF), c_loc(coordsF))
    call print_results_F(nAtoms, "After helpme_compute_E_recF", energyF, forcesF, virialF)
    energyF = helpme_compute_EF_recF(pmeF, nAtoms, angMom, c_loc(chargesF), c_loc(coordsF), c_loc(forcesF))
    call print_results_F(nAtoms, "After helpme_compute_EF_recF", energyF, forcesF, virialF)
    energyF = helpme_compute_EFV_recF(pmeF, nAtoms, angMom, c_loc(chargesF), c_loc(coordsF), c_loc(forcesF), c_loc(virialF))
    call print_results_F(nAtoms, "After helpme_compute_EFV_recF", energyF, forcesF, virialF)
    call helpme_compute_P_recF(pmeF, nAtoms, 0, c_loc(chargesF), c_loc(coordsF), nAtoms,&
                               c_loc(coordsF), 1, c_loc(potentialAndGradientF));
    write(*,*) "Potential and its gradient:"
    do atom = 1,nAtoms
        write(*,'(4F16.10)') potentialAndGradientF(:,atom)
    enddo
    write(*,*)

    call helpme_destroyF(pmeF)

    call check_value_F(real(expectedEnergy, c_float), energyF, toleranceF,&
                      __FILE__, __LINE__)
    call check_matrix_F(3, 6, real(expectedForces, c_float), forcesF, toleranceF,&
                      __FILE__, __LINE__)
    call check_matrix_F(6, 1, real(expectedVirial, c_float), virialF, toleranceF,&
                      __FILE__, __LINE__)
    call check_matrix_F(4, 6, real(expectedPotential, c_float), potentialAndGradientF, toleranceF,&
                      __FILE__, __LINE__)


    deallocate(coordsD, chargesD, forcesD, potentialAndGradientD)
    deallocate(coordsF, chargesF, forcesF, potentialAndGradientF)

end program testfortran
