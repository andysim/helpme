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


program testfortran
    use iso_c_binding
    use helpme
    implicit none

    type(c_ptr) :: pmeD, pmeF
    real(c_float) :: alphaF
    real(c_double) :: alphaD
    integer(c_int) :: rPower, splineOrder, aDim, bDim, cDim, angMom
    integer(c_size_t) :: nAtoms, atom
    real(c_float), allocatable, target :: coordsF(:,:), chargesF(:), forcesF(:,:), potentialAndFieldF(:,:)
    real(c_double), allocatable, target :: coordsD(:,:), chargesD(:), forcesD(:,:), potentialAndFieldD(:,:)
    real(c_float), target :: scaleFactorF, energyF, virialF(6)
    real(c_double), target :: scaleFactorD, energyD, virialD(6)

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
    allocate(coordsD(3,nAtoms), chargesD(nAtoms), forcesD(3,nAtoms), potentialAndFieldD(4,nAtoms))
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
    potentialAndFieldD = 0d0

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
                               c_loc(coordsD), 1, c_loc(potentialAndFieldD));
    write(*,*) "Potential and field:"
    do atom = 1,nAtoms
        write(*,'(4F16.10)') potentialAndFieldD(:,atom)
    enddo
    write(*,*)
    call helpme_destroyD(pmeD)


    !
    ! Instantiate single precision PME object
    !
    allocate(coordsF(3,nAtoms), chargesF(nAtoms), forcesF(3,nAtoms), potentialAndFieldF(4,nAtoms))
    scaleFactorF =  332.0716
    coordsF = real(coordsD, c_float)
    chargesF = real(chargesD, c_float)
    alphaF = real(alphaD)

    energyF = 0.0
    forcesF = 0.0
    virialF = 0.0
    potentialAndFieldF = 0.0

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
                               c_loc(coordsF), 1, c_loc(potentialAndFieldF));
    write(*,*) "Potential and field:"
    do atom = 1,nAtoms
        write(*,'(4F16.10)') potentialAndFieldF(:,atom)
    enddo
    write(*,*)
    call helpme_destroyF(pmeF)

    deallocate(coordsD, chargesD, forcesD, potentialAndFieldD)
    deallocate(coordsF, chargesF, forcesF, potentialAndFieldF)

end program testfortran
