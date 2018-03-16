! BEGINLICENSE
!
! This file is part of helPME, which is distributed under the BSD 3-clause license,
! as described in the LICENSE file in the top level directory of this project.
!
! Author: Andrew C. Simmonett
!
! ENDLICENSE

program testfortran
    use iso_c_binding
    use helpme
    implicit none

    type(c_ptr) :: pmeD, pmeF
    real(c_float) :: alphaF
    real(c_double) :: alphaD
    integer(c_int) :: rPower, splineOrder, aDim, bDim, cDim, angMom
    integer(c_size_t) :: nAtoms
    real(c_float), allocatable, target :: coordsF(:,:), chargesF(:), forcesF(:,:)
    real(c_double), allocatable, target :: coordsD(:,:), chargesD(:), forcesD(:,:)
    real(c_float), target :: scaleFactorF, energyF
    real(c_double), target :: scaleFactorD, energyD

    angMom = 0
    rPower = 1
    nAtoms = 6
    splineOrder = 5
    aDim = 32
    bDim = 32
    cDim = 32

    ! Instantiate double precision PME object
    allocate(coordsD(3,nAtoms), chargesD(nAtoms), forcesD(3,nAtoms))
    scaleFactorD =  332.0716d0
    forcesD = 0d0
    alphaD = 0.3d0
    coordsD = reshape( [ 2.0d0, 2.0d0, 2.0d0, &
                         2.5d0, 2.0d0, 3.0d0, &
                         1.5d0, 2.0d0, 3.0d0, &
                         0.0d0, 0.0d0, 0.0d0, &
                         0.5d0, 0.0d0, 1.0d0, &
                         0.5d0, 0.0d0, 1.0d0 ], [ 3, size(coordsD,2) ] )
    chargesD = [ -0.834d0, 0.417d0, 0.417d0, -0.834d0, 0.417d0, 0.417d0 ]
    pmeD = helpme_createD()
    call helpme_setupD(pmeD, rPower, alphaD, splineOrder, aDim, bDim, cDim, scaleFactorD, 1)
    call helpme_set_lattice_vectorsD(pmeD, 20d0, 20d0, 20d0, 90d0, 90d0, 90d0, XAligned)
    energyD = helpme_compute_EF_recD(pmeD, nAtoms, angMom, c_loc(chargesD), c_loc(coordsD), c_loc(forcesD))

    ! Instantiate single precision PME object
    allocate(coordsF(3,nAtoms), chargesF(nAtoms), forcesF(3,nAtoms))
    scaleFactorF =  332.0716
    forcesF = 0.0
    alphaF = real(alphaD)
    coordsF = real(coordsD, c_float)
    chargesF = real(chargesD, c_float)
    pmeF = helpme_createF()
    call helpme_setupF(pmeF, rPower, alphaF, splineOrder, aDim, bDim, cDim, scaleFactorF, 1)
    call helpme_set_lattice_vectorsF(pmeF, 20.0, 20.0, 20.0, 90.0, 90.0, 90.0, XAligned)
    energyF = helpme_compute_EF_recF(pmeF, nAtoms, angMom, c_loc(chargesF), c_loc(coordsF), c_loc(forcesF))


    deallocate(coordsD, coordsF, chargesD, chargesF, forcesD, forcesF)
end program testfortran
