module helpme
    use iso_c_binding
    implicit none

    enum, bind(c)
        enumerator :: ShapeMatrix = 0
        enumerator :: XAligned = 1
    end enum

    public ShapeMatrix, XAligned

    interface

        function helpme_createD() bind(C, name="helpme_createD")
            use iso_c_binding
            type(c_ptr) :: helpme_createD
        end function

        function helpme_createF() bind(C, name="helpme_createF")
            use iso_c_binding
            type(c_ptr) :: helpme_createF
        end function

        subroutine helpme_destroyD(pme) bind(C, name="helpme_destroyD")
            use iso_c_binding
            type(c_ptr), value :: pme
        end subroutine

        subroutine helpme_destroyF(pme) bind(C, name="helpme_destroyF")
            use iso_c_binding
            type(c_ptr), value :: pme
        end subroutine

        subroutine helpme_setupD(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor, nThreads)&
                            bind(C, name="helpme_setupD")
            use iso_c_binding
            type(c_ptr), value :: pme
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, nThreads
            real(c_double), value :: kappa, scaleFactor
        end subroutine

        subroutine helpme_setupF(pme, rPower, kappa, splineOrder, aDim, bDim, cDim, scaleFactor, nThreads)&
                            bind(C, name="helpme_setupF")
            use iso_c_binding
            type(c_ptr), value :: pme
            integer(c_int), value :: rPower, splineOrder, aDim, bDim, cDim, nThreads
            real(c_float), value :: kappa, scaleFactor
        end subroutine

        subroutine helpme_set_lattice_vectorsD(pme, A, B, C, alpha, beta, gamma, lattice)&
                            bind(C, name="helpme_set_lattice_vectorsD")
            use iso_c_binding
            type(c_ptr), value :: pme
            real(c_double), value :: A, B, C, alpha, beta, gamma
            integer(kind(ShapeMatrix)), value :: lattice
        end subroutine

        subroutine helpme_set_lattice_vectorsF(pme, A, B, C, alpha, beta, gamma, lattice)&
                            bind(C, name="helpme_set_lattice_vectorsF")
            use iso_c_binding
            type(c_ptr), value :: pme
            real(c_float), value :: A, B, C, alpha, beta, gamma
            integer(kind(ShapeMatrix)), value :: lattice
        end subroutine

        function helpme_compute_E_recD(pme, nAtoms, parameterAngMom, parameters, coordinates)&
                            bind(C, name="helpme_compute_E_recD")
            use iso_c_binding
            real(c_double) helpme_compute_E_recD
            type(c_ptr), value :: pme, parameters, coordinates
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

        function helpme_compute_E_recF(pme, nAtoms, parameterAngMom, parameters, coordinates)&
                            bind(C, name="helpme_compute_E_recF")
            use iso_c_binding
            real(c_float) helpme_compute_E_recF
            type(c_ptr), value :: pme, parameters, coordinates
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

        function helpme_compute_EF_recD(pme, nAtoms, parameterAngMom, parameters, coordinates, forces)&
                            bind(C, name="helpme_compute_EF_recD")
            use iso_c_binding
            real(c_double) helpme_compute_EF_recD
            type(c_ptr), value :: pme, parameters, coordinates, forces
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

        function helpme_compute_EF_recF(pme, nAtoms, parameterAngMom, parameters, coordinates, forces)&
                            bind(C, name="helpme_compute_EF_recF")
            use iso_c_binding
            real(c_float) helpme_compute_EF_recF
            type(c_ptr), value :: pme, parameters, coordinates, forces
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

        function helpme_compute_EFV_recD(pme, nAtoms, parameterAngMom, parameters, coordinates, forces, virial)&
                            bind(C, name="helpme_compute_EFV_recD")
            use iso_c_binding
            real(c_double) helpme_compute_EFV_recD
            type(c_ptr), value :: pme, parameters, coordinates, forces, virial
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

        function helpme_compute_EFV_recF(pme, nAtoms, parameterAngMom, parameters, coordinates, forces, virial)&
                            bind(C, name="helpme_compute_EFV_recF")
            use iso_c_binding
            real(c_float) helpme_compute_EFV_recF
            type(c_ptr), value :: pme, parameters, coordinates, forces, virial
            integer(c_size_t), value :: nAtoms
            integer(c_int),  value :: parameterAngMom
        end function

    end interface
end module helpme
