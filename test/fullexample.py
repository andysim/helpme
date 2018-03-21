import unittest

import helpmelib as pme
import numpy as np

def print_results(label, e, f, v):
    np.set_printoptions(precision=10, linewidth=100)
    print(label)
    print("Energy = {:16.10f}".format(e))
    print("Forces:")
    print(f)
    print("Virial:")
    print(v)
    print()


class TestHelpme(unittest.TestCase):
    def test_serial(self):
        # Instatiate double precision PME object
        coords = np.array([
            [ 2.00000,  2.00000, 2.00000],
            [ 2.50000,  2.00000, 3.00000],
            [ 1.50000,  2.00000, 3.00000],
            [ 0.00000,  0.00000, 0.00000],
            [ 0.50000,  0.00000, 1.00000],
            [-0.50000,  0.00000, 1.00000]
        ], dtype=np.float64)
        charges = np.array([[-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]], dtype=np.float64).T

        energy = 0
        forces = np.zeros((6,3),dtype=np.float64)
        virial = np.zeros((1,6),dtype=np.float64)

        pmeD = pme.PMEInstanceD()
        pmeD.setup(1, 0.3, 5, 32, 32, 32, 332.0716, 1)
        mat = pme.MatrixD
        pmeD.set_lattice_vectors(20, 20, 20, 90, 90, 90, pmeD.LatticeType.XAligned)
        # Compute just the energy
        print_results("Before pmeD.compute_E_rec", energy, forces, virial)
        energy = pmeD.compute_E_rec(0, mat(charges), mat(coords))
        print_results("After pmeD.compute_E_rec", energy, forces, virial)
        # Compute the energy and forces
        energy = pmeD.compute_EF_rec(0, mat(charges), mat(coords), mat(forces))
        print_results("After pmeD.compute_EF_rec", energy, forces, virial)
        # Compute the energy, forces and virial
        energy = pmeD.compute_EFV_rec(0, mat(charges), mat(coords), mat(forces), mat(virial))
        print_results("After pmeD.compute_EFV_rec", energy, forces, virial)

    def test_float(self):
        # Instatiate single precision PME object
        coords = np.array([
            [ 2.00000,  2.00000, 2.00000],
            [ 2.50000,  2.00000, 3.00000],
            [ 1.50000,  2.00000, 3.00000],
            [ 0.00000,  0.00000, 0.00000],
            [ 0.50000,  0.00000, 1.00000],
            [-0.50000,  0.00000, 1.00000]
        ], dtype=np.float32)
        charges = np.array([[-0.834, 0.417, 0.417, -0.834, 0.417, 0.417]], dtype=np.float32).T

        energy = 0
        forces = np.zeros((6,3),dtype=np.float32)
        virial = np.zeros((1,6),dtype=np.float64)

        pmeF = pme.PMEInstanceF()
        pmeF.setup(1, 0.3, 5, 32, 32, 32, 332.0716, 1)
        mat = pme.MatrixF
        pmeF.set_lattice_vectors(20, 20, 20, 90, 90, 90, pmeF.LatticeType.XAligned)
        # Compute just the energy
        print_results("Before pmeF.compute_E_rec", energy, forces, virial)
        energy = pmeF.compute_E_rec(0, mat(charges), mat(coords))
        print_results("After pmeF.compute_E_rec", energy, forces, virial)
        # Compute the energy and forces
        energy = pmeF.compute_EF_rec(0, mat(charges), mat(coords), mat(forces))
        print_results("After pmeF.compute_EF_rec", energy, forces, virial)
        # Compute the energy, forces and virial
        energy = pmeF.compute_EFV_rec(0, mat(charges), mat(coords), mat(forces), mat(virial))
        print_results("After pmeF.compute_EFV_rec", energy, forces, virial)

if __name__ == '__main__':
    unittest.main()
