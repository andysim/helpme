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
    def setUp(self):
        self.toleranceD = 1e-8;
        self.toleranceF = 1e-4;
        self.expectedEnergy = 5.864957414;
        self.expectedForces = np.array([[-1.20630693, -1.49522843, 12.65589187],
                                        [ 1.00695882,  0.88956328, -5.08428301],
                                        [ 0.69297661,  1.09547848, -5.22771480],
                                        [-2.28988057, -2.10832506, 10.18914165],
                                        [ 0.81915340,  0.92013663, -6.43738026],
                                        [ 0.97696467,  0.69833887, -6.09492437]]);
        self.expectedVirial = np.array([[0.65613058, 0.49091167, 0.61109732,
                                         2.26906257, 2.31925449, -10.04901641]]);
        self.expectedPotential = np.array([[ 1.18119329, -0.72320559, -0.89641992, 7.58746515],
                                          [ 7.69247982, -1.20738468, -1.06662264, 6.09626260],
                                          [ 8.73449635, -0.83090721, -1.31352336, 6.26824317],
                                          [-9.98483179, -1.37283008, -1.26398385, 6.10859811],
                                          [-3.50591589, -0.98219832, -1.10328133, 7.71868137],
                                          [-2.39904512, -1.17142047, -0.83733677, 7.30806279]]);

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
        potentialAndGradient = np.zeros((6,4),dtype=np.float64)

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
        # Compute the reciprocal space potential and its gradient
        pmeD.compute_P_rec(0, mat(charges), mat(coords), mat(coords), 1, mat(potentialAndGradient))
        print("Potential and its gradient:")
        print(potentialAndGradient, "\n")

        self.assertTrue(np.allclose([self.expectedEnergy], [energy], atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedForces, forces, atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedVirial, virial, atol=self.toleranceD))
        self.assertTrue(np.allclose(self.expectedPotential, potentialAndGradient, atol=self.toleranceD))


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
        virial = np.zeros((1,6),dtype=np.float32)
        potentialAndGradient = np.zeros((6,4),dtype=np.float32)

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
        # Compute the reciprocal space potential and its gradient
        pmeF.compute_P_rec(0, mat(charges), mat(coords), mat(coords), 1, mat(potentialAndGradient))
        print("Potential and its gradient:")
        print(potentialAndGradient, "\n")

        self.assertTrue(np.allclose([self.expectedEnergy], [energy], atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedForces, forces, atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedVirial, virial, atol=self.toleranceF))
        self.assertTrue(np.allclose(self.expectedPotential, potentialAndGradient, atol=self.toleranceF))

if __name__ == '__main__':
    unittest.main()
