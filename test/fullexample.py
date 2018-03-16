import helpmelib as pme
import numpy as np

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
forces = np.zeros((6,3),dtype=np.float64)
pmeD = pme.PMEInstanceD()
pmeD.setup(1, 0.3, 5, 32, 32, 32, 332.0716, 1)
mat = pme.MatrixD
pmeD.set_lattice_vectors(20, 20, 20, 90, 90, 90, pmeD.LatticeType.XAligned)
energy = pmeD.compute_EF_rec(0, mat(charges), mat(coords), mat(forces))

# Instatiate single precision PME object
coords = np.array([
    [ 2.00000,  2.00000, 2.00000],
    [ 2.50000,  2.00000, 3.00000],
    [ 1.50000,  2.00000, 3.00000],
    [ 0.00000,  0.00000, 0.00000],
    [ 0.50000,  0.00000, 1.00000],
    [-0.50000,  0.00000, 1.00000]
], dtype=np.float32)
charges = np.array([[-0.834], [0.417], [0.417], [-0.834], [0.417], [0.417]], dtype=np.float32)
forces = np.zeros((6,3),dtype=np.float32)
pmeF = pme.PMEInstanceF()
pmeF.setup(1, 0.3, 5, 32, 32, 32, 332.0716, 1)
mat = pme.MatrixF
pmeF.set_lattice_vectors(20, 20, 20, 90, 90, 90, pmeF.LatticeType.XAligned)
energy = pmeF.compute_EF_rec(0, mat(charges), mat(coords), mat(forces))
