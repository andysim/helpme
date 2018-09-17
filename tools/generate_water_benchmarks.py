import os
import sys
import numpy as np
from scipy.stats import special_ortho_group
from random import uniform as rand
from simtk import unit
from simtk.openmm import app
from simtk import openmm

# TIP3P parameters
charges = "-0.834 0.417 0.417\n"
c6s = "24.3929177025 0.0274557263 0.0274557263\n"

grid_density = 1.0 # grid points / Angstrom

def find_grid_size(min_size):
    """ Finds the minimum grid size that is at least the size of the input grid, but has
        only factors of 2,3,5,7 raised to any power and either 11 or 13 appearing once. """
    prime_factors = [2, 3, 5, 7]
    current_size = min_size
    while 1:
        remainder = current_size
        for prime_factor in prime_factors:
            while remainder > 1 and remainder % prime_factor == 0:
                remainder /= prime_factor
        if remainder == 1 or remainder == 11 or remainder == 13:
            return current_size
        current_size += 1

# There are dim X dim X dim molecules in the box
for dim in range(16, 60, 4):
    n_waters = dim**3
    n_atoms = 3*n_waters
    print('Generating waterbox%d...' % n_atoms)
    datadir = '../test/'
    cppfile = datadir + 'waterbox%d_benchmark.cpp' % n_atoms
    crdfile = datadir + 'data/waterbox%d_coords.txt' % n_atoms
    c6sfile = datadir + 'data/waterbox%d_c6s.txt' % n_atoms
    chgfile = datadir + 'data/waterbox%d_charges.txt' % n_atoms
    if os.path.isfile(crdfile) or os.path.isfile(c6sfile) or os.path.isfile(chgfile):
        print('\twaterbox%d already exists...' % n_atoms)
        continue

    with open(chgfile, 'w') as fp:
        fp.write('%d 1\n%s' % (n_atoms, n_waters*charges))

    with open(c6sfile, 'w') as fp:
        fp.write('%d 1\n%s' % (n_atoms, n_waters*c6s))

    initial_topology = app.Topology()
    initial_positions = unit.Quantity((), unit.angstroms)
    m = app.Modeller(initial_topology, initial_positions)
    ff = app.ForceField('tip3p.xml')
    m.addSolvent(ff, numAdded=n_waters, model='tip3p')
    topology = m.getTopology()
    positions = m.getPositions()
    box_vectors = topology.getPeriodicBoxVectors()
    system = ff.createSystem(topology, nonbondedMethod=app.PME, rigidWater=True)

    ## Do a short equilibration run to get it around 300K
    integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picosecond, 2.0*unit.femtosecond)
    simulation = app.Simulation(topology, system, integrator)
    simulation.context.setPeriodicBoxVectors(*box_vectors)
    simulation.context.setPositions(positions)
    print('\tMinimizing...')
    simulation.minimizeEnergy(tolerance = 100*unit.kilojoule/unit.mole)
    simulation.reporters.append(app.StateDataReporter(sys.stdout, 250, step=True, potentialEnergy=True, temperature=True))
    simulation.step(25000)
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()

    with open(crdfile, 'w') as fp:
        fp.write('%d 3\n' % n_atoms)
        for atom in positions:
            x = atom[0].value_in_unit(unit.angstroms)
            y = atom[1].value_in_unit(unit.angstroms)
            z = atom[2].value_in_unit(unit.angstroms)
            fp.write("%16.10f%16.10f%16.10f\n" % (x, y, z))

    with open(cppfile, 'w') as fp:
        box_length = int(grid_density * box_vectors[0][0].value_in_unit(unit.angstroms))
        grid_dim = find_grid_size(box_length)
        fp.write("""\
// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#define BOX_SIZE_A %16.10f
#define BOX_SIZE_B %16.10f
#define BOX_SIZE_C %16.10f
#define DEFAULT_GRID_A %d
#define DEFAULT_GRID_B %d
#define DEFAULT_GRID_C %d
#define FILENAME "%s"

// Make the benchmark from the generic template
#include "make_benchmark.hpp"
""" % (box_length, box_length, box_length, grid_dim, grid_dim, grid_dim, 'waterbox%d'%n_atoms))
