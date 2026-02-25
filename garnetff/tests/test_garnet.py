# Garnet tests

from garnetff import garnet
from openff.toolkit.topology import Molecule, Topology
from openff.pablo import topology_from_pdb
from openmm import LangevinMiddleIntegrator
from openmm.app import PME, HBonds, Simulation, PDBFile
from openmm.unit import nanometer, picosecond, kelvin, kilojoules_per_mole
import os

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")

def get_potential_energy(simulation):
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy()

def test_pdb():
    pdb_fp = os.path.join(data_dir, "gb3.pdb")
    topology = topology_from_pdb(pdb_fp)

    system, top_openmm = garnet.topology_to_openmm_system(
        topology,
        nonbondedMethod=PME,
        nonbondedCutoff=1*nanometer,
        constraints=HBonds,
        rigidWater=True,
    )
    assert system.getNumParticles() == 9906

    temp = 300*kelvin
    dt = 0.004*picosecond
    integrator = LangevinMiddleIntegrator(temp, 1/picosecond, dt)
    simulation = Simulation(top_openmm, system, integrator)
    pdb = PDBFile(pdb_fp)
    simulation.context.setPositions(pdb.positions)
    assert get_potential_energy(simulation) < -100000 * kilojoules_per_mole

    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temp)
    simulation.step(100)
    assert get_potential_energy(simulation) < -100000 * kilojoules_per_mole

def test_smiles():
    smiles = "[H]O[H]"
    mol = Molecule.from_smiles(smiles, hydrogens_are_explicit=True)
    topology = Topology.from_molecules(molecules=[mol])

    system, top_openmm = garnet.topology_to_openmm_system(topology)
    assert system.getNumParticles() == 3

def test_sdf():
    mol = Molecule.from_file(os.path.join(data_dir, "zw_l_alanine.sdf"))
    topology = Topology.from_molecules(molecules=[mol])

    system, top_openmm = garnet.topology_to_openmm_system(topology)
    assert system.getNumParticles() == 13
