# Run a simulation in OpenMM with a trained force field

import MDAnalysis as mda
from openff.toolkit import Topology
from openff.toolkit.topology import Molecule
import networkx as nx
from openmm.app import ForceField, PDBFile, PME, CutoffPeriodic, HBonds, Simulation, StateDataReporter, DCDReporter, CheckpointReporter
from openmm import Vec3, LangevinMiddleIntegrator, MonteCarloBarostat, Platform
from openmm.unit import *
from rdkit import Chem
import os
import sys

run_name = sys.argv[1] # e.g. "64_7_ep12" or "amber14"
with open("proteins.txt") as f:
    proteins = [l.rstrip() for l in f.readlines()]
protein = proteins[int(sys.argv[2]) - 1]
rep_n = int(sys.argv[3])

out_prefix = f"dcd/{run_name}/{protein}_{rep_n}"
dcd_fp        = out_prefix + ".dcd"
checkpoint_fp = out_prefix + ".chk"
struc_fp = f"structures/{protein}.pdb"
n_steps = 1250000000 # 5 μs
n_steps_log = 125000 # 500 ps
temp = 300*kelvin
dt = 0.004*picoseconds
hydrogen_mass = 2
rigid_water = True
use_barostat = True
guess_bonds = True
platform = Platform.getPlatformByName("CUDA")

if run_name != "amber14":
    ff_fp = f"ff_xml/{protein}_{run_name}.xml"
    rdkit_mol = Chem.MolFromPDBFile(struc_fp, removeHs=False, proximityBonding=guess_bonds)
    mol_sys = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True, hydrogens_are_explicit=True)
    graph_sys = mol_sys.to_networkx()

    molecule_inds_sys = [0] * mol_sys.n_atoms
    for ci, cc in enumerate(nx.connected_components(graph_sys)):
        for ai in cc:
            molecule_inds_sys[ai] = ci + 1
    assert 0 not in molecule_inds_sys

    for (a, mi) in zip(mol_sys.atoms, molecule_inds_sys):
        a.metadata["residue_name"] = str(mi)
        a.metadata["residue_number"] = 1
        a.metadata["insertion_code"] = " "

    top = Topology()
    top.add_molecule(mol_sys)
    top_omm = top.to_openmm()
    u = mda.Universe(struc_fp)
    cell = u.dimensions[:3] / 10
    box = Quantity(value=(
        Vec3(x=cell[0], y=0.0    , z=0.0    ),
        Vec3(x=0.0    , y=cell[1], z=0.0    ),
        Vec3(x=0.0    , y=0.0    , z=cell[2]),
    ), unit=nanometer)
    top_omm.setPeriodicBoxVectors(box)

    forcefield = ForceField(ff_fp)
    positions = top.get_positions().to_openmm()
else:
    pdb = PDBFile(struc_fp)
    forcefield = ForceField("amber14-all.xml", "amber14/tip3p.xml")
    top_omm = pdb.topology
    positions = pdb.positions

system = forcefield.createSystem(
    top_omm,
    nonbondedMethod=PME,
    nonbondedCutoff=1*nanometer,
    constraints=HBonds,
    rigidWater=rigid_water,
    hydrogenMass=hydrogen_mass,
)
if use_barostat:
    system.addForce(MonteCarloBarostat(1*bar, temp))
integrator = LangevinMiddleIntegrator(temp, 1/picosecond, dt)
simulation = Simulation(top_omm, system, integrator, platform)

if os.path.isfile(checkpoint_fp):
    print("Restarting from checkpoint")
    simulation.loadCheckpoint(checkpoint_fp)
    n_steps_to_run = n_steps - simulation.currentStep
    append_dcd = True
else:
    simulation.context.setPositions(positions)
    print("Starting minimisation")
    simulation.minimizeEnergy()
    simulation.context.setVelocitiesToTemperature(temp)
    n_steps_to_run = n_steps
    append_dcd = False

simulation.reporters.append(DCDReporter(dcd_fp, n_steps_log, append=append_dcd))
simulation.reporters.append(CheckpointReporter(checkpoint_fp, n_steps_log))
simulation.reporters.append(StateDataReporter(sys.stdout, n_steps_log, step=True,
        potentialEnergy=True, temperature=True))
print("Running for", n_steps_to_run, "steps")
simulation.step(n_steps_to_run)
