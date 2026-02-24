# Run Garnet training simulations
# Change non-bonded 1-4 scaling (NonbondedForce line) in
#   /path/to/miniconda3/envs/openmm/lib/python3.11/site-packages/openmmforcefields/ffxml/amber/gaff/ffxml/gaff-2.11.xml
#   to match TIP3P force field file value (0.833333)
# Change chem_shift_path to point to the nmr conda environment
# Arguments are output directory and force field XML file, if none given runs with GAFF/TIP3P
#   for condensed phase and Amber14/TIP3P for protein

from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmforcefields.generators import GAFFTemplateGenerator
from openff.toolkit import Topology
from openff.toolkit.topology import Molecule
import networkx as nx
import MDAnalysis as mda
from rdkit import Chem
import os
import sys
import subprocess
from time import time

out_dir = sys.argv[1]
if len(sys.argv) > 2:
    ff_xml_fp = sys.argv[2]
    use_default_ff = False
else:
    use_default_ff = True

chem_shift_path = "XLA_FLAGS=--xla_gpu_cuda_data_dir=/path/to/miniconda3/envs/nmr /path/to/miniconda3/envs/nmr/bin/python"
run_val_systems = False
n_steps = 1250000 # 2.5 ns
n_steps_save = 5000 # 10 ps
n_steps_protein = 15000000 # 30 ns
n_steps_save_protein = 5000 # 10 ps
dt = 0.002*picoseconds
dist_cutoff = 1*nanometer
nonbonded_method = PME
pressure = 1*bar
temp_mixing = 298.15 # K
temp_protein = 298.0 # K
constraints = HBonds # None, HBonds or HAngles
rigidWater = True
enth_vap_temps = list(range(285, 325+1, 10)) # K
data_dir = "condensed_data"
platform = Platform.getPlatformByName("CUDA")
platform_gas = Platform.getPlatformByName("Reference")

enth_vap_data = [
    # smiles     val_only
    ["O"       , False], # Water
    ["CC(=O)C" , True ], # Acetone
    ["c1ccccc1", False], # Benzene
    ["CO"      , False], # Methanol
]

enth_mixing_data = [
    # smiles_1     smiles_2     val_only
    ["C1COCCN1"  , "O"        , False],
    ["CCC(C)=O"  , "Nc1ccccc1", True ],
    ["CNCCO"     , "O"        , True ],
    ["CN1CCCC1=O", "ClCCCl"   , False],
    ["CCCCO"     , "OC1=NCCC1", False],
    ["C=CCCCC"   , "CCCO"     , False],
]

protein_data = [
    # name  n_res
    ["gb3", 56],
]

if os.path.isdir(out_dir):
    raise Exception(f"output directory {out_dir} already exists")

os.mkdir(out_dir)
for d in ["vapourisation_gas", "vapourisation_liquid", "mixing_single",
          "mixing_combined", "protein"]:
    os.mkdir(os.path.join(out_dir, d))

def generate_forcefield(smiles_list):
    if use_default_ff:
        # Use GAFF to get small molecule parameters, except use TIP3P for water
        molecules = []
        for smiles in smiles_list:
            if smiles != "O":
                molecules.append(Molecule.from_smiles(smiles))
        forcefield = ForceField("tip3p.xml")
        if len(molecules) > 0:
            gaff = GAFFTemplateGenerator(molecules=molecules)
            forcefield.registerTemplateGenerator(gaff.generator)
    else:
        forcefield = ForceField(ff_xml_fp)
    return forcefield

for smiles, val_system in enth_vap_data:
    if val_system and not run_val_systems:
        print("vapourisation_liquid ", smiles, "- skipped")
        continue
    forcefield = generate_forcefield([smiles])
    for temp_vap in enth_vap_temps:
        pdb = PDBFile(f"{data_dir}/starting_structures/vapourisation_liquid_{smiles}.pdb")
        out_prefix = f"{out_dir}/vapourisation_liquid/{smiles}_{temp_vap}K"
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=nonbonded_method,
            nonbondedCutoff=dist_cutoff,
            constraints=constraints,
            rigidWater=rigidWater,
        )
        system.addForce(MonteCarloBarostat(pressure, temp_vap*kelvin))
        integrator = LangevinMiddleIntegrator(temp_vap*kelvin, 1/picosecond, dt)
        simulation = Simulation(pdb.topology, system, integrator, platform)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(DCDReporter(f"{out_prefix}.dcd", n_steps_save))
        simulation.reporters.append(StateDataReporter(f"{out_prefix}.log",
                            n_steps_save, step=True, potentialEnergy=True, temperature=True))

        t_start = time()
        simulation.step(n_steps)
        print("vapourisation_liquid ", smiles, temp_vap, "K -",
              round(time() - t_start, 2), "s")

        pdb = PDBFile(f"{data_dir}/monomers/{smiles}.pdb")
        out_prefix = f"{out_dir}/vapourisation_gas/{smiles}_{temp_vap}K"
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=NoCutoff,
            nonbondedCutoff=dist_cutoff,
            constraints=constraints,
            rigidWater=rigidWater,
        )
        integrator = LangevinMiddleIntegrator(temp_vap*kelvin, 1/picosecond, dt)
        simulation = Simulation(pdb.topology, system, integrator, platform_gas)
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(DCDReporter(f"{out_prefix}.dcd", n_steps_save))
        simulation.reporters.append(StateDataReporter(f"{out_prefix}.log",
                            n_steps_save, step=True, potentialEnergy=True, temperature=True))

        t_start = time()
        simulation.step(n_steps)
        print("vapourisation_gas ", smiles, temp_vap, "K -",
              round(time() - t_start, 2), "s")

for smiles_1, smiles_2, val_system in enth_mixing_data:
    if val_system and not run_val_systems:
        print("mixing_combined ", smiles_1, smiles_2, "- skipped")
        continue
    forcefield = generate_forcefield([smiles_1, smiles_2])
    for smiles in [smiles_1, smiles_2]:
        pdb = PDBFile(f"{data_dir}/starting_structures/mixing_single_{smiles}.pdb")
        out_prefix = f"{out_dir}/mixing_single/{smiles}"
        # Don't re-run duplicates
        if not os.path.isfile(f"{out_prefix}.dcd"):
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=nonbonded_method,
                nonbondedCutoff=dist_cutoff,
                constraints=constraints,
                rigidWater=rigidWater,
            )
            system.addForce(MonteCarloBarostat(pressure, temp_mixing*kelvin))
            integrator = LangevinMiddleIntegrator(temp_mixing*kelvin, 1/picosecond, dt)
            simulation = Simulation(pdb.topology, system, integrator, platform)
            simulation.context.setPositions(pdb.positions)
            simulation.minimizeEnergy()
            simulation.reporters.append(DCDReporter(f"{out_prefix}.dcd", n_steps_save))
            simulation.reporters.append(StateDataReporter(f"{out_prefix}.log",
                                n_steps_save, step=True, potentialEnergy=True, temperature=True))

            t_start = time()
            simulation.step(n_steps)
            print("mixing_single ", smiles, temp_mixing, "K -",
                  round(time() - t_start, 2), "s")

    pdb = PDBFile(f"{data_dir}/starting_structures/mixing_combined_{smiles_1}_{smiles_2}.pdb")
    out_prefix = f"{out_dir}/mixing_combined/{smiles_1}_{smiles_2}"
    system = forcefield.createSystem(
        pdb.topology,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=dist_cutoff,
        constraints=constraints,
        rigidWater=rigidWater,
    )
    system.addForce(MonteCarloBarostat(pressure, temp_mixing*kelvin))
    integrator = LangevinMiddleIntegrator(temp_mixing*kelvin, 1/picosecond, dt)
    simulation = Simulation(pdb.topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(DCDReporter(f"{out_prefix}.dcd", n_steps_save))
    simulation.reporters.append(StateDataReporter(f"{out_prefix}.log",
                        n_steps_save, step=True, potentialEnergy=True, temperature=True))

    t_start = time()
    simulation.step(n_steps)
    print("mixing_combined ", smiles_1, smiles_2, temp_mixing, "K -",
          round(time() - t_start, 2), "s")

def protein_setup(pdb_fp):
    rdkit_mol = Chem.MolFromPDBFile(pdb_fp, removeHs=False, proximityBonding=True)
    mol_sys = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True,
                                  hydrogens_are_explicit=True)
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
    u = mda.Universe(pdb_fp)
    cell = u.dimensions[:3] / 10
    box = Quantity(value=(
        Vec3(x=cell[0], y=0.0    , z=0.0    ),
        Vec3(x=0.0    , y=cell[1], z=0.0    ),
        Vec3(x=0.0    , y=0.0    , z=cell[2]),
    ), unit=nanometer)
    top_omm.setPeriodicBoxVectors(box)

    forcefield = ForceField(ff_xml_fp)
    positions = top.get_positions().to_openmm()
    return forcefield, top_omm, positions

for protein, n_res in protein_data:
    pdb_fp = f"{data_dir}/{protein}/{protein}.pdb"
    out_prefix = f"{out_dir}/protein/{protein}"
    if use_default_ff:
        pdb = PDBFile(pdb_fp)
        forcefield = ForceField("amber14-all.xml", "tip3p.xml")
        topology, positions = pdb.topology, pdb.positions
    else:
        forcefield, topology, positions = protein_setup(pdb_fp)

    system = forcefield.createSystem(
        topology,
        nonbondedMethod=nonbonded_method,
        nonbondedCutoff=dist_cutoff,
        constraints=constraints,
        rigidWater=rigidWater,
    )
    system.addForce(MonteCarloBarostat(pressure, temp_protein*kelvin))
    integrator = LangevinMiddleIntegrator(temp_protein*kelvin, 1/picosecond, dt)
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(DCDReporter(f"{out_prefix}.dcd", n_steps_save_protein))
    simulation.reporters.append(StateDataReporter(f"{out_prefix}.log",
                        n_steps_save_protein, step=True, potentialEnergy=True, temperature=True))

    t_start = time()
    simulation.step(n_steps_protein)
    print("protein ", protein, temp_protein, "K -",
          round(time() - t_start, 2), "s")

    chem_shift_fp = f"{out_prefix}_chem_shifts.txt"
    cmd = f"{chem_shift_path} chem_shift.py {protein} {out_dir} {n_res} > {chem_shift_fp}"
    subprocess.run(cmd, shell=True)

# This empty file is used to indicate that the simulations have completed
with open(f"{out_dir}/done.txt", "w") as of:
    of.write("")
