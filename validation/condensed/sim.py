# Run simulations to calculate vapourisation enthalpy
# Argument is output directory and force field name
# Run with conda activate garnet
#   Install garnet following instructions on garnet GitHub, 
#   and then install openmmforcefields in the env as well

from openmm.app import *
from openmm import *
from openmm.unit import *
from openmmforcefields.generators import GAFFTemplateGenerator
from openmmforcefields.generators import SMIRNOFFTemplateGenerator
from openff.toolkit import Topology
from openff.toolkit.topology import Molecule
from garnetff import garnet
import os
import sys
from time import time
import csv

out_dir = sys.argv[1] # e.g. simulations/gaff
ff_name = sys.argv[2] # should be "gaff" or "openff" or "garnet"

# Simulation setup
n_steps = 100000000 # 200 ns 
n_steps_save = 5000 # 10 ps
dt = 0.002*picoseconds
dist_cutoff = 1*nanometer
nonbonded_method = PME
pressure = 1*bar
constraints = HBonds # None, HBonds or HAngles
rigidWater = True
data_dir = "condensed_data"
platform = Platform.getPlatformByName("CUDA")
platform_gas = Platform.getPlatformByName("Reference")

# Read smiles and Hvap temperature for benchmark molecules
# Simulations will be run at Hvap exp temperature
enth_vap_data = []
with open("condensed_data/exp_data/density_hvap_benchmark.csv", newline="") as f:
    reader = csv.DictReader(f, delimiter=",")
    for row in reader:
        smiles = row["SMILES"].strip()
        temp = float(row["T_vap_C"]) + 273.15 # temp in K
        if smiles:
            enth_vap_data.append([smiles, temp])

if os.path.isdir(out_dir):
    raise Exception(f"output directory {out_dir} already exists")

os.mkdir(out_dir)
for d in ["vapourisation_gas", "vapourisation_liquid"]:
    os.mkdir(os.path.join(out_dir, d))

def generate_forcefield(smiles_list, ff_name):
    # Use OpenFF or GAFF to get small molecule parameters, except use TIP3P for water
    molecules = []
    for smiles in smiles_list:
        if smiles != "O":
            molecules.append(Molecule.from_smiles(smiles))
    forcefield = ForceField("tip3p.xml")
    if len(molecules) > 0:
        if ff_name == "gaff":
            gaff = GAFFTemplateGenerator(molecules=molecules)
            forcefield.registerTemplateGenerator(gaff.generator)
        elif ff_name == "openff":
            smirnoff = SMIRNOFFTemplateGenerator(molecules=molecules, forcefield="openff-2.2.1")
            forcefield.registerTemplateGenerator(smirnoff.generator)
        else:
            raise ValueError("Small molecule force field name must be 'gaff' or 'openff' (or 'garnet')") 
    return forcefield

def system_setup(pdb_fp, smiles, ff_name, gas_phase=False):

    # Read PDB with OpenMM
    pdb = PDBFile(pdb_fp)

    # Set nonbonded method
    nonbonded = NoCutoff if gas_phase else nonbonded_method
    
    # System setup with TemplateGenerator
    if ff_name in ["gaff","openff"]:

        # Generate FF
        forcefield = generate_forcefield([smiles], ff_name)
        
        # Create system
        topology = pdb.topology
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=nonbonded, 
            nonbondedCutoff=dist_cutoff,
            constraints=constraints,
            rigidWater=rigidWater,
        )
    
    elif ff_name=="garnet":

        # Calculate Garnet parameters and create system
        # Must be an OpenFF topology 
        molecule_openff = Molecule.from_smiles(smiles)
        topology_openff = Topology.from_openmm(
            pdb.topology,
            unique_molecules=[molecule_openff],
        )

        system, topology = garnet.topology_to_openmm_system(
            topology_openff,
            nonbondedMethod=nonbonded, 
            nonbondedCutoff=dist_cutoff,
            constraints=constraints,
            rigidWater=rigidWater,
        )
    
    else:
        raise ValueError("Small molecule force field name must be 'gaff', 'openff' or 'garnet'") 
    
    return system, topology, pdb

for smiles, temp_vap in enth_vap_data:

    # Run simulations of liquid phase

    pdb_fp = f"{data_dir}/starting_structures/vapourisation_liquid_{smiles}.pdb"
    out_prefix = f"{out_dir}/vapourisation_liquid/{smiles}_{temp_vap}K"

    system, topology, pdb = system_setup(pdb_fp, smiles, ff_name, gas_phase=False)

    system.addForce(MonteCarloBarostat(pressure, temp_vap*kelvin))
    integrator = LangevinMiddleIntegrator(temp_vap*kelvin, 1/picosecond, dt)
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(DCDReporter(f"{out_prefix}.dcd", n_steps_save))
    simulation.reporters.append(
        StateDataReporter(
            f"{out_prefix}.log",
            n_steps_save, step=True, potentialEnergy=True, temperature=True, density=True,
        )
    )

    t_start = time()
    simulation.step(n_steps)
    print("vapourisation_liquid ", smiles, temp_vap, "K -", round(time() - t_start, 2), "s")

    # Run simulations of gas phase

    pdb_fp = f"{data_dir}/monomers/{smiles}.pdb"
    out_prefix = f"{out_dir}/vapourisation_gas/{smiles}_{temp_vap}K"

    system, topology, pdb = system_setup(pdb_fp, smiles, ff_name, gas_phase=True)

    integrator = LangevinMiddleIntegrator(temp_vap*kelvin, 1/picosecond, dt)
    simulation = Simulation(topology, system, integrator, platform_gas)
    simulation.context.setPositions(pdb.positions)
    simulation.minimizeEnergy()
    simulation.reporters.append(DCDReporter(f"{out_prefix}.dcd", n_steps_save))
    simulation.reporters.append(
        StateDataReporter(
            f"{out_prefix}.log",
            n_steps_save, step=True, potentialEnergy=True, temperature=True,
        )
    )

    t_start = time()
    simulation.step(n_steps)
    print("vapourisation_gas ", smiles, temp_vap, "K -", round(time() - t_start, 2), "s")

# This empty file is used to indicate that the simulations have completed
with open(f"{out_dir}/done.txt", "w") as of:
    of.write("")
