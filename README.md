# Garnet force field

If you use the force field, please cite the paper:

...

## Using the force field

The model was trained in Julia as described below.
To make it easier to parameterise molecules, it has been ported to PyTorch for inference.

### Installation

Example conda/mamba commands:
```bash
conda create -n garnet python=3.12
conda activate garnet
conda install -c conda-forge pip 'openff-toolkit-base>=0.18.0' rustworkx rdkit openmm pyxdg gemmi
pip install torch torch_geometric igraph
pip install git+https://github.com/openforcefield/openff-pablo.git@v0.2.0
pip install garnetff
```

### Assigning parameters

Garnet integrates with the [OpenFF](https://docs.openforcefield.org/en/latest) and [OpenMM](https://openmm.org) software ecosystem.
The recommended route to obtaining parameters for a system is to create an [OpenFF Toolkit](https://docs.openforcefield.org/projects/toolkit/en/stable) `Topology`, then call `topology_to_openmm_system` on it.
The `Topology` can come from a PDB file of the whole system, a SMILES string or a structure file.
In the first case, using [OpenFF Pablo](https://openff-pablo.readthedocs.io/en/stable) to read the PDB file can be more flexible than using `Topology.from_pdb` (for example, it supports nucleic acids and post-translational modifications).

```python
from garnetff import garnet
from openff.pablo import topology_from_pdb
from openmm.app import PME, HBonds
from openmm.unit import nanometer

pdb_fp = "data/gb3.pdb"
topology = topology_from_pdb(pdb_fp)

system, top_openmm = garnet.topology_to_openmm_system(
    topology,
    nonbondedMethod=PME,
    nonbondedCutoff=1*nanometer,
    constraints=HBonds,
    rigidWater=True,
)

# Run an OpenMM simulation
from openmm import LangevinMiddleIntegrator
from openmm.app import Simulation, PDBFile
from openmm.unit import picosecond, kelvin

temp = 300*kelvin
dt = 0.004*picosecond
integrator = LangevinMiddleIntegrator(temp, 1/picosecond, dt)
simulation = Simulation(top_openmm, system, integrator)
pdb = PDBFile(pdb_fp)
simulation.context.setPositions(pdb.positions)

simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temp)
simulation.step(1000)
```
Disulfide bridges need to be explicitly given via `CONECT` records, unlike when reading in with OpenMM directly.
The `Topology` can also be written to an OpenMM force field XML file:
```python
garnet.topology_to_openmm_xml("gb3.xml", topology)
```
In this case each molecule is written out as a single residue.
This can cause problems for polymers when trying to set up an OpenMM system with the force field file, since OpenMM will look for separate residue entries based on the residue names in the PDB file.
Writing force field files like this is therefore only recommended for non-polymers, though OpenMM [may add support for this in future](https://github.com/openmm/openmm/issues/5178).
The optional keyword arguments `mol_names`, to specify the name of each molecule, and `prefix`, to give unique atom names to an XML file and avoid clashes with other files, are available.
However, currently only one force field XML can be loaded with `ForceField` from OpenMM [due to the way custom non-bonded forces work](https://github.com/openmm/openmm/issues/5154).

From a SMILES string:
```python
from garnetff import garnet
from openff.toolkit.topology import Molecule, Topology

smiles = "[H]O[H]"
mol = Molecule.from_smiles(smiles, hydrogens_are_explicit=True)
topology = Topology.from_molecules(molecules=[mol])

system, top_openmm = garnet.topology_to_openmm_system(topology)
```

From a structure file (see [the OpenFF toolkit docs](https://docs.openforcefield.org/projects/toolkit/en/stable/api/generated/openff.toolkit.topology.Molecule.html#openff.toolkit.topology.Molecule.from_file) for available formats):
```python
from garnetff import garnet
from openff.toolkit.topology import Molecule, Topology

mol = Molecule.from_file("data/zw_l_alanine.sdf")
topology = Topology.from_molecules(molecules=[mol])

system, top_openmm = garnet.topology_to_openmm_system(topology)
```

The default options for `topology_to_openmm_system` are `nonbondedMethod=NoCutoff`, `nonbondedCutoff=1*nanometer`, `constraints=None` and `rigidWater=False`.
It is important to select these parameters carefully as described in the [the OpenMM documentation](https://docs.openmm.org/latest/userguide/application/02_running_sims.html#simulation-parameters).
Assigning parameters is usually fast (a few seconds), but for some molecules it can take a minute or so due to graph isomorphism checks.
If you want to use Garnet with simulation software other than OpenMM, you could try conversion software such as [ParmEd](https://parmed.github.io/ParmEd/html/index.html) or [OpenFF Interchange](https://docs.openforcefield.org/projects/interchange/en/stable).

## Training

See the [training](/training/) directory for instructions on how to train the force field from scratch.
Training made use of the [Molly.jl](https://juliamolsim.github.io/Molly.jl/stable) software.

## Feedback

We are interested in how people are using Garnet and in cases where it performs well or poorly.
Do let us know by opening an issue or [via email](https://www2.mrc-lmb.cam.ac.uk/groups/greener/#contact), user requests will inform future development.
