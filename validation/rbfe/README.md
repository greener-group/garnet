# Relative binding free energy prediction with Garnet

## Background

This repository contains scripts and data used to evaluate Garnet for relative binding free energy (RBFE) estimation with alchemical free energy calculations. 

Below, we provide instructions for running RBFE calculations with Garnet using software from Open Free Energy (OpenFE), which we adapted to be compatible with Garnet force field terms and parameters. Our adaptions follows OpenFE’s RBFE protocol closely, but substitutes the Lennard-Jones potential and the Gapsys soft-core potential with a double exponential potential and a related soft-core potential. 

OpenFE’s RFBE protocol can be run with our Garnet force field by following the few steps below and requires minimal user intervention, aside from protein and ligand structure preparation. OpenFE have shown that their default RFBE protocol is robust and competitive, and we have shown that Garnet works well with OpenFE’s protocol. 

We refer to the OpenFE documentation [1] and Industry Benchmarking Project paper and GitHub for details on OpenFE [2,3] and to our paper [4] for further description and benchmarking results of Garnet. 

## Directory content

Python scripts for running the calculations described below can be found in the `scripts/` directory, and input data used for our benchmarking can be found in the `data/` directory. We note that:
- `plan_rbfe_network.py` was copied from the OpenFE Industry Benchmarking Project Github [2,3] and modified to be compatible with Garnet. 
- Protein, ligand and cofactor structure files in `data/` were copied from the OpenFE Industry Benchmarking Project Github [2,3] and modified to be compatible with Garnet. 

Additionally, we provide Garnet RBFE predictions in `results/` in the form of:
- ΔΔG predictions for edges in the ligand transformation networks (`garnet_calc_edges_ddg.csv`).
- ΔΔG predictions for all ligand pairs in each ligand transformation series (`garnet_calc_all_pairwise_ddg.csv`).
- ΔG predictions estimated using ΔΔG results for edges in transformation networks (`garnet_calc_dg.csv`).

RBFE benchmarking plots shown in our paper can be reproduced with the Jupyter notebook in `results/garnet_openfe_fep_plus.ipynb`.

## Installation

See instructions on how to install our `openfe` fork at https://github.com/greener-group/openfe. This installation will create the `openfe` conda environment. 

You will also need to install the environment `garnet_rbfe` as follows:
```
conda create -n garnet_rbfe python=3.12
conda activate garnet_rbfe
conda install -c conda-forge pip 'openff-toolkit-base>=0.18.0' rustworkx rdkit openmm pyxdg gemmi
pip install torch torch_geometric igraph
pip install git+https://github.com/openforcefield/openff-pablo.git@v0.2.0
pip install garnetff
conda install -c conda-forge click
```

## Instructions (short)

To apply the Garnet force field within the OpenFE framework, the following steps are required:

**1. Prepare structure and Garnet force field parameter files**
```
conda activate garnet_rbfe
python write_xml_and_pdb.py --pdb protein.pdb --ligands ligands.sdf --cofactors cofactors.sdf --output output_dir
```

(The cofactors flag is optional). 

**2. Plan ligand transformation network and simulations with OpenFE tools**
```
conda activate openfe
python plan_rbfe_network.py --ligands ligands_sdf --pdb protein.pdb --output output_dir --ff_path garnet.xml --interpolate_14_exceptions
```
Will create directory with information about the ligand transformation network and the individual transformations that need to be run.

**3. Run and analyse simulations with OpenFE tools**
```
conda activate openfe
openfe quickrun transformation.json -o output.json -d output_dir
openfe gather results/  --report ddg -o results_ddg.tsv
openfe gather results/ --report dg -o results_dg.tsv
```
The transformation.json file will be in the output directory specified in the previous step.

`quickrun` and `gather` are OpenFE's command line options to run and analyse RBFE calculations. Runs started with `quickrun` must be completed before `gather` is used. The raw output of the RBFE calculations should be in the `results/` directory. 

These steps are described in detail in OpenFE’s documentation.

## Instructions (detailed)

**1. Prepare structure and Garnet force field parameter files**

Running OpenFE’s RBFE protocol with Garnet parameters requires a pdb file with protein coordinates and an sdf file specifying coordinates of all ligands in the ligand series of interest. Optionally, a second sdf file with co-factor coordinates can be provided. Ligand structures are required to be pre-aligned in the ligand binding site. OpenFE’s protocol automatically prepares systems prior to simulation by assembling the input components into a system. The system is automatically solvated, and ions are added to reach a specified ionic strength. 

Garnet force field parameters must be generated for all components of the system, which means that Garnet must be run on the input pdb and sdf files. Additionally, the input pdb file needs to be reformatted to represent each protein chain as a single residue and to specify CONECT records for all bonds. This is necessary to match the generated force field parameters to the correct protein atoms during an OpenMM-based system creation. Force field parameters and reformatting of the pdb file are performed with

```
conda activate garnet_rbfe
python write_xml_and_pdb.py --pdb input_protein.pdb --ligands ligands.sdf --cofactors cofactors.sdf --output output_dir
```
The cofactors flag is optional. The script should output garnet.xml and protein.pdb. The script works for pdb files that contain protein(s) and crystallographic water molecules. Multiple protein chains can exist in the pdb file. Note that the input pdb file must be compatible with the function `topology_from_pdb` from OpenFF Pablo. 

**2. Plan ligand transformation network and simulations with OpenFE tools**

It is now necessary to define the ligand transformations for which simulations should be run. In OpenFE, the ligand transformation network is automatically planned. To plan the network and provide OpenFE with instructions on how to run simulations, run

```
conda activate openfe
python plan_rbfe_network.py --ligands ligands_sdf --pdb protein.pdb --output output_dir --ff_path garnet.xml --interpolate_14_exceptions
```

The script should produce a transformation.json file for all ligand transformations in the ligand transformation network. The script was adapted from a planning script from the OpenFE Industry Benchmarking Project GitHub [2,3]. 

If you wish to run the RBFE protocol with non-default simulation or alchemical settings, the function get_settings (or get_settings_charge_changes) can be modified. We refer to OpenFE [1] for an overview of possible settings. 

If you do not modify the script, RBFE calculations will be run with OpenFE v1.8.0 default settings, except Garnet force field parameters will be used and 1-4 electrostatic interactions will be linearly interpolated between alchemical (lambda) states. Simulations will run on the GPU in replicates of 1. 

Note that transformations involving net charge changes will be run for longer and using more lambda windows than other transformations. 

**3. Run and analyse simulations with OpenFE tools**

The OpenFE command line interface can now be used to first execute and then analyse simulations. As described by OpenFE, execution is performed with:

```
conda activate openfe
openfe quickrun transformation.json -o output.json -d output_dir
```
The transformation.json file will be in the output directory specified in the previous step.

Once simulations are done running, output data can be analysed with:

```
conda activate openfe
openfe gather results/  --report ddg -o results_ddg.tsv
openfe gather results/ --report dg -o results_dg.tsv
```

`quickrun` and `gather` are OpenFE's command line options to run and analyse RBFE calculations. The raw output of the RBFE calculations should be in the `results` directory. 

These steps are described in details in OpenFE’s documentation.

## Caveats

- We have noticed that the setup described above does not work well for transformations that involve ligand net charge changes, as described in our paper [4]. 
- For the first step outlined above, pdb files are read with the function `topology_from_pdb` from OpenFF Pablo. This requires input pdb files to follow certain naming rules, and it is possible that your input file needs to be modified slightly, i.e. according to the potential error message, before Garnet parameterisation can be performed.
- CONECT records for disulfide bonds must be explicitly defined in the input pdb file. 
- It is important that the force field file is called `garnet.xml`, as this will make our modified version of OpenFE run with the "correct" functional forms, i.e. the double exponential potential and its soft-core potential, and Garnet parameters.

## References

1. OpenFE documentation: https://docs.openfree.energy/en/latest/

2. OpenFE Industry Benchmarking Project paper: Baumann, Horton, Henry, et al. Large-scale collaborative assessment of binding free energy calculations for drug discovery using OpenFE. ChemRxiv. 18 December 2025. DOI: https://doi.org/10.26434/chemrxiv-2025-7sthd

3. OpenFE Industry Benchmarking Project GitHub: OpenFE Industry Benchmarking Project Github: https://github.com/OpenFreeEnergy/IndustryBenchmarks2024

4. Garnet paper: Blanco-González, Schulze, Rovers, Greener. Training a force field for proteins and small molecules from scratch. arXiv. 2026. DOI: https://doi.org/10.48550/arXiv.2603.16770
