# Set up starting structures for condensed phase simulations
# Run from openmm82 conda env, openff-toolkit v0.18.0, Packmol 21.2.3, rdkit 2024.03.5

from openff.toolkit.topology import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem
from glob import glob
import os
import subprocess
import textwrap
import csv

out_dir = "condensed_data"
box_length = 3.0 # nm
packmol_tol = 2.0 # Å
skip_packmol = False

# Read experimental densities
enth_vap_data = []
with open("condensed_data/exp_data/density_hvap_benchmark.csv", newline="") as f:
    reader = csv.DictReader(f, delimiter=",")
    for row in reader:
        smiles = row["SMILES"].strip()
        density = float(row["density_exp"])
        if smiles:
            enth_vap_data.append([smiles, density])

def calc_n_molecules(mass, density):
    box_m3 = (box_length * 1e-9) ** 3
    box_ml = 1e6 * box_m3
    box_g = density * box_ml
    n_moles = box_g / mass
    return int(round(n_moles * 6.02214076e23))

# Water needs the residue name HOH to be kept rigid in OpenMM
def rename_water_res(pdb_str, smiles):
    if smiles == "O":
        return pdb_str.replace("UNL", "HOH")
    else:
        return pdb_str

for d in ["monomers", "packmol_inputs", "packmol_logs", "starting_structures"]:
    for file in glob(f"{out_dir}/{d}/*"):
        os.remove(file)

for smiles, dens in enth_vap_data:

    mol_id_gas = f"vapourisation_gas_{smiles}"
    mol_id_liq = f"vapourisation_liquid_{smiles}"
    mol = Molecule.from_smiles(smiles, allow_undefined_stereo=True,
                                hydrogens_are_explicit=False)

    mass = sum([float(a.mass / a.mass.units) for a in mol.atoms])
    n_mol = calc_n_molecules(mass, dens)

    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    AllChem.EmbedMolecule(rdkit_mol)
    monomer_fp = f"{out_dir}/monomers/{smiles}.pdb"
    with open(monomer_fp, "w") as ofp:
        ofp.write(rename_water_res(Chem.MolToPDBBlock(rdkit_mol), smiles))

    packmol_input_fp = f"{out_dir}/packmol_inputs/{mol_id_liq}.inp"
    with open(packmol_input_fp, "w") as ofp:
        ofp.write(textwrap.dedent(
            f"""\
            tolerance {packmol_tol}
            output {out_dir}/starting_structures/{mol_id_liq}.pdb
            filetype pdb
            connect yes
            pbc {10 * box_length} {10 * box_length} {10 * box_length}
            structure {monomer_fp}
                number {n_mol}
            end structure
            """,
        ))

    if not skip_packmol:
        subprocess.run(f'packmol < "{packmol_input_fp}" > "{out_dir}/packmol_logs/{mol_id_liq}.log"', shell=True)