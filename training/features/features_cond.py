# Read molecules and record elements, formal charges, aromatic atoms,
#   n bonded atoms, bonds, angles, propers, impropers and molecule indices
# Output uses one-based indexing
# See also https://github.com/openmm/spice-models/blob/main/five-et/createSpiceDataset.py
# Also set up starting structures for condensed phase simulations
# Assumes all formal charges are zero, since charges are not spread to equivalent atoms

from openff.toolkit.topology import Molecule
from rdkit import Chem
from rdkit.Chem import AllChem
from glob import glob
import os
import subprocess
import textwrap

out_fp = "features_cond.tsv"
out_dir = "../condensed_data"
box_length = 3.0 # nm
packmol_tol = 2.0 # Å
skip_packmol = False

enth_vap_data = [
    # smiles     dens
    ["O"       , 1.0   ], # Water
    ["CC(=O)C" , 0.7845], # Acetone
    ["c1ccccc1", 0.8765], # Benzene
    ["CO"      , 0.792 ], # Methanol
]

enth_mixing_data = [
    # smiles_1     smiles_2     dens_1   dens_2
    ["C1COCCN1"  , "O"        , 0.996  , 1.0    ],
    ["CCC(C)=O"  , "Nc1ccccc1", 0.8050 , 1.0173 ],
    ["CNCCO"     , "O"        , 0.93618, 1.0    ],
    ["CN1CCCC1=O", "ClCCCl"   , 1.02823, 1.24637],
    ["CCCCO"     , "OC1=NCCC1", 0.81   , 0.99428],
    ["C=CCCCC"   , "CCCO"     , 0.66934, 0.803  ],
]

element_indices = {
    1 : 1 , 3 : 2 , 5 : 3 , 6 : 4 , 7 : 5 , 8 : 6 , # H  Li B  C  N  O
    9 : 7 , 11: 8 , 12: 9 , 14: 10, 15: 11, 16: 12, # F  Na Mg Si P  S
    17: 13, 19: 14, 20: 15, 35: 16, 53: 17,         # Cl K  Ca Br I
}

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

def repeat_inter(inds, repeat_i, n_atoms):
    mapped_inds = [i + repeat_i * n_atoms for i in inds]
    return "/".join(str(i) for i in mapped_inds)

def pad_empty_str(s):
    if len(s) == 0:
        return "-"
    else:
        return s

for d in ["monomers", "packmol_inputs", "packmol_logs", "starting_structures"]:
    for file in glob(f"{out_dir}/{d}/*"):
        os.remove(file)

with open(out_fp, "w") as of:
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

        assert mol.n_atoms > 0

        str_element = ",".join(str(element_indices[a.atomic_number])
                               for a in mol.atoms)

        str_charge = ",".join(str(float(a.formal_charge / a.formal_charge.units))
                              for a in mol.atoms)

        str_aromatic = ",".join("1" if a.is_aromatic else "0"
                                for a in mol.atoms)

        str_nbonded = ",".join(str(len(list(a.bonded_atoms)))
                               for a in mol.atoms)

        bonds = []
        for bond_set, offset in [[mol.bonds, 1]]:
            for bond in bond_set:
                bonds.append([bond.atom1_index + offset, bond.atom2_index + offset])

        angles = []
        for angle_set, offset in [[list(mol.angles), 1]]:
            for angle in angle_set:
                angles.append([angle[0].molecule_atom_index + offset,
                               angle[1].molecule_atom_index + offset,
                               angle[2].molecule_atom_index + offset])

        propers = []
        for proper_set, offset in [[list(mol.propers), 1]]:
            for proper in proper_set:
                propers.append([proper[0].molecule_atom_index + offset,
                                proper[1].molecule_atom_index + offset,
                                proper[2].molecule_atom_index + offset,
                                proper[3].molecule_atom_index + offset])

        impropers = []
        for improper_set, offset in [[list(mol.amber_impropers), 1]]:
            for improper in improper_set:
                central_i = improper[0].molecule_atom_index + offset
                other_is = sorted(a.molecule_atom_index + offset for a in improper[1:])
                joined_is = [central_i, *other_is]
                if joined_is not in impropers:
                    impropers.append(joined_is)

        str_bond     = ",".join(repeat_inter(inds, 0, mol.n_atoms) for inds in bonds)
        str_angle    = ",".join(repeat_inter(inds, 0, mol.n_atoms) for inds in angles)
        str_proper   = ",".join(repeat_inter(inds, 0, mol.n_atoms) for inds in propers)
        str_improper = ",".join(repeat_inter(inds, 0, mol.n_atoms) for inds in impropers)
        molecule_inds = [1] * mol.n_atoms
        str_molecule_inds = ",".join(str(mi) for mi in molecule_inds)

        str_out = "\t".join(pad_empty_str(s) for s in [mol_id_gas, str_element, str_charge,
                                str_aromatic, str_nbonded, str_bond, str_angle,
                                str_proper, str_improper, str_molecule_inds])
        assert len(str_out.split("\t")) == 10
        of.write(str_out + "\n")

        str_element_rep, str_charge_rep, str_aromatic_rep, str_nbonded_rep = "", "", "", ""
        str_bond_rep, str_angle_rep, str_proper_rep, str_improper_rep = "", "", "", ""
        str_molecule_inds = ""

        for repeat_i in range(n_mol):
            sep = "," if repeat_i > 0 else ""
            str_element_rep  += sep + str_element
            str_charge_rep   += sep + str_charge
            str_aromatic_rep += sep + str_aromatic
            str_nbonded_rep  += sep + str_nbonded
            str_bond_rep  += sep + ",".join(repeat_inter(inds, repeat_i, mol.n_atoms)
                                            for inds in bonds)
            str_angle_rep += sep + ",".join(repeat_inter(inds, repeat_i, mol.n_atoms)
                                            for inds in angles)
            if len(propers) > 0:
                str_proper_rep += sep + ",".join(repeat_inter(inds, repeat_i, mol.n_atoms)
                                                 for inds in propers)
            if len(impropers) > 0:
                str_improper_rep += sep + ",".join(repeat_inter(inds, repeat_i, mol.n_atoms)
                                                   for inds in impropers)
            str_molecule_inds += sep + ",".join(str(mi + repeat_i) for mi in molecule_inds)

        str_out = "\t".join(pad_empty_str(s) for s in [mol_id_liq, str_element_rep, str_charge_rep,
                                str_aromatic_rep, str_nbonded_rep, str_bond_rep, str_angle_rep,
                                str_proper_rep, str_improper_rep, str_molecule_inds])
        assert len(str_out.split("\t")) == 10
        of.write(str_out + "\n")

    for smiles_1, smiles_2, dens_1, dens_2 in enth_mixing_data:
        mol_id_1 = f"mixing_single_{smiles_1}"
        mol_id_2 = f"mixing_single_{smiles_2}"
        mol_id_com = f"mixing_combined_{smiles_1}_{smiles_2}"
        mol_1 = Molecule.from_smiles(smiles_1, allow_undefined_stereo=True,
                                     hydrogens_are_explicit=False)
        mol_2 = Molecule.from_smiles(smiles_2, allow_undefined_stereo=True,
                                     hydrogens_are_explicit=False)

        mass_1 = sum([float(a.mass / a.mass.units) for a in mol_1.atoms])
        mass_2 = sum([float(a.mass / a.mass.units) for a in mol_2.atoms])
        combined_mass = mass_1 + mass_2
        combined_density = (dens_1 + dens_2) / 2
        n_mol_1 = calc_n_molecules(mass_1, dens_1)
        n_mol_2 = calc_n_molecules(mass_2, dens_2)
        n_pairs = calc_n_molecules(combined_mass, combined_density)

        rdkit_mol_1 = Chem.MolFromSmiles(smiles_1)
        rdkit_mol_2 = Chem.MolFromSmiles(smiles_2)
        rdkit_mol_1 = Chem.AddHs(rdkit_mol_1)
        rdkit_mol_2 = Chem.AddHs(rdkit_mol_2)
        AllChem.EmbedMolecule(rdkit_mol_1)
        AllChem.EmbedMolecule(rdkit_mol_2)
        monomer_fp_mol_1 = f"{out_dir}/monomers/{smiles_1}.pdb"
        monomer_fp_mol_2 = f"{out_dir}/monomers/{smiles_2}.pdb"
        with open(monomer_fp_mol_1, "w") as ofp:
            ofp.write(rename_water_res(Chem.MolToPDBBlock(rdkit_mol_1), smiles_1))
        with open(monomer_fp_mol_2, "w") as ofp:
            ofp.write(rename_water_res(Chem.MolToPDBBlock(rdkit_mol_2), smiles_2))

        packmol_input_fp_mol_1 = f"{out_dir}/packmol_inputs/{mol_id_1}.inp"
        packmol_input_fp_mol_2 = f"{out_dir}/packmol_inputs/{mol_id_2}.inp"
        packmol_input_fp_mixture = f"{out_dir}/packmol_inputs/{mol_id_com}.inp"

        with open(packmol_input_fp_mol_1, "w") as ofp:
            ofp.write(textwrap.dedent(
                f"""\
                tolerance {packmol_tol}
                output {out_dir}/starting_structures/{mol_id_1}.pdb
                filetype pdb
                connect yes
                pbc {10 * box_length} {10 * box_length} {10 * box_length}
                structure {monomer_fp_mol_1}
                  number {n_mol_1}
                end structure
                """,
            ))

        with open(packmol_input_fp_mol_2, "w") as ofp:
            ofp.write(textwrap.dedent(
                f"""\
                tolerance {packmol_tol}
                output {out_dir}/starting_structures/{mol_id_2}.pdb
                filetype pdb
                connect yes
                pbc {10 * box_length} {10 * box_length} {10 * box_length}
                structure {monomer_fp_mol_2}
                  number {n_mol_2}
                end structure
                """,
            ))

        with open(packmol_input_fp_mixture, "w") as ofp:
            ofp.write(textwrap.dedent(
                f"""\
                tolerance {packmol_tol}
                output {out_dir}/starting_structures/{mol_id_com}.pdb
                filetype pdb
                connect yes
                pbc {10 * box_length} {10 * box_length} {10 * box_length}
                structure {monomer_fp_mol_1}
                  number {n_pairs}
                end structure
                structure {monomer_fp_mol_2}
                  number {n_pairs}
                end structure
                """,
            ))

        if not skip_packmol:
            subprocess.run(f'packmol < "{packmol_input_fp_mol_1}" > "{out_dir}/packmol_logs/{mol_id_1}.log"', shell=True)
            subprocess.run(f'packmol < "{packmol_input_fp_mol_2}" > "{out_dir}/packmol_logs/{mol_id_2}.log"', shell=True)
            subprocess.run(f'packmol < "{packmol_input_fp_mixture}" > "{out_dir}/packmol_logs/{mol_id_com}.log"', shell=True)

        assert mol_1.n_atoms > 0
        assert mol_2.n_atoms > 0

        for mol_id, mol, n_mol in [[mol_id_1, mol_1, n_mol_1], [mol_id_2, mol_2, n_mol_2]]:
            str_element = ",".join(str(element_indices[a.atomic_number])
                                   for a in mol.atoms)

            str_charge = ",".join(str(float(a.formal_charge / a.formal_charge.units))
                                  for a in mol.atoms)

            str_aromatic = ",".join("1" if a.is_aromatic else "0"
                                    for a in mol.atoms)

            str_nbonded = ",".join(str(len(list(a.bonded_atoms)))
                                   for a in mol.atoms)

            bonds = []
            for bond_set, offset in [[mol.bonds, 1]]:
                for bond in bond_set:
                    bonds.append([bond.atom1_index + offset, bond.atom2_index + offset])

            angles = []
            for angle_set, offset in [[list(mol.angles), 1]]:
                for angle in angle_set:
                    angles.append([angle[0].molecule_atom_index + offset,
                                angle[1].molecule_atom_index + offset,
                                angle[2].molecule_atom_index + offset])

            propers = []
            for proper_set, offset in [[list(mol.propers), 1]]:
                for proper in proper_set:
                    propers.append([proper[0].molecule_atom_index + offset,
                                    proper[1].molecule_atom_index + offset,
                                    proper[2].molecule_atom_index + offset,
                                    proper[3].molecule_atom_index + offset])

            impropers = []
            for improper_set, offset in [[list(mol.amber_impropers), 1]]:
                for improper in improper_set:
                    central_i = improper[0].molecule_atom_index + offset
                    other_is = sorted(a.molecule_atom_index + offset for a in improper[1:])
                    joined_is = [central_i, *other_is]
                    if joined_is not in impropers:
                        impropers.append(joined_is)

            str_element_rep, str_charge_rep, str_aromatic_rep, str_nbonded_rep = "", "", "", ""
            str_bond_rep, str_angle_rep, str_proper_rep, str_improper_rep = "", "", "", ""
            molecule_inds = [1] * mol.n_atoms
            str_molecule_inds = ""

            for repeat_i in range(n_mol):
                sep = "," if repeat_i > 0 else ""
                str_element_rep  += sep + str_element
                str_charge_rep   += sep + str_charge
                str_aromatic_rep += sep + str_aromatic
                str_nbonded_rep  += sep + str_nbonded
                str_bond_rep  += sep + ",".join(repeat_inter(inds, repeat_i, mol.n_atoms)
                                                for inds in bonds)
                str_angle_rep += sep + ",".join(repeat_inter(inds, repeat_i, mol.n_atoms)
                                                for inds in angles)
                if len(propers) > 0:
                    str_proper_rep += sep + ",".join(repeat_inter(inds, repeat_i, mol.n_atoms)
                                                     for inds in propers)
                if len(impropers) > 0:
                    str_improper_rep += sep + ",".join(repeat_inter(inds, repeat_i, mol.n_atoms)
                                                       for inds in impropers)
                str_molecule_inds += sep + ",".join(str(mi + repeat_i) for mi in molecule_inds)

            str_out = "\t".join(pad_empty_str(s) for s in [mol_id, str_element_rep, str_charge_rep,
                                    str_aromatic_rep, str_nbonded_rep, str_bond_rep, str_angle_rep,
                                    str_proper_rep, str_improper_rep, str_molecule_inds])
            assert len(str_out.split("\t")) == 10
            of.write(str_out + "\n")

        str_element = ",".join(str(element_indices[a.atomic_number])
                               for a in mol_1.atoms + mol_2.atoms)

        str_charge = ",".join(str(float(a.formal_charge / a.formal_charge.units))
                              for a in mol_1.atoms + mol_2.atoms)

        str_aromatic = ",".join("1" if a.is_aromatic else "0"
                                for a in mol_1.atoms + mol_2.atoms)

        str_nbonded = ",".join(str(len(list(a.bonded_atoms)))
                               for a in mol_1.atoms + mol_2.atoms)

        bonds = []
        for bond_set, offset in [[mol_1.bonds, 1], [mol_2.bonds, mol_1.n_atoms + 1]]:
            for bond in bond_set:
                bonds.append([bond.atom1_index + offset, bond.atom2_index + offset])

        angles = []
        for angle_set, offset in [[list(mol_1.angles), 1], [list(mol_2.angles), mol_1.n_atoms + 1]]:
            for angle in angle_set:
                angles.append([angle[0].molecule_atom_index + offset,
                               angle[1].molecule_atom_index + offset,
                               angle[2].molecule_atom_index + offset])

        propers = []
        for proper_set, offset in [[list(mol_1.propers), 1], [list(mol_2.propers), mol_1.n_atoms + 1]]:
            for proper in proper_set:
                propers.append([proper[0].molecule_atom_index + offset,
                                proper[1].molecule_atom_index + offset,
                                proper[2].molecule_atom_index + offset,
                                proper[3].molecule_atom_index + offset])

        impropers = []
        for improper_set, offset in [[list(mol_1.amber_impropers), 1], [list(mol_2.amber_impropers), mol_1.n_atoms + 1]]:
            for improper in improper_set:
                central_i = improper[0].molecule_atom_index + offset
                other_is = sorted(a.molecule_atom_index + offset for a in improper[1:])
                joined_is = [central_i, *other_is]
                if joined_is not in impropers:
                    impropers.append(joined_is)

        str_element_rep, str_charge_rep, str_aromatic_rep, str_nbonded_rep = "", "", "", ""
        str_bond_rep, str_angle_rep, str_proper_rep, str_improper_rep = "", "", "", ""
        molecule_inds = [1] * mol_1.n_atoms + [2] * mol_2.n_atoms
        str_molecule_inds = ""
        n_atoms_com = mol_1.n_atoms + mol_2.n_atoms

        for repeat_i in range(n_pairs):
            sep = "," if repeat_i > 0 else ""
            str_element_rep  += sep + str_element
            str_charge_rep   += sep + str_charge
            str_aromatic_rep += sep + str_aromatic
            str_nbonded_rep  += sep + str_nbonded
            str_bond_rep  += sep + ",".join(repeat_inter(inds, repeat_i, n_atoms_com)
                                            for inds in bonds)
            str_angle_rep += sep + ",".join(repeat_inter(inds, repeat_i, n_atoms_com)
                                            for inds in angles)
            if len(propers) > 0:
                str_proper_rep += sep + ",".join(repeat_inter(inds, repeat_i, n_atoms_com)
                                                 for inds in propers)
            if len(impropers) > 0:
                str_improper_rep += sep + ",".join(repeat_inter(inds, repeat_i, n_atoms_com)
                                                   for inds in impropers)
            str_molecule_inds += sep + ",".join(str(mi + repeat_i * 2) for mi in molecule_inds)

        str_out = "\t".join(pad_empty_str(s) for s in [mol_id_com, str_element_rep, str_charge_rep,
                                str_aromatic_rep, str_nbonded_rep, str_bond_rep, str_angle_rep,
                                str_proper_rep, str_improper_rep, str_molecule_inds])
        assert len(str_out.split("\t")) == 10
        of.write(str_out + "\n")
