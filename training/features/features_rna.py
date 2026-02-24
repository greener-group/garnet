# Read molecules and record elements, formal charges, aromatic atoms,
#   n bonded atoms, bonds, angles, propers, impropers and molecule indices
# Output uses one-based indexing
# See also https://github.com/openmm/spice-models/blob/main/five-et/createSpiceDataset.py
# This is for the Takaba2024 RNA data from https://zenodo.org/records/8148817

import h5py
from openff.toolkit.topology import Molecule
import networkx as nx
from copy import deepcopy

out_fp = "features_rna.tsv"
out_fp_label = "rna_label_to_key.tsv"

hdf5_files_to_labels = {
    "../RNA-DIVERSE-OPENFF-DEFAULT.hdf5"      : "rna_diverse",
    "../RNA-NUCLEOSIDE-OPENFF-DEFAULT.hdf5"   : "rna_nucleoside",
    "../RNA-TRINUCLEOTIDE-OPENFF-DEFAULT.hdf5": "rna_trinucleotide",
}

element_indices = {
    1 : 1 , 3 : 2 , 5 : 3 , 6 : 4 , 7 : 5 , 8 : 6 , # H  Li B  C  N  O
    9 : 7 , 11: 8 , 12: 9 , 14: 10, 15: 11, 16: 12, # F  Na Mg Si P  S
    17: 13, 19: 14, 20: 15, 35: 16, 53: 17,         # Cl K  Ca Br I
}

# Finds identical atoms according to the molecular graph for atoms with
#   formal charge not equal to zero, includes self-reference
def find_identical_atoms(graph):
    n_atoms = len(graph)
    identical_atoms = [[] for _ in range(n_atoms)]
    for i in range(n_atoms):
        fci = graph.nodes[i]["formal_charge"]
        ei  = graph.nodes[i]["atomic_number"]
        if fci / fci.u != 0:
            identical_atoms[i].append(i)
            g1 = deepcopy(graph)
            g1.remove_node(i)
            for j in range(n_atoms):
                ej = graph.nodes[j]["atomic_number"]
                if i != j and ei == ej:
                    g2 = deepcopy(graph)
                    g2.remove_node(j)
                    iso = nx.vf2pp_isomorphism(g1, g2, node_label="atomic_number")
                    if iso is not None:
                        identical_atoms[i].append(j)
    return identical_atoms

def pad_empty_str(s):
    if len(s) == 0:
        return "-"
    else:
        return s

with open(out_fp, "w") as of, open(out_fp_label, "w") as of2:
    for hdf5_fp in hdf5_files_to_labels:
        f = h5py.File(hdf5_fp, "r")
        for mi, mol_id in enumerate(sorted(f.keys())):
            mol_id_clean = f"{hdf5_files_to_labels[hdf5_fp]}_{mi + 1}"
            print(mi + 1, "/", len(f), "-", mol_id_clean)
            of2.write(f"{mol_id_clean}\t{mol_id}\n")

            smiles = f[mol_id]["smiles"][0]
            mol = Molecule.from_mapped_smiles(smiles, allow_undefined_stereo=True)
            graph = mol.to_networkx()

            assert mol.n_atoms > 0
            assert [a.atomic_number for a in mol.atoms] == list(f[mol_id]["atomic_numbers"])

            identical_atoms = find_identical_atoms(graph)
            formal_charges = [float(a.formal_charge / a.formal_charge.units) for a in mol.atoms]
            spread_charges = [0.0] * mol.n_atoms
            for i in range(mol.n_atoms):
                if len(identical_atoms[i]) > 0:
                    for j in identical_atoms[i]:
                        spread_charges[j] += formal_charges[i] / len(identical_atoms[i])

            str_element = ",".join(str(element_indices[a.atomic_number]) for a in mol.atoms)

            str_charge = ",".join(str(round(c, 4)) for c in spread_charges)

            str_aromatic = ",".join("1" if a.is_aromatic else "0" for a in mol.atoms)

            str_nbonded = ",".join(str(len(list(a.bonded_atoms))) for a in mol.atoms)

            str_bond = ""
            for bond in mol.bonds:
                if len(str_bond) > 0:
                    str_bond += ","
                str_bond += "/".join(str(i + 1) for i in [bond.atom1_index, bond.atom2_index])

            str_angle = ""
            for angle in list(mol.angles):
                if len(str_angle) > 0:
                    str_angle += ","
                str_angle += "/".join(str(i + 1) for i in [angle[0].molecule_atom_index,
                                        angle[1].molecule_atom_index, angle[2].molecule_atom_index])

            str_proper = ""
            for proper in list(mol.propers):
                if len(str_proper) > 0:
                    str_proper += ","
                str_proper += "/".join(str(i + 1) for i in [proper[0].molecule_atom_index,
                                        proper[1].molecule_atom_index, proper[2].molecule_atom_index,
                                        proper[3].molecule_atom_index])

            impropers = []
            for improper in list(mol.amber_impropers):
                central_i = improper[0].molecule_atom_index
                other_is = sorted(a.molecule_atom_index for a in improper[1:])
                joined_is = "/".join(str(i + 1) for i in [central_i, *other_is])
                if joined_is not in impropers:
                    impropers.append(joined_is)
            str_improper = ",".join(impropers)

            molecule_inds = [0] * mol.n_atoms
            for ci, cc in enumerate(nx.connected_components(graph)):
                for ai in cc:
                    molecule_inds[ai] = ci + 1
            assert 0 not in molecule_inds
            str_molecule_inds = ",".join(str(mol_i) for mol_i in molecule_inds)

            str_out = "\t".join(pad_empty_str(s) for s in [mol_id_clean, str_element, str_charge,
                            str_aromatic, str_nbonded, str_bond, str_angle,
                            str_proper, str_improper, str_molecule_inds])
            assert len(str_out.split("\t")) == 10
            of.write(str_out + "\n")
