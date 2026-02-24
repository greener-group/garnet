# Read molecules and record elements, formal charges, aromatic atoms,
#   n bonded atoms, bonds, angles, propers, impropers and molecule indices
# Output uses one-based indexing
# See also https://github.com/openmm/spice-models/blob/main/five-et/createSpiceDataset.py
# This is for the GEMS data and uses MDAnalysis to determine bonds based on coordinates
# Not all data is used due to conversion issues

import apsw
import MDAnalysis as mda
import numpy as np
from openff.toolkit.topology import Molecule
import networkx as nx
from rdkit import Chem
import os
from copy import deepcopy

out_fp = "features_gems.tsv"
pdb_fp = "gems_crambin_pdbs/crambin_xxx.pdb"
temp_fp = "temp_mda.xyz"
data_fp = "../crambin.db"
fudge_factor = 0.59

element_indices = {
    1 : 1 , 3 : 2 , 5 : 3 , 6 : 4 , 7 : 5 , 8 : 6 , # H  Li B  C  N  O
    9 : 7 , 11: 8 , 12: 9 , 14: 10, 15: 11, 16: 12, # F  Na Mg Si P  S
    17: 13, 19: 14, 20: 15, 35: 16, 53: 17,         # Cl K  Ca Br I
}

# Finds identical atoms according to the molecular graph for atoms with
#   formal charge not equal to zero, includes self-reference
def find_identical_atoms(graph, molecule_inds):
    n_atoms = len(graph)
    identical_atoms = [[] for _ in range(n_atoms)]
    for i in range(n_atoms):
        fci = graph.nodes[i]["formal_charge"]
        ei  = graph.nodes[i]["atomic_number"]
        mol_i = molecule_inds[i]
        if fci / fci.u != 0:
            identical_atoms[i].append(i)
            g1 = deepcopy(graph)
            g1.remove_node(i)
            for j in range(n_atoms):
                ej = graph.nodes[j]["atomic_number"]
                mol_j = molecule_inds[j]
                # Assume that identical atoms are separated by fewer than 5 bonds
                # Algorithm did not complete otherwise
                if i != j and ei == ej and mol_i == mol_j and nx.shortest_path_length(graph, i, j) < 5:
                    g2 = deepcopy(graph)
                    g2.remove_node(j)
                    iso = nx.vf2pp_isomorphism(g1, g2, node_label="atomic_number")
                    if iso is not None:
                        identical_atoms[i].append(j)
    return identical_atoms

atomic_number_to_element = {1: "H", 6: "C", 7: "N", 8: "O", 16: "S"}

# From read_db.py at https://zenodo.org/records/10720941
class Database:
    def __init__(self, filename):
        self.cursor = apsw.Connection(filename, flags=apsw.SQLITE_OPEN_READONLY).cursor()

    def __len__(self):
        return self.cursor.execute('''SELECT * FROM metadata WHERE id=1''').fetchone()[-1]

    def __getitem__(self, idx):
        data = self.cursor.execute('''SELECT * FROM data WHERE id='''+str(idx)).fetchone()
        return self._unpack_data_tuple(data)

    def _deblob(self, buffer, dtype, shape=None):
        array = np.frombuffer(buffer, dtype)
        if not np.little_endian:
            array = array.byteswap()
        array.shape = shape
        return np.copy(array)

    def _unpack_data_tuple(self, data):
        n = len(data[3])//4 # A single int32 is 4 bytes long.
        q = np.asarray([0.0 if data[1] is None else data[1]], dtype=np.float32)
        s = np.asarray([0.0 if data[2] is None else data[2]], dtype=np.float32)
        z = self._deblob(data[3], dtype=np.int32,   shape=(n,))
        r = self._deblob(data[4], dtype=np.float32, shape=(n, 3))
        e = np.asarray([0.0 if data[5] is None else data[5]], dtype=np.float32)
        f = self._deblob(data[6], dtype=np.float32, shape=(n, 3))
        d = self._deblob(data[7], dtype=np.float32, shape=(1, 3))
        return q, s, z, r, e, f, d
    
database = Database(data_fp)

def pad_empty_str(s):
    if len(s) == 0:
        return "-"
    else:
        return s

with open(out_fp, "w") as of:
    for mi in range(len(database)):
        mol_id = f"crambin_{mi + 1}"

        gems_total_charge, _, atomic_numbers, coords, _, _, _ = database[mi]
        gems_total_charge = int(gems_total_charge[0])
        n_atoms = coords.shape[0]
        xyz_str = f"{n_atoms}\n\n"
        for ai in range(n_atoms):
            el = atomic_number_to_element[atomic_numbers[ai]]
            xyz_str += f"{el} {coords[ai, 0]} {coords[ai, 1]} {coords[ai, 2]}\n"
        with open(temp_fp, "w") as of2:
            of2.write(xyz_str)
        # Higher fudge_factor means disulphide bridges are covalent
        # Going even higher means more hydrogens get two bonds
        u = mda.Universe(temp_fp, guess_bonds=True, fudge_factor=fudge_factor)
        os.remove(temp_fp)
        try:
            rdkit_mol = u.atoms.convert_to.rdkit()
        except Chem.AtomValenceException:
            error_str = "valence issue on rdkit conversion"
            of.write(f"{mol_id}\t{error_str}\n")
            print(mi + 1, "/", len(database), "-", mol_id, "-", error_str)
            continue
        Chem.MolToPDBFile(rdkit_mol, pdb_fp.replace("xxx", str(mi + 1)))

        mol = Molecule.from_rdkit(rdkit_mol)
        graph = mol.to_networkx()

        assert mol.n_atoms > 0 and mol.n_atoms == len(u.atoms)
        assert len(mol.bonds) == len(u.bonds)

        molecule_inds = [0] * mol.n_atoms
        for ci, cc in enumerate(nx.connected_components(graph)):
            for ai in cc:
                molecule_inds[ai] = ci + 1
        assert 0 not in molecule_inds

        identical_atoms = find_identical_atoms(graph, molecule_inds)
        formal_charges = [float(a.formal_charge / a.formal_charge.units) for a in mol.atoms]
        if sum(formal_charges) != gems_total_charge:
            error_str = f"GEMS total charge is {gems_total_charge} but rdkit total charge is {sum(formal_charges)}"
            of.write(f"{mol_id}\t{error_str}\n")
            print(mi + 1, "/", len(database), "-", mol_id, "-", error_str)
            continue

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

        str_molecule_inds = ",".join(str(mol_i) for mol_i in molecule_inds)

        str_out = "\t".join(pad_empty_str(s) for s in [mol_id, str_element, str_charge,
                        str_aromatic, str_nbonded, str_bond, str_angle,
                        str_proper, str_improper, str_molecule_inds])
        assert len(str_out.split("\t")) == 10
        of.write(str_out + "\n")
        print(mi + 1, "/", len(database), "-", mol_id, "- fine")
