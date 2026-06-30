import h5py
import numpy as np

from rdkit          import Chem
from rdkit.Chem     import rdMolAlign
from rdkit.Geometry import Point3D


def _mol_from_smiles_match_natoms(smiles: str, n_atoms: int) -> Chem.Mol:
    """
    Build an RDKit Mol whose atom count matches n_atoms.
    Tries AddHs first (common when coords include H), then without.
    """
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError(f"RDKit failed to parse SMILES: {smiles!r}")

    mol_h = Chem.AddHs(base, addCoords=False)
    if mol_h.GetNumAtoms() == n_atoms:
        return mol_h

    if base.GetNumAtoms() == n_atoms:
        return base

    raise ValueError(
        f"Atom-count mismatch for SMILES {smiles!r}: "
        f"RDKit(AddHs)={mol_h.GetNumAtoms()}, RDKit(noHs)={base.GetNumAtoms()}, expected={n_atoms}. "
        "This usually means your stored coords use a different atom ordering/explicit-H convention than SMILES."
    )


def _add_conformer_from_coords(mol: Chem.Mol, coords: np.ndarray) -> int:
    """
    Add a conformer to mol from coords (shape (N,3)), return conformer id.
    """
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f"coords must have shape (N,3); got {coords.shape}")
    if coords.shape[0] != mol.GetNumAtoms():
        raise ValueError(
            f"coords atom count {coords.shape[0]} != mol atom count {mol.GetNumAtoms()}"
        )

    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, z = coords[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return mol.AddConformer(conf, assignId=True)


def _pdb_single_model_two_chains(
    mol: Chem.Mol,
    conf_id_qm: int,
    conf_id_ff: int,
    chain_qm: str = "A",
    chain_ff: str = "B",
    remark_qm: str = "QM_MIN",
    remark_ff: str = "FF_MIN",
) -> str:
    """
    Single PDB MODEL containing two copies of the same molecule as two chains,
    using two different conformers.
    """
    pdb_qm = Chem.MolToPDBBlock(mol, confId=conf_id_qm)
    pdb_ff = Chem.MolToPDBBlock(mol, confId=conf_id_ff)

    # Keep only coordinate-bearing records; drop CONECT/END/etc to avoid duplication
    def atom_lines(pdb_block: str) -> list[str]:
        out = []
        for ln in pdb_block.splitlines():
            if ln.startswith(("ATOM  ", "HETATM")):
                out.append(ln)
        return out

    qm_lines = atom_lines(pdb_qm)
    ff_lines = atom_lines(pdb_ff)

    # Set chain IDs and renumber atom serials so the combined file is valid-ish
    def set_chain_and_renumber(lines: list[str], chain_id: str, start_serial: int) -> list[str]:
        out = []
        serial = start_serial
        for ln in lines:
            # Atom serial is columns 7-11 (1-based); chain is column 22
            ln = f"{ln[:6]}{serial:5d}{ln[11:]}"
            if len(ln) >= 22:
                ln = ln[:21] + chain_id[0] + ln[22:]
            out.append(ln)
            serial += 1
        return out

    qm_lines = set_chain_and_renumber(qm_lines, chain_qm, 1)
    ff_lines = set_chain_and_renumber(ff_lines, chain_ff, 1 + len(qm_lines))

    # Optional: shift residue numbers for the second chain so some viewers separate them cleanly
    def bump_resseq(lines: list[str], delta: int) -> list[str]:
        out = []
        for ln in lines:
            # resSeq is columns 23-26 (1-based) => indices 22:26
            if len(ln) >= 26:
                try:
                    resseq = int(ln[22:26])
                    ln = ln[:22] + f"{resseq + delta:4d}" + ln[26:]
                except ValueError:
                    pass
            out.append(ln)
        return out

    ff_lines = bump_resseq(ff_lines, 1000)

    header = [
        f"REMARK   1 {remark_qm} (chain {chain_qm})",
        f"REMARK   1 {remark_ff} (chain {chain_ff})",
        "MODEL        1",
    ]
    footer = ["ENDMDL", "END"]

    return "\n".join(header + qm_lines + ff_lines + footer) + "\n"


class BenchmarkResults:
    """
    Compact container for benchmark results, with efficient HDF5 I/O.

    Coordinates are stored in ragged form:
        coords_qm  : (total_atoms, 3)
        coords_min : (total_atoms, 3)
        n_atoms    : (n_mols,)
        offsets    : (n_mols+1,)
    """
    def __init__(self, backend: str, dataset_name: str):
        self.backend = backend
        self.dataset_name = dataset_name

        self.names   = []
        self.smiles  = []
        self.n_atoms = []

        self._coords_qm_chunks  = []
        self._coords_min_chunks = []
        self._partial_charge_chunks = []

        self.rmsd_cart      = []
        self.rmsd_bonds     = []
        self.rmsd_angles    = []
        self.rmsd_propers   = []
        self.rmsd_impropers = []
        self.tfd            = []

        self.energy_qm      = []
        self.energy_initial = []
        self.energy_min     = []

        self.dipole_qm = []
        self.dipole_min = []
        self.dipole_tensor_qm = []
        self.dipole_tensor_min = []
        self.inertia_moments_qm = []
        self.inertia_moments_min = []
        self.total_charge = []

    def add(self, record: dict):
        """
        Add one molecule result as returned by process_single_molecule.
        """
        name   = record["name"]
        smiles = record["smiles"]

        coords_qm  = record["coords"]["qm"]
        coords_min = record["coords"]["min"]

        n = coords_qm.shape[0]
        assert coords_qm.shape == coords_min.shape == (n, 3)

        self.names.append(name)
        self.smiles.append(smiles)
        self.n_atoms.append(n)

        self._coords_qm_chunks.append(coords_qm.astype(np.float32, copy=False))
        self._coords_min_chunks.append(coords_min.astype(np.float32, copy=False))
        self._partial_charge_chunks.append(
            record["dipoles"]["partial_charges"].astype(np.float32, copy=False)
        )

        self.rmsd_cart.append(float(record["rmsd_cart"]))
        self.rmsd_bonds.append(float(record["rmsd_bonds"]))
        self.rmsd_angles.append(float(record["rmsd_angles"]))
        self.rmsd_propers.append(float(record["rmsd_propers"]))
        self.rmsd_impropers.append(float(record["rmsd_impropers"]))
        self.tfd.append(float(record["tfd"]))

        self.energy_qm.append(float(record["energy_qm"]))
        self.energy_initial.append(float(record["energy_initial"]))
        self.energy_min.append(float(record["energy_min"]))

        dipoles = record["dipoles"]
        self.dipole_qm.append(dipoles["qm"]["dipole"])
        self.dipole_min.append(dipoles["min"]["dipole"])
        self.dipole_tensor_qm.append(dipoles["qm"]["tensor"])
        self.dipole_tensor_min.append(dipoles["min"]["tensor"])
        self.inertia_moments_qm.append(dipoles["qm"]["inertia_moments"])
        self.inertia_moments_min.append(dipoles["min"]["inertia_moments"])
        self.total_charge.append(float(dipoles["total_charge"]))

    def _finalize_coords(self):
        """
        Build concatenated coordinate arrays and offsets.
        """
        if not self._coords_qm_chunks:
            self.coords_qm  = np.zeros((0, 3), dtype=np.float32)
            self.coords_min = np.zeros((0, 3), dtype=np.float32)
            self.partial_charges = np.zeros(0, dtype=np.float32)
            self.offsets    = np.zeros(1, dtype=np.int64)
            self.n_atoms    = np.zeros(0, dtype=np.int64)
            return

        self.n_atoms = np.asarray(self.n_atoms, dtype=np.int64)
        self.coords_qm  = np.vstack(self._coords_qm_chunks)
        self.coords_min = np.vstack(self._coords_min_chunks)
        self.partial_charges = np.concatenate(self._partial_charge_chunks)

        self.offsets = np.zeros(len(self.n_atoms) + 1, dtype=np.int64)
        self.offsets[1:] = np.cumsum(self.n_atoms)

    def to_hdf5(self, filename: str, compression: str = "gzip"):
        """
        Save all results to an HDF5 file for later postprocessing.
        """
        self._finalize_coords()

        names  = np.asarray(self.names,  dtype=h5py.string_dtype(encoding="utf-8"))
        smiles = np.asarray(self.smiles, dtype=h5py.string_dtype(encoding="utf-8"))

        rmsd_cart      = np.asarray(self.rmsd_cart,      dtype=np.float32)
        rmsd_bonds     = np.asarray(self.rmsd_bonds,     dtype=np.float32)
        rmsd_angles    = np.asarray(self.rmsd_angles,    dtype=np.float32)
        rmsd_propers   = np.asarray(self.rmsd_propers,   dtype=np.float32)
        rmsd_impropers = np.asarray(self.rmsd_impropers, dtype=np.float32)
        tfd            = np.asarray(self.tfd,            dtype=np.float32)

        energy_qm      = np.asarray(self.energy_qm,      dtype=np.float32)
        energy_initial = np.asarray(self.energy_initial, dtype=np.float32)
        energy_min     = np.asarray(self.energy_min,     dtype=np.float32)

        dipole_qm = np.asarray(self.dipole_qm, dtype=np.float32)
        dipole_min = np.asarray(self.dipole_min, dtype=np.float32)
        dipole_tensor_qm = np.asarray(self.dipole_tensor_qm, dtype=np.float32)
        dipole_tensor_min = np.asarray(self.dipole_tensor_min, dtype=np.float32)
        inertia_moments_qm = np.asarray(self.inertia_moments_qm, dtype=np.float32)
        inertia_moments_min = np.asarray(self.inertia_moments_min, dtype=np.float32)
        total_charge = np.asarray(self.total_charge, dtype=np.float32)

        with h5py.File(filename, "w") as f:
            meta = f.create_group("meta")
            meta.attrs["backend"]      = self.backend
            meta.attrs["dataset_name"] = self.dataset_name
            meta.attrs["dipole_units"] = "elementary_charge * angstrom"
            meta.attrs["dipole_origin"] = "center_of_mass"
            meta.attrs["dipole_frame"] = "right-handed principal_inertia_axes"
            meta.attrs["dipole_tensor"] = "outer(dipole, dipole)"

            f.create_dataset("names",  data=names)
            f.create_dataset("smiles", data=smiles)

            f.create_dataset("n_atoms", data=self.n_atoms)
            f.create_dataset("offsets", data=self.offsets)

            f.create_dataset("coords_qm",  data=self.coords_qm,
                             compression=compression)
            f.create_dataset("coords_min", data=self.coords_min,
                             compression=compression)
            f.create_dataset("partial_charges", data=self.partial_charges,
                             compression=compression)

            f.create_dataset("rmsd_cart",      data=rmsd_cart)
            f.create_dataset("rmsd_bonds",     data=rmsd_bonds)
            f.create_dataset("rmsd_angles",    data=rmsd_angles)
            f.create_dataset("rmsd_propers",   data=rmsd_propers)
            f.create_dataset("rmsd_impropers", data=rmsd_impropers)
            f.create_dataset("tfd",            data=tfd)

            f.create_dataset("energy_qm",      data=energy_qm)
            f.create_dataset("energy_initial", data=energy_initial)
            f.create_dataset("energy_min",     data=energy_min)

            f.create_dataset("dipole_qm", data=dipole_qm)
            f.create_dataset("dipole_min", data=dipole_min)
            f.create_dataset("dipole_tensor_qm", data=dipole_tensor_qm)
            f.create_dataset("dipole_tensor_min", data=dipole_tensor_min)
            f.create_dataset("inertia_moments_qm", data=inertia_moments_qm)
            f.create_dataset("inertia_moments_min", data=inertia_moments_min)
            f.create_dataset("total_charge", data=total_charge)

    @classmethod
    def from_hdf5(cls, filename: str) -> "BenchmarkResults":
        """
        Load results from an HDF5 file created by to_hdf5.
        """
        with h5py.File(filename, "r") as f:
            meta = f["meta"]
            backend      = meta.attrs["backend"]
            dataset_name = meta.attrs["dataset_name"]

            obj = cls(backend=backend, dataset_name=dataset_name)

            obj.names   = [s for s in f["names"][()]]
            obj.smiles  = [s for s in f["smiles"][()]]
            obj.n_atoms = f["n_atoms"][()]
            obj.offsets = f["offsets"][()]

            obj.coords_qm  = f["coords_qm"][()]
            obj.coords_min = f["coords_min"][()]
            obj.partial_charges = (
                f["partial_charges"][()] if "partial_charges" in f else
                np.full(obj.offsets[-1], np.nan, dtype=np.float32)
            )

            obj.rmsd_cart      = f["rmsd_cart"][()].tolist()
            obj.rmsd_bonds     = f["rmsd_bonds"][()].tolist()
            obj.rmsd_angles    = f["rmsd_angles"][()].tolist()
            obj.rmsd_propers   = f["rmsd_propers"][()].tolist()
            obj.rmsd_impropers = f["rmsd_impropers"][()].tolist()
            obj.tfd            = f["tfd"][()].tolist()

            obj.energy_qm      = f["energy_qm"][()].tolist()
            obj.energy_initial = f["energy_initial"][()].tolist()
            obj.energy_min     = f["energy_min"][()].tolist()

            n = len(obj.n_atoms)
            obj.dipole_qm = (
                f["dipole_qm"][()].tolist() if "dipole_qm" in f else
                np.full((n, 3), np.nan, dtype=np.float32).tolist()
            )
            obj.dipole_min = (
                f["dipole_min"][()].tolist() if "dipole_min" in f else
                np.full((n, 3), np.nan, dtype=np.float32).tolist()
            )
            obj.dipole_tensor_qm = (
                f["dipole_tensor_qm"][()].tolist() if "dipole_tensor_qm" in f else
                np.full((n, 3, 3), np.nan, dtype=np.float32).tolist()
            )
            obj.dipole_tensor_min = (
                f["dipole_tensor_min"][()].tolist() if "dipole_tensor_min" in f else
                np.full((n, 3, 3), np.nan, dtype=np.float32).tolist()
            )
            obj.inertia_moments_qm = (
                f["inertia_moments_qm"][()].tolist() if "inertia_moments_qm" in f else
                np.full((n, 3), np.nan, dtype=np.float32).tolist()
            )
            obj.inertia_moments_min = (
                f["inertia_moments_min"][()].tolist() if "inertia_moments_min" in f else
                np.full((n, 3), np.nan, dtype=np.float32).tolist()
            )
            obj.total_charge = (
                f["total_charge"][()].tolist() if "total_charge" in f else
                np.full(n, np.nan, dtype=np.float32).tolist()
            )

        return obj

    def get_coords_qm_for_mol(self, idx: int) -> np.ndarray:
        """
        Convenience accessor: QM coordinates for molecule `idx`.
        """
        start = self.offsets[idx]
        end   = self.offsets[idx + 1]
        return self.coords_qm[start:end]

    def get_coords_min_for_mol(self, idx: int) -> np.ndarray:
        """
        Convenience accessor: minimized coordinates for molecule `idx`.
        """
        start = self.offsets[idx]
        end   = self.offsets[idx + 1]
        return self.coords_min[start:end]

    def get_partial_charges_for_mol(self, idx: int) -> np.ndarray:
        """
        Convenience accessor: partial charges for molecule `idx`.
        """
        start = self.offsets[idx]
        end   = self.offsets[idx + 1]
        return self.partial_charges[start:end]

    def save_aligned_qm_ff_pdb(
        self,
        idx: int,
        filename: str,
        align_to: str = "qm",
        qm_chain: str = "A",
        ff_chain: str = "B"):

        """
        Create a 2-model PDB for molecule/conformer `idx`:
          - MODEL 1: QM-minimized (chain qm_chain)
          - MODEL 2: FF-minimized (chain ff_chain)
        with one aligned onto the other.

        Parameters
        ----------
        idx : int
            Index into this BenchmarkResults object (one record = one conformer).
        filename : str
            Output PDB path.
        align_to : {"qm", "ff"}
            Which structure is the reference frame.
        qm_chain, ff_chain : str
            Chain IDs to help visualize overlap in PyMOL/VMD/etc.
        """
        if align_to not in ("qm", "ff"):
            raise ValueError("align_to must be 'qm' or 'ff'")

        smiles = self.smiles[idx]
        n_atoms = int(self.n_atoms[idx])

        coords_qm = self.get_coords_qm_for_mol(idx)
        coords_ff = self.get_coords_min_for_mol(idx)

        mol = _mol_from_smiles_match_natoms(smiles, n_atoms)

        # Add conformers: (qm, ff)
        qm_cid = _add_conformer_from_coords(mol, coords_qm)
        ff_cid = _add_conformer_from_coords(mol, coords_ff)

        # Align one conformer onto the other in-place
        if align_to == "qm":
            rdMolAlign.AlignMol(mol, mol, prbCid=ff_cid, refCid=qm_cid)
        else:
            rdMolAlign.AlignMol(mol, mol, prbCid=qm_cid, refCid=ff_cid)

        pdb = _pdb_single_model_two_chains(
            mol,
            conf_id_qm=qm_cid,
            conf_id_ff=ff_cid,
            chain_qm=qm_chain,
            chain_ff=ff_chain,
            remark_qm="QM_MIN",
            remark_ff="FF_MIN",
        )

        with open(filename, "w", encoding="utf-8") as f:
            f.write(pdb)

    def save_aligned_qm_ff_pdbs(
        self,
        idx: int,
        qm_filename: str,
        ff_filename: str,
        align_to: str = "qm",):
        """
        Write two separate PDBs (QM and FF) in the same reference frame.
        """
        if align_to not in ("qm", "ff"):
            raise ValueError("align_to must be 'qm' or 'ff'")

        smiles = self.smiles[idx]
        n_atoms = int(self.n_atoms[idx])

        coords_qm = self.get_coords_qm_for_mol(idx)
        coords_ff = self.get_coords_min_for_mol(idx)

        mol = _mol_from_smiles_match_natoms(smiles, n_atoms)
        qm_cid = _add_conformer_from_coords(mol, coords_qm)
        ff_cid = _add_conformer_from_coords(mol, coords_ff)

        # Align in-place
        if align_to == "qm":
            rdMolAlign.AlignMol(mol, mol, prbCid=ff_cid, refCid=qm_cid)
        else:
            rdMolAlign.AlignMol(mol, mol, prbCid=qm_cid, refCid=ff_cid)

        # Write each conformer separately
        with open(qm_filename, "w", encoding="utf-8") as f:
            f.write(Chem.MolToPDBBlock(mol, confId=qm_cid))
        with open(ff_filename, "w", encoding="utf-8") as f:
            f.write(Chem.MolToPDBBlock(mol, confId=ff_cid))
