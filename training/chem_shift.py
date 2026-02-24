# Calculate backbone chemical shifts from a trajectory
# Arguments are the protein, the trajectory directory and the number of residues

import MDAnalysis as mda
import nmrgnn
import os
import sys

protein = sys.argv[1]
traj_dir = sys.argv[2]
n_res = int(sys.argv[3])

u = mda.Universe(
    os.path.join("condensed_data", protein, f"{protein}.pdb"),
    os.path.join(traj_dir, "protein", f"{protein}.dcd"),
)

prot = u.select_atoms("protein")
atoms_CA = u.select_atoms("name CA")
atoms_C = u.select_atoms(f"name C and resnum 1-{n_res-1}")
atoms_N = u.select_atoms(f"name N and resnum 2-{n_res}")
atoms_H = u.select_atoms(f"name H and resnum 2-{n_res}")

model = nmrgnn.load_model()

for ts in u.trajectory:
    x = nmrgnn.universe2graph(prot)
    peaks = model(x).numpy()
    s = ""
    for inds in [atoms_CA.ix, atoms_C.ix, atoms_N.ix, atoms_H.ix]:
        for x in peaks[inds]:
            s += f"{x:.4f} "
    assert len(s.split()) == (n_res * 4 - 3)
    print(s)
