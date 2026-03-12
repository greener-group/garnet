# Record RMSD over validation simulations

import MDAnalysis as mda
from MDAnalysis.analysis import rms
import matplotlib.pyplot as plt
import numpy as np
import os

run_name = "64_7_ep12"
n_reps = 3
smooth_n = 10
selector = "backbone"

with open("proteins.txt") as f:
    proteins = [l.rstrip() for l in f.readlines()]

fig, axs = plt.subplots(1, 4, figsize=(10, 3))

def smooth_array(xs):
    return [np.mean(xs[max(i - smooth_n, 0):min(i + smooth_n + 1, len(xs))]) for i in range(len(xs))]

for pi, protein in enumerate(proteins):
    struc_fp = f"structures/{protein}.pdb"
    ax = axs[pi]
    xmax = 0
    for rep_n in range(1, n_reps + 1):
        traj_fp = f"dcd/{run_name}/{protein}_{rep_n}.dcd"
        if os.path.isfile(traj_fp):
            u = mda.Universe(struc_fp, traj_fp)
            R = rms.RMSD(u, u, select=selector, ref_frame=0)
            R.run()
            res = R.results.rmsd[:, 2]
            res_smooth = smooth_array(res)
            final_rmsd = res[-1]
            print(traj_fp, "-", final_rmsd, "Å")

            xs = [(i + 1) / 2000 for i in range(len(res))]
            xmax = max(xmax, max(xs))
            ax.plot(xs, res_smooth, label=f"Repeat {rep_n}")

    ax.set_xlim(0, min(xmax, 5))
    ax.set_ylim(0, 10)
    ax.set_xlabel("Time / μs")
    ax.set_ylabel("RMSD / Å" if pi == 0 else None)
    ax.set_title(protein)

#plt.legend()
plt.tight_layout()
plt.savefig(f"plots/{run_name}/rmsd.pdf")
