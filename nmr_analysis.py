#%%
import os
import re

import argparse

import numpy      as np
import pickle     as pk
import MDAnalysis as mda

from os         import path
from src.hbonds import karplus_dict, backbone_torsions, karplus_J, karplus_hbonds, backbone_amide_hbond_between, karplus_extrema 

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default="./NMR_Data/J_coupling/ESPALOMA",
        help="Path to where NMR data is stored",
    )
    parser.add_argument(
        "--simulations",
        type=str,
        default="/cephfs2/jgreener/dms/typing/val_folded/",
        help="Path to simulations master folder",
    )
    parser.add_argument(
        "--replica",
        type=int,
        default=1,
        help="Simulation replica nr."
    )
    parser.add_argument(
        "--outfile",
        type=str,
        default="results_hbonds_garnet.pk",
        help="File to output results."
    )
    args = parser.parse_args()

    data_files = [_ for _ in os.listdir(args.datasets) if _.endswith(".dat")]

    torsion_dict = {}
    for file in data_files:

        fields  = re.split(r"[_.]+", file)[:-1]
        protein = fields[0]
        torsion = fields[1:]

        if not protein in torsion_dict:
            torsion_dict[protein] = [torsion]
        else:
            torsion_dict[protein].append(torsion)

    if os.path.isdir(path.join(args.simulations, f"dcd/64_7_ep12")):
        trjpath = path.join(args.simulations, f"dcd/64_7_ep12")
    else:
        trjpath = path.join(args.simulations, f"dcd/espaloma")

    u_gb3  = mda.Universe(path.join(args.simulations, "structures/gb3.pdb"),
                          path.join(trjpath, f"gb3_{args.replica}.dcd"), 
                          in_memory   = True)

    u_ubq  = mda.Universe(path.join(args.simulations, "structures/ubiquitin.pdb"),
                          path.join(trjpath, f"ubiquitin_{args.replica}.dcd"),
                          in_memory = True)

    u_bpti = mda.Universe(path.join(args.simulations, "structures/bpti.pdb"),
                          path.join(trjpath, f"bpti_{args.replica}.dcd"),
                          in_memory = True)

    u_hewl = mda.Universe(path.join(args.simulations, "structures/hewl.pdb"),
                          path.join(trjpath, f"hewl_{args.replica}.dcd"),
                          in_memory = True)

    universe_dict = {"gb3":u_gb3, "ubq":u_ubq, "bpti":u_bpti, "hewl":u_hewl}

    if os.path.isfile(args.outfile):
        with open(args.outfile, "rb") as infile:
            RESULTS = pk.load(infile)
    else:
        RESULTS = {"gb3":{},
                   "ubq":{},
                   "bpti":{},
                   "hewl":{}}

    time_trim = 1000000 # ps, discard 1 microsecond

    for protein, torsions in torsion_dict.items():
        
        u = universe_dict[protein]

        t_max = u.trajectory[-1].time

        idx_trim = int(time_trim * len(u.trajectory) / t_max)

        for frame in u.trajectory[idx_trim:]:

            pct = int((frame.time - time_trim) / (t_max - time_trim) * 100)
            t_ns = round(frame.time / 1000, 3)

            msg = f"Protein: {protein}; time {t_ns} ns; {pct}% done"
            print(msg.ljust(80), end="\r", flush=True)

            prot_atoms = u.select_atoms("protein")

            # H bonds
            if protein == "gb3" or protein == "ubq":

                if not "HBD" in RESULTS[protein].keys():
                    RESULTS[protein]["HBD"] = {}

                file = path.join(args.datasets, f"../HBONDS/{protein}_hbonds.txt")
                data = np.loadtxt(file)
                
                res_is = data[:,0].astype(int) - 1
                res_js = data[:,1].astype(int) - 1

                for n, (i, j) in enumerate(zip(res_is, res_js)):

                    res_i = prot_atoms.residues[i]
                    res_j = prot_atoms.residues[j]

                    is_hb, direction, R, theta, phi = backbone_amide_hbond_between(res_i, res_j)

                    if is_hb:
                        
                        J = karplus_hbonds(R, theta, phi)

                        d = (J , data[n,2])

                        if (i, j) not in RESULTS[protein]["HBD"].keys():
                            RESULTS[protein]["HBD"][(i, j)] = [d]
                        else:
                            RESULTS[protein]["HBD"][(i, j)].append(d)

            for torsion in torsions:

                torsion = tuple(torsion)

                if not torsion in karplus_dict.keys():
                    continue
                if not torsion in RESULTS[protein].keys():
                    RESULTS[protein][torsion] = {}
                
                pattern = "_".join(torsion)
                file    = f"{protein}_{pattern}.dat"
                
                data = np.loadtxt(path.join(args.datasets, file))
                resindex = data[:,0].astype(int)

                for residue in prot_atoms.residues:
                    
                    rnum    = residue.resnum
                    idx_vec = np.where(resindex == rnum)[0]
                    if len(idx_vec) == 0:
                        continue
                    j_idx = idx_vec[0]

                    if not rnum in RESULTS[protein][torsion].keys():
                        RESULTS[protein][torsion][rnum] = []

                    if torsion in backbone_torsions:
                        phi = residue.phi_selection()
                        if phi is not None:
                            
                            phi_val = np.deg2rad(phi.dihedral.value())
                            params  = karplus_dict[torsion]["ALL"]
                            minJ, maxJ = karplus_extrema(params)

                            j_exp = data[j_idx,1]
                            if np.isnan(j_exp):
                                continue
                            if j_exp < minJ:
                                j_exp = minJ
                            elif j_exp > maxJ:
                                j_exp = maxJ

                            j_comp = karplus_J(phi_val, params)

                            RESULTS[protein][torsion][rnum].append([j_comp, j_exp, minJ, maxJ])

                    else:
                        resname = residue.resname
                        chi1 = residue.chi1_selection()
                        if chi1 is not None:

                            chi1_val    = np.deg2rad(chi1.dihedral.value())
                            params_dict = karplus_dict[torsion]

                            if resname in params_dict.keys():
                                params = params_dict[resname]
                                minJ, maxJ = karplus_extrema(params)

                                j_exp = data[j_idx,1]
                                if np.isnan(j_exp):
                                    continue
                                if j_exp < minJ:
                                    j_exp = minJ
                                elif j_exp > maxJ:
                                    j_exp = maxJ

                                j_comp = karplus_J(chi1_val, params)

                                RESULTS[protein][torsion][rnum].append([j_comp, j_exp, minJ, maxJ])

                            else:
                                if "ALL" in params_dict.keys():
                                    params = params_dict["ALL"]
                                    minJ, maxJ = karplus_extrema(params)
                                    j_exp = data[j_idx,1]
                                    if np.isnan(j_exp):
                                        continue
                                    if j_exp < minJ:
                                        j_exp = minJ
                                    elif j_exp > maxJ:
                                        j_exp = maxJ

                                    j_comp = karplus_J(chi1_val, params)

                                    RESULTS[protein][torsion][rnum].append([j_comp, j_exp, minJ, maxJ])

    with open(args.outfile, "wb") as fout:

        pk.dump(RESULTS, fout, protocol=4)
