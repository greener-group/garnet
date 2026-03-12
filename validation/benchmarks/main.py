#!/usr/bin/env python
#%%
import os

# 1. Enforce thread limits BEFORE importing numpy or torch
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import gc
import argparse
import multiprocessing as mp

import numpy as np
import qcportal
import matplotlib.pyplot as plt

from ase import Atoms
from ase.optimize import BFGS

from openmm import Platform, VerletIntegrator, LocalEnergyMinimizer, NonbondedForce
from openmm.app import Simulation
from openmm.unit import picosecond, kilojoules_per_mole, nanometer
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.qcsubmit.results import OptimizationResultCollection

from src.run_model import garnet
from src.geometry  import compute_internal_coordinates, compute_tfd_from_coords, kabsch_align
from src.stats     import freedman_diaconis_bins, histogram_cdf
from src.io        import BenchmarkResults

# -----------------------------------------------------------------------------
# Global parallelism configuration / env vars
# -----------------------------------------------------------------------------
#%%
# Number of logical CPU cores
N_LOGICAL = 16

# Threads per OpenMM simulation
THREADS_PER_SIM = 4

# Number of worker processes (Default for CPU paths)
N_PROCS_DEFAULT   = max(1, N_LOGICAL // THREADS_PER_SIM)

# OpenMM CPU threading
os.environ.setdefault("OPENMM_CPU_THREADS", str(THREADS_PER_SIM))

# -----------------------------------------------------------------------------
# Globals shared across workers
# -----------------------------------------------------------------------------

QCARCHIVE_URL = "https://api.qcarchive.molssi.org:443/"
DATASET_NAME  = "OpenFF Industry Benchmark Season 1 v1.2"

_client     = None

# Assigned via worker initializer
_BACKEND    = None 
_DEVICE     = "cpu"

# Lazies per backend
_esp_model  = None
_mace_calc  = None
_ff         = ForceField("openff-2.2.1.offxml")

# These imports are backend-dependent; we import lazily
_espaloma_mod = None
_torch_mod    = None

#%%
# -----------------------------------------------------------------------------
# Utility / stats helpers
# -----------------------------------------------------------------------------

def compute_rmsd(coords_A, coords_B):
    # Preserves float64 precision natively instead of MDTraj's float32 coercion
    diff = coords_A - coords_B
    return np.sqrt(np.mean(np.sum(diff**2, axis=-1)))

def rmsd_1d(x0, x1):
    return np.sqrt(np.mean((x1 - x0) ** 2))

def rmsd_periodic(phi0, phi1):
    d = phi1 - phi0
    d = (d + np.pi) % (2 * np.pi) - np.pi  # wrap to (-π, π]
    return np.sqrt(np.mean(d ** 2))

# -----------------------------------------------------------------------------
# Lazy singletons: QCArchive client + dataset
# -----------------------------------------------------------------------------

def get_client():
    global _client
    if _client is None:
        _client = qcportal.PortalClient(QCARCHIVE_URL)
    return _client

# -----------------------------------------------------------------------------
# Backend-specific model accessors
# -----------------------------------------------------------------------------

def get_espaloma_model():
    global _esp_model, _espaloma_mod, _torch_mod, _DEVICE
    if _esp_model is None:
        import espaloma as esp
        import torch

        _espaloma_mod = esp
        _torch_mod    = torch

        model = esp.get_model("latest")
        model.to(_DEVICE)
        _esp_model = model
    return _esp_model, _espaloma_mod, _torch_mod

def build_system_espaloma(mol, top):
    global _DEVICE
    model, esp, torch = get_espaloma_model()

    mgraph = esp.Graph(mol)
    mgraph.heterograph = mgraph.heterograph.to(_DEVICE)

    with torch.no_grad():
        model(mgraph.heterograph)

    system = esp.graphs.deploy.openmm_system_from_graph(mgraph)
    return system, mgraph

def get_mace_model():
    global _mace_calc, _DEVICE
    if _mace_calc is None:
        from mace.calculators import mace_off
        _mace_calc = mace_off(model="medium", device=_DEVICE, default_dtype="float64")
    return _mace_calc

# -----------------------------------------------------------------------------
# OpenMM simulation helper
# -----------------------------------------------------------------------------

def make_simulation(topology, system):
    global _DEVICE
    integrator = VerletIntegrator(0.002 * picosecond)
    
    if "cuda" in _DEVICE:
        platform = Platform.getPlatformByName("CUDA")
        device_idx = _DEVICE.split(":")[-1] if ":" in _DEVICE else "0"
        properties = {"DeviceIndex": device_idx}
    else:
        platform = Platform.getPlatformByName("CPU")
        properties = {"Threads": str(THREADS_PER_SIM)}

    sim = Simulation(topology, system, integrator, platform, properties)
    return sim

# -----------------------------------------------------------------------------
# Single-molecule computation
# -----------------------------------------------------------------------------

def process_single_molecule(rec_id: int):
    backend = _BACKEND

    client = get_client()
    entry  = client.get_optimizations(rec_id, include="final_molecule")
    final_molecule = entry.final_molecule

    name   = final_molecule.dict()["name"]
    mol    = Molecule.from_qcschema(final_molecule, allow_undefined_stereo = True)
    qm_energy = entry.dict()["energies"][-1] * 2625.4996394799 # High-precision Hartree to kJ/mol

    # =========================================================================
    # UNIVERSAL SANITIZATION BLOCK
    # Enforces strict valence, aromaticity, and stereochemistry definitions 
    # across all molecules before backend-specific parsing occurs.
    # =========================================================================
    from rdkit import Chem
    rdmol = mol.to_rdkit()
    Chem.SanitizeMol(rdmol)
    Chem.AssignStereochemistry(rdmol, force=True, cleanIt=True)
    
    # Re-instantiate the OpenFF molecule from the perfectly clean RDKit object
    mol = Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)
    # =========================================================================

    top       = Topology.from_molecules(molecules=[mol])
    conf      = mol.conformers[0]
    coords_qm = conf.m_as("angstrom") # Enforce strictly Angstrom extraction
    smiles    = mol.to_smiles(mapped=True)

    ic0 = compute_internal_coordinates(mol, coords_qm)

    if backend in ["espaloma", "openff", "garnet"]:
        if backend == "espaloma":
            system, mgraph = build_system_espaloma(mol, top)
            
        elif backend == "openff":
            # Assign charges using the production NAGL model directly.
            # The molecule is already sanitized, so this will not crash.
            mol.assign_partial_charges(
                partial_charge_method="openff-gnn-am1bcc-1.0.0.pt"
            )
            
            # Generate the OpenMM system and FORCE it to use the pre-calculated charges
            system = _ff.create_openmm_system(
                top, 
                charge_from_molecules=[mol]
            )
            mgraph = None
            
        elif backend == "garnet":
            system, top = garnet.topology_to_openmm_system(top)
            mgraph = None
        
        for force in system.getForces():
            if isinstance(force, NonbondedForce):
                force.setNonbondedMethod(NonbondedForce.NoCutoff)

        simulation = make_simulation(top, system)
        simulation.context.setPositions(coords_qm * 0.1) # to nm

        state  = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
        energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)

        LocalEnergyMinimizer.minimize(
            simulation.context,
            tolerance=5.0 * kilojoules_per_mole / nanometer,
            maxIterations=1500,
        )

        state_minim = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
        energy_min  = state_minim.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        coords_min  = state_minim.getPositions(asNumpy=True)
        coords_min  = kabsch_align(coords_qm, coords_min.value_in_unit(nanometer).astype(np.float32) * 10) 
        ic1         = compute_internal_coordinates(mol, coords_min)

        del state, state_minim
        del system, simulation
        if mgraph is not None:
            del mgraph

    elif backend == "mace":
        symbols = [atom.symbol for atom in mol.atoms]
        atoms = Atoms(symbols=symbols, positions=coords_qm)
        atoms.calc = get_mace_model()

        ev_to_kj_mol = 96.48530749925793
        omm_tolerance = 5.0
        ase_fmax = omm_tolerance / (ev_to_kj_mol * 10.0)

        energy = atoms.get_potential_energy() * ev_to_kj_mol

        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=ase_fmax, steps=1500)

        energy_min = atoms.get_potential_energy() * ev_to_kj_mol
        coords_min = atoms.get_positions().astype(np.float32)
        coords_min = kabsch_align(coords_qm, coords_min)
        ic1 = compute_internal_coordinates(mol, coords_min)

    else:
        raise ValueError(f"Unknown backend {_BACKEND!r}")
    
    rmsd_cart      = compute_rmsd(coords_min, coords_qm)
    rmsd_bonds     = rmsd_1d(ic1["bonds"],     ic0["bonds"])
    rmsd_angles    = rmsd_periodic(ic1["angles"],    ic0["angles"])
    rmsd_propers   = rmsd_periodic(ic1["propers"],   ic0["propers"])
    rmsd_impropers = rmsd_periodic(ic1["impropers"], ic0["impropers"])
    tfd            = compute_tfd_from_coords(mol, coords_qm, coords_min)

    del ic0, ic1
    gc.collect()

    return {
        "name": name,
        "smiles":smiles,
        "coords":{"qm":coords_qm,
                  "min":coords_min},
        "rmsd_cart": rmsd_cart,
        "rmsd_bonds": rmsd_bonds,
        "rmsd_angles": rmsd_angles,
        "rmsd_propers": rmsd_propers,
        "rmsd_impropers": rmsd_impropers,
        "energy_qm": qm_energy,
        "energy_initial": energy,
        "energy_min": energy_min,
        "tfd":tfd,
    }

# -----------------------------------------------------------------------------
# Batch worker + helpers
# -----------------------------------------------------------------------------

def init_worker(backend, device_string):
    global _BACKEND, _DEVICE
    _BACKEND = backend

    devices = [d.strip() for d in device_string.split(",")]
    if len(devices) > 1:
        worker_id = mp.current_process()._identity[0]
        _DEVICE = devices[(worker_id - 1) % len(devices)]
    else:
        _DEVICE = devices[0]

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    try:
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

def process_batch(ids_batch):
    print("PROCESS_BATCH", flush=True)
    out = []
    for rec_id in ids_batch:
        try:
            res = process_single_molecule(rec_id)
            out.append(res)
        except Exception as e:
            print(e, flush=True)
            # print(f"WARNING: Skipped molecule {rec_id} due to toolkit error.", flush=True)
            out.append({"name": rec_id, "error": repr(e)})

    return out

def chunk_list(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

#%%
# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main(backend: str, device: str, max_mols: int | None = None, n_workers: int | None = None):
    num_processes = n_workers if n_workers is not None else N_PROCS_DEFAULT
    batch_size = int(num_processes * 2)

    os.environ["OPENMM_CPU_THREADS"] = str(THREADS_PER_SIM)

    print(f"Backend: {backend} | Device: {device}")
    print(f"Logical cores: {N_LOGICAL}")
    print(f"Using {num_processes} worker processes.")

    dataset = OptimizationResultCollection.parse_file("Datasets/Industry/OpenFF_1.2.json")
    results = dataset.entries[QCARCHIVE_URL]
    rec_ids = [result.record_id for result in results]

    if max_mols != 0:
        rec_ids = rec_ids[:max_mols]

    batches = list(chunk_list(rec_ids, batch_size))
    all_results = []

    ctx = mp.get_context("spawn")

    with ctx.Pool(
        processes=num_processes, 
        maxtasksperchild=10,
        initializer=init_worker,
        initargs=(backend, device)
    ) as pool:
        for batch_res in pool.map(process_batch, batches):
            all_results.extend(batch_res)

    filtered = [r for r in all_results if "error" not in r]

    rmsd_array = np.array([
        [
            r["rmsd_cart"],
            r["rmsd_bonds"],
            r["rmsd_angles"],
            r["rmsd_propers"],
            r["rmsd_impropers"],
            r["tfd"],
        ]
        for r in filtered
    ])

    bench = BenchmarkResults(backend=backend, dataset_name=DATASET_NAME)
    for r in filtered:
        bench.add(r)

    return rmsd_array, bench


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def graph_rmsd(rmsd,
               title    = "Structure",
               xlabel   = r"RMSD / $\mathbf{\AA}$",
               ylabel   = "P(RMSD)",
               ylabel_2 = "CDF(RMSD)",
               savepath = None):

    fig, ax = plt.subplots(figsize=(7, 7), layout="tight")
    ax_2 = ax.twinx()

    ax.set_title(title, fontsize=20, fontweight="bold")

    n, b, _ = ax.hist(
        rmsd,
        bins=freedman_diaconis_bins(rmsd),
        density=True,
        color="royalblue",
        edgecolor="k",
    )

    cdf = histogram_cdf(n, b)
    ax_2.plot(b, cdf, c="k")
    ax_2.set_ylim(0)

    ax.set_xlabel(xlabel, fontsize=18, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=18, fontweight="bold")
    ax_2.set_ylabel(ylabel_2, fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=16)
    ax_2.tick_params(labelsize=16)

    if savepath is not None:
        fig.savefig(savepath, dpi = 150)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["espaloma", "openff", "garnet", "mace"],
        default="espaloma",
        help="Which model to use for building the OpenMM/ASE system.",
    )
    parser.add_argument(
        "--max-mols",
        type=int,
        default=0,
        help="Optional cap on number of molecules to process.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional HDF5 file to save benchmark results.",
    )
    parser.add_argument(
        "--graph",
        type=bool,
        default=False,
        help="Wether to graph the results fro the benchmark."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device for PyTorch backends (e.g., 'cpu', 'cuda', 'cuda:0', 'cuda:1')."
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=None,
        help="Manual override for worker pool size to prevent VRAM exhaustion on GPUs."
    )
    args = parser.parse_args()

    rmsd_array, bench = main(
        backend=args.backend, 
        max_mols=args.max_mols, 
        device=args.device,
        n_workers=args.n_workers
    )

    if args.out is not None:
        bench.to_hdf5(args.out)
        print(f"Saved benchmark results to {args.out}")

    if args.graph:
        graph_rmsd(rmsd_array[:, 0], title = "Structure", xlabel=r"RMSD / $\mathbf{\AA}$", savepath=f"structure_{args.backend}.png")
        graph_rmsd(rmsd_array[:, 1], title = "Bonds",     xlabel=r"RMSD / $\mathbf{\AA}$", savepath=f"bonds_{args.backend}.png")
        graph_rmsd(rmsd_array[:, 2], title = "Angles",    xlabel="RMSD / rad", savepath=f"angles_{args.backend}.png")
        graph_rmsd(rmsd_array[:, 3], title = "Propers",   xlabel="RMSD / rad", savepath=f"propers_{args.backend}.png")
        graph_rmsd(rmsd_array[:, 4], title = "Impropers", xlabel="RMSD / rad", savepath=f"impropers_{args.backend}.png")
        graph_rmsd(rmsd_array[:, 5], title = "TFD",       xlabel="TFD", ylabel="P(TFD)", ylabel_2="CDF(TFD)", savepath=f"tfd_{args.backend}.png")
# %%
