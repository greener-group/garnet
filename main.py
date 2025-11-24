#!/usr/bin/env python
#%%
import os
import gc
import argparse
import multiprocessing as mp

import numpy as np
import mdtraj as md
import qcportal
import qcengine
import matplotlib.pyplot as plt

from openmm import Platform, VerletIntegrator, LocalEnergyMinimizer, NonbondedForce
from openmm.app import Simulation
from openmm.unit import picosecond, kilojoules_per_mole, nanometer
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.qcsubmit.results import OptimizationResultCollection

from src.geometry import compute_internal_coordinates, compute_tfd_from_coords, kabsch_align
from src.stats    import freedman_diaconis_bins, histogram_cdf
from src.io       import BenchmarkResults

# -----------------------------------------------------------------------------
# Global parallelism configuration / env vars
# -----------------------------------------------------------------------------
#%%
# Number of logical CPU cores
N_LOGICAL = 16#os.cpu_count() or 1

# Threads per OpenMM simulation
THREADS_PER_SIM = 4

# Number of worker processes
N_PROCS   = max(1, N_LOGICAL // THREADS_PER_SIM)
BATCH_SIZE = int(N_PROCS * 2)

# Make sure we don't grossly oversubscribe
assert THREADS_PER_SIM * N_PROCS <= 4 * N_LOGICAL  # you can tighten if you want

# OpenMM CPU threading
os.environ.setdefault("OPENMM_CPU_THREADS", str(THREADS_PER_SIM))

# Avoid BLAS / OpenMP contention
os.environ.setdefault("OMP_NUM_THREADS", f"{THREADS_PER_SIM}")
os.environ.setdefault("MKL_NUM_THREADS", f"{THREADS_PER_SIM}")
os.environ.setdefault("OPENBLAS_NUM_THREADS", f"{THREADS_PER_SIM}")

# Disable GPUs by default (for espaloma / torch if used)
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# -----------------------------------------------------------------------------
# Globals shared across workers
# -----------------------------------------------------------------------------

QCARCHIVE_URL = "https://api.qcarchive.molssi.org:443/"
DATASET_NAME  = "OpenFF Industry Benchmark Season 1 v1.2"
DATASET_TYPE  = "optimization"

ang_to_nm = 0.1

_client     = qcportal.PortalClient(QCARCHIVE_URL)
_dataset    = None
_BACKEND    = None  # "espaloma" or "openff"

# Lazies per backend
_esp_model  = None
_ff         = ForceField("openff-2.2.1.offxml")

# These imports are backend-dependent; we import lazily
_espaloma_mod = None
_torch_mod    = None
_forcefield_class = None

#%%
# -----------------------------------------------------------------------------
# Utility / stats helpers
# -----------------------------------------------------------------------------

def compute_rmsd(coords_A, coords_B):
    traj0 = md.Trajectory(coords_A[None, :, :], None)
    traj1 = md.Trajectory(coords_B[None, :, :], None)
    return md.rmsd(traj1, traj0)[0]


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


def get_dataset():
    global _dataset
    if _dataset is None:
        _dataset = OptimizationResultCollection.parse_file("Datasets/Industry/OpenFF_1.2.json")
    return _dataset


# -----------------------------------------------------------------------------
# Backend-specific model accessors
# -----------------------------------------------------------------------------

def get_espaloma_model():
    global _esp_model, _espaloma_mod, _torch_mod
    if _esp_model is None:
        # Lazy imports so openff-only runs don't require these installed
        import espaloma as esp
        import torch

        _espaloma_mod = esp
        _torch_mod    = torch

        model = esp.get_model("latest")
        model.to("cpu")
        _esp_model = model
    return _esp_model, _espaloma_mod, _torch_mod

def get_forcefield():
    ff = ForceField("openff-2.2.1.offxml")
    return ff

# -----------------------------------------------------------------------------
# OpenMM simulation helper
# -----------------------------------------------------------------------------

def make_simulation(topology, system):
    integrator = VerletIntegrator(0.002 * picosecond)
    platform   = Platform.getPlatformByName("CPU")
    properties = {"Threads": str(THREADS_PER_SIM)}
    sim = Simulation(topology, system, integrator, platform, properties)
    return sim

# -----------------------------------------------------------------------------
# Backend-specific system builders
# -----------------------------------------------------------------------------

def build_system_espaloma(mol, top):
    model, esp, torch = get_espaloma_model()

    mgraph = esp.Graph(mol)
    mgraph.heterograph = mgraph.heterograph.to("cpu")

    with torch.no_grad():
        model(mgraph.heterograph)

    system = esp.graphs.deploy.openmm_system_from_graph(mgraph)
    return system, mgraph  # mgraph can be deleted after use

def build_system_openff(mol, top):
    ff = get_forcefield()
    system = ff.create_openmm_system(top)
    return system, None

# -----------------------------------------------------------------------------
# Single-molecule computation
# -----------------------------------------------------------------------------

def process_single_molecule(rec_id: int):
    backend = _BACKEND

    entry          = _client.get_optimizations(rec_id, include="final_molecule")
    final_molecule = entry.final_molecule

    name   = final_molecule.dict()["name"]
    mol    = Molecule.from_qcschema(final_molecule)
    top    = Topology.from_molecules(molecules=[mol])
    conf      = mol.conformers[0]
    coords_qm = conf.m_as(conf.units) # This is in Angstroms
    smiles = mol.to_smiles(mapped=True)
    qm_energy = entry.dict()["energies"][-1] #* 2625.5 # To kJ/mol

    if backend == "espaloma":
        system, mgraph = build_system_espaloma(mol, top)
    elif backend == "openff":
        system = _ff.create_openmm_system(top)
        mgraph = None
    else:
        raise ValueError(f"Unknown backend {_BACKEND!r}")
    
    for force in system.getForces():
        if isinstance(force, NonbondedForce):
            force.setNonbondedMethod(NonbondedForce.NoCutoff)

    simulation = make_simulation(top, system)
    simulation.context.setPositions(coords_qm * 0.1) # This expects coords in nm

    # Initial state
    state  = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
    energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    ic0    = compute_internal_coordinates(mol, coords_qm)

    # Minimize
    LocalEnergyMinimizer.minimize(
        simulation.context,
        tolerance=5e-9 * kilojoules_per_mole / nanometer,
        maxIterations=1500,
    )

    # Minimized state
    state_minim = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
    energy_min  = state_minim.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    coords_min  = state_minim.getPositions(asNumpy=True)
    coords_min  = kabsch_align(coords_qm, coords_min.value_in_unit(nanometer).astype(np.float32) * 10) # Back to angstroms
    ic1         = compute_internal_coordinates(mol, coords_min)

    # RMSDs
    rmsd_cart      = compute_rmsd(coords_min, coords_qm)
    rmsd_bonds     = rmsd_1d(ic1["bonds"],     ic0["bonds"])
    rmsd_angles    = rmsd_periodic(ic1["angles"],    ic0["angles"])
    rmsd_propers   = rmsd_periodic(ic1["propers"],   ic0["propers"])
    rmsd_impropers = rmsd_periodic(ic1["impropers"], ic0["impropers"])
    tfd            = compute_tfd_from_coords(mol, coords_qm, coords_min)

    # Clean up
    del state, state_minim, ic0, ic1
    del system, simulation
    if mgraph is not None:
        del mgraph
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

def process_batch(ids_batch):
    out = []
    for rec_id in ids_batch:
        try:
            res = process_single_molecule(rec_id)
            out.append(res)
        except Exception as e:
            print(e)
            out.append({"name": rec_id, "error": repr(e)})
    return out


def chunk_list(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]

#%%
# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def main(backend: str, max_mols: int | None = None):
    global _BACKEND
    _BACKEND = backend

    os.environ["OPENMM_CPU_THREADS"] = str(THREADS_PER_SIM)

    print(f"Backend: {backend}")
    print(f"Logical cores: {N_LOGICAL}")
    print(f"Using {N_PROCS} processes × {THREADS_PER_SIM} OpenMM threads "
          f"= {N_PROCS * THREADS_PER_SIM} OpenMM threads total")

    client  = qcportal.PortalClient(QCARCHIVE_URL)
    dataset = OptimizationResultCollection.parse_file("Datasets/Industry/OpenFF_1.2.json")
    results = dataset.entries[QCARCHIVE_URL]
    rec_ids = [result.record_id for result in results]

    if max_mols != 0:
        rec_ids = rec_ids[:max_mols]

    batches = list(chunk_list(rec_ids, BATCH_SIZE))

    all_results = []
    with mp.Pool(processes=N_PROCS, maxtasksperchild=10) as pool:
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
               xlabel   = "RMSD / nm",
               ylabel   = "P(RMSD)",
               ylabel_2 = "CDF(RMSD)"):

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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        choices=["espaloma", "openff"],
        default="espaloma",
        help="Which model to use for building the OpenMM system.",
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
    args = parser.parse_args()

    rmsd_array, bench = main(backend=args.backend, max_mols=args.max_mols)

    if args.out is not None:
        bench.to_hdf5(args.out)
        print(f"Saved benchmark results to {args.out}")

    # Example plotting
    graph_rmsd(rmsd_array[:, 0], title = "Structure", xlabel="RMSD / nm")
    graph_rmsd(rmsd_array[:, 1], title = "Bonds",     xlabel="RMSD / nm")
    graph_rmsd(rmsd_array[:, 2], title = "Angles",    xlabel="RMSD / rad")
    graph_rmsd(rmsd_array[:, 3], title = "Propers",   xlabel="RMSD / rad")
    graph_rmsd(rmsd_array[:, 4], title = "Impropers", xlabel="RMSD / rad")
    graph_rmsd(rmsd_array[:, 5], title = "TFD",       xlabel="TFD", ylabel="P(TFD)", ylabel_2="CDF(TFD)")

    plt.show()
