# Small-Molecule + NMR Benchmark

This repository contains the benchmarking pipeline used to compare classical/ML force fields on:

1. Small molecules from the OpenFF Industry Benchmark Season 1 v1.2 optimization set ([DOI](https://doi.org/10.5281/zenodo.15801401)).
2. Protein NMR J-coupling and H-bond observables (GB3, ubiquitin, BPTI, HEWL).

The code here has been used to produce the HDF5 benchmark result files (not included for size limitations, available upon request), the analysis figures (`cdf_smallmols.pdf`, `violin_smallmols.pdf`, `violins_dde.pdf`, `heatmap_*.pdf`), and the NMR figures (`hbonds_*.pdf`, `ane_*.pdf`) present in this directory.

## What Each Script Does

- `main.py`: runs the small-molecule minimization benchmark for one backend (`openff`, `espaloma`, `garnet`, `mace`) and writes an HDF5 result file.
- `postprocess.py`: loads the four model HDF5 files, builds comparative plots, runs motif enrichment, and writes per-model enrichment CSV files + heatmaps.
- `enrich_analysis.py`: aggregates enrichment CSV files and writes manuscript-ready LaTeX tables into `latex_tables_formatted/`.
- `nmr_analysis.py`: computes J-coupling/H-bond observables from MD trajectories and writes a `results_jcoup_*.pk` file.
- `nmr_graphs.py`: loads `results_jcoup_*.pk` files and generates NMR comparison figures.

## Repository Inputs Expected By The Pipeline

- `Datasets/Industry/OpenFF_1.2.json` for small-molecule records.
- `NMR_Data/J_coupling/...` and `NMR_Data/J_coupling/HBONDS/...` for NMR experimental references.
- A simulation root for `nmr_analysis.py` with this layout:
  - `structures/gb3.pdb`, `structures/ubiquitin.pdb`, `structures/bpti.pdb`, `structures/hewl.pdb`.
  - `dcd/<run_name>/<protein>_<replica>.dcd` (the script checks `dcd/64_7_ep12` first, then `dcd/espaloma`).

## Environment Setup

A helper script is provided:

```bash
bash install_env.sh
```

Notes:

- It is a GPU-oriented conda/pip setup and may need adaptation for your system. Some of the dependencies really do not like to play along together depending on the specific computer architecture.
- `install_env.sh` currently uses `python Models/Espaloma/setup.py install`, change that path to your local espaloma repo.
- `src/run_model.py` contains a hardcoded Garnet checkpoint path (`trained_model_fp`); update it to your local model path before running `--backend garnet`.

## Small-Molecule Workflow

Run one command per backend:

```bash
python main.py --backend <openff / espaloma / mace / garnet> --out <out file> --device <cpu / gpu / gpu:device> --n-workers <paraller workers>
```

Optional flags:

- `--max-mols <int>`: limit number of molecules (0 means full dataset).
- `--graph True`: emit quick per-metric PNG hist/CDF plots from `main.py`.

The output HDF5 files store coordinates, energies, and geometry metrics (RMSD bonds/angles/torsions + TFD) in `BenchmarkResults` format (`src/io.py`).

## Small-Molecule Postprocessing Workflow

After the four backend HDF5 files are present. Make sure the names of the files generated in the previous step match:

```bash
python postprocess.py
```

This generates:

- `cdf_smallmols.pdf`
- `violin_smallmols.pdf`
- `violins_dde.pdf`
- `enrichment_tables/enrichment_<Model>_<Metric>_<best|worst>_1_percent.csv`
- `heatmap_<metric_attr>_<tier>.pdf`

Then generate LaTeX tables:

```bash
python enrich_analysis.py
```

Outputs go to `latex_tables_formatted/`.

## NMR Workflow

1. Run `nmr_analysis.py` once per force field trajectory set to create pickle results.

Example:

```bash
python nmr_analysis.py --datasets NMR_Data/J_coupling/ESPALOMA --simulations /path/to/amber_runs    --replica 1 --outfile results_jcoup_amber.pk
python nmr_analysis.py --datasets NMR_Data/J_coupling/ESPALOMA --simulations /path/to/garnet_runs   --replica 1 --outfile results_jcoup_garnet.pk
python nmr_analysis.py --datasets NMR_Data/J_coupling/ESPALOMA --simulations /path/to/espaloma_runs --replica 1 --outfile results_jcoup_espaloma.pk
```

2. Plot comparisons:

```bash
python nmr_graphs.py
```

This produces:

- `hbonds_rmse_bar.pdf`
- `hbonds_rmse_violin.pdf`
- `hbonds_chisq_bar.pdf`
- `hbonds_chisq_violin.pdf`
- `ane_bar.pdf`
- `ane_violin.pdf`

## Practical Notes

- Many scripts use fixed filenames at the top of the file. If you change output names, update those constants.
- The repo contains generated artifacts from previous runs.
- Threading/device behavior is managed in `main.py`; adjust `THREADS_PER_SIM` and `--n-workers` to match your CPU/GPU memory budget.
