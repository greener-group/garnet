import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.chem import compute_motif_enrichment_table, motifs
from src.graphs import style_violins, violin_prune_outliers
from src.io import BenchmarkResults
from src.stats import freedman_diaconis_bins, histogram_cdf

MODEL_FILES = {
    "OpenFF": "openff_gpu.hdf5",
    "Espaloma": "espaloma_new.hdf5",
    "MACE-OFF": "mace_f64.hdf5",
    "Garnet": "garnet_gpu.hdf5",
}

MODEL_ORDER = ("OpenFF", "Espaloma", "MACE-OFF", "Garnet")
MODEL_COLORS = {
    "OpenFF": "royalblue",
    "Espaloma": "firebrick",
    "MACE-OFF": "seagreen",
    "Garnet": "gold",
}

METRIC_SPECS = [
    ("rmsd_cart", "Structure", r"RMSD / $\\mathbf{\\AA}$"),
    ("rmsd_bonds", "Bonds", r"RMSD / $\\mathbf{\\AA}$"),
    ("rmsd_angles", "Angles", "RMSD / rad"),
    ("rmsd_propers", "Propers", "RMSD / rad"),
    ("rmsd_impropers", "Impropers", "RMSD / rad"),
    ("tfd", "Torsion Fingerprints", "TFD"),
]


def load_benchmarks():
    return {
        model_name: BenchmarkResults.from_hdf5(path)
        for model_name, path in MODEL_FILES.items()
    }


def get_conformer_groups(dataset):
    if not dataset.smiles:
        return []

    groups = []
    current = dataset.smiles[0]
    buff = [0]
    for i, smiles in enumerate(dataset.smiles[1:], start=1):
        if smiles != current:
            groups.append(buff)
            current = smiles
            buff = [i]
        else:
            buff.append(i)

    groups.append(buff)
    return groups


def get_ddE(bmark, conformers, debug=False):
    dE = []
    MSD = []

    for group_idx, conf in enumerate(conformers):
        qm = np.array(bmark.energy_qm)[conf]

        local_idx = np.argmin(qm)
        global_idx = conf[local_idx]

        min_0 = bmark.energy_min[global_idx]
        qm_0 = bmark.energy_qm[global_idx]

        if debug and group_idx < 5:
            print(f"--- Group {group_idx} | SMILES: {bmark.smiles[conf[0]]} ---")
            print(f"  Global indices in group: {conf}")
            print(f"  Local argmin: {local_idx} -> Global index: {global_idx}")
            print(f"  Reference QM Energy  (qm_0) : {qm_0:.2f} kJ/mol")
            print(f"  Reference Min Energy (min_0): {min_0:.2f} kJ/mol")

        n = 0
        tmp = []
        for c in conf:
            if c == global_idx:
                continue

            emin = bmark.energy_min[c]
            eqm = bmark.energy_qm[c]

            dmin = emin - min_0
            dqm = eqm - qm_0
            dde = dmin - dqm

            if bmark.rmsd_cart[c] <= 1.0:
                tmp.append(dde)
                n += 1
            dE.append(dde)

            if debug and group_idx < 5:
                print(
                    f"    Conformer {c}: dqm = {dqm:>7.2f}, dmin = {dmin:>7.2f} -> ddE = {dde:>7.2f}"
                )

        if n > 1:
            MSD.append(sum(tmp) / (n - 1))

    return dE, MSD


def compute_dde_by_model(benchmarks):
    dde_by_model = {}
    for model_name in MODEL_ORDER:
        conformers = get_conformer_groups(benchmarks[model_name])
        dE, _ = get_ddE(benchmarks[model_name], conformers)
        dde_by_model[model_name] = dE
    return dde_by_model


def pdf_cdf(series):
    n, bins = np.histogram(series, bins=freedman_diaconis_bins(series), density=True)
    x = 0.5 * (bins[1:] + bins[:-1])
    return x, n, histogram_cdf(n, bins)[:-1]


def plot_cdf(data, ax):
    for color, series in zip((MODEL_COLORS[m] for m in MODEL_ORDER), data):
        x, _, cdf = pdf_cdf(series)
        ax.plot(x, cdf, lw=3, c=color)


def _style_grid_ticks(ax_grid, x_major_direction="in"):
    nrows, ncols = ax_grid.shape
    for r in range(nrows):
        for c in range(ncols):
            ax = ax_grid[r, c]
            ax.minorticks_on()
            ax.tick_params(
                labelsize=12,
                axis="y",
                which="major",
                length=8.0,
                direction="in",
            )
            ax.tick_params(
                labelsize=12,
                axis="y",
                which="minor",
                length=4.0,
                direction="in",
            )
            ax.tick_params(
                labelsize=12,
                axis="x",
                which="major",
                length=8.0,
                direction=x_major_direction,
            )
            ax.tick_params(
                labelsize=12,
                axis="x",
                which="minor",
                length=4.0 if x_major_direction == "in" else 0.0,
                direction="in",
            )


def plot_cdf_grid(benchmarks, output_path="cdf_smallmols.pdf"):
    legend_handles = [
        mpatches.Patch(color=MODEL_COLORS[m], label=m)
        for m in MODEL_ORDER
    ]

    fig, ax = plt.subplots(2, 3, figsize=(9, 7), layout="tight", sharey=True)

    for idx, (metric_attr, metric_title, x_label) in enumerate(METRIC_SPECS):
        row, col = divmod(idx, 3)
        metric_series = [getattr(benchmarks[m], metric_attr) for m in MODEL_ORDER]
        plot_cdf(metric_series, ax[row, col])

        ax[row, col].set_ylim(0)
        ax[row, col].set_title(metric_title, fontsize=14, fontweight="bold")
        ax[row, col].set_xlabel(x_label, fontsize=14, fontweight="bold")

    ax[0, 0].set_ylabel("CDF", fontsize=14, fontweight="bold")
    ax[1, 0].set_ylabel("CDF", fontsize=14, fontweight="bold")
    ax[1, 1].legend(handles=legend_handles, fontsize=12)

    _style_grid_ticks(ax, x_major_direction="in")
    fig.savefig(output_path, transparent=True)


def plot_dde_violin(dde_by_model, output_path="violins_dde.pdf"):
    fig, ax = plt.subplots(figsize=(6, 5), layout="tight")

    ax.axhline(0.0, ls="--", lw=2, color="firebrick", alpha=0.5)

    data = [dde_by_model[m] for m in MODEL_ORDER]
    parts = violin_prune_outliers(ax, data, mad_k=10, allow_negative=True)
    style_violins(parts)

    ax.set_xticks([1, 2, 3, 4], list(MODEL_ORDER))
    ax.set_ylabel(r"$\\mathbf{\\Delta \\Delta E\ /\ kJ \\cdot mol^{-1}}$", fontsize=14, fontweight="bold")

    ax.minorticks_on()
    ax.tick_params(labelsize=14, axis="y", which="major", length=8.0, direction="in")
    ax.tick_params(labelsize=14, axis="y", which="minor", length=4.0, direction="in")
    ax.tick_params(labelsize=14, axis="x", which="major", length=8.0, direction="out")
    ax.tick_params(labelsize=14, axis="x", which="minor", length=0.0, direction="in")

    ax.set_ylim(-100, 100)
    fig.savefig(output_path, transparent=True)


def plot_metric_violin_grid(benchmarks, output_path="violin_smallmols.pdf"):
    fig, ax = plt.subplots(2, 3, figsize=(9, 7), layout="tight")

    for idx, (metric_attr, metric_title, y_label) in enumerate(METRIC_SPECS):
        row, col = divmod(idx, 3)
        metric_series = [getattr(benchmarks[m], metric_attr) for m in MODEL_ORDER]

        parts = violin_prune_outliers(ax[row, col], metric_series, mad_k=10)
        style_violins(parts)

        ax[row, col].set_title(metric_title, fontsize=14, fontweight="bold")
        ax[row, col].set_xticks((1, 2, 3, 4), MODEL_ORDER, rotation=45)
        ax[row, col].set_ylabel(y_label, fontsize=12, fontweight="bold")

    _style_grid_ticks(ax, x_major_direction="out")
    fig.savefig(output_path, transparent=True)


def run_motif_enrichment(
    benchmarks,
    out_dir="enrichment_tables",
    bayes_samples=20000,
    random_state=42,
):
    os.makedirs(out_dir, exist_ok=True)
    all_significant_results = []

    for model_name in MODEL_ORDER:
        bmark = benchmarks[model_name]

        for metric_attr, metric_name, _ in METRIC_SPECS:
            metric_data = getattr(bmark, metric_attr)

            df_best = compute_motif_enrichment_table(
                rmsd=metric_data,
                smiles=bmark.smiles,
                motifs=motifs,
                quantile=1,
                bayes_samples=bayes_samples,
                random_state=random_state,
            )

            sig_best = df_best[
                (df_best["count_all"] >= 10)
                & (df_best["enrichment_low_vs_high"] >= 1.25)
                & (df_best["bayes_prob_enrichment_gt_1"] >= 0.99)
            ].copy()

            if not sig_best.empty:
                sig_best["model"] = model_name
                sig_best["metric"] = metric_name
                sig_best["tier"] = "best_1_percent"
                sig_best["enrichment_score"] = sig_best["enrichment_low_vs_high"]
                sig_best["confidence"] = sig_best["bayes_prob_enrichment_gt_1"]

                out_path = os.path.join(
                    out_dir,
                    f"enrichment_{model_name}_{metric_name}_best_1_percent.csv",
                )
                sig_best.to_csv(out_path, index=False)
                all_significant_results.append(sig_best)

            df_worst = compute_motif_enrichment_table(
                rmsd=metric_data,
                smiles=bmark.smiles,
                motifs=motifs,
                quantile=99,
                bayes_samples=bayes_samples,
                random_state=random_state,
            )

            df_worst["enrichment_high_vs_low"] = 1.0 / df_worst["enrichment_low_vs_high"]
            df_worst["bayes_prob_high_gt_low"] = 1.0 - df_worst["bayes_prob_enrichment_gt_1"]

            sig_worst = df_worst[
                (df_worst["count_all"] >= 10)
                & (df_worst["enrichment_high_vs_low"] >= 1.25)
                & (df_worst["bayes_prob_high_gt_low"] >= 0.99)
            ].copy()

            if not sig_worst.empty:
                sig_worst["model"] = model_name
                sig_worst["metric"] = metric_name
                sig_worst["tier"] = "worst_1_percent"
                sig_worst["enrichment_score"] = sig_worst["enrichment_high_vs_low"]
                sig_worst["confidence"] = sig_worst["bayes_prob_high_gt_low"]

                out_path = os.path.join(
                    out_dir,
                    f"enrichment_{model_name}_{metric_name}_worst_1_percent.csv",
                )
                sig_worst.to_csv(out_path, index=False)
                all_significant_results.append(sig_worst)

    if not all_significant_results:
        return pd.DataFrame()

    return pd.concat(all_significant_results, ignore_index=True)


def plot_comparative_heatmap(df, metric, tier, savepath=None):
    subset = df[(df["metric"] == metric) & (df["tier"] == tier)]
    if subset.empty:
        return

    pivot = subset.pivot(index="motif", columns="model", values="enrichment_score")
    pivot = pivot.fillna(1.0)

    log_pivot = np.log2(pivot)

    fig, ax = plt.subplots(figsize=(6, len(pivot) * 0.4 + 1.5), layout="tight")

    sns.heatmap(
        log_pivot,
        annot=pivot,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": r"$\log_2(\mathrm{Enrichment})$"},
        ax=ax,
    )

    ax.set_title(
        f"Motif Enrichment: {metric}\n({tier.replace('_', ' ').title()})",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.tick_params(axis="y", rotation=0)
    ax.tick_params(axis="x", rotation=45)

    if savepath:
        fig.savefig(savepath, dpi=150, transparent=True)
    plt.close(fig)


def generate_enrichment_heatmaps(df_master):
    if df_master.empty:
        return

    for metric_attr, metric_name, _ in METRIC_SPECS:
        for tier in ("best_1_percent", "worst_1_percent"):
            plot_comparative_heatmap(
                df_master,
                metric_name,
                tier,
                savepath=f"heatmap_{metric_attr}_{tier}.pdf",
            )


def main():
    benchmarks = load_benchmarks()
    dde_by_model = compute_dde_by_model(benchmarks)

    plot_cdf_grid(benchmarks)
    plot_dde_violin(dde_by_model)
    plot_metric_violin_grid(benchmarks)

    df_master = run_motif_enrichment(benchmarks)
    generate_enrichment_heatmaps(df_master)


if __name__ == "__main__":
    main()
