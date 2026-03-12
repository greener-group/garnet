#%%
import numpy              as np
import pickle             as pk
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches

from src.hbonds import karplus_dict, backbone_torsions
from src.stats  import moving_block_bootstrap

#%%
# Load results
with open("results_jcoup_garnet.pk", "rb") as f:
    RESULTS_GARNET = pk.load(f)

with open("results_jcoup_amber.pk", "rb") as f:
    RESULTS_AMBER = pk.load(f)

with open("results_jcoup_espaloma.pk", "rb") as f:
    RESULTS_ESPALOMA = pk.load(f)

#%%
# Helpers (keep core logic; reduce duplication)

PROTEINS = ["gb3", "ubq", "bpti", "hewl"]
TORSIONS = [tuple(t) for t in karplus_dict.keys()]

MODELS = {
    "Amber":    RESULTS_AMBER,
    "Garnet":   RESULTS_GARNET,
    "Espaloma": RESULTS_ESPALOMA,
}

MODEL_NAMES = tuple(MODELS.keys())
HBD_PROTEINS = ("gb3", "ubq")
MODEL_POSITIONS = np.array([1, 2, 3])
ANE_GROUP_CENTERS = [2, 7, 12]


def _set_model_xticks(ax):
    ax.set_xticks(MODEL_POSITIONS, MODEL_NAMES)


def _set_ane_group_xticks(ax):
    ax.set_xticks(ANE_GROUP_CENTERS, MODEL_NAMES)


def _flatten_hbd(results, prot):
    out = []
    for resids, values in results[prot]["HBD"].items():
        for v in values:
            out.append([v[0], v[1]])
    return np.asarray(out)


def _collect_jcoup_samples(results, prot, torsions):
    all_x = []
    bbn_x = []
    sdc_x = []

    for torsion in torsions:
        if torsion not in results[prot].keys():
            continue

        for resid, values in results[prot][torsion].items():
            for v in values:
                row = [v[0], v[1], v[2], v[3]]
                if torsion in backbone_torsions:
                    all_x.append(row)
                    bbn_x.append(row)
                else:
                    all_x.append(row)
                    sdc_x.append(row)

    return all_x, bbn_x, sdc_x


def _bootstrap(data, estimator, block_size=None, n_boot=2000, alpha=0.001):
    return moving_block_bootstrap(
        data,
        estimator=estimator,
        block_size=block_size,  # let it pick based on autocorrelation of first column
        n_boot=n_boot,
        alpha=alpha,
    )


def _style_axes_ticks(ax):
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
        direction="out",
    )
    ax.tick_params(
        labelsize=12,
        axis="x",
        which="minor",
        length=0.0,
        direction="in",
    )


def _violin_with_ci(
    ax,
    boot_samples_list,
    ci_list,
    positions,
    facecolors,
    cap_width=0.25,
    edgecolor="k",
    alpha=0.85,
    linewidth=1.5,
    median_marker_size=25,
):
    parts = ax.violinplot(
        boot_samples_list,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    ax.scatter(
        positions,
        [np.median(b) for b in boot_samples_list],
        zorder=1e6,
        c="w",
        s=median_marker_size,
    )

    ax.vlines(
        positions,
        [ci[0] for ci in ci_list],
        [ci[1] for ci in ci_list],
        color="k",
        linewidth=linewidth,
    )
    ax.hlines(
        [ci[0] for ci in ci_list],
        positions - cap_width / 2,
        positions + cap_width / 2,
        color="k",
        linewidth=linewidth,
    )
    ax.hlines(
        [ci[1] for ci in ci_list],
        positions - cap_width / 2,
        positions + cap_width / 2,
        color="k",
        linewidth=linewidth,
    )

    for n, body in enumerate(parts["bodies"]):
        body.set_facecolor(facecolors[n])
        body.set_edgecolor(edgecolor)
        body.set_alpha(alpha)

    return parts


#%%
# Build HBD datasets (GB3 / Ubq)
hbd = {
    "Amber": {
        "gb3": _flatten_hbd(RESULTS_AMBER, "gb3"),
        "ubq": _flatten_hbd(RESULTS_AMBER, "ubq"),
    },
    "Garnet": {
        "gb3": _flatten_hbd(RESULTS_GARNET, "gb3"),
        "ubq": _flatten_hbd(RESULTS_GARNET, "ubq"),
    },
    "Espaloma": {
        "gb3": _flatten_hbd(RESULTS_ESPALOMA, "gb3"),
        "ubq": _flatten_hbd(RESULTS_ESPALOMA, "ubq"),
    },
}

#%%
# Build torsion datasets for all proteins / models
all_by_model = {m: {p: [] for p in PROTEINS} for m in MODEL_NAMES}
bbn_by_model = {m: {p: [] for p in PROTEINS} for m in MODEL_NAMES}
sdc_by_model = {m: {p: [] for p in PROTEINS} for m in MODEL_NAMES}

for model_name, results in MODELS.items():
    for prot in PROTEINS:
        all_x, bbn_x, sdc_x = _collect_jcoup_samples(results, prot, TORSIONS)
        all_by_model[model_name][prot] = all_x
        bbn_by_model[model_name][prot] = bbn_x
        sdc_by_model[model_name][prot] = sdc_x

#%%
# Estimators (unchanged)
def chisq_estimator(sample):

    diffsq = (sample[:,0] - sample[:,1])**2
    out    = diffsq / (0.12**2)

    return np.mean(out)

def rmse_estimator(sample):
    # sample shape: (N, 2) with columns [measured, truth]
    diff = sample[:, 0] - sample[:, 1]
    return np.sqrt(np.mean(diff**2))

def ANE_estimator(sample):
    """
    sample: array of shape (N, 4)
            columns: j_comp, j_exp, j_min, j_max
    """
    sample = np.asarray(sample, dtype=float)

    j_comp = sample[:, 0]
    j_exp  = sample[:, 1]
    j_min  = sample[:, 2]
    j_max  = sample[:, 3]

    j_exp_clamped = np.clip(j_exp, j_min, j_max)

    denom = j_max - j_min

    out = np.abs(j_comp - j_exp_clamped) / denom

    return out.mean()

#%%
# RMSE on HBonds (GB3 / Ubq)
rmse_boot = {m: {} for m in MODEL_NAMES}

for model_name in MODEL_NAMES:
    for prot in HBD_PROTEINS:
        (rmse_hat, var_rmse_hat,
         boot_rmses, ci_rmse,
         L_opt, tau_int) = _bootstrap(hbd[model_name][prot], rmse_estimator)

        rmse_boot[model_name][prot] = {
            "rmse": rmse_hat,
            "var": var_rmse_hat,
            "boot": boot_rmses,
            "ci": ci_rmse,
            "L_opt": L_opt,
            "tau_int": tau_int,
        }

#%%
# Violin of RMSE bootstraps (GB3 / Ubq)
cap_width = 0.25
loc_x = MODEL_POSITIONS

fig, ax = plt.subplots(1, 2, figsize=(7, 5), sharex=True, sharey=True, layout="tight")

for j, prot in enumerate(HBD_PROTEINS):
    boots = [rmse_boot[m][prot]["boot"] for m in MODEL_NAMES]
    cis = [rmse_boot[m][prot]["ci"] for m in MODEL_NAMES]

    parts = _violin_with_ci(
        ax[j],
        boots,
        cis,
        positions=loc_x,
        facecolors=["#785ef0"] * len(boots),
        cap_width=cap_width,
        linewidth=3,
        median_marker_size=45,
    )

    _set_model_xticks(ax[j])

for body in parts["bodies"]:
    body.set_facecolor("#785ef0")
    body.set_edgecolor("k")
    body.set_alpha(0.85)

ax[0].set_ylabel(r"RMSE[$\mathbf{^3J_{N,C^{\prime}}}$] / Hz", fontsize=14, fontweight="bold")

_style_axes_ticks(ax[0])
_style_axes_ticks(ax[1])

fig.savefig("hbonds_rmse_violin.pdf", transparent = True)

#%%
# Bar plot of RMSE (GB3 / Ubq)
fig, ax = plt.subplots(1, 2, figsize=(7, 5), sharex=True, sharey=True, layout="tight")

for j, prot in enumerate(HBD_PROTEINS):
    vals = [rmse_boot[m][prot]["rmse"] for m in MODEL_NAMES]
    cis = [rmse_boot[m][prot]["ci"] for m in MODEL_NAMES]
    yerr = [
        [v - ci[0] for v, ci in zip(vals, cis)],
        [ci[1] - v for v, ci in zip(vals, cis)],
    ]

    ax[j].set_title(f"{prot.upper()} HBonds", fontsize=16, fontweight="bold")
    ax[j].bar(
        MODEL_POSITIONS,
        vals,
        yerr=yerr,
        color="#785ef0",
        edgecolor="k",
        width=0.5,
        capstyle="round",
        error_kw={"capsize": 16},
    )
    _set_model_xticks(ax[j])
    _style_axes_ticks(ax[j])

ax[0].set_ylabel(r"RMSE[$\mathbf{^3J_{N,C^{\prime}}}$] / Hz", fontsize=14, fontweight="bold")

fig.savefig("hbonds_rmse_bar.pdf", transparent = True)

#%%
# Chi^2 on HBonds (GB3 / Ubq)
chisq_boot = {m: {} for m in MODEL_NAMES}

for model_name in MODEL_NAMES:
    for prot in HBD_PROTEINS:
        (chisq_hat, var_chisq_hat,
         boot_chisqs, ci_chisq,
         L_opt, tau_int) = _bootstrap(hbd[model_name][prot], chisq_estimator)

        chisq_boot[model_name][prot] = {
            "chisq": chisq_hat,
            "var": var_chisq_hat,
            "boot": boot_chisqs,
            "ci": ci_chisq,
            "L_opt": L_opt,
            "tau_int": tau_int,
        }

#%%
# Bar plot of Chi^2 (GB3 / Ubq)
fig, ax = plt.subplots(1, 2, figsize=(7, 5), sharex=True, sharey=True, layout="tight")

for j, prot in enumerate(HBD_PROTEINS):
    vals = [chisq_boot[m][prot]["chisq"] for m in MODEL_NAMES]
    cis = [chisq_boot[m][prot]["ci"] for m in MODEL_NAMES]
    yerr = [
        [v - ci[0] for v, ci in zip(vals, cis)],
        [ci[1] - v for v, ci in zip(vals, cis)],
    ]

    ax[j].set_title(f"{prot.upper()} HBonds", fontsize=16, fontweight="bold")
    ax[j].bar(
        MODEL_POSITIONS,
        vals,
        yerr=yerr,
        color="#785ef0",
        edgecolor="k",
        width=0.5,
        capstyle="round",
        error_kw={"capsize": 16},
    )
    _set_model_xticks(ax[j])
    _style_axes_ticks(ax[j])

ax[0].set_ylabel(r"$\mathbf{\chi^2 [ ^3J_{N,C^{\prime}}]}$", fontsize=14, fontweight="bold")

fig.savefig("hbonds_chisq_bar.pdf", transparent = True)

#%%
# Violin of Chi^2 bootstraps (GB3 / Ubq)
cap_width = 0.25
loc_x = MODEL_POSITIONS

fig, ax = plt.subplots(1, 2, figsize=(7, 5), sharex=True, sharey=True, layout="tight")

for j, prot in enumerate(HBD_PROTEINS):
    ax[j].set_title(prot.upper(), fontsize=16, fontweight="bold")

    boots = [chisq_boot[m][prot]["boot"] for m in MODEL_NAMES]
    cis = [chisq_boot[m][prot]["ci"] for m in MODEL_NAMES]

    _violin_with_ci(
        ax[j],
        boots,
        cis,
        positions=loc_x,
        facecolors=["#785ef0"] * len(boots),
        cap_width=cap_width,
        linewidth=3,
        median_marker_size=45,
    )

    _set_model_xticks(ax[j])
    _style_axes_ticks(ax[j])

ax[0].set_ylabel(r"$\mathbf{\chi^2 [ ^3J_{N,C^{\prime}}]}$", fontsize=14, fontweight="bold")

fig.savefig("hbonds_chisq_violin.pdf", transparent = True)

#%%
# ANE bootstraps for torsions (all/bbn/sdc) for each protein and model
ane_boot = {m: {} for m in MODEL_NAMES}

for prot in PROTEINS:
    for model_name in MODEL_NAMES:

        # All torsions
        (ane_hat_all, var_ane_hat_all,
         boot_anes_all, ci_ane_all,
         L_opt_all, tau_int_all) = _bootstrap(
            np.asarray(all_by_model[model_name][prot], dtype=float),
            ANE_estimator,
        )

        # Backbone
        if prot == "hewl":
            ane_hat_bbn     = np.nan
            var_ane_hat_bbn = np.nan
            ci_ane_bbn      = (np.nan, np.nan)
            boot_anes_bbn   = [np.nan]
        else:
            (ane_hat_bbn, var_ane_hat_bbn,
             boot_anes_bbn, ci_ane_bbn,
             L_opt_bbn, tau_int_bbn) = _bootstrap(
                np.asarray(bbn_by_model[model_name][prot], dtype=float),
                ANE_estimator,
            )

        # Side chains
        (ane_hat_sdc, var_ane_hat_sdc,
         boot_anes_sdc, ci_ane_sdc,
         L_opt_sdc, tau_int_sdc) = _bootstrap(
            np.asarray(sdc_by_model[model_name][prot], dtype=float),
            ANE_estimator,
        )

        ane_boot[model_name][prot] = {
            "all": {"ane": ane_hat_all, "var": var_ane_hat_all, "boot": boot_anes_all, "ci": ci_ane_all},
            "bbn": {"ane": ane_hat_bbn, "var": var_ane_hat_bbn, "boot": boot_anes_bbn, "ci": ci_ane_bbn},
            "sdc": {"ane": ane_hat_sdc, "var": var_ane_hat_sdc, "boot": boot_anes_sdc, "ci": ci_ane_sdc},
        }

#%%
# Colors / legend (same scheme)
c_all   = "#785ef0"
c_bbn   = "#98245c"
c_sdc   = "#fe6100"

legend_handles = [
    mpatches.Patch(color=c_all, label="All"),
    mpatches.Patch(color=c_bbn, label="Backbone"),
    mpatches.Patch(color=c_sdc, label="Sidechain"),
]

#%%
# Violin plots of ANE bootstraps (2x2 proteins)
cap_width = 0.25

def _ane_positions_and_colors(include_backbone=True, model_index=0):
    # Positions are grouped by model: 1..3, 6..8, 11..13 (gap between models).
    base = 1 + model_index * 5
    if include_backbone:
        positions = [base, base + 1, base + 2]
        colors = [c_all, c_bbn, c_sdc]
    else:
        positions = [base, base + 2]  # keep visual spacing
        colors = [c_all, c_sdc]
    return positions, colors


def _plot_ane_violin(ax, prot):
    if prot == "hewl":
        include_bbn = False
    else:
        include_bbn = True

    boots = []
    cis   = []
    pos   = []
    cols  = []

    for i, model_name in enumerate(MODEL_NAMES):
        p_i, c_i = _ane_positions_and_colors(include_backbone=include_bbn, model_index=i)
        pos.extend(p_i)
        cols.extend(c_i)

        if include_bbn:
            boots.extend([
                ane_boot[model_name][prot]["all"]["boot"],
                ane_boot[model_name][prot]["bbn"]["boot"],
                ane_boot[model_name][prot]["sdc"]["boot"],
            ])
            cis.extend([
                ane_boot[model_name][prot]["all"]["ci"],
                ane_boot[model_name][prot]["bbn"]["ci"],
                ane_boot[model_name][prot]["sdc"]["ci"],
            ])
        else:
            boots.extend([
                ane_boot[model_name][prot]["all"]["boot"],
                ane_boot[model_name][prot]["sdc"]["boot"],
            ])
            cis.extend([
                ane_boot[model_name][prot]["all"]["ci"],
                ane_boot[model_name][prot]["sdc"]["ci"],
            ])

    _violin_with_ci(
        ax,
        boots,
        cis,
        positions=np.asarray(pos, dtype=float),
        facecolors=cols,
        cap_width=cap_width,
        linewidth=1.5,
        median_marker_size=25,
    )

    _set_ane_group_xticks(ax)


fig, ax = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True, layout="tight")

ax[0,0].set_title("GB3", fontsize=16, fontweight="bold")
_plot_ane_violin(ax[0,0], "gb3")

ax[0,1].set_title("Ubq", fontsize=16, fontweight="bold")
_plot_ane_violin(ax[0,1], "ubq")

ax[1,0].set_title("BPTI", fontsize=16, fontweight="bold")
_plot_ane_violin(ax[1,0], "bpti")

ax[1,1].set_title("HEWL", fontsize=16, fontweight="bold")
_plot_ane_violin(ax[1,1], "hewl")

for r in range(2):
    for c in range(2):
        _style_axes_ticks(ax[r,c])

ax[0,0].set_ylabel(r"ANE[$\mathbf{^3J}$]", fontsize=14, fontweight="bold")
ax[1,0].set_ylabel(r"ANE[$\mathbf{^3J}$]", fontsize=14, fontweight="bold")

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    fontsize=12,
    frameon=False,
    bbox_to_anchor=(0.5, -0.05),
)

fig.savefig("ane_violin.pdf", transparent = True)

#%%
# Bar plots of ANE (2x2 proteins)
def _ane_bar_positions(include_backbone=True, model_index=0):
    base = 1 + model_index * 5
    if include_backbone:
        positions = [base, base + 1, base + 2]
    else:
        positions = [base, base + 2]
    return positions


def _plot_ane_bar(ax, prot):
    include_bbn = (prot != "hewl")

    xs = []
    ys = []
    yerr_lo = []
    yerr_hi = []
    cols = []

    for i, model_name in enumerate(MODEL_NAMES):
        positions = _ane_bar_positions(include_backbone=include_bbn, model_index=i)

        if include_bbn:
            keys = ["all", "bbn", "sdc"]
            cols_i = [c_all, c_bbn, c_sdc]
        else:
            keys = ["all", "sdc"]
            cols_i = [c_all, c_sdc]

        for x, k, col in zip(positions, keys, cols_i):
            v = ane_boot[model_name][prot][k]["ane"]
            ci = ane_boot[model_name][prot][k]["ci"]
            xs.append(x)
            ys.append(v)
            yerr_lo.append(v - ci[0])
            yerr_hi.append(ci[1] - v)
            cols.append(col)

    ax.bar(
        xs,
        ys,
        yerr=[yerr_lo, yerr_hi],
        color=cols,
        edgecolor="k",
        capstyle="round",
        error_kw={"capsize": 6},
    )

    _set_ane_group_xticks(ax)


fig, ax = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True)

ax[0,0].set_title("GB3", fontsize=16, fontweight="bold")
_plot_ane_bar(ax[0,0], "gb3")

ax[0,1].set_title("Ubq", fontsize=16, fontweight="bold")
_plot_ane_bar(ax[0,1], "ubq")

ax[1,0].set_title("BPTI", fontsize=16, fontweight="bold")
_plot_ane_bar(ax[1,0], "bpti")

ax[1,1].set_title("HEWL", fontsize=16, fontweight="bold")
_plot_ane_bar(ax[1,1], "hewl")

for r in range(2):
    for c in range(2):
        _style_axes_ticks(ax[r,c])

ax[0,0].set_ylabel(r"ANE[$\mathbf{^3J}$]", fontsize=14, fontweight="bold")
ax[1,0].set_ylabel(r"ANE[$\mathbf{^3J}$]", fontsize=14, fontweight="bold")

fig.tight_layout(rect = [0, 0.08, 1, 1])

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=3,
    fontsize=12,
    frameon=False,
    bbox_to_anchor=(0.53, 0.02),
)

fig.savefig("ane_bar.pdf", transparent = True)

# %%
