#%%
import numpy as np
import matplotlib.pyplot as plt

from itertools  import combinations
from src.io     import BenchmarkResults
from src.stats  import freedman_diaconis_bins, histogram_cdf
from src.graphs import violin_prune_outliers, style_violins

# %%
openff_data   = "openff_2048.hdf5"
espaloma_data = "espaloma_complete.hdf5"

openff_bmark   = BenchmarkResults.from_hdf5(openff_data)
espaloma_bmark = BenchmarkResults.from_hdf5(espaloma_data)
# %%
def get_conformer_groups(dataset):
    groups  = []
    current = dataset.smiles[0]
    buff    = [0]
    for i, smiles in enumerate(dataset.smiles[1:], start = 1):
        if smiles != current:
            current = smiles
            groups.append(buff)
            buff = [i]
            continue
        buff.append(i)
    return groups

# %%
conformers_openff   = get_conformer_groups(openff_bmark)
conformers_espaloma = get_conformer_groups(espaloma_bmark)

#%%
def get_ddE(bmark, conformers):
    dE  = []
    MSD = []
    for conf in conformers:

        qm  = np.array(bmark.energy_qm)[conf]
        idx = np.argmin(qm)
        
        min_0 = bmark.energy_min[idx]
        qm_0  = bmark.energy_qm[idx]
        n = 0
        tmp = []
        for i, c in enumerate(conf):
            if i == idx:
                continue
            emin = bmark.energy_min[c]
            eqm  = bmark.energy_qm[c]
            dmin = emin - min_0
            dqm  = eqm  - qm_0
            dde  = dmin - dqm
            if bmark.rmsd_cart[c] <= 1.0:
                tmp.append(dde)
                n += 1
            dE.append(dde)

        if n > 1:
            MSD.append(sum(tmp)/(n-1))
    return dE, MSD

# %%
dE_openff, MSD_openff     = get_ddE(openff_bmark,   conformers_openff)
dE_espaloma, MSD_espaloma = get_ddE(espaloma_bmark, conformers_espaloma)

#%%
def pdf_cdf(series):
    n, bins = np.histogram(series, bins = freedman_diaconis_bins(series), density = True)
    x = 0.5*(bins[1:] + bins[:-1])
    return x, n, histogram_cdf(n, bins)[:-1]

#%%
def plot_cdf(data, ax):

    colors = ["royalblue", "firebrick", "gold"]
    labels = ["OpenFF", "Espaloma", "Our Model"]

    for n, series in enumerate(data):
        x, _, cdf = pdf_cdf(series)
        ax.plot(x, cdf, lw = 3, c = colors[n], label = labels[n])


#%%
fig, ax = plt.subplots(2, 3, figsize=(9, 7), layout = "tight")

plot_cdf([openff_bmark.rmsd_cart, espaloma_bmark.rmsd_cart], ax[0,0])
ax[0,0].set_ylim(0)
ax[0,0].set_title("Structure", fontsize = 14, fontweight = "bold")
ax[0,0].set_xlabel(r"RMSD / $\mathbf{\AA}$", fontsize = 14, fontweight = "bold")
ax[0,0].set_ylabel("CDF", fontsize = 14, fontweight = "bold")
ax[0,0].tick_params(labelsize = 12)
ax[0,0].legend(fontsize = 12)

plot_cdf([openff_bmark.rmsd_bonds, espaloma_bmark.rmsd_bonds], ax[0,1])
ax[0,1].set_ylim(0)
ax[0,1].set_title("Bonds", fontsize = 14, fontweight = "bold")
ax[0,1].set_xlabel(r"RMSD / $\mathbf{\AA}$", fontsize = 14, fontweight = "bold")
ax[0,1].set_ylabel("CDF", fontsize = 14, fontweight = "bold")
ax[0,1].tick_params(labelsize = 12)
ax[0,1].legend(fontsize = 12)

plot_cdf([openff_bmark.rmsd_angles, espaloma_bmark.rmsd_angles], ax[0,2])
ax[0,2].set_ylim(0)
ax[0,2].set_title("Angles", fontsize = 14, fontweight = "bold")
ax[0,2].set_xlabel(r"RMSD / rad", fontsize = 14, fontweight = "bold")
ax[0,2].set_ylabel("CDF", fontsize = 14, fontweight = "bold")
ax[0,2].tick_params(labelsize = 12)
ax[0,2].legend(fontsize = 12)

plot_cdf([openff_bmark.rmsd_propers, espaloma_bmark.rmsd_propers], ax[1,0])
ax[1,0].set_ylim(0)
ax[1,0].set_title("Propers", fontsize = 14, fontweight = "bold")
ax[1,0].set_xlabel(r"RMSD / rad", fontsize = 14, fontweight = "bold")
ax[1,0].set_ylabel("CDF", fontsize = 14, fontweight = "bold")
ax[1,0].tick_params(labelsize = 12)
ax[1,0].legend(fontsize = 12)

plot_cdf([openff_bmark.rmsd_impropers, espaloma_bmark.rmsd_impropers], ax[1,1])
ax[1,1].set_ylim(0)
ax[1,1].set_title("Impropers", fontsize = 14, fontweight = "bold")
ax[1,1].set_xlabel(r"RMSD / rad", fontsize = 14, fontweight = "bold")
ax[1,1].set_ylabel("CDF", fontsize = 14, fontweight = "bold")
ax[1,1].tick_params(labelsize = 12)
ax[1,1].legend(fontsize = 12)

plot_cdf([openff_bmark.tfd, espaloma_bmark.tfd], ax[1,2])
ax[1,2].set_ylim(0)
ax[1,2].set_title("Torsion Fingerprints", fontsize = 14, fontweight = "bold")
ax[1,2].set_xlabel(r"TFD", fontsize = 14, fontweight = "bold")
ax[1,2].set_ylabel("CDF", fontsize = 14, fontweight = "bold")
ax[1,2].tick_params(labelsize = 12)
ax[1,2].legend(fontsize = 12)


# %%
fig, ax = plt.subplots(figsize = (7,7), layout = "tight")
ax2 = ax.twinx()

n, b, _ = ax.hist(MSD_openff, bins = freedman_diaconis_bins(dE_openff), density= True, color = "royalblue", edgecolor = "k", alpha = 0.5)
cdf = histogram_cdf(n, b)
ax2.plot(b, cdf, c = "royalblue", lw = 3)

n, b, _ = ax.hist(MSD_espaloma, bins = freedman_diaconis_bins(dE_espaloma), density= True, color = "firebrick", edgecolor = "k", alpha = 0.5)
cdf = histogram_cdf(n, b)
ax2.plot(b, cdf, c = "firebrick", lw = 3)

ax2.set_ylim(0)

# %%
fig, ax = plt.subplots(2, 3, figsize=(9, 7), layout="tight")

parts = violin_prune_outliers(
    ax[0, 0],
    [openff_bmark.rmsd_cart, espaloma_bmark.rmsd_cart],
)
style_violins(parts)
ax[0,0].set_title("Structure", 
                  fontsize = 14, fontweight = "bold")
ax[0,0].set_xticks((1, 2), ("OpenFF", "Espaloma"))
ax[0,0].set_ylabel(r"RMSD / $\mathbf{\AA}$",
                    fontsize = 12, fontweight = "bold")
ax[0,0].tick_params(labelsize = 12)

parts = violin_prune_outliers(
    ax[0, 1],
    [openff_bmark.rmsd_bonds, espaloma_bmark.rmsd_bonds],
)
style_violins(parts)
ax[0,1].set_title("Bonds", 
                  fontsize = 14, fontweight = "bold")
ax[0,1].set_xticks((1, 2), ("OpenFF", "Espaloma"))
ax[0,1].set_ylabel(r"RMSD / $\mathbf{\AA}$",
                    fontsize = 12, fontweight = "bold")
ax[0,1].tick_params(labelsize = 12)

parts = violin_prune_outliers(
    ax[0, 2],
    [openff_bmark.rmsd_angles, espaloma_bmark.rmsd_angles],
)
style_violins(parts)
ax[0,2].set_title("Angles", 
                  fontsize = 14, fontweight = "bold")
ax[0,2].set_xticks((1, 2), ("OpenFF", "Espaloma"))
ax[0,2].set_ylabel(r"RMSD / rad",
                    fontsize = 12, fontweight = "bold")
ax[0,2].tick_params(labelsize = 12)

parts = violin_prune_outliers(
    ax[1, 0],
    [openff_bmark.rmsd_propers, espaloma_bmark.rmsd_propers],
)
style_violins(parts)
ax[1,0].set_title("Propers", 
                  fontsize = 14, fontweight = "bold")
ax[1,0].set_xticks((1, 2), ("OpenFF", "Espaloma"))
ax[1,0].set_ylabel(r"RMSD / rad",
                    fontsize = 12, fontweight = "bold")
ax[1,0].tick_params(labelsize = 12)

parts = violin_prune_outliers(
    ax[1, 1],
    [openff_bmark.rmsd_impropers, espaloma_bmark.rmsd_impropers],
)
style_violins(parts)
ax[1,1].set_title("Impropers", 
                  fontsize = 14, fontweight = "bold")
ax[1,1].set_xticks((1, 2), ("OpenFF", "Espaloma"))
ax[1,1].set_ylabel(r"RMSD / rad",
                    fontsize = 12, fontweight = "bold")
ax[1,1].tick_params(labelsize = 12)

parts = violin_prune_outliers(
    ax[1, 2],
    [openff_bmark.tfd, espaloma_bmark.tfd],
)
style_violins(parts)
ax[1,2].set_title("Torsion Fingerprint", 
                  fontsize = 14, fontweight = "bold")
ax[1,2].set_xticks((1, 2), ("OpenFF", "Espaloma"))
ax[1,2].set_ylabel(r"TFD",
                    fontsize = 12, fontweight = "bold")
ax[1,2].tick_params(labelsize = 12)

plt.show()

#%%
from src.chem import motifs, compute_motif_enrichment_table

#%%
df_enrich = compute_motif_enrichment_table(
    rmsd=espaloma_bmark.rmsd_cart,
    smiles=espaloma_bmark.smiles,
    motifs=motifs,
    quantile=25,          # low-RMSD = best 25%
    bayes_samples=20000,
    random_state=42,
)

print(df_enrich.head(20))
# %%

df_filter = df_enrich[
    (df_enrich["count_all"] >= 10) &
    (df_enrich["enrichment_low_vs_high"] >= 1.25) &
    (df_enrich["fisher_p_low_gt_high"] <= 1e-1) &
    (df_enrich["bayes_prob_enrichment_gt_1"] >= 0.99)
]

# %%
