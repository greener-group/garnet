import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import fisher_exact

motifs = {

    # --- Carbonyl-containing groups ---
    "aldehyde"        : Chem.MolFromSmarts("[CH1](=O)[#6]"),
    "ketone"          : Chem.MolFromSmarts("[#6]C(=O)[#6]"),
    "carboxylic_acid" : Chem.MolFromSmarts("[CX3](=O)[OH]"),
    "ester"           : Chem.MolFromSmarts("[CX3](=O)O[#6]"),
    "amide"           : Chem.MolFromSmarts("[CX3](=O)N"),
    "acyl_halide"     : Chem.MolFromSmarts("[CX3](=O)[Cl,Br,I,F]"),
    "anhydride"       : Chem.MolFromSmarts("[CX3](=O)O[CX3](=O)"),
    "imide"           : Chem.MolFromSmarts("[CX3](=O)N([CX3]=O)"),
    "urea"            : Chem.MolFromSmarts("NC(=O)N"),
    "carbonate"       : Chem.MolFromSmarts("OC(=O)O"),
    "thioester"       : Chem.MolFromSmarts("[CX3](=O)S[#6]"),
    "thiourea"        : Chem.MolFromSmarts("NC(=S)N"),
    "isocyanate"      : Chem.MolFromSmarts("[NX2]=C=O"),

    # --- Alcohols, thiols, amines, etc. ---
    "alcohol"         : Chem.MolFromSmarts("[CX4][OH]"),
    "primary_amine"   : Chem.MolFromSmarts("[N;H1,H2;!$(NC(=O))][#6]"),
    "enol"            : Chem.MolFromSmarts("[#6][#6]([OH])=[#6]"),
    "enoxide"         : Chem.MolFromSmarts("[#6][#6]([O-])=[#6]"),
    "secondary_amine" : Chem.MolFromSmarts("[NX3;H1;!$(NC(=O));!-]"),
    "tertiary_amine"  : Chem.MolFromSmarts("[NX3;H0;!$(NC(=O));!-]"),
    "aniline"         : Chem.MolFromSmarts("c[NX3H2,NX2H1-,NX4H3+]"),
    "phenol"          : Chem.MolFromSmarts("c[OX2H,OX1-,OX3H2+]"),
    "thiol"           : Chem.MolFromSmarts("[#6][SX2H,SX1-]"),
    "disulfide"       : Chem.MolFromSmarts("S-S"),

    # --- Multiple bonds / reactive groups ---
    "alkene"          : Chem.MolFromSmarts("C=C"),
    "alkyne"          : Chem.MolFromSmarts("C#C"),
    "conjugated"      : Chem.MolFromSmarts("C=C-C=C"),
    "allene"          : Chem.MolFromSmarts("C=C=C"),
    "nitrile"         : Chem.MolFromSmarts("[CX2]#N"),
    "isothiocyanate"  : Chem.MolFromSmarts("[NX2]=C=S"),

    # --- Aromatics and rings ---
    "aromatic_ring"   : Chem.MolFromSmarts("a1aaaaa1"),
    "benzene"         : Chem.MolFromSmarts("c1ccccc1"),
    "phenyl"          : Chem.MolFromSmarts("[cX3]1ccccc1"),
    "fused_aromatic"  : Chem.MolFromSmarts("a1aaa2aaaaa2a1"),

    # --- Heterocycles: 5- and 6-membered ---
    "pyridine"        : Chem.MolFromSmarts("c1ncccc1"),
    "pyrimidine"      : Chem.MolFromSmarts("c1ncncc1"),
    "pyrrole"         : Chem.MolFromSmarts("c1nccc1"),
    "imidazole"       : Chem.MolFromSmarts("c1ncnc1"),
    "pyrazole"        : Chem.MolFromSmarts("c1nncc1"),
    "thiazole"        : Chem.MolFromSmarts("c1ncsc1"),
    "oxazole"         : Chem.MolFromSmarts("c1ncoc1"),
    "isoxazole"       : Chem.MolFromSmarts("c1nocc1"),
    "indole"          : Chem.MolFromSmarts("c1ccc2nccc2c1"),
    "indazole"        : Chem.MolFromSmarts("c1ccc2nncc2c1"),

    # --- Charged groups ---
    "phosphate"       : Chem.MolFromSmarts("[P;X4](=O)([OX1,OX2H,OX2H0-])([OX1,OX2H,OX2H0-])[OX1,OX2H,OX2H0-]"),
    "quaternary_amine": Chem.MolFromSmarts("[NX4+]"),
    "sulfonium"       : Chem.MolFromSmarts("[SX3+]"),
    "carboxylate"     : Chem.MolFromSmarts("[CX3](=O)[O-]"),
    "sulfonate"       : Chem.MolFromSmarts("[SX4](=O)(=O)[O-]"),

    # --- Sulfur & phosphorus chemistry ---
    "sulfide"         : Chem.MolFromSmarts("[#6]S[#6]"),
    "sulfoxide"       : Chem.MolFromSmarts("S(=O)[#6]"),
    "sulfone"         : Chem.MolFromSmarts("[SX4](=O)(=O)"),
    "sulfonamide"     : Chem.MolFromSmarts("[SX4](=O)(=O)[NX3]"),
    "phosphonate"     : Chem.MolFromSmarts("[PX4](=O)([OX2])([OX2])"),

    # --- Halogens & organometallic flags ---
    "halogen"         : Chem.MolFromSmarts("[F,Cl,Br,I]"),
    "fluorinated_C"   : Chem.MolFromSmarts("C(F)(F)F"),
    "boronic_acid"    : Chem.MolFromSmarts("[BX3](O)(O)"),

    # --- Miscellaneous motifs ---
    "ether"           : Chem.MolFromSmarts("C-O-C"),
    "acetal"          : Chem.MolFromSmarts("[CX4](OC)(OC)"),
    "epoxide"         : Chem.MolFromSmarts("C1OC1"),
    "azide"           : Chem.MolFromSmarts("N=[N+]=[N-]"),
    "diazo"           : Chem.MolFromSmarts("N=N"),
    "guanidine"       : Chem.MolFromSmarts("NC(=N)N"),
    "barbiturate"     : Chem.MolFromSmarts("O=C1NC(=O)NC(=O)1"),

}


def _count_motifs(smiles_list, motifs):
    """
    Count how many molecules in smiles_list contain each motif at least once.
    Returns a dict motif_name -> count.
    """
    counts = {name: 0 for name in motifs}
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        for name, patt in motifs.items():
            if mol.HasSubstructMatch(patt):
                counts[name] += 1
    return counts


def _enrichment_CI_logratio(a, N_low, c, N_high, z=1.96):
    """
    Approximate confidence interval for enrichment ratio:
        E = (a/N_low) / (c/N_high)
    using log-ratio normal approximation.

    Returns (lo, hi). If not defined (zero counts), returns (np.nan, np.nan).
    """
    if a == 0 or c == 0 or N_low == 0 or N_high == 0:
        return (np.nan, np.nan)

    E = (a / N_low) / (c / N_high)
    # standard error for log(E)
    se = np.sqrt((1 / a - 1 / N_low) + (1 / c - 1 / N_high))
    lo = np.exp(np.log(E) - z * se)
    hi = np.exp(np.log(E) + z * se)
    return (lo, hi)


def _bayesian_enrichment(a, N_low, c, N_high, nsamples=20000, rng=None):
    """
    Beta-Binomial model for motif frequencies in low vs high sets.

    p_low  ~ Beta(a+1, N_low-a+1)
    p_high ~ Beta(c+1, N_high-c+1)

    Returns:
        median_enr, (ci_lo, ci_hi), prob_enriched
    where enrichment = p_low / p_high.
    """
    # No trials or no occurrences -> no information
    if N_low == 0 or N_high == 0 or (a + c) == 0:
        return (np.nan, (np.nan, np.nan), np.nan)

    if rng is None:
        rng = np.random.default_rng()

    p_low = rng.beta(a + 1,  N_low - a + 1,  size=nsamples)
    p_high = rng.beta(c + 1, N_high - c + 1, size=nsamples)

    eps = 1e-12
    enr_samples = p_low / (p_high + eps)

    median = np.median(enr_samples)
    ci_lo, ci_hi = np.percentile(enr_samples, [2.5, 97.5])
    prob_enriched = np.mean(enr_samples > 1.0)
    return median, (ci_lo, ci_hi), prob_enriched


def compute_motif_enrichment_table(
    rmsd,
    smiles,
    motifs,
    quantile=25,
    bayes_samples=20000,
    random_state=None,
):
    """
    Parameters
    ----------
    rmsd : array-like, shape (N,)
        RMSD values for each molecule.
    smiles : array-like, length N
        SMILES strings corresponding to rmsd.
    motifs : dict[str, rdkit.Chem.Mol]
        Dictionary of motif name -> SMARTS-compiled RDKit Mol (substructure pattern).
    quantile : float
        Quantile threshold (0-100) for defining "low-RMSD" set.
        E.g. 25 = molecules with RMSD <= 25th percentile are "low".
    bayes_samples : int
        Number of samples for Bayesian enrichment estimation.
    random_state : int or None
        Seed for RNG used in Bayesian step.

    Returns
    -------
    df : pandas.DataFrame
        One row per motif with counts, frequencies, enrichments,
        Fisher p-values, and Bayesian statistics.
    """
    rmsd = np.asarray(rmsd, dtype=float)
    smiles = np.asarray(smiles, dtype=str)
    assert rmsd.shape[0] == smiles.shape[0], "rmsd and smiles size mismatch"

    # Split into low vs high RMSD
    cutoff = np.percentile(rmsd, quantile)
    low_mask = rmsd <= cutoff
    high_mask = ~low_mask

    smiles_all = smiles
    smiles_low = smiles[low_mask]
    smiles_high = smiles[high_mask]

    N_all = len(smiles_all)
    N_low = len(smiles_low)
    N_high = len(smiles_high)

    # Count motif presence
    all_counts = _count_motifs(smiles_all, motifs)
    low_counts = _count_motifs(smiles_low, motifs)
    high_counts = _count_motifs(smiles_high, motifs)

    # RNG for Bayesian
    rng = np.random.default_rng(random_state)

    rows = []
    for name in motifs.keys():
        a = low_counts[name]
        c = high_counts[name]
        count_all = all_counts[name]

        # frequencies
        freq_all = count_all / N_all if N_all > 0 else np.nan
        freq_low = a / N_low if N_low > 0 else np.nan
        freq_high = c / N_high if N_high > 0 else np.nan

        # enrichments
        enr_low_vs_all = (
            freq_low / freq_all if (freq_all is not None and freq_all > 0) else np.nan
        )
        enr_low_vs_high = (
            freq_low / freq_high if (freq_high is not None and freq_high > 0) else np.nan
        )

        # Fisher exact test (low vs high, motif enrichment in low)
        if N_low > 0 and N_high > 0 and (a + c) > 0:
            b = N_low - a
            d = N_high - c
            table = [[a, b], [c, d]]
            _, fisher_p = fisher_exact(table, alternative="greater")
        else:
            fisher_p = np.nan

        # Approximate CI for enrichment (low vs high)
        ci_lo, ci_hi = _enrichment_CI_logratio(a, N_low, c, N_high)

        # Bayesian enrichment
        bayes_median, (bayes_ci_lo, bayes_ci_hi), bayes_prob = _bayesian_enrichment(
            a, N_low, c, N_high, nsamples=bayes_samples, rng=rng
        )

        rows.append(
            {
                "motif": name,
                "count_all": count_all,
                "count_low": a,
                "count_high": c,
                "freq_all": freq_all,
                "freq_low": freq_low,
                "freq_high": freq_high,
                "enrichment_low_vs_all": enr_low_vs_all,
                "enrichment_low_vs_high": enr_low_vs_high,
                "fisher_p_low_gt_high": fisher_p,
                "enrichment_CI_low": ci_lo,
                "enrichment_CI_high": ci_hi,
                "bayes_median_enrichment": bayes_median,
                "bayes_CI_low": bayes_ci_lo,
                "bayes_CI_high": bayes_ci_hi,
                "bayes_prob_enrichment_gt_1": bayes_prob,
            }
        )

    df = pd.DataFrame(rows)
    # Example default sort: most confident enrichments first
    df = df.sort_values(
        by=["bayes_prob_enrichment_gt_1", "enrichment_low_vs_high"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return df