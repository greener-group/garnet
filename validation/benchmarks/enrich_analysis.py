import glob
import os

import numpy as np
import pandas as pd

MODELS = ("OpenFF", "Espaloma", "MACE-OFF", "Garnet")
TIERS = ("best_1_percent", "worst_1_percent")


def _write_text_file(filename, lines):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def calculate_cis_and_format(row):
    """
    Calculate CIs depending on tier and format enrichment as a display string.
    """
    if row["tier"] == "worst_1_percent":
        ci_low = 1.0 / row["bayes_CI_high"] if row["bayes_CI_high"] > 0 else np.inf
        ci_high = 1.0 / row["bayes_CI_low"] if row["bayes_CI_low"] > 0 else np.inf
    else:
        ci_low = row["bayes_CI_low"]
        ci_high = row["bayes_CI_high"]

    score = row["enrichment_score"]

    if np.isinf(score) or np.isnan(score):
        fmt_str = "inf"
    else:
        fmt_str = f"{score:.2f} [{ci_low:.2f}, {ci_high:.2f}]"

    return pd.Series([ci_low, ci_high, fmt_str])


def load_and_aggregate_tables(data_dir="enrichment_tables"):
    """Load all enrichment CSV files and concatenate into a master dataframe."""
    all_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}.")

    df_master = pd.concat((pd.read_csv(file) for file in all_files), ignore_index=True)
    df_master["enrichment_score"] = df_master["enrichment_score"].astype(float)
    df_master[["ci_lower", "ci_upper", "enrichment_str"]] = df_master.apply(
        calculate_cis_and_format,
        axis=1,
    )
    return df_master


def format_sci(p):
    """Format p-values as manuscript-style scientific notation for LaTeX."""
    if pd.isna(p):
        return "--"
    if 0 < p < 0.001:
        exp = int(np.floor(np.log10(p)))
        base = p / (10**exp)
        return f"${base:.2f}\\times 10^{{{exp}}}$"
    if p == 0:
        return "$< 10^{-300}$"
    return f"{p:.4f}"


def generate_motif_compare_latex(df, metric, tier, filename):
    """Generate grouped comparative motif table (Table 1 style)."""
    subset = df[(df["metric"] == metric) & (df["tier"] == tier)].copy()
    if subset.empty:
        return

    motif_stats = (
        subset.groupby("motif")
        .agg(n_all=("count_all", "max"), model_count=("model", "nunique"))
        .reset_index()
    )

    pivot = subset.pivot(index="motif", columns="model", values="enrichment_str").fillna("--")
    for model_name in MODELS:
        if model_name not in pivot.columns:
            pivot[model_name] = "--"

    merged = pd.merge(motif_stats, pivot, on="motif").sort_values(
        by=["model_count", "n_all"],
        ascending=[False, False],
    )

    tier_text = "lowest 1\\% errors" if tier == "best_1_percent" else "highest 1\\% errors"

    tex = [
        "\\begin{table}[h!]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{5pt}",
        "  \\renewcommand{\\arraystretch}{1.15}",
        "  \\begin{tabular}{l r c c c c}",
        "    \\hline",
        "    Motif & $N_\\mathrm{all}$ & OpenFF & Espaloma & MACE-OFF & Garnet \\\\",
        "    \\hline",
    ]

    current_count = 5
    for count in [4, 3, 2, 1]:
        group = merged[merged["model_count"] == count]
        if group.empty:
            continue

        if current_count <= 4:
            tex.append("    \\hdashline")

        for _, row in group.iterrows():
            motif_name = row["motif"].replace("_", "\\_")
            values = [str(row[m]).replace(" [", "\\,\\,[") for m in MODELS]
            tex.append(
                f"    {motif_name} & {int(row['n_all'])} & {values[0]} & {values[1]} & {values[2]} & {values[3]} \\\\",
            )

        current_count = count

    tex.extend(
        [
            "    \\hline",
            "  \\end{tabular}",
            f"  \\caption{{Comparative motif enrichment ({metric}, {tier_text}).}}",
            f"  \\label{{tab:compare_{metric.lower()}_{tier.split('_')[0]}}}",
            "\\end{table}",
        ],
    )

    _write_text_file(filename, tex)


def generate_motif_split_latex(df, metric, tier, filename):
    """Generate per-model motif table (Table 2 style)."""
    subset = df[(df["metric"] == metric) & (df["tier"] == tier)].copy()
    if subset.empty:
        return

    tier_text = "lowest 1\\% errors" if tier == "best_1_percent" else "highest 1\\% errors"
    ratio_text = "$E_{\\mathrm{low/high}}$" if tier == "best_1_percent" else "$E_{\\mathrm{high/low}}$"
    prob_text = "$P(E>1)$" if tier == "best_1_percent" else "$P(E_{\\mathrm{high}}>E_{\\mathrm{low}})$"

    tex = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\small",
        "  \\setlength{\\tabcolsep}{6pt}",
        "  \\renewcommand{\\arraystretch}{1.15}",
        "  \\begin{tabular}{llrrrr}",
        "    \\hline",
        f"    Force field & Motif & $N_\\mathrm{{all}}$ & {ratio_text} (Bayes) & $p_\\mathrm{{Fisher}}$ & {prob_text} \\\\",
        "    \\hline",
    ]

    for model_name in MODELS:
        model_df = subset[subset["model"] == model_name].sort_values(
            by="enrichment_score",
            ascending=False,
        )
        if model_df.empty:
            continue

        first = True
        for _, row in model_df.iterrows():
            ff_name = model_name if first else " "
            motif_name = row["motif"].replace("_", "\\_")
            e_str = str(row["enrichment_str"]).replace(" [", "\\,[")

            p_val = row.get("fisher_p_low_gt_high", np.nan)
            if tier == "worst_1_percent" and not pd.isna(p_val):
                p_val = 1.0 - p_val

            conf = row["confidence"]
            tex.append(
                f"    {ff_name} & {motif_name} & {int(row['count_all'])} & {e_str} & {format_sci(p_val)} & {conf:.3f} \\\\",
            )
            first = False

        tex.append("    \\hline")

    tex.extend(
        [
            "  \\end{tabular}",
            f"  \\caption{{Motif enrichment for {tier_text}.}}",
            f"  \\label{{tab:split_{metric.lower()}_{tier.split('_')[0]}}}",
            "\\end{table}",
        ],
    )

    _write_text_file(filename, tex)


def generate_master_agglutinated_latex(df, tier, filename):
    """
    Generate a longtable aggregating all metrics for one tier.
    """
    subset = df[df["tier"] == tier].copy()
    if subset.empty:
        return

    n_all_map = subset.groupby("motif")["count_all"].max().to_dict()

    pivot = subset.pivot_table(
        index=["motif", "metric"],
        columns="model",
        values="enrichment_str",
        aggfunc="first",
    ).fillna("--")

    for model_name in MODELS:
        if model_name not in pivot.columns:
            pivot[model_name] = "--"
    pivot = pivot[list(MODELS)]

    tier_text = "lowest 1\\% errors" if tier == "best_1_percent" else "highest 1\\% errors"
    title_text = "Motif Successes" if tier == "best_1_percent" else "Motif Blind Spots"
    ratio_text = "$E_{\\mathrm{low/high}}$" if tier == "best_1_percent" else "$E_{\\mathrm{high/low}}$"
    label = f"tab:agglutinated_{tier.split('_')[0]}"

    caption = (
        f"Comparative motif enrichment across all structural metrics ({tier_text}). "
        f"Entries report the Bayesian posterior median enrichment ratio {ratio_text} "
        f"with a 95\\% credible interval in brackets; ``--'' indicates lack of "
        "significant enrichment for that specific model-metric pair."
    )

    tex = [
        "\\newpage",
        "\\FloatBarrier",
        "\\begingroup",
        "\\small",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\renewcommand{\\arraystretch}{1.1}",
        "% Force centering for over-wide longtables",
        "\\setlength{\\LTleft}{-20cm plus 1fill}",
        "\\setlength{\\LTright}{-20cm plus 1fill}",
        "",
        "\\begin{longtable}[c]{ ll | r | c c c c}",
        f"    \\caption[]{{\\bfseries {title_text} ({tier_text.title()})}} \\\\",
        "    \\hline",
        "    Motif & Metric & $N_\\mathrm{all}$ & OpenFF & Espaloma & MACE-OFF & Garnet \\\\",
        "    \\hline",
        "    \\hline",
        "    \\endfirsthead",
        "",
        "    \\multicolumn{7}{c}%",
        "    {{\\bfseries \\tablename\\ \\thetable{} -- continued from previous page}} \\\\",
        "    \\hline",
        "    Motif & Metric & $N_\\mathrm{all}$ & OpenFF & Espaloma & MACE-OFF & Garnet \\\\",
        "    \\hline",
        "    \\hline",
        "    \\endhead",
        "",
        "    \\hline",
        "    \\multicolumn{7}{r}{{Continued on next page}} \\\\",
        "    \\endfoot",
        "",
        "    \\hline",
        "    \\endlastfoot",
        "",
    ]

    motif_order = subset.groupby("motif")["enrichment_score"].count().sort_values(ascending=False).index

    last_motif = None
    for motif in motif_order:
        motif_rows = pivot.loc[motif]
        first_row = True

        if last_motif is not None:
            tex.append("    \\hline")

        for metric, row in motif_rows.iterrows():
            motif_display = motif.replace("_", "\\_") if first_row else ""
            n_display = str(int(n_all_map[motif])) if first_row else ""
            vals = [str(row[m]).replace(" [", "\\,\\,[") for m in MODELS]
            tex.append(
                f"    {motif_display} & {metric} & {n_display} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} \\\\",
            )
            first_row = False

        last_motif = motif

    tex.extend(
        [
            "    \\hline",
            "    \\hline",
            f"    \\caption{{{caption}}}",
            f"    \\label{{{label}}}",
            "\\end{longtable}",
            "\\endgroup",
        ],
    )

    _write_text_file(filename, tex)
    print(f"Saved agglutinated master table: {filename}")


def main():
    out_dir = "latex_tables_formatted"
    os.makedirs(out_dir, exist_ok=True)

    df_master = load_and_aggregate_tables("enrichment_tables")
    metrics = df_master["metric"].unique()

    for metric in metrics:
        for tier in TIERS:
            generate_motif_compare_latex(
                df_master,
                metric,
                tier,
                os.path.join(out_dir, f"{metric}_{tier}_compare.tex"),
            )
            generate_motif_split_latex(
                df_master,
                metric,
                tier,
                os.path.join(out_dir, f"{metric}_{tier}_split.tex"),
            )

    for tier in TIERS:
        generate_master_agglutinated_latex(
            df_master,
            tier,
            os.path.join(out_dir, f"agglutinated_{tier}.tex"),
        )


if __name__ == "__main__":
    main()
