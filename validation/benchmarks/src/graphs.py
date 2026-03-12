import numpy as np

def violin_prune_outliers(ax, data_list, positions=None, widths=0.7,

                              violin_kwargs=None,
                              whisker_kwargs=None,
                              outlier_kwargs=None,
                              median_kwargs=None,
                              mad_k=3.0, allow_negative = False):
    """
    Violin plots where the violin is drawn from data pruned of outliers
    defined using a MAD-based rule:
        outlier <=> |x - median| > mad_k * MAD,
    whiskers span exactly [median - mad_k*MAD, median + mad_k*MAD],
    and outliers are plotted as scatter.
    """
    if positions is None:
        positions = np.arange(1, len(data_list) + 1)

    if violin_kwargs is None:
        violin_kwargs = {}
    if whisker_kwargs is None:
        whisker_kwargs = dict(color="k", linewidth=1.3)
    if outlier_kwargs is None:
        outlier_kwargs = dict(marker="x", s=5, alpha=0.35,
                              edgecolors="none", color="k")
    if median_kwargs is None:
        median_kwargs = dict(marker="o", s=20, color="k", zorder=3)

    # Precompute stats, inliers, and outliers
    inliers_list = []
    # (median, lower_fence, upper_fence, outliers)
    stats = []

    for data in data_list:
        d = np.asarray(data)
        d = d[~np.isnan(d)]  # drop NaNs
        if d.size == 0:
            inliers_list.append(d)
            stats.append((np.nan, np.nan, np.nan, np.array([])))
            continue

        med = np.median(d)
        abs_dev = np.abs(d - med)
        mad = np.median(abs_dev)

        if mad == 0:
            # All points identical or nearly so: treat everything as inlier
            inliers = d
            outliers = np.array([])
            lower_fence = med
            upper_fence = med
        else:
            threshold = mad_k * mad
            mask_in = abs_dev <= threshold
            inliers = d[mask_in]
            outliers = d[~mask_in]

            # Fallback: if somehow all are outliers, use original
            if inliers.size == 0:
                inliers = d
                outliers = np.array([])

            lower_fence = med - threshold
            upper_fence = med + threshold

        inliers_list.append(inliers)
        stats.append((med, lower_fence, upper_fence, outliers))

    # Draw violins using only inliers
    parts = ax.violinplot(
        inliers_list,
        positions=positions,
        widths=widths,
        showextrema=False,
        showmedians=False,
        **violin_kwargs
    )

    # Whiskers, medians, outliers
    for x, (med, lower_fence, upper_fence, outliers) in zip(positions, stats):
        if np.isnan(lower_fence) or np.isnan(upper_fence):
            continue

        if (not allow_negative) and lower_fence < 0:
            lower_fence = 0.0

        # Whisker line
        ax.vlines(x, lower_fence, upper_fence, **whisker_kwargs)

        # Caps
        cap_width = widths * 0.25
        ax.hlines(lower_fence, x - cap_width / 2, x + cap_width / 2,
                  **whisker_kwargs)
        ax.hlines(upper_fence, x - cap_width / 2, x + cap_width / 2,
                  **whisker_kwargs)

        # Median
        ax.scatter([x], [med], **median_kwargs)

        # Outliers as scatter with jitter
        if outliers.size > 0:
            jitter = (np.random.rand(outliers.size) - 0.5) * widths * 0.3
            ax.scatter(x + jitter, outliers, **outlier_kwargs)

    return parts

def style_violins(parts, facecolor="royalblue"):
    for pc in parts["bodies"]:
        pc.set_facecolor(facecolor)
        pc.set_edgecolor("k")
        pc.set_alpha(0.75)
