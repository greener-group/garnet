import numpy as np
import matplotlib.pyplot as plt

def violin_prune_outliers(ax, data_list, positions=None, widths=0.7,
                          violin_kwargs=None,
                          whisker_kwargs=None,
                          outlier_kwargs=None,
                          median_kwargs=None):
    """
    Violin plots where the violin is drawn from data pruned of outliers
    (outside Q1 ± 1.5*IQR), whiskers span exactly [Q1-1.5*IQR, Q3+1.5*IQR],
    and outliers are plotted as scatter.
    """
    if positions is None:
        positions = np.arange(1, len(data_list) + 1)

    if violin_kwargs is None:
        violin_kwargs = {}
    if whisker_kwargs is None:
        whisker_kwargs = dict(color="k", linewidth=1.3)
    if outlier_kwargs is None:
        outlier_kwargs = dict(marker="o", s=15, alpha=0.6, edgecolors="none", color="k")
    if median_kwargs is None:
        median_kwargs = dict(marker="o", s=20, color="k", zorder=3)

    # Precompute stats, inliers, and outliers
    inliers_list = []
    stats = []  # (q1, q3, lower_fence, upper_fence, outliers, median)

    for data in data_list:
        d = np.asarray(data)
        d = d[~np.isnan(d)]  # drop NaNs
        if d.size == 0:
            # empty dataset; just store empty inliers/outliers
            inliers_list.append(d)
            stats.append((np.nan, np.nan, np.nan, np.nan, np.array([]), np.nan))
            continue

        q1, q3 = np.percentile(d, [25, 75])
        iqr = q3 - q1

        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        mask_in = (d >= lower_fence) & (d <= upper_fence)
        inliers = d[mask_in]
        outliers = d[~mask_in]

        # Fallback: if somehow all are outliers, use original
        if inliers.size == 0:
            inliers = d
            outliers = np.array([])

        median = np.median(inliers)

        inliers_list.append(inliers)
        stats.append((q1, q3, lower_fence, upper_fence, outliers, median))

    # Draw violins using only inliers
    parts = ax.violinplot(
        inliers_list,
        positions=positions,
        widths=widths,
        showextrema=False,
        showmedians=False,
        **violin_kwargs
    )

    # Whiskers (fences), medians, outliers
    for x, (q1, q3, lower_fence, upper_fence, outliers, median) in zip(positions, stats):
        if np.isnan(lower_fence) or np.isnan(upper_fence):
            continue

        # Whisker line from Q1−1.5IQR to Q3+1.5IQR
        ax.vlines(x, lower_fence, upper_fence, **whisker_kwargs)

        # Caps
        cap_width = widths * 0.25
        ax.hlines(lower_fence, x - cap_width / 2, x + cap_width / 2, **whisker_kwargs)
        ax.hlines(upper_fence, x - cap_width / 2, x + cap_width / 2, **whisker_kwargs)

        # Median (of inliers)
        ax.scatter([x], [median], **median_kwargs)

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