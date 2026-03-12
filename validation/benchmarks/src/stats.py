import numpy as np

def freedman_diaconis_bins(x):
    """
    Return the optimal number of histogram bins using
    the Freedman–Diaconis rule.

    Parameters
    ----------
    x : array-like
        1D array of samples.

    Returns
    -------
    int
        Number of bins.
    """
    x = np.asarray(x).ravel()
    n = x.size
    if n < 2:
        return 1

    # Interquartile range
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        return 1

    # Bin width
    bw = 2 * iqr / np.cbrt(n)
    if bw == 0:
        return 1

    data_range = x.max() - x.min()
    return int(np.ceil(data_range / bw))


def histogram_cdf(n, b):
    """
    Compute the cumulative distribution function from a normalized histogram.

    Parameters
    ----------
    n : array-like
        Histogram bin heights (density=True).
    b : array-like
        Bin edges, shape (len(n)+1,).

    Returns
    -------
    cdf : ndarray
        CDF evaluated at each bin edge, shape (len(n)+1,).
        cdf[0] = 0, cdf[-1] = 1.
    """
    n = np.asarray(n)
    b = np.asarray(b)

    widths = b[1:] - b[:-1]
    cdf = np.zeros_like(b, dtype=float)
    cdf[1:] = np.cumsum(n * widths)
    return cdf

def pdf_cdf(series):
    n, bins = np.histogram(series, bins = freedman_diaconis_bins(series), density = True)
    x = 0.5*(bins[1:] + bins[:-1])
    return x, n, histogram_cdf(n, bins)[:-1]

# --------- Autocorrelation and integrated autocorrelation time ---------

def autocorrelation_fft(x, max_lag=None):
    """
    Fast autocorrelation estimate using FFT, normalized so that acf[0] = 1.

    Parameters
    ----------
    x : array_like, shape (N,)
    max_lag : int or None
        Maximum lag to return (inclusive). If None, use N-1.

    Returns
    -------
    acf : ndarray, shape (max_lag+1,)
        acf[0] = 1, acf[k] ~ Corr(x_t, x_{t+k}).
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    x = x - x.mean()

    # Next power of 2 for zero-padding (for speed)
    nfft = 1 << (2 * N - 1).bit_length()

    fx = np.fft.rfft(x, n=nfft)
    sxx = fx * np.conjugate(fx)
    acf_full = np.fft.irfft(sxx, n=nfft)[:N]
    acf_full /= acf_full[0]

    if max_lag is None or max_lag >= N:
        max_lag = N - 1

    return acf_full[:max_lag + 1]


def integrated_autocorrelation_time(acf):
    """
    Estimate integrated autocorrelation time using Geyer's
    initial positive sequence (IPS) method.

    Parameters
    ----------
    acf : array_like
        Autocorrelation function, acf[0] should be 1.

    Returns
    -------
    tau_int : float
        Integrated autocorrelation time (in units of steps).
    """
    acf = np.asarray(acf, dtype=float)
    if acf[0] <= 0:
        raise ValueError("acf[0] must be positive (usually 1).")

    # Γ_k = ρ_{2k-1} + ρ_{2k}, using 0-based indexing:
    # acf[1] + acf[2], acf[3] + acf[4], ...
    # Stop when Γ_k becomes non-positive.
    # acf length M -> up to floor((M-1)/2) Γ_k terms.
    gamma = acf[1:-1:2] + acf[2::2]
    if gamma.size == 0:
        # Not enough lags; fall back to simple sum until first nonpositive
        positive = acf[1:] > 0
        if not np.any(positive):
            return 0.5  # effectively uncorrelated
        M = np.where(~positive)[0]
        if M.size == 0:
            M = acf.size - 1
        else:
            M = M[0]
        tau_int = 0.5 + acf[1:M + 1].sum()
        return max(tau_int, 0.5)

    # Find first k where Gamma_k <= 0
    nonpos = np.where(gamma <= 0)[0]
    if nonpos.size == 0:
        # All Gamma_k positive; use all available lags
        last_k = gamma.size - 1
    else:
        last_k = max(0, nonpos[0] - 1)

    # Use lags up to 2 * last_k + 2 (inclusive index)
    M = min(2 * (last_k + 1) + 1, acf.size - 1)

    tau_int = 0.5 + acf[1:M + 1].sum()

    # Numerical safety: tau_int should be at least 0.5
    return max(tau_int, 0.5)


def estimate_block_size_from_acf(x, max_lag=None, c=2.0, max_frac=0.1):
    """
    Estimate an optimal block size for moving-block bootstrap from
    the integrated autocorrelation time.

    Heuristic:
        L_opt ≈ c * τ_int,
    with clipping so that 1 <= L_opt <= max_frac * N.

    Parameters
    ----------
    x : array_like, shape (N,)
        Time series data.
    max_lag : int or None
        Maximum lag for ACF (passed to autocorrelation_fft). If None,
        use min(N//2, 5000) as a reasonable default.
    c : float
        Multiplier for τ_int (typical values 1–5; 2 is a reasonable default).
    max_frac : float
        Maximum fraction of total length allowed for the block size.

    Returns
    -------
    block_size : int
        Estimated block size.
    tau_int : float
        Estimated integrated autocorrelation time.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if max_lag is None:
        max_lag = min(N // 2, 5000)

    acf = autocorrelation_fft(x, max_lag=max_lag)
    tau_int = integrated_autocorrelation_time(acf)

    # Heuristic: L ~ c * tau_int, clipped
    L = int(np.round(c * tau_int))
    L = max(1, L)
    L = min(L, max(1, int(max_frac * N)))

    return L, tau_int

def moving_block_bootstrap(
    data,
    estimator,
    block_size=None,
    n_boot=1000,
    rng=None,
    c=2.0,
    max_frac=0.1,
    max_lag=None,
    alpha=0.05,
    acf_series=None,
    estimator_args=None,
    estimator_kwargs=None,
):
    """
    Generic moving-block bootstrap for a scalar estimator.

    Parameters
    ----------
    data : array_like, shape (N, ...) 
        Time series data; blocking is along axis 0.
    estimator : callable
        Function f(sample, *estimator_args, **estimator_kwargs) -> scalar.
        'sample' has the same shape as 'data', but resampled along axis 0.
    block_size : int or None
        Length of each block. If None, estimated from acf_series via τ_int.
    n_boot : int
        Number of bootstrap replicates.
    rng : np.random.Generator or None
        RNG for reproducibility.
    c, max_frac, max_lag :
        Passed to estimate_block_size_from_acf if block_size is None.
    alpha : float
        Significance level for the (1 - alpha) confidence interval.
    acf_series : array_like or None
        1D series used to estimate autocorrelation and block size.
        If None:
            - If data.ndim == 1, use data.
            - Else, use data[:, 0].
    estimator_args : tuple or None
        Positional arguments passed to estimator.
    estimator_kwargs : dict or None
        Keyword arguments passed to estimator.

    Returns
    -------
    theta_hat : float
        Estimator applied to the original data.
    var_theta_hat : float
        Bootstrap estimate of Var(theta_hat).
    boot_thetas : ndarray, shape (n_boot,)
        Bootstrap replicates of the estimator.
    ci : tuple(float, float)
        (lower, upper) percentile confidence interval for theta.
    block_size : int
        Block size actually used.
    tau_int : float or None
        Estimated integrated autocorrelation time (None if block_size was given).
    """
    data = np.asarray(data)
    N = data.shape[0]

    if estimator_args is None:
        estimator_args = ()
    if estimator_kwargs is None:
        estimator_kwargs = {}

    # Choose series for ACF / τ_int if needed
    if block_size is None:
        if acf_series is None:
            if data.ndim == 1:
                acf_series = data
            else:
                acf_series = data[:, 0]
        acf_series = np.asarray(acf_series, dtype=float)
        block_size, tau_int = estimate_block_size_from_acf(
            acf_series, max_lag=max_lag, c=c, max_frac=max_frac
        )
    else:
        tau_int = None

    if block_size > N:
        raise ValueError("block_size must be <= length of data along axis 0")

    if rng is None:
        rng = np.random.default_rng()

    # Precompute block start indices
    starts = np.arange(0, N - block_size + 1)
    n_blocks_available = starts.size
    blocks_per_rep = int(np.ceil(N / block_size))

    boot_thetas = np.empty(n_boot, dtype=float)

    for b in range(n_boot):
        idx = rng.integers(0, n_blocks_available, size=blocks_per_rep)
        # Concatenate blocks along axis 0
        sample = np.concatenate(
            [data[s:s + block_size] for s in starts[idx]],
            axis=0
        )
        sample = sample[:N]  # truncate to original length
        boot_thetas[b] = estimator(sample, *estimator_args, **estimator_kwargs)

    # Estimator on original data
    theta_hat = estimator(data, *estimator_args, **estimator_kwargs)
    var_theta_hat = boot_thetas.var(ddof=1)

    # Percentile CI
    lower = np.quantile(boot_thetas, alpha / 2.0)
    upper = np.quantile(boot_thetas, 1.0 - alpha / 2.0)
    ci = (lower, upper)

    return theta_hat, var_theta_hat, boot_thetas, ci, block_size, tau_int
