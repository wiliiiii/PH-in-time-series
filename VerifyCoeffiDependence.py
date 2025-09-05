# -*- coding: utf-8 -*-
import sys
print("Using Python:", sys.executable)

import numpy as np
from gtda.time_series import SingleTakensEmbedding
from gtda.homology import VietorisRipsPersistence

# 1) Data 
x_periodic = np.linspace(0, 50, 1001)
y_periodic = 0.6*np.cos(x_periodic)+0.8*np.sin(np.pi*x_periodic)

# 2) Search for optimal parameters
max_embedding_dimension = 30
max_time_delay = 30

stride = 3

embedder_search = SingleTakensEmbedding(
    parameters_type="search",
    time_delay=max_time_delay,
    dimension=max_embedding_dimension,
    stride=stride,
)

def fit_embedder(embedder: SingleTakensEmbedding, y: np.ndarray, verbose: bool=True) -> np.ndarray:
    """Fits a Takens embedder and displays optimal search parameters."""
    y_embedded = embedder.fit_transform(y)

    if verbose:
        print(f"Shape of embedded time series: {y_embedded.shape}")
        print(
            f"Optimal embedding dimension is {embedder.dimension_} "
            f"and time delay is {embedder.time_delay_}"
        )

    return y_embedded

y_embedded_search = fit_embedder(embedder_search, y_periodic)

# Use optimal parameters directly
embedding_dimension_periodic = embedder_search.dimension_
embedding_time_delay_periodic = embedder_search.time_delay_

embedder_periodic = SingleTakensEmbedding(
    parameters_type="fixed",
    n_jobs=2,
    time_delay=embedding_time_delay_periodic,
    dimension=embedding_dimension_periodic,
    stride=stride,
)

# y_periodic_embedded: (n_windows, dimension)
Y = embedder_periodic.fit_transform(y_periodic)

# 3) Utility function: extract dimension and compute statistics 
def stats_for_dim(diag_sample: np.ndarray, dim: int):
    """
    diag_sample: shape (n_points, 3) with columns (birth, death, hom_dim)
    Returns: L2, L3, max, mean, count, RMS
    """
    D = diag_sample[diag_sample[:, 2] == dim, :2]   # (birth, death)
    D = D[np.isfinite(D[:, 1])]
    if D.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0, 0.0
    pers = D[:, 1] - D[:, 0]
    l2   = float(np.linalg.norm(pers, 2))                 # (sum pers^2)^(1/2)
    l3   = float(np.sum(pers**3) ** (1.0/3.0))            # (sum pers^3)^(1/3)
    pmax = float(pers.max())
    pmean= float(pers.mean())
    n    = pers.size
    rms  = l2 / np.sqrt(n)
    return l2, l3, pmax, pmean, int(n), rms

def compute_diagram(points, coeff=7, homology_dimensions=(0,1,2), n_jobs=6):
    vr = VietorisRipsPersistence(homology_dimensions=list(homology_dimensions),
                                 coeff=coeff, n_jobs=n_jobs)
    D = vr.fit_transform(points[None, :, :])[0]  # shape: (n_pts, 3)
    return D

# 4) Example: output H1/H2 stats at p=7 
coeff_for_summary = 7
diag = compute_diagram(Y, coeff=coeff_for_summary, homology_dimensions=(1,2))
l2_H1, l3_H1, max_H1, mean_H1, n1, rms_H1 = stats_for_dim(diag, dim=1)
l2_H2, l3_H2, max_H2, mean_H2, n2, rms_H2 = stats_for_dim(diag, dim=2)


# 5) Extra tool A: check if H1 PD is identical across primes p<=13 
def PD_equal_H1_across_primes(points, primes=(2,3,5,7,11,13), tol=1e-9, n_jobs=6):
    base = None
    X = points[None, :, :]
    for p in primes:
        vr = VietorisRipsPersistence(homology_dimensions=[1], coeff=p, n_jobs=n_jobs)
        D = vr.fit_transform(X)[0]
        D = D[np.isfinite(D[:, 1])]
        D = D[D[:, 2] == 1][:, :2]
        order = np.lexsort((D[:, 1], D[:, 0]))  # sort by (birth, death)
        D = D[order]
        if base is None:
            base = D
        else:
            same_shape = (D.shape == base.shape)
            same_vals  = same_shape and (np.max(np.abs(D - base)) <= tol if D.size else True)
            if not same_vals:
                return False, p
    return True, None

same, badp = PD_equal_H1_across_primes(Y, primes=(2,3,5,7,11,13), tol=1e-9, n_jobs=6)
print("H1 PD equal across primes <=13?:", same, "| first differing prime:", badp)

# 6) Extra tool B: list all differing primes compared to p=2 
def PD_H1_differences_across_primes(points, primes=(2,3,5,7,11,13), tol=1e-9, n_jobs=6):
    """
    Returns (diff_primes, details, base_prime)
      - diff_primes: list of primes where PD differs from the base prime
      - details: [(p, n_bars_p, n_bars_base, max_abs_diff_if_same_len_or_None), ...]
      - base_prime: prime chosen as baseline (default primes[0])
    """
    X = points[None, :, :]

    def pd_H1_for_prime(p):
        vr = VietorisRipsPersistence(homology_dimensions=[1], coeff=p, n_jobs=n_jobs)
        D = vr.fit_transform(X)[0]
        D = D[np.isfinite(D[:, 1])]
        D = D[D[:, 2] == 1][:, :2]
        if D.size:
            D = D[np.lexsort((D[:, 1], D[:, 0]))]
        return D

    PDs = {p: pd_H1_for_prime(p) for p in primes}
    base_p = primes[0]
    base_D = PDs[base_p]

    diff_primes, details = [], []
    for p in primes[1:]:
        D = PDs[p]
        same_shape = (D.shape == base_D.shape)
        if same_shape:
            if D.size == 0 and base_D.size == 0:
                same_vals, max_diff = True, 0.0
            else:
                max_diff = float(np.max(np.abs(D - base_D))) if D.size else 0.0
                same_vals = (max_diff <= tol)
        else:
            same_vals, max_diff = False, None
        if not same_vals:
            diff_primes.append(p)
        details.append((p, D.shape[0], base_D.shape[0], max_diff))
    return diff_primes, details, base_p

diff_ps, info, base_p = PD_H1_differences_across_primes(Y, primes=(2,3,5,7,11,13), tol=1e-9, n_jobs=6)
if not diff_ps:
    print(f"H1 PD are identical for all primes ≤13 (compared to p={base_p}).")
else:
    print(f"H1 PD differ (vs p={base_p}) at primes:", diff_ps)
    for p, n_p, n_base, maxdiff in info:
        if p in diff_ps:
            if maxdiff is None:
                print(f"  p={p}: bar count differs (n_p={n_p}, n_base={n_base})")
            else:
                print(f"  p={p}: same bar count ({n_p}), max |Δ| = {maxdiff:.3e}")
