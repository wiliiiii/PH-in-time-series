from gtda.time_series import SingleTakensEmbedding   # giotto-tda
import numpy as np
from gtda.homology import VietorisRipsPersistence

# 1) create the dataset
x_periodic = np.linspace(0, 50, 1001)
y_periodic = np.cos(5*x_periodic) + np.sin(7*x_periodic)

# 2) Takens embedding
embedding_dimension_periodic = 6
embedding_time_delay_periodic = 12
stride = 4

embedder_periodic = SingleTakensEmbedding(
    parameters_type="fixed",
    n_jobs=2,
    time_delay=embedding_time_delay_periodic,
    dimension=embedding_dimension_periodic,
    stride=stride,
)

y_periodic_embedded = embedder_periodic.fit_transform(y_periodic)


# 3) calculate persistent homology
y_periodic_embedded_batch = y_periodic_embedded[None, :, :]
homology_dimensions = [0, 1, 2]
vr = VietorisRipsPersistence(
    homology_dimensions=homology_dimensions,
    coeff=2,  # characteristic of the coefficient field
    n_jobs=6
)

diagrams = vr.fit_transform(y_periodic_embedded_batch)   # shape: (1, n_pts, 3) -> (b, d, dim)

def stats_for_dim(diag_sample: np.ndarray, dim: int):
    """
    diag_sample: diagram of a single sample, shape (n_points, 3), columns are (birth, death, hom_dim)
    dim: homology dimension to analyze (1 or 2)
    Return: L2 norm, L3 norm, max persistence, mean persistence, number of bars
    """
    # select the target dimension and remove bars with death=inf
    D = diag_sample[diag_sample[:, 2] == dim, :2]          # (birth, death)
    D = D[np.isfinite(D[:, 1])]
    if D.size == 0:
        return 0.0, 0.0, 0.0, 0.0, 0

    pers = D[:, 1] - D[:, 0]                               # bar lengths (>=0)
    # L^2 and L^3 norms
    l2 = float(np.linalg.norm(pers, ord=2))                # (sum pers^2)^(1/2)
    l3 = float(np.sum(pers**3) ** (1.0/3.0))               # (sum pers^3)^(1/3)
    # other statistics
    pmax = float(pers.max())
    pmean = float(pers.mean())
    return l2, l3, pmax, pmean, pers.size

diag0 = diagrams[0]   # only one sample

l2_H1, l3_H1, max_H1, mean_H1, n1 = stats_for_dim(diag0, dim=1)
l2_H2, l3_H2, max_H2, mean_H2, n2 = stats_for_dim(diag0, dim=2)

# RMS (optional): L2 / sqrt(n)
rms_H1 = l2_H1 / np.sqrt(max(n1, 1))
rms_H2 = l2_H2 / np.sqrt(max(n2, 1))

print("H1 (1D classes):")
print(f"  count = {n1},  L2 = {l2_H1:.6f},  L3 = {l3_H1:.6f},  RMS = {rms_H1:.6f},  "
      f"max = {max_H1:.6f},  mean = {mean_H1:.6f}")

print("H2 (2D classes):")
print(f"  count = {n2},  L2 = {l2_H2:.6f},  L3 = {l3_H2:.6f},  RMS = {rms_H2:.6f},  "
      f"max = {max_H2:.6f},  mean = {mean_H2:.6f}")
