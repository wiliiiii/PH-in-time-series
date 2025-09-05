from gtda.time_series import SingleTakensEmbedding   # giotto-tda
from gtda.plotting import plot_point_cloud           
import numpy as np
from sklearn.decomposition import PCA
from gtda.homology import VietorisRipsPersistence

# 1) create the dataset
x_periodic = np.linspace(0, 50, 1001)
y_periodic = np.cos(5*x_periodic)+np.cos(np.pi*x_periodic)

# 2) search function to find the optimal parameters: tau and d
max_embedding_dimension = 30
max_time_delay = 30

stride = 4

# Takens embedding with parameter search
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

# Example run
y_periodic_embedded = fit_embedder(embedder_search, y_periodic)


# 3) Takens embedding
embedding_dimension_periodic = embedder_search.dimension_
embedding_time_delay_periodic = embedder_search.time_delay_

embedder_periodic = SingleTakensEmbedding(
    parameters_type="fixed",
    n_jobs=2,
    time_delay=embedding_time_delay_periodic,
    dimension=embedding_dimension_periodic,
    stride=stride,
)

# y_periodic_embedded
y_periodic_embedded = embedder_periodic.fit_transform(y_periodic)

# 4) Visualization with or without PCA
if embedding_dimension_periodic == 3:
    # Directly plot the embedding in 3D
    fig = plot_point_cloud(y_periodic_embedded)
else:
    # Use PCA to reduce to 3D for visualization
    pca = PCA(n_components=3)
    y_periodic_embedded_pca = pca.fit_transform(y_periodic_embedded)
    fig = plot_point_cloud(y_periodic_embedded_pca)

fig.show()
 


# 5) calculate the persistence homology
y_periodic_embedded_batch = y_periodic_embedded[None, :, :]  

homology_dimensions = [0, 1, 2]
periodic_persistence = VietorisRipsPersistence(
    homology_dimensions=homology_dimensions,
    coeff=2, # the characteristic of the coefficient field
    n_jobs=6
)

print("Persistence diagram signal")
periodic_persistence.fit_transform_plot(y_periodic_embedded_batch)
