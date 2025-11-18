"""
SOM_HeartFailure_Clustering.py

Aim: Apply a Kohonen Self-Organizing Map (SOM) to cluster heart failure
patients and interpret clusters w.r.t. DEATH_EVENT.

Usage:
- Place 'heart_failure_clinical_records_dataset.csv' in the same folder.
- Run: python SOM_HeartFailure_Clustering.py

This script is written for clarity and teaching: each major step has a
separate section and comments explaining the logic.

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans
from collections import defaultdict

# ---------------------------
# 1) Data Preparation
# ---------------------------

# Load dataset (assumes file is in working directory)
DATA_PATH = 'Heart_failure_clinical_records_dataset.csv'

def load_and_prepare(path=DATA_PATH, target_col='DEATH_EVENT'):
    """Load CSV, handle missing, separate target, scale features.

    Returns: X (ndarray), y (ndarray), df (original dataframe)
    """
    df = pd.read_csv(path)

    # Basic info & missing handling
    # If missing values exist, drop rows (other strategies possible: impute)
    if df.isnull().sum().sum() > 0:
        print('Missing values found - dropping rows with missing data')
        df = df.dropna().reset_index(drop=True)

    # Separate features and target
    if target_col in df.columns:
        y = df[target_col].values
        X_df = df.drop(columns=[target_col])
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    # Keep list of feature names for later interpretation
    feature_names = list(X_df.columns)

    # Standardize numeric features -> SOM benefits from similarly scaled inputs
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df.values)

    return X_scaled, y, df, feature_names, scaler

# ---------------------------
# 2) SOM Implementation
# ---------------------------

class SOM:
    """
    A simple implementation of a rectangular Kohonen Self-Organizing Map.

    Parameters
    ----------
    m, n : int
        Grid size (m rows x n columns)
    dim : int
        Dimensionality of the input vectors
    learning_rate : float
        Initial learning rate
    sigma : float
        Initial neighborhood radius (recommended: max(m,n)/2)
    random_state : int or None
        For reproducibility
    """

    def __init__(self, m, n, dim, learning_rate=0.5, sigma=None, random_state=None):
        self.m = m
        self.n = n
        self.dim = dim
        self.learning_rate = learning_rate
        self.sigma = sigma if sigma is not None else max(m, n) / 2.0
        self.random_state = random_state

        rng = np.random.RandomState(random_state)
        # Weight vectors: shape (m, n, dim)
        self.weights = rng.randn(m, n, dim)

        # Precompute coordinate grid for neurons
        self._neuron_locations = np.array([[i, j] for i in range(m) for j in range(n)])

    def _decay_learning_rate(self, initial_lr, iter, max_iter):
        # Exponential decay
        return initial_lr * np.exp(-iter / max_iter)

    def _decay_sigma(self, initial_sigma, iter, max_iter):
        return initial_sigma * np.exp(-iter / max_iter)

    def _find_bmu(self, x):
        """Return coordinates of Best Matching Unit for input x."""
        # Compute L2 distance from x to all weights
        diff = self.weights - x  # broadcast
        sq_dist = np.sum(diff ** 2, axis=2)  # shape (m, n)
        bmu_idx = np.unravel_index(np.argmin(sq_dist), (self.m, self.n))
        return bmu_idx

    def _neighborhood_func(self, bmu_idx, sigma):
        """Compute neighborhood influence for every neuron given BMU and sigma.

        Returns an (m, n) array of influence factors in (0,1].
        """
        grid = self._neuron_locations.reshape(self.m, self.n, 2)
        bmu_loc = np.array(bmu_idx)
        # squared distance on grid (not input space)
        dist_sq = np.sum((grid - bmu_loc) ** 2, axis=2)
        # Gaussian neighborhood
        return np.exp(-dist_sq / (2 * (sigma ** 2)))

    def train(self, X, num_iterations=1000, verbose=False):
        """Train SOM with input data X (shape: [n_samples, dim])."""
        n_samples = X.shape[0]
        initial_lr = self.learning_rate
        initial_sigma = self.sigma

        for it in range(num_iterations):
            # select a sample (random order helps convergence)
            x = X[np.random.randint(0, n_samples)]

            # decay parameters
            lr = self._decay_learning_rate(initial_lr, it, num_iterations)
            sigma = self._decay_sigma(initial_sigma, it, num_iterations)

            bmu = self._find_bmu(x)
            theta = self._neighborhood_func(bmu, sigma)  # shape (m,n)
            theta = theta[:, :, np.newaxis]  # for broadcasting to weights

            # weight update rule: w += lr * theta * (x - w)
            diff = x - self.weights
            self.weights += lr * theta * diff

            if verbose and (it % (num_iterations // 5 + 1) == 0):
                print(f'Iteration {it+1}/{num_iterations} | lr={lr:.4f} | sigma={sigma:.4f}')

    def map_vects(self, X):
        """Map each input vector to BMU coordinates. Returns list of coords and flat index."""
        bmu_coords = []
        bmu_indices = []
        for x in X:
            bmu = self._find_bmu(x)
            bmu_coords.append(bmu)
            bmu_indices.append(bmu[0] * self.n + bmu[1])
        return np.array(bmu_coords), np.array(bmu_indices)

    def get_weights_matrix(self):
        """Return weights reshaped to (m*n, dim) and neuron grid coords."""
        W = self.weights.reshape(self.m * self.n, self.dim)
        coords = np.array([[i, j] for i in range(self.m) for j in range(self.n)])
        return W, coords

# ---------------------------
# 3) Training pipeline
# ---------------------------

def run_som_pipeline(csv_path=DATA_PATH,
                     grid_size=(15, 15),
                     num_iterations=5000,
                     seed=42):
    """Full pipeline: load data, train SOM, cluster neurons, evaluate, visualize."""
    X, y, df, feature_names, scaler = load_and_prepare(csv_path)
    m, n = grid_size
    som = SOM(m, n, X.shape[1], learning_rate=0.5, sigma=max(m, n) / 2.0, random_state=seed)

    print('Training SOM...')
    som.train(X, num_iterations=num_iterations, verbose=True)
    print('Training finished.')

    # Map patients to neurons
    bmu_coords, bmu_indices = som.map_vects(X)

    # ---------------------------
    # 4) Clustering neurons -> assign cluster labels to patients
    # ---------------------------
    W, coords = som.get_weights_matrix()  # shape (m*n, dim)

    # Use KMeans on neuron weights to group neurons into clusters
    # Try k in range 2..6 and pick best by silhouette (on patient-level labels)
    best_k = None
    best_score = -1
    best_labels = None

    # Note: silhouette requires at least 2 clusters and less than n_samples
    for k in range(2, 7):
        kmeans = KMeans(n_clusters=k, random_state=seed).fit(W)
        neuron_labels = kmeans.labels_  # length m*n
        # assign each patient the cluster of its BMU neuron
        patient_labels = neuron_labels[bmu_indices]
        try:
            s = silhouette_score(X, patient_labels)
        except Exception:
            s = -1
        if s > best_score:
            best_score = s
            best_k = k
            best_labels = neuron_labels

    print(f'Best k (by silhouette on patient assignments): {best_k} (score={best_score:.4f})')

    # use best labels to assign patients
    patient_cluster_labels = best_labels[bmu_indices]

    # ---------------------------
    # 5) Evaluation
    # ---------------------------
    sil = silhouette_score(X, patient_cluster_labels)
    dbi = davies_bouldin_score(X, patient_cluster_labels)
    print(f'Patient-level Silhouette Score: {sil:.4f}')
    print(f'Patient-level Davies-Bouldin Index: {dbi:.4f}')

    # Compare cluster membership with DEATH_EVENT
    cluster_stats = pd.DataFrame({'cluster': patient_cluster_labels, 'DEATH_EVENT': y})
    summary = cluster_stats.groupby('cluster')['DEATH_EVENT'].agg(['count', 'sum'])
    summary['death_rate'] = summary['sum'] / summary['count']
    print('\nCluster summary (count, deaths, death_rate):')
    print(summary)

    # ---------------------------
    # 6) Visualization
    # ---------------------------
    # 6a: U-Matrix (inter-neuron distance heatmap)
    umatrix = compute_u_matrix(som)

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(umatrix, cmap='viridis')
    ax.set_title('SOM U-Matrix (inter-neuron distances)')
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.show()

    # 6b: Feature heatmaps for a few important features (choose a subset if many)
    num_features = len(feature_names)
    # Select up to 6 most informative features by variance (heuristic)
    feature_variances = np.var(df.drop(columns=['DEATH_EVENT']).values, axis=0)
    top_idx = np.argsort(-feature_variances)[:6]
    selected_features = [feature_names[i] for i in top_idx]

    W_grid = som.weights  # shape (m,n,dim)
    for i, feat in enumerate(selected_features):
        feat_idx = feature_names.index(feat)
        heat = W_grid[:, :, feat_idx]
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(heat, cmap='coolwarm')
        ax.set_title(f'Feature heatmap: {feat}')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

    # 6c: Cluster distribution (counts)
    counts = pd.Series(patient_cluster_labels).value_counts().sort_index()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of patients')
    ax.set_title('Patient count per cluster')
    plt.tight_layout()
    plt.show()

    # 6d: Death rate per cluster bar chart
    fig, ax = plt.subplots()
    summary['death_rate'].plot(kind='bar', ax=ax)
    ax.set_ylabel('Death rate')
    ax.set_title('Death rate per SOM cluster')
    plt.tight_layout()
    plt.show()

    # Return useful objects for further inspection
    return {
        'som': som,
        'patient_cluster_labels': patient_cluster_labels,
        'neuron_labels': best_labels,
        'summary': summary,
        'feature_names': feature_names,
        'scaler': scaler,
        'bmu_coords': bmu_coords,
        'bmu_indices': bmu_indices
    }

# Utility: compute U-matrix

def compute_u_matrix(som_obj):
    """Compute the U-Matrix for a SOM instance.

    U-matrix is the average distance between a neuron and its immediate
    neighbors, giving a visualization of cluster boundaries.
    """
    m, n = som_obj.m, som_obj.n
    U = np.zeros((m, n))
    W = som_obj.weights
    for i in range(m):
        for j in range(n):
            neighbors = []
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n:
                    neighbors.append(W[ni, nj])
            if neighbors:
                dists = [np.linalg.norm(W[i, j] - nb) for nb in neighbors]
                U[i, j] = np.mean(dists)
            else:
                U[i, j] = 0.0
    return U

# ---------------------------
# If run as script
# ---------------------------
if __name__ == '__main__':
    # Adjust grid size and iterations to available compute
    results = run_som_pipeline(csv_path=DATA_PATH, grid_size=(15, 15), num_iterations=4000, seed=42)
    print('\nPipeline completed. Objects returned:')
    for k in results.keys():
        print('-', k)

    # Short suggestion on interpreting clusters (printed for student)
    print('\nInterpretation hints:')
    print('- High U-Matrix values between neurons indicate cluster borders.')
    print('- Feature heatmaps show where values of a feature are high/low on the map.')
    print('- Compare death_rate per cluster (shown above) to identify high-risk groups.')
    print("- Inspect patients in clusters with high death rate and look at their\n  average feature values (age, ejection_fraction, serum_creatinine, etc.) to form a medical risk profile.")
