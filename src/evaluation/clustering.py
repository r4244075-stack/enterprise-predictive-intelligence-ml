"""Customer segmentation: K-Means++, Agglomerative Ward Hierarchical Clustering, DBSCAN, and cluster validity metrics."""

from __future__ import annotations
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist


def kmeans_plus_plus(X: np.ndarray, k: int, n_init: int = 10, max_iter: int = 100, random_state: int = 42):
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape
    rng = np.random.default_rng(random_state)
    best_inertia = np.inf
    best_labels = None
    best_centers = None

    for _ in range(n_init):
        centers = np.empty((k, d), dtype=np.float64)
        centers[0] = X[rng.integers(0, n)]
        dist_sq = np.sum((X - centers[0]) ** 2, axis=1)
        for c in range(1, k):
            probs = dist_sq / max(np.sum(dist_sq), 1e-12)
            centers[c] = X[rng.choice(n, p=probs)]
            dist_sq = np.minimum(dist_sq, np.sum((X - centers[c]) ** 2, axis=1))

        for _it in range(max_iter):
            dists = cdist(X, centers, metric="sqeuclidean")
            labels = np.argmin(dists, axis=1)
            new_centers = np.zeros_like(centers)
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    new_centers[c] = X[mask].mean(axis=0)
                else:
                    new_centers[c] = X[rng.integers(0, n)]
            if np.allclose(new_centers, centers, atol=1e-5):
                centers = new_centers
                break
            centers = new_centers

        dists = cdist(X, centers, metric="sqeuclidean")
        labels = np.argmin(dists, axis=1)
        inertia = float(np.sum(dists[np.arange(n), labels]))
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centers

    return best_labels, best_centers, best_inertia


def silhouette_score_sampled(X: np.ndarray, labels: np.ndarray, sample_size: int = 4000, random_state: int = 42) -> float:
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    valid = labels >= 0
    X = X[valid]
    labels = labels[valid]
    uniq = np.unique(labels)
    if len(uniq) < 2 or len(X) < 10:
        return -1.0
    if len(X) > sample_size:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=sample_size, replace=False)
        X = X[idx]
        labels = labels[idx]
    D = cdist(X, X, metric="euclidean")
    s = np.zeros(len(X), dtype=np.float64)
    for i in range(len(X)):
        own = labels == labels[i]
        a = np.sum(D[i, own]) / max(np.sum(own) - 1, 1)
        b = np.inf
        for c in uniq:
            if c == labels[i]:
                continue
            other = labels == c
            if np.any(other):
                b = min(b, float(np.mean(D[i, other])))
        s[i] = (b - a) / max(a, b, 1e-12)
    return float(np.mean(s))


def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
    valid = labels >= 0
    X = X[valid]
    labels = labels[valid]
    uniq = np.unique(labels)
    k = len(uniq)
    if k < 2:
        return float("inf")
    centroids = np.array([X[labels == c].mean(axis=0) for c in uniq])
    scatters = np.array([np.mean(np.linalg.norm(X[labels == c] - centroids[i], axis=1)) for i, c in enumerate(uniq)])
    cd = cdist(centroids, centroids, metric="euclidean")
    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i != j:
                R[i, j] = (scatters[i] + scatters[j]) / max(cd[i, j], 1e-12)
    return float(np.mean(np.max(R, axis=1)))


def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    valid = labels >= 0
    X = X[valid]
    labels = labels[valid]
    uniq = np.unique(labels)
    k = len(uniq)
    n = len(X)
    if k < 2 or n <= k:
        return 0.0
    overall_mean = X.mean(axis=0)
    bgss = 0.0
    wgss = 0.0
    for c in uniq:
        Xc = X[labels == c]
        mc = Xc.mean(axis=0)
        bgss += len(Xc) * float(np.sum((mc - overall_mean) ** 2))
        wgss += float(np.sum((Xc - mc) ** 2))
    return float((bgss / (k - 1)) / max(wgss / (n - k), 1e-12))


def hierarchical_ward_clustering(X: np.ndarray, k: int = 4, fit_sample: int = 3000, random_state: int = 42):
    """Fit Ward linkage on stratified/random subsample and assign full dataset to nearest Ward centroid."""
    rng = np.random.default_rng(random_state)
    n = len(X)
    sub_idx = rng.choice(n, size=min(n, fit_sample), replace=False)
    Z = linkage(X[sub_idx], method="ward")
    sub_labels = fcluster(Z, t=k, criterion="maxclust") - 1
    centroids = np.array([X[sub_idx][sub_labels == c].mean(axis=0) for c in range(k)])
    full_labels = np.argmin(cdist(X, centroids, metric="sqeuclidean"), axis=1)
    return full_labels, centroids, Z


def dbscan_clustering(X: np.ndarray, eps: float = 1.6, min_samples: int = 25) -> np.ndarray:
    """Fast cKDTree DBSCAN implementation."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    tree = cKDTree(X)
    neighbors = tree.query_ball_tree(tree, r=eps)
    labels = np.full(n, -1, dtype=int)
    visited = np.zeros(n, dtype=bool)
    cluster_id = 0

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        nbrs = neighbors[i]
        if len(nbrs) < min_samples:
            continue
        labels[i] = cluster_id
        queue = list(nbrs)
        qi = 0
        while qi < len(queue):
            j = queue[qi]
            qi += 1
            if not visited[j]:
                visited[j] = True
                jnbrs = neighbors[j]
                if len(jnbrs) >= min_samples:
                    queue.extend(jnbrs)
            if labels[j] == -1:
                labels[j] = cluster_id
        cluster_id += 1
    return labels


def pca_2d(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    var_ratio = (S ** 2) / max(np.sum(S ** 2), 1e-12)
    return Xc @ Vt[:2].T, var_ratio[:2]
