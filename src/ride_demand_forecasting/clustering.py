"""Pickup-zone assignment via KMeans, fit on training coordinates only.

Fitting on train-only coordinates (rather than train+test combined, as the
exploratory notebook does) avoids leaking test-set pickup locations into the
cluster centers.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from ride_demand_forecasting.config import N_PICKUP_ZONES, RANDOM_STATE


def fit_pickup_zones(
    train_df: pd.DataFrame,
    n_zones: int = N_PICKUP_ZONES,
    sample_weight: np.ndarray | None = None,
) -> KMeans:
    """Fit on one row per pickup, or on one row per distinct pickup point with
    `sample_weight` set to how many pickups happened there - the two are equivalent."""
    kmeans = KMeans(n_clusters=n_zones, random_state=RANDOM_STATE, n_init="auto")
    kmeans.fit(train_df[["start_lat", "start_lng"]], sample_weight=sample_weight)
    return kmeans


def assign_pickup_zones(df: pd.DataFrame, kmeans: KMeans) -> np.ndarray:
    return kmeans.predict(df[["start_lat", "start_lng"]])


def weighted_silhouette(coords: np.ndarray, labels: np.ndarray, weights: np.ndarray) -> float:
    """Mean silhouette of the dataset in which point `i` appears `weights[i]` times,
    computed without building it.

    That is the score a per-pickup clustering would get, which scikit-learn's
    `silhouette_score` can't give for weighted points.
    """
    n = len(coords)
    rows = np.arange(n)
    distances = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    clusters, cluster_of = np.unique(labels, return_inverse=True)
    membership = (cluster_of[None, :] == np.arange(len(clusters))[:, None]) * weights
    cluster_weight = membership.sum(axis=1)

    # distance_to[i, c]: summed distance from point i to every (repeated) member of c.
    distance_to = distances @ membership.T
    own_weight = cluster_weight[cluster_of]
    # Mean distance to the other members of its own cluster - its own copies are at
    # distance zero, and one of them is the point itself.
    within = distance_to[rows, cluster_of] / np.maximum(own_weight - 1, 1)
    mean_to = distance_to / cluster_weight
    mean_to[rows, cluster_of] = np.inf
    nearest_other = mean_to.min(axis=1)

    scale = np.maximum(within, nearest_other)
    scores = np.where(scale > 0, (nearest_other - within) / np.where(scale > 0, scale, 1), 0.0)
    # A cluster of a single pickup has no within-cluster distance; by convention it
    # scores zero.
    scores[own_weight <= 1] = 0.0
    return float(np.average(scores, weights=weights))
