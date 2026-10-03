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
