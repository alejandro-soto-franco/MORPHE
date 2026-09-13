"""Tests for the five evaluation metrics and the reconstructed missing helpers.

See morphe.evaluation.metrics's module docstring: six helper functions this
module needs (``kl_to_similarity``, ``compute_hist``, ``compute_ratio``,
``compare_cluster_feature_centroids``, ``catplot2``,
``save_inferred_map_to_df``) are referenced but never defined anywhere in the
upstream repository.
"""

import numpy as np
import torch

from morphe.evaluation import metrics as ev


def _type_map(pattern: np.ndarray) -> torch.Tensor:
    return torch.tensor(pattern, dtype=torch.long).unsqueeze(0)


def test_kl_to_similarity_bounds():
    assert ev.kl_to_similarity(0.0, 0.0, 1.0) == 1.0
    assert ev.kl_to_similarity(1.0, 0.0, 1.0) == 0.0
    assert ev.kl_to_similarity(2.0, 0.0, 1.0) == 0.0  # clipped


def test_compute_ratio_sums_to_one():
    feats = np.array([1, 1, 2, 3, 3, 3])
    ratio = ev.compute_ratio(feats, n_types=5)
    assert ratio.shape == (5,)
    assert abs(ratio.sum() - 1.0) < 1e-6
    assert abs(ratio[0] - 2 / 6) < 1e-6  # type 1
    assert abs(ratio[2] - 3 / 6) < 1e-6  # type 3


def test_compute_hist_rows_sum_to_one():
    coords = np.array([[i, 0] for i in range(6)])
    feats = np.array([1, 1, 1, 2, 2, 2])
    hist = ev.compute_hist(feats, coords, k=3, n_types=3)
    assert hist.shape == (6, 3)
    np.testing.assert_allclose(hist.sum(axis=1), 1.0, atol=1e-5)


def test_compare_cluster_feature_centroids_aligns_identical_clusters():
    hist = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = np.array([0, 0, 1, 1])

    row_order, col_order, cost_matrix, total_dist = ev.compare_cluster_feature_centroids(
        hist, labels, hist, labels, n_clusters=2
    )
    assert total_dist == 0.0
    assert list(row_order) == list(col_order)


def test_rgb_centroid_distance_score_identical_maps_is_one():
    rgb = torch.rand(3, 4, 4)
    types = _type_map(np.array([[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 3, 3], [0, 0, 3, 3]]))
    score = ev.rgb_centroid_distance_score(rgb, rgb, types, types)
    assert abs(score - 1.0) < 1e-6


def test_cell_type_distribution_identical_maps_is_one():
    types = _type_map(np.array([[1, 1, 2, 2], [1, 1, 2, 2], [0, 0, 3, 3], [0, 0, 3, 3]]))
    score = ev.cell_type_distribution(types, types)
    assert abs(score - 1.0) < 1e-6


def test_cell_type_distribution_empty_maps_returns_one():
    empty = _type_map(np.zeros((4, 4), dtype=int))
    assert ev.cell_type_distribution(empty, empty) == 1.0


def test_save_inferred_map_to_df_shape():
    types = _type_map(np.array([[1, 2], [3, 4]]))
    df = ev.save_inferred_map_to_df(types, region_id=7)
    assert len(df) == 4
    assert set(df["region_id"]) == {7}
    assert set(df["cell_type"]) == {1, 2, 3, 4}
