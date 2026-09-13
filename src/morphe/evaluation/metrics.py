"""Five metrics comparing a generated tissue map against the true map.

Ported from ``Evaluation/Evaluation.ipynb``: RGB centroid distance, neighbour
KMeans composition matching, cell density, spatial structure (LPIPS) and
cell-type distribution.

Upstream defect, not one of the five confirmed bugs but found while
porting and documented here: the notebook calls six helper functions
(``kl_to_similarity``, ``compute_hist``, ``compute_ratio``,
``compare_cluster_feature_centroids``, ``catplot2``,
``save_inferred_map_to_df``) that are referenced but never defined anywhere
in the repository, so ``Evaluation.ipynb`` cannot run end to end as shipped
(consistent with its stored outputs being empty). The five score functions'
own logic is ported verbatim; the six missing helpers are reconstructed here
from their call sites and docstrings, and are the one place in this module
that is a reconstruction rather than a port.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import rel_entr
from sklearn.cluster import MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

CELL_TYPE_NAMES = [
    "B",
    "CD4+ T cell",
    "CD57+ Enterocyte",
    "CD66+ Enterocyte",
    "CD7+ Immune",
    "CD8+ T",
    "Cycling TA",
    "DC",
    "Endothelial",
    "Enterocyte",
    "Goblet",
    "ICC",
    "Lymphatic",
    "M1 Macrophage",
    "M2 Macrophage",
    "MUC1+ Enterocyte",
    "NK",
    "Nerve",
    "Neuroendocrine",
    "Neutrophil",
    "Paneth",
    "Plasma",
    "Smooth muscle",
    "Stroma",
    "TA",
]


# ---------------------------------------------------------------------------
# Reconstructed helpers (see module docstring)
# ---------------------------------------------------------------------------


def kl_to_similarity(avg_dist: float, min_kl: float = 0.0, max_kl: float = 1.0) -> float:
    """Map a non-negative distance to a similarity in ``[0, 1]`` (lower distance -> higher score)."""
    if max_kl <= min_kl:
        raise ValueError("max_kl must exceed min_kl")
    normalised = np.clip((avg_dist - min_kl) / (max_kl - min_kl), 0.0, 1.0)
    return float(1.0 - normalised)


def compute_hist(feats: np.ndarray, coords: np.ndarray, k: int = 50, n_types: int = 25) -> np.ndarray:
    """Per-point local cell-type composition histogram over its ``k`` nearest neighbours.

    ``feats`` holds 1-indexed cell types (0 reserved for background and
    already filtered out by the caller). Returns ``[N, n_types]``, each row
    normalised to sum to 1.
    """
    n_neighbors = min(k, len(coords))
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(coords)
    _, idx = nbrs.kneighbors(coords)
    neighbor_types = feats[idx]

    hist = np.zeros((len(coords), n_types), dtype=np.float32)
    for t in range(n_types):
        hist[:, t] = (neighbor_types == (t + 1)).sum(axis=1)
    hist /= hist.sum(axis=1, keepdims=True) + 1e-8
    return hist


def compute_ratio(feats: np.ndarray, n_types: int = 25) -> np.ndarray:
    """Global fraction of each (1-indexed) cell type in ``feats``."""
    counts = np.array([(feats == (t + 1)).sum() for t in range(n_types)], dtype=np.float64)
    return counts / (counts.sum() + 1e-8)


def compare_cluster_feature_centroids(
    hist_true: np.ndarray,
    labels_true: np.ndarray,
    hist_gen: np.ndarray,
    labels_gen: np.ndarray,
    n_clusters: int,
    metric: str = "euclidean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Align true/generated cluster labels by nearest composition centroid.

    Returns ``(row_order, col_order, cost_matrix, total_dist)`` from the
    Hungarian assignment between the two sets of cluster centroids.
    """

    def _centroids(hist: np.ndarray, labels: np.ndarray) -> np.ndarray:
        out = np.zeros((n_clusters, hist.shape[1]), dtype=np.float64)
        for c in range(n_clusters):
            mask = labels == c
            if np.any(mask):
                out[c] = hist[mask].mean(axis=0)
        return out

    centroids_true = _centroids(hist_true, labels_true)
    centroids_gen = _centroids(hist_gen, labels_gen)

    # pyrefly: ignore [no-matching-overload]
    cost_matrix = cdist(centroids_true, centroids_gen, metric=metric)
    row_order, col_order = linear_sum_assignment(cost_matrix)
    total_dist = float(cost_matrix[row_order, col_order].sum())
    return row_order, col_order, cost_matrix, total_dist


def save_inferred_map_to_df(type_map: torch.Tensor, region_id: int) -> pd.DataFrame:
    """Flatten a ``[1, H, W]`` cell-type map into a long ``(x, y, cell_type, region_id)`` frame."""
    arr = type_map.squeeze().cpu().numpy()
    H, W = arr.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    return pd.DataFrame(
        {"x": xs.flatten(), "y": ys.flatten(), "cell_type": arr.flatten(), "region_id": region_id}
    )


# ---------------------------------------------------------------------------
# Metric 1: RGB centroid distance
# ---------------------------------------------------------------------------


def rgb_centroid_distance_score(
    map_true: torch.Tensor,
    map_gen: torch.Tensor,
    type_true: torch.Tensor,
    type_gen: torch.Tensor,
    penalty_value: float = 2.5,
) -> float:
    """Similarity of the per-cell-type mean RGB ("centroid") between the two maps."""
    map_true_np = map_true.permute(1, 2, 0).cpu().numpy() if map_true.shape[0] == 3 else map_true
    map_gen_np = map_gen.permute(1, 2, 0).cpu().numpy() if map_gen.shape[0] == 3 else map_gen

    type_true_np = type_true.squeeze().cpu().numpy()
    type_gen_np = type_gen.squeeze().cpu().numpy()

    centroids_true: dict[int, np.ndarray] = {}
    centroids_gen: dict[int, np.ndarray] = {}
    all_types = set(np.unique(type_true_np)).union(np.unique(type_gen_np))
    all_types.discard(0)

    for t in all_types:
        coords_true = np.argwhere(type_true_np == t)
        coords_gen = np.argwhere(type_gen_np == t)
        if len(coords_true) > 0:
            # pyrefly: ignore [no-matching-overload]
            centroids_true[t] = map_true_np[coords_true[:, 0], coords_true[:, 1]].mean(axis=0)
        if len(coords_gen) > 0:
            # pyrefly: ignore [no-matching-overload]
            centroids_gen[t] = map_gen_np[coords_gen[:, 0], coords_gen[:, 1]].mean(axis=0)

    distances = []
    for t in sorted(all_types):
        if t in centroids_true and t in centroids_gen:
            distances.append(np.linalg.norm(centroids_true[t] - centroids_gen[t]))
        else:
            # pyrefly: ignore [bad-argument-type]
            distances.append(penalty_value)

    avg_dist = float(np.mean(distances)) if distances else 0.0
    return kl_to_similarity(avg_dist, min_kl=0.0, max_kl=1.0)


# ---------------------------------------------------------------------------
# Metric 2: neighbour KMeans composition matching
# ---------------------------------------------------------------------------


def compute_cluster_celltype_composition(
    cluster_labels: np.ndarray, cell_types: np.ndarray, n_clusters: int = 5, n_types: int = 34
) -> np.ndarray:
    comp = np.zeros((n_clusters, n_types), dtype=np.float32)
    for i in range(len(cluster_labels)):
        comp[cluster_labels[i], cell_types[i] - 1] += 1
    return comp


def neighbor_kmeans_composition_matching_score(
    true_map: torch.Tensor,
    gen_map1: torch.Tensor,
    gen_map2: torch.Tensor,
    gen_map3: torch.Tensor,
    k: int = 50,
    n_clusters: int = 5,
) -> list[float]:
    """KMeans-based local composition similarity, one score per generated map."""
    maps = [gen_map1, gen_map2, gen_map3]
    scores = []

    map_true = true_map.squeeze(0)
    H, W = map_true.shape
    coords_true = np.array([(x, y) for y in range(H) for x in range(W)])
    feats_true = map_true.flatten().cpu().numpy()
    mask_true = feats_true != 0
    feats_true = feats_true[mask_true]
    coords_true = coords_true[mask_true]

    hist_true = compute_hist(feats_true, coords_true, k=k)
    km_true = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(hist_true)
    cluster_labels_true = km_true.labels_

    for map_gen in maps:
        map_gen = map_gen.squeeze(0)
        coords_gen = np.array([(x, y) for y in range(H) for x in range(W)])
        feats_gen = map_gen.flatten().cpu().numpy()
        mask_gen = feats_gen != 0
        feats_gen = feats_gen[mask_gen]
        coords_gen = coords_gen[mask_gen]

        hist_gen = compute_hist(feats_gen, coords_gen, k=k)
        km_gen = MiniBatchKMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
        cluster_labels_gen = km_gen.fit_predict(hist_gen)

        _, _, _, total_dist = compare_cluster_feature_centroids(
            hist_true, cluster_labels_true, hist_gen, cluster_labels_gen, n_clusters=n_clusters
        )
        scores.append(1 / (1 + total_dist))

    return scores


# ---------------------------------------------------------------------------
# Metric 3: cell density
# ---------------------------------------------------------------------------


def _get_coords_and_feats(types: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    H, W = types.shape[1:]
    coords = np.array([(i, j) for i in range(H) for j in range(W)])
    feats = types.squeeze(0).flatten().cpu().numpy()
    mask = feats != 0
    return coords[mask], feats[mask]


def _distance_distributions(
    coords: np.ndarray, feats: np.ndarray, k: int, n_types: int = 25
) -> list[np.ndarray | None]:
    dists_by_type: list[np.ndarray | None] = [None] * n_types
    for t in range(n_types):
        points = coords[feats == t]
        if len(points) < k + 1:
            continue
        nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
        dists, _ = nbrs.kneighbors(points)
        dists_by_type[t] = dists[:, 1:].mean(axis=1)
    return dists_by_type


def _extract_all_distances(cell_map: torch.Tensor, k: int) -> np.ndarray | None:
    coords = np.argwhere(cell_map.squeeze().cpu().numpy() != 0)
    if len(coords) <= k:
        return None
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(coords)
    dists, _ = nbrs.kneighbors(coords)
    return dists[:, 1:].mean(axis=1)


def cell_density_score_new(
    true_map: torch.Tensor,
    gen_map1: torch.Tensor,
    gen_map2: torch.Tensor,
    gen_map3: torch.Tensor,
    k: int = 20,
    bin_width: float = 5.0,
    penalty_value: float = 1.0,
    alpha: float = 0.5,
) -> list[float]:
    """Blends per-cell-type and global kNN-distance-distribution similarity (1/(1+KL))."""
    maps = [true_map, gen_map1, gen_map2, gen_map3]
    all_dists_by_map = [_distance_distributions(*_get_coords_and_feats(m), k=k) for m in maps]

    num_gen = 3
    score_matrix: list[list[float]] = [[] for _ in range(num_gen)]

    for t in range(25):
        all_dists = [d[t] for d in all_dists_by_map]
        if any(x is None for x in all_dists):
            for i in range(num_gen):
                score_matrix[i].append(1 / (1 + penalty_value))
            continue

        # pyrefly: ignore [no-matching-overload]
        true_max = max(np.max(all_dists[0]), 1e-8)
        bin_edges = np.arange(0, true_max + bin_width, bin_width)

        hists = []
        for d in all_dists:
            # pyrefly: ignore [bad-argument-type]
            hist, _ = np.histogram(d, bins=bin_edges)
            hist = hist.astype(np.float32) + 1e-8
            hist /= hist.sum()
            hists.append(hist)

        true_hist = hists[0]
        for i in range(num_gen):
            kl = np.sum(rel_entr(true_hist, hists[i + 1]))
            score_matrix[i].append(1 / (1 + kl))

    type_scores = [np.mean(s) for s in score_matrix]

    true_dists = _extract_all_distances(true_map, k)
    global_scores = []
    max_global = true_dists.max() if true_dists is not None else 1.0
    bin_edges_global = np.arange(0, max_global + bin_width, bin_width)

    for gen_map in [gen_map1, gen_map2, gen_map3]:
        gen_dists = _extract_all_distances(gen_map, k)
        if true_dists is None or gen_dists is None:
            global_scores.append(1 / (1 + penalty_value))
            continue

        true_hist, _ = np.histogram(true_dists, bins=bin_edges_global)
        gen_hist, _ = np.histogram(gen_dists, bins=bin_edges_global)
        true_hist = true_hist.astype(np.float32) + 1e-8
        gen_hist = gen_hist.astype(np.float32) + 1e-8
        true_hist /= true_hist.sum()
        gen_hist /= gen_hist.sum()

        kl = np.sum(rel_entr(true_hist, gen_hist))
        global_scores.append(1 / (1 + kl))

    # pyrefly: ignore [bad-return]
    return [alpha * type_scores[i] + (1 - alpha) * global_scores[i] for i in range(num_gen)]


# ---------------------------------------------------------------------------
# Metric 4: spatial structure (LPIPS)
# ---------------------------------------------------------------------------


def spatial_structure_score(true_map: torch.Tensor, gen_map: torch.Tensor, lpips_model: Any) -> float:
    """``1 - LPIPS``: higher means more perceptually similar RGB maps.

    ``lpips_model`` is a ``lpips.LPIPS`` instance, passed in rather than
    constructed at import time (the upstream notebook built one, on
    ``"cuda"``, as a module-level side effect).
    """
    device = next(lpips_model.parameters()).device
    m1 = torch.as_tensor(true_map).squeeze().to(device)
    m2 = torch.as_tensor(gen_map).squeeze().to(device)

    if m1.ndim != 3 or m1.shape[0] != 3:
        raise ValueError("input map must have shape [3, H, W]")

    m1 = (m1 * 2 - 1).unsqueeze(0)
    m2 = (m2 * 2 - 1).unsqueeze(0)

    with torch.no_grad():
        score = lpips_model(m1, m2).item()

    return 1 - score


# ---------------------------------------------------------------------------
# Metric 5: cell-type distribution
# ---------------------------------------------------------------------------


def cell_type_distribution(true_map: torch.Tensor, generated_map: torch.Tensor) -> float:
    """Global cell-type frequency similarity, ``1/(1 + KL^2)`` with a missing-type penalty."""
    true_map_np = true_map.squeeze().cpu().numpy()
    generated_map_np = generated_map.squeeze().cpu().numpy()

    unique_true = np.unique(true_map_np)
    unique_gen = np.unique(generated_map_np)
    all_labels = np.union1d(unique_true, unique_gen)
    all_labels = all_labels[all_labels != 0]

    if all_labels.size == 0:
        return 1.0

    num_classes = int(all_labels.max()) + 1
    true_hist, _ = np.histogram(true_map_np, bins=np.arange(1, num_classes + 1))
    gen_hist, _ = np.histogram(generated_map_np, bins=np.arange(1, num_classes + 1))

    if true_hist.sum() == 0 or gen_hist.sum() == 0:
        return 0.0

    eps = 1e-8
    true_hist = (true_hist + eps) / (true_hist.sum() + eps * len(true_hist))
    gen_hist = (gen_hist + eps) / (gen_hist.sum() + eps * len(gen_hist))

    missing_mask = (true_hist > eps) & (gen_hist <= eps)
    kl_div = np.sum(true_hist * np.log(true_hist / (gen_hist + eps)))
    kl_div += 2 * np.sum(true_hist[missing_mask])

    return float(1 / (1 + kl_div * kl_div))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def evaluate_generated_maps(
    true_img_path: str,
    gen_img_paths: tuple[str, str, str],
    autoencoder: Any,
    lpips_model: Any,
    load_png: Any,
    infer_cell_map: Any,
) -> dict[str, list[float]]:
    """Score up to three generated maps against one true map on all five metrics.

    ``load_png``/``infer_cell_map`` are injected (rather than imported here)
    to keep this module free of a hard PIL/autoencoder-checkpoint dependency
    at import time; pass
    :func:`morphe.embeddings.interpret_cellmap.load_and_recover_z3d_png` and
    :func:`morphe.embeddings.interpret_cellmap.infer_cell_map`.
    """
    true_map = load_png(true_img_path)
    gen_maps = [load_png(p) for p in gen_img_paths]

    type_true = infer_cell_map(true_map, autoencoder)
    type_gens = [infer_cell_map(m, autoencoder) for m in gen_maps]

    rgb_scores = [
        rgb_centroid_distance_score(true_map, gm, type_true, tg)
        for gm, tg in zip(gen_maps, type_gens, strict=True)
    ]
    kmeans_scores = neighbor_kmeans_composition_matching_score(type_true, *type_gens, k=10, n_clusters=20)
    density_scores = cell_density_score_new(type_true, *type_gens, k=50)
    structure_scores = [spatial_structure_score(true_map, gm, lpips_model) for gm in gen_maps]
    distribution_scores = [cell_type_distribution(type_true, tg) for tg in type_gens]

    return {
        "rgb_centroid_distance": rgb_scores,
        "neighbor_kmeans_composition": kmeans_scores,
        "cell_density": density_scores,
        "spatial_structure": structure_scores,
        "cell_type_distribution": distribution_scores,
    }
