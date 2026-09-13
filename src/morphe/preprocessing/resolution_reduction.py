"""Reduces the coordinate grid of a spatial-omics cell table.

Ported from ``preprocessing/resolution_reduction.ipynb``. Fine-grained
pixel coordinates leave most grid points empty; this searches for the
coarsest coordinate grid at which every cell in every region still maps to a
distinct ``(x, y)`` pair, halving the search interval each round.

The upstream notebook hardcoded its input path to
``/Users/zachrobers/Spatial_LLM/notebooks/conflict_cleaned_CODEX_HuBMAP_alldata_Dryad_merged.csv``,
a machine-local, already-merged file distinct from the raw per-donor Dryad
release; callers now pass an explicit path.
"""

from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

Coord = tuple[float, float]


def calculate_dimensions(passing: Coord, failing: Coord) -> Coord:
    """Midpoint between a known-good and a known-failing grid size.

    Returns ``passing`` unchanged once the midpoint would equal ``failing``
    (the search has converged to one grid step of resolution).
    """
    x_p, y_p = passing
    x_f, y_f = failing

    new = (round((x_p + x_f) * 0.5), round((y_p + y_f) * 0.5))
    return passing if new == failing else new


def compute_new_coord_pair(
    coord_pair: Coord, dimensions: Coord, original_dimensions: Coord
) -> tuple[int, int]:
    """Rescale a coordinate pair from ``original_dimensions`` onto ``dimensions``."""
    c1, c2 = coord_pair
    x, y = dimensions
    x_i, y_i = original_dimensions

    return round(c1 * (x / x_i)), round(c2 * (y / y_i))


def reduce_dimensions(df_initial: pd.DataFrame, original_dimensions: Coord) -> tuple[pd.DataFrame, Coord]:
    """Binary-search the coarsest grid at which no two cells in any region collide.

    Returns the rescaled dataframe (only ``x``/``y`` change) and the chosen
    grid dimensions.
    """
    failing_dimensions: Coord = (1, 1)
    passing_dimensions = original_dimensions
    dimensions: Coord = failing_dimensions
    restart_outer_loop = True

    while dimensions != passing_dimensions:
        if restart_outer_loop:
            failing_dimensions = dimensions
        else:
            passing_dimensions = dimensions

        dimensions = calculate_dimensions(passing_dimensions, failing_dimensions)

        df = copy.deepcopy(df_initial)
        grouped_by_region = df.groupby("unique_region")
        restart_outer_loop = False

        for _region_name, region_data in grouped_by_region:
            coords: set[tuple[int, int]] = set()

            for idx, row in region_data.iterrows():
                coord_pair = (row["x"], row["y"])
                new_coord_pair = compute_new_coord_pair(coord_pair, dimensions, original_dimensions)

                if new_coord_pair in coords:
                    restart_outer_loop = True
                    break

                # pyrefly: ignore [unsupported-operation]
                df.at[idx, "x"], df.at[idx, "y"] = new_coord_pair
                coords.add(new_coord_pair)

            if restart_outer_loop:
                break

    # pyrefly: ignore [unbound-name]
    return df, dimensions


def coordinate_fill_ratios(df: pd.DataFrame) -> dict[str, float]:
    """Fraction of the bounding grid occupied by a cell, per region.

    Used both before and after :func:`reduce_dimensions` to report how much
    denser the reduced grid is.
    """
    ratios = {}
    for region, region_data in df.groupby("unique_region"):
        max_x = region_data["x"].max()
        max_y = region_data["y"].max()
        total_possible = (max_x + 1) * (max_y + 1)
        ratios[region] = len(region_data) / total_possible
    # pyrefly: ignore [bad-return]
    return ratios


def run(input_csv: str | Path, output_csv: str | Path, original_dimensions: Coord) -> Coord:
    """Load ``input_csv``, reduce its coordinate grid and write ``output_csv``."""
    df = pd.read_csv(input_csv, index_col=0)
    new_df, new_dimensions = reduce_dimensions(df, original_dimensions)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(output_csv, index=False)
    return new_dimensions
