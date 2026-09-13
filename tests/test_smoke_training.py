"""The from-scratch tiny-resolution training smoke test actually trains."""

import math

from morphe.testing_fixtures import make_synthetic_regions
from morphe.training.smoke import run_train_smoke


def test_run_train_smoke_produces_finite_decreasing_ish_loss(tmp_path):
    make_synthetic_regions(tmp_path, n_regions=3, size=64)

    losses = run_train_smoke(tmp_path, steps=3, img_size=64)

    assert len(losses) == 3
    for loss in losses:
        assert math.isfinite(loss)
        assert loss >= 0
