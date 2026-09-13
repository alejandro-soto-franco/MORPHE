"""Regression test: CoordEncoder's checkpoint layout must match the
published weights (flat keys), not the training module's own class shape.

See the README's "Additional defects found while porting" entry.
"""

from morphe.models.coord_encoder import CoordEncoder


def test_coord_encoder_state_dict_has_flat_keys():
    model = CoordEncoder()
    keys = set(model.state_dict().keys())

    assert keys == {"0.weight", "0.bias"}
    assert not any(k.startswith("net.") for k in keys)
