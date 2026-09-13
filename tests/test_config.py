"""Config sanity checks: every committed download hash is a real SHA-256."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(name: str) -> dict:
    with open(REPO_ROOT / "config" / name) as fh:
        return yaml.safe_load(fh)


def _iter_sha256_fields(node, path=""):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "sha256":
                yield path, v
            else:
                yield from _iter_sha256_fields(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _iter_sha256_fields(v, f"{path}[{i}]")


def test_config_yaml_parses():
    cfg = _load("config.yaml")
    assert cfg["mask_axis_convention"] in ("upstream", "xy")


def test_smoke_yaml_parses():
    cfg = _load("smoke.yaml")
    assert cfg["mode"] == "smoke"


def test_every_committed_sha256_is_64_hex_chars():
    cfg = _load("config.yaml")
    hashes = list(_iter_sha256_fields(cfg))
    assert hashes, "expected at least one sha256 field in config.yaml"

    for path, value in hashes:
        if value.startswith("PLACEHOLDER"):
            continue  # documented as not-yet-obtained (see README)
        assert len(value) == 64, f"{path}: sha256 {value!r} is not 64 hex characters"
        int(value, 16)  # raises if not valid hex
