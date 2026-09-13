"""Configuration loading for the MORPHE workflow.

Every path, seed and hyperparameter that the original notebooks hardcoded
(Colab drive paths, ``runwayml/stable-diffusion-v1-5``, placeholder file
names) lives in ``config/config.yaml`` or ``config/smoke.yaml`` and is loaded
through :func:`load_config`. Nothing in the library imports a literal path.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


@dataclasses.dataclass(frozen=True)
class PretrainedConfig:
    """A pinned diffusers model id and revision.

    The upstream notebooks load ``runwayml/stable-diffusion-v1-5``, which
    HuggingFace removed. ``model_id`` points at a maintained mirror and
    ``revision`` pins an exact commit hash so the resolved weights never
    drift under us.
    """

    model_id: str
    revision: str


@dataclasses.dataclass(frozen=True)
class SplitConfig:
    """An explicit, region-based held-out split.

    ``val_regions`` lists the region identifiers held out for validation;
    every other region under ``root`` is used for training. Recording the
    split in config (rather than a second hardcoded directory that may not
    exist) makes the held-out set reproducible and auditable.
    """

    root: str
    val_regions: tuple[str, ...]
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class DownloadSpec:
    """A single verified download: URL, destination and SHA-256."""

    url: str
    dest: str
    sha256: str


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    fields = {f.name for f in dataclasses.fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in fields}
    for k, v in list(kwargs.items()):
        if k == "val_regions" and isinstance(v, list):
            kwargs[k] = tuple(v)
    return cls(**kwargs)


class MorpheConfig:
    """Thin wrapper over the parsed YAML config.

    Access nested values with :meth:`get` (dotted path) or use the typed
    helpers (:meth:`pretrained`, :meth:`split`) for the sections that have a
    dataclass shape. Everything else stays plain ``dict``/``list`` data,
    since the config spans several unrelated pipelines.
    """

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root

    def get(self, path: str, default: Any = dataclasses.MISSING) -> Any:
        node: Any = self._data
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                if default is dataclasses.MISSING:
                    raise KeyError(f"config path {path!r} not found")
                return default
            node = node[part]
        return node

    def pretrained(self, path: str = "pretrained") -> PretrainedConfig:
        return _dict_to_dataclass(PretrainedConfig, self.get(path))

    def split(self, path: str) -> SplitConfig:
        return _dict_to_dataclass(SplitConfig, self.get(path))

    def resolve_path(self, value: str) -> Path:
        """Resolve a config path value against ``config.paths.data_root``/``results_root``."""
        p = Path(value)
        if p.is_absolute():
            return p
        return self.root / p


def load_config(path: str | Path) -> MorpheConfig:
    path = Path(path)
    with open(path) as fh:
        data = yaml.safe_load(fh)
    return MorpheConfig(data, root=path.resolve().parent.parent)
