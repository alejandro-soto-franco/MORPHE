"""Downloads the four inference-relevant files of the published stage-1 checkpoint.

Verifies each against the SHA-256 committed in config (see
``outpainting_2d_imputation.published_checkpoint`` in config/config.yaml).
"""

import hashlib
from pathlib import Path

from huggingface_hub import hf_hub_download

# pyrefly: ignore [unknown-name]
cfg = snakemake.config["outpainting_2d_imputation"]["published_checkpoint"]
# pyrefly: ignore [unknown-name]
repo = snakemake.params.repo
# pyrefly: ignore [unknown-name]
dest = Path(snakemake.output.unet).parent
dest.mkdir(parents=True, exist_ok=True)

for entry in cfg["files"]:
    local_path = hf_hub_download(repo_id=repo, filename=entry["path"], local_dir=str(dest))

    digest = hashlib.sha256()
    with open(local_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)

    actual = digest.hexdigest()
    expected = entry["sha256"]
    if len(expected) != 64:
        raise ValueError(f"config sha256 for {entry['path']} is not 64 hex characters: {expected!r}")
    if actual != expected:
        raise ValueError(f"{entry['path']}: sha256 mismatch (expected {expected}, got {actual})")

    print(f"verified {entry['path']}: {actual}")
