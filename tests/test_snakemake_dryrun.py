"""Dry-run the Snakemake DAG on both configs (required by the workflow spec)."""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _dry_run(configfile: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "snakemake",
            "-s",
            "workflow/Snakefile",
            "--configfile",
            configfile,
            "-n",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )


def test_smoke_config_dry_run():
    result = _dry_run("config/smoke.yaml")
    assert result.returncode == 0, result.stdout + result.stderr


def test_full_config_dry_run():
    result = _dry_run("config/config.yaml")
    assert result.returncode == 0, result.stdout + result.stderr
