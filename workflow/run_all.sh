#!/usr/bin/env bash
# Wrapper for `pixi run all`. Two gotchas this works around:
#   1. pixi's own task-string tokenizer mis-splits a bare `--resources gpu=1`
#      argument, so it lives in a real script instead.
#   2. `--resources` takes nargs='+' and, if it comes before the `all` target
#      with nothing recognised in between, greedily swallows `all` too
#      ("dictionary update sequence element #1 has length 1"). The target
#      must come before `--resources` on the command line.
set -euo pipefail

export OMP_NUM_THREADS=6
exec nice -n 19 snakemake \
    -s workflow/Snakefile \
    --configfile config/config.yaml \
    --cores 6 \
    all \
    --resources gpu=1
