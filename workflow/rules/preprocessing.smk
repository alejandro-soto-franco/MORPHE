"""Resolution reduction of a spatial-omics cell table.

Not wired into ``rule all``: the upstream notebook's exact input (a
pre-merged, machine-local CSV) is not part of the published Dryad release.
Point ``preprocessing.input_csv`` in config at your own merged table and run
``pixi run snakemake reduce_resolution`` directly. See the README's
"Differences from upstream" section.
"""

rule reduce_resolution:
    input:
        csv=config["preprocessing"]["input_csv"],
    output:
        csv=config["preprocessing"]["output_csv"],
    params:
        original_dimensions=tuple(config["preprocessing"]["original_dimensions"]),
    script:
        "../scripts/reduce_resolution.py"
