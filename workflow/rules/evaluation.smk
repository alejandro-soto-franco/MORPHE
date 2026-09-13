"""Standalone evaluation metrics, given any true/generated map pair.

``evaluate_outpainting_2d_imputation`` (in outpainting_2d_imputation.smk) is
the concrete instance wired into ``rule all``; this rule is the same script
usable against any other pair of maps.
"""

rule evaluate_maps:
    input:
        true_map=config["evaluation"].get("true_map", "data/evaluation/true.png"),
        gen_maps=config["evaluation"].get(
            "gen_maps", ["data/evaluation/gen1.png", "data/evaluation/gen2.png", "data/evaluation/gen3.png"]
        ),
        autoencoder_checkpoint=config["embeddings"]["autoencoder"]["checkpoint_path"],
    output:
        metrics="results/evaluation/metrics.json",
    script:
        "../scripts/evaluate_maps.py"
