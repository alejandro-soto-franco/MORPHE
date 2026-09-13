"""Arbitrary-shape inpainting: train and infer.

Not wired into ``rule all``: no published checkpoint exists for this use
case's held-out region set on this machine's data budget. Run
``pixi run snakemake train_arbitrary_inpainting`` directly with
``arbitrary_inpainting.split.root`` pointed at your own region images.
"""

rule train_arbitrary_inpainting:
    output:
        directory(config["arbitrary_inpainting"]["checkpoint_dir"]),
    script:
        "../scripts/train_arbitrary_inpainting.py"
