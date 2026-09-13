"""Pixel-diffusion decoder: precompute (latent, image) pairs, then train.

Not wired into ``rule all``: requires a directory of full-resolution region
images beyond the six shipped samples. Run
``pixi run snakemake train_pixel_decoder`` directly once
``pixel_decoder.train_index``/``val_index`` point at a real precomputed set
(see :func:`morphe.inference.pixel_decoder.precompute_latents`).
"""

rule train_pixel_decoder:
    input:
        train_index=config["pixel_decoder"]["train_index"],
        val_index=config["pixel_decoder"]["val_index"],
    output:
        directory(config["pixel_decoder"]["checkpoint_dir"]),
    script:
        "../scripts/train_pixel_decoder.py"
