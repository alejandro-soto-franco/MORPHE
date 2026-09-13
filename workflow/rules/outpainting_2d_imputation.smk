"""Stage-1 latent diffusion: outpainting and 2D imputation.

``train_smoke_outpainting_2d_imputation`` and ``evaluate_outpainting_2d_imputation``
are wired into ``rule all``. The smoke run builds tiny, randomly initialised
diffusers models directly (no download) on synthetic fixture images, so it
needs neither the published checkpoint nor real data. The full run uses the
published checkpoint and the six real region images in
``Assets/sample_regions``.
"""

rule make_smoke_fixture:
    output:
        directory("results/smoke/fixture_regions"),
    script:
        "../scripts/make_smoke_fixture.py"

rule download_stage1_checkpoint:
    output:
        unet="{dest}/model_1.safetensors",
        coord="{dest}/model_2.safetensors",
        cond="{dest}/model_3.safetensors",
        vae="{dest}/model.safetensors",
    params:
        repo=config["outpainting_2d_imputation"]["published_checkpoint"]["repo"],
    script:
        "../scripts/download_stage1_checkpoint.py"

rule train_smoke_outpainting_2d_imputation:
    # Always the synthetic fixture, in both smoke and full mode: this must
    # stay a fast, tiny-resolution smoke check regardless of config, unlike
    # infer_outpainting/infer_imputation_2d below (the real-data reproduction).
    input:
        rules.make_smoke_fixture.output[0],
    output:
        touch("results/checkpoints/outpainting_2d_imputation/.smoke_done"),
    script:
        "../scripts/train_smoke_outpainting_2d_imputation.py"

rule infer_outpainting:
    input:
        unet=os.path.join(config["outpainting_2d_imputation"]["published_checkpoint"]["dest"], "model_1.safetensors"),
        coord=os.path.join(config["outpainting_2d_imputation"]["published_checkpoint"]["dest"], "model_2.safetensors"),
        cond=os.path.join(config["outpainting_2d_imputation"]["published_checkpoint"]["dest"], "model_3.safetensors"),
        region="Assets/sample_regions/region_0.png",
    output:
        image="results/outpainting/region_0_outpainted.png",
    # The single 8 GB GPU cannot fit two SD1.5-sized models at once (fp16,
    # ~2 GB each including activations, but this box also runs other agents'
    # jobs); this and infer_imputation_2d must not run concurrently.
    resources:
        gpu=1,
    script:
        "../scripts/infer_outpainting.py"

rule infer_imputation_2d:
    input:
        unet=os.path.join(config["outpainting_2d_imputation"]["published_checkpoint"]["dest"], "model_1.safetensors"),
        coord=os.path.join(config["outpainting_2d_imputation"]["published_checkpoint"]["dest"], "model_2.safetensors"),
        cond=os.path.join(config["outpainting_2d_imputation"]["published_checkpoint"]["dest"], "model_3.safetensors"),
        vae=os.path.join(config["outpainting_2d_imputation"]["published_checkpoint"]["dest"], "model.safetensors"),
        region="Assets/sample_regions/region_1.png",
    output:
        image="results/imputation_2d/region_1_imputed.png",
    resources:
        gpu=1,
    script:
        "../scripts/infer_imputation_2d.py"

rule evaluate_outpainting_2d_imputation:
    input:
        outpainted=rules.infer_outpainting.output.image,
        imputed=rules.infer_imputation_2d.output.image,
    output:
        metrics="results/evaluation/outpainting_2d_imputation_metrics.json",
    script:
        "../scripts/evaluate_outpainting_2d_imputation.py"
