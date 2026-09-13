"""SD2 and FluxFill outpainting finetune baselines.

Not wired into ``rule all``: comparison baselines against MORPHE's own
outpainting model, not MORPHE itself; both need a training image set beyond
the data budget.
"""

rule train_sd2_baseline:
    output:
        directory("results/checkpoints/finetune_sd2"),
    script:
        "../scripts/train_sd2_baseline.py"

rule train_fluxfill_baseline:
    output:
        directory("results/checkpoints/finetune_fluxfill"),
    script:
        "../scripts/train_fluxfill_baseline.py"
