"""3D imputation (z-stack slice interpolation): train.

Not wired into ``rule all``: requires a MERFISH whole-brain z-stack (see
README datasets section), outside this repository's data budget.
"""

rule train_imputation_3d:
    input:
        train_dir=config["imputation_3d"]["train_dir"],
        val_dir=config["imputation_3d"]["val_dir"],
    output:
        directory(config["imputation_3d"]["checkpoint_dir"]),
    script:
        "../scripts/train_imputation_3d.py"
