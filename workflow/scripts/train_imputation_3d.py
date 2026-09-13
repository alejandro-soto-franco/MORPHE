from morphe.training.slice3d_trainer import Slice3DTrainer

# pyrefly: ignore [unknown-name]
cfg = snakemake.config["imputation_3d"]
# pyrefly: ignore [unknown-name]
pretrained = snakemake.config["pretrained"]

trainer = Slice3DTrainer(
    # pyrefly: ignore [unknown-name]
    train_dir=snakemake.input.train_dir,
    # pyrefly: ignore [unknown-name]
    val_dir=snakemake.input.val_dir,
    pretrained_model_id=pretrained["model_id"],
    pretrained_revision=pretrained["revision"],
    # pyrefly: ignore [unknown-name]
    checkpoint_dir=snakemake.output[0],
    cond_dim=cfg["cond_dim"],
    lr=cfg["lr"],
)
trainer.train(epochs=cfg["epochs"], patience=cfg["patience"])
