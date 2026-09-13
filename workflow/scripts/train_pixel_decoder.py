from morphe.training.pixel_decoder_trainer import Cascade512Trainer

# pyrefly: ignore [unknown-name]
cfg = snakemake.config["pixel_decoder"]
# pyrefly: ignore [unknown-name]
pretrained = snakemake.config["pretrained"]

trainer = Cascade512Trainer(
    # pyrefly: ignore [unknown-name]
    train_index=snakemake.input.train_index,
    # pyrefly: ignore [unknown-name]
    val_index=snakemake.input.val_index,
    # pyrefly: ignore [unknown-name]
    checkpoint_dir=snakemake.output[0],
    pretrained_model_id=pretrained["model_id"],
    pretrained_revision=pretrained["revision"],
    vis_dir=cfg["vis_dir"],
    batch_size=cfg["batch_size"],
    lr=cfg["lr"],
)
trainer.train(epochs=cfg["epochs"], patience=cfg["patience"], vis_steps=cfg.get("vis_steps", 50))
