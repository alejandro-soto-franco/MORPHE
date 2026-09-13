from morphe.finetune.sd2 import SD21InpaintTrainer

# pyrefly: ignore [unknown-name]
cfg = snakemake.config["finetune"]["sd2"]

trainer = SD21InpaintTrainer(
    # pyrefly: ignore [unknown-name]
    train_root=snakemake.config.get("finetune_train_root", "data/finetune/train"),
    # pyrefly: ignore [unknown-name]
    val_root=snakemake.config.get("finetune_val_root", "data/finetune/val"),
    sd2_model_id=cfg["model_id"],
    clip_vision_model_id=cfg["clip_vision_model_id"],
    # pyrefly: ignore [unknown-name]
    save_dir=snakemake.output[0],
)
# pyrefly: ignore [unknown-name]
trainer.train(epochs=snakemake.config.get("finetune_epochs", 20))
