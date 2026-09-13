from morphe.finetune.fluxfill import FluxFillLoraTrainer

# pyrefly: ignore [unknown-name]
cfg = snakemake.config["finetune"]["fluxfill"]

trainer = FluxFillLoraTrainer(
    # pyrefly: ignore [unknown-name]
    train_root=snakemake.config.get("finetune_train_root", "data/finetune/train"),
    # pyrefly: ignore [unknown-name]
    val_root=snakemake.config.get("finetune_val_root", "data/finetune/val"),
    hf_repo=cfg["model_id"],
    # pyrefly: ignore [unknown-name]
    save_dir=snakemake.output[0],
    lora_rank=cfg["lora_rank"],
)
# pyrefly: ignore [unknown-name]
trainer.train(epochs=snakemake.config.get("finetune_epochs", 10))
