from morphe.datasets.arbitrary_inpainting import split_by_region
from morphe.training.arbitrary_inpainting_trainer import ArbitraryInpaintTrainer

# pyrefly: ignore [unknown-name]
cfg = snakemake.config["arbitrary_inpainting"]
# pyrefly: ignore [unknown-name]
pretrained = snakemake.config["pretrained"]

train_ds, val_ds = split_by_region(
    root_dir=cfg["split"]["root"],
    val_regions=tuple(cfg["split"]["val_regions"]),
    masks_per_image_train=cfg["train_masks_per_image"],
    masks_per_image_val=cfg["val_masks_per_image"],
)

trainer = ArbitraryInpaintTrainer(
    train_dataset=train_ds,
    val_dataset=val_ds,
    # pyrefly: ignore [unknown-name]
    checkpoint_dir=snakemake.output[0],
    pretrained_model_id=pretrained["model_id"],
    pretrained_revision=pretrained["revision"],
    lr=cfg["lr"],
)
trainer.train(epochs=cfg["epochs"])
