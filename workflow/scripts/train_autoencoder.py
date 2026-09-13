import torch

from morphe.embeddings.autoencoder import embed_to_rgb, train_autoencoder

# pyrefly: ignore [unknown-name]
cfg = snakemake.config["embeddings"]["autoencoder"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, result_df = train_autoencoder(
    # pyrefly: ignore [unknown-name]
    prob_csv=snakemake.input.probs_csv,
    # pyrefly: ignore [unknown-name]
    checkpoint_path=snakemake.output.checkpoint,
    epochs=cfg["epochs"],
    batch_size=cfg["batch_size"],
    lr=cfg["lr"],
    val_ratio=cfg["val_ratio"],
    alpha=cfg["alpha"],
    seed=cfg["seed"],
    device=device,
)

rgb_df = embed_to_rgb(model, result_df, device=device)
# pyrefly: ignore [unknown-name]
rgb_df.to_csv(snakemake.output.rgb_csv, index=False)
