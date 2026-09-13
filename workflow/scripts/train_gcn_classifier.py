from pathlib import Path

import torch

from morphe.embeddings.gcn_classifier import predict_probabilities, run_training

# pyrefly: ignore [unknown-name]
cfg = snakemake.config["embeddings"]["gcn"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, df, feature_cols = run_training(
    dataset_csv=cfg["dataset_csv"],
    drop_cols=cfg["drop_cols"],
    hidden_channels=cfg["hidden_channels"],
    epochs=cfg["epochs"],
    lr=cfg["lr"],
    weight_decay=cfg["weight_decay"],
    k_neighbors=cfg["k_neighbors"],
    device=device,
)

result_df = predict_probabilities(model, df, feature_cols, device, k_neighbors=cfg["k_neighbors"])
# pyrefly: ignore [unknown-name]
Path(snakemake.output.probs_csv).parent.mkdir(parents=True, exist_ok=True)
# pyrefly: ignore [unknown-name]
result_df.to_csv(snakemake.output.probs_csv, index=False)
