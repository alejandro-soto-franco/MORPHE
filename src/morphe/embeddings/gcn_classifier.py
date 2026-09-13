"""Graph-convolutional cell-type classifier.

Ported from ``Embeddings/01_GCN_Classifier.ipynb``. Builds a k-nearest-neighbour
spatial graph per tissue region and trains an APPNP-propagated MLP to predict
cell type from marker expression, producing per-cell class probabilities.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import APPNP, GraphNorm


class RegionGraphDataset(Dataset):
    """One graph per tissue region, edges from a k-nearest-neighbour spatial graph."""

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str],
        label_col: str = "Cell Type",
        region_col: str = "unique_region",
        pos_cols: tuple[str, str] = ("x", "y"),
        k_neighbors: int = 20,
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.region_col = region_col
        self.pos_cols = list(pos_cols)
        self.k = k_neighbors

        self.label_encoder = LabelEncoder()
        # pyrefly: ignore [unsupported-operation]
        self.df[label_col] = self.label_encoder.fit_transform(self.df[label_col])
        self.label_col = label_col

        self.region_ids = self.df[self.region_col].unique()
        self.data_list: list[Data] = []

        for region_id in self.region_ids:
            region_df = self.df[self.df[self.region_col] == region_id]

            features = region_df[self.feature_cols].values
            labels = region_df[self.label_col].values
            positions = region_df[self.pos_cols].values

            num_nodes = features.shape[0]

            if num_nodes <= 1:
                edge_index = torch.empty((2, 0), dtype=torch.long)
            else:
                knn = NearestNeighbors(n_neighbors=min(self.k + 1, num_nodes), algorithm="ball_tree")
                knn.fit(positions)
                _, indices = knn.kneighbors(positions)

                edge_list = [[i, nbr] for i, nbrs in enumerate(indices) for nbr in nbrs if nbr != i]
                edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1).unique(dim=1)

            x = torch.tensor(features, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)

            data = Data(x=x, edge_index=edge_index, y=y)
            data.region_id = region_id
            self.data_list.append(data)

    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


class GCNClassifier(nn.Module):
    """MLP encoder followed by APPNP propagation and a linear classification head."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        dropout: float = 0.1,
        alpha: float = 0.9,
        K: int = 20,
    ):
        super().__init__()

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.norm1 = GraphNorm(hidden_channels)

        self.lin2 = nn.Linear(hidden_channels, hidden_channels)
        self.norm2 = GraphNorm(hidden_channels)

        self.propagation = APPNP(K=K, alpha=alpha, dropout=dropout)

        self.dropout = dropout
        self.out_lin = nn.Linear(hidden_channels, num_classes)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.lin1(x)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.lin2(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.propagation(x, edge_index)
        return self.out_lin(x)


def train_one_epoch(
    model: GCNClassifier, loader: DataLoader, optimizer: optim.Optimizer, device: torch.device
) -> float:
    model.train()
    total_loss = 0.0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model: GCNClassifier, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct, total = 0, 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.y.size(0)
    return correct / total


@torch.no_grad()
def predict_probabilities(
    model: GCNClassifier,
    df: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
    k_neighbors: int = 20,
) -> pd.DataFrame:
    """Run inference and return a dataframe of per-class probabilities plus x/y/region."""
    dataset = RegionGraphDataset(df=df, feature_cols=feature_cols, k_neighbors=k_neighbors)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model.eval()
    all_probs = []
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        probs = torch.softmax(out, dim=1).cpu().numpy()
        all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    result_df = pd.DataFrame(all_probs, columns=[f"prob_class{i}" for i in range(all_probs.shape[1])])
    result_df["x"] = df["x"].values
    result_df["y"] = df["y"].values
    result_df["region"] = df["unique_region"].values
    cols = ["x", "y", "region"] + [f"prob_class{i}" for i in range(all_probs.shape[1])]
    return result_df[cols]


def load_feature_columns(df: pd.DataFrame, drop_cols: list[str]) -> list[str]:
    """Feature columns are every column not in ``drop_cols`` (dataset-specific metadata)."""
    return [c for c in df.columns if c not in drop_cols]


def run_training(
    dataset_csv: str | Path,
    drop_cols: list[str],
    hidden_channels: int = 768,
    epochs: int = 40,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    k_neighbors: int = 20,
    device: torch.device | str = "cpu",
) -> tuple[GCNClassifier, pd.DataFrame, list[str]]:
    """Train the GCN classifier end to end and return the model, source dataframe and feature columns."""
    df = pd.read_csv(dataset_csv, index_col=0)
    feature_cols = load_feature_columns(df, drop_cols)

    dataset = RegionGraphDataset(df=df, feature_cols=feature_cols, k_neighbors=k_neighbors)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model = GCNClassifier(
        in_channels=len(feature_cols),
        hidden_channels=hidden_channels,
        num_classes=len(np.unique(df["Cell Type"])),
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
        # pyrefly: ignore [bad-argument-type]
        loss = train_one_epoch(model, loader, optimizer, device)
        # pyrefly: ignore [bad-argument-type]
        acc = evaluate(model, loader, device)
        print(f"epoch {epoch}, loss {loss:.4f}, train acc {acc:.4f}")

    return model, df, feature_cols
