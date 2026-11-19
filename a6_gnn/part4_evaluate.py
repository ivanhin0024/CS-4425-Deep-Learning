"""
Part 4: Model Evaluation - GCN and GAT on the held-out PROTEINS test set

Run:  python part4_evaluate.py   (after part3_train.py)
  - Loads the lowest-validation-loss checkpoints gcn_best.pt and gat_best.pt
  - Predicts on the test set (argmax over class scores)
  - Reports accuracy, precision, recall and F1 (sklearn.metrics)
  - Adds a "Part 4" section to README.md

The test set is used ONLY here, once, for final evaluation. Nothing here
changes model weights, epochs or hyperparameters.
"""
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from part2_preprocessing import load_splits, get_loaders

HIDDEN_CHANNELS = 64     # must match Part 3
HEADS = 4                # must match Part 3
DROPOUT = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ----------------------------------------------------------------------
# Model definitions (identical to Part 3, so the checkpoints load exactly;
# load_state_dict raises an error if anything does not match)
# ----------------------------------------------------------------------
class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=HEADS,
                 dropout=DROPOUT):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
                             heads=1, concat=False)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        return self.lin(x)


# ----------------------------------------------------------------------
# Data: same split as Parts 2 and 3
# ----------------------------------------------------------------------
dataset, train_ds, val_ds, test_ds = load_splits()
_, _, test_loader = get_loaders(train_ds, val_ds, test_ds)


# ----------------------------------------------------------------------
# 1. Load the selected checkpoints
# ----------------------------------------------------------------------
def load_model(model, path):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    print(f"Loaded {path}: epoch {ckpt['epoch']}, val loss {ckpt['val_loss']:.4f}")
    return model, ckpt["epoch"], ckpt["val_loss"]


gcn, gcn_epoch, gcn_val = load_model(
    GCNClassifier(dataset.num_features, HIDDEN_CHANNELS, dataset.num_classes),
    "gcn_best.pt")
gat, gat_epoch, gat_val = load_model(
    GATClassifier(dataset.num_features, HIDDEN_CHANNELS, dataset.num_classes),
    "gat_best.pt")


# ----------------------------------------------------------------------
# 2. Predict on the test set: scores -> class labels with argmax(dim=1)
# ----------------------------------------------------------------------
@torch.no_grad()
def predict(model, loader):
    model.eval()
    y_true, y_pred = [], []
    for data in loader:
        data = data.to(device)
        scores = model(data.x, data.edge_index, data.batch)  # [num_graphs, num_classes]
        preds = scores.argmax(dim=1)                          # index of largest score
        y_true.extend(data.y.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    return y_true, y_pred


# ----------------------------------------------------------------------
# 3. Metrics
#   Binary  = class 1 treated as the positive class (sklearn default)
#   Macro   = unweighted average over both classes (fairer with the
#             ~60/40 class imbalance)
# ----------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    m = {"accuracy": accuracy_score(y_true, y_pred)}
    for avg in ["binary", "macro"]:
        m[f"precision_{avg}"] = precision_score(y_true, y_pred, average=avg, zero_division=0)
        m[f"recall_{avg}"] = recall_score(y_true, y_pred, average=avg, zero_division=0)
        m[f"f1_{avg}"] = f1_score(y_true, y_pred, average=avg, zero_division=0)
    m["cm"] = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return m


results = {}
for name, model in [("GCN", gcn), ("GAT", gat)]:
    y_true, y_pred = predict(model, test_loader)
    results[name] = compute_metrics(y_true, y_pred)

# Majority-class baseline (always predict the most common training label)
majority = int(torch.bincount(train_ds.y).argmax())
majority_acc = accuracy_score(y_true, [majority] * len(y_true))

print(f"\n=== Test-set results ({len(test_ds)} graphs) ===")
print(f"{'Metric':<22}{'GCN':>8}{'GAT':>8}")
rows = [("Accuracy", "accuracy"),
        ("Precision (class 1)", "precision_binary"),
        ("Recall (class 1)", "recall_binary"),
        ("F1-score (class 1)", "f1_binary"),
        ("Precision (macro)", "precision_macro"),
        ("Recall (macro)", "recall_macro"),
        ("F1-score (macro)", "f1_macro")]
for label, key in rows:
    print(f"{label:<22}{results['GCN'][key]:>8.4f}{results['GAT'][key]:>8.4f}")
print(f"\nMajority-class baseline accuracy (always predict class {majority}): "
      f"{majority_acc:.4f}")

for name in ["GCN", "GAT"]:
    cm = results[name]["cm"]
    print(f"\n{name} confusion matrix (rows = true, cols = predicted):")
    print(f"            pred 0  pred 1")
    print(f"  true 0  {cm[0,0]:>8}{cm[0,1]:>8}")
    print(f"  true 1  {cm[1,0]:>8}{cm[1,1]:>8}")


# ----------------------------------------------------------------------
# README.md: add / replace Part 4 section
# ----------------------------------------------------------------------
table = "\n".join(
    f"| {label} | {results['GCN'][key]:.4f} | {results['GAT'][key]:.4f} |"
    for label, key in rows
)


def cm_md(name):
    cm = results[name]["cm"]
    return (f"| **{name}** | Pred 0 | Pred 1 |\n|---|---|---|\n"
            f"| True 0 | {cm[0,0]} | {cm[0,1]} |\n"
            f"| True 1 | {cm[1,0]} | {cm[1,1]} |")


section = f"""## Part 4: Model Evaluation

### 4.1 Selected Checkpoints

Each model was restored from the checkpoint with the **lowest validation loss**
from Part 3. The test set was not used to choose epochs, tune hyperparameters
or update weights; it was used once, here, for final evaluation.

| Model | Selected epoch | Validation loss |
|---|---|---|
| GCN | {gcn_epoch} | {gcn_val:.4f} |
| GAT | {gat_epoch} | {gat_val:.4f} |

### 4.2 Converting Outputs to Class Labels

For each graph, the model outputs one score per class (shape
`[num_graphs, {dataset.num_classes}]`). The predicted label is the index of the
largest score: `scores.argmax(dim=1)`.

### 4.3 Test-Set Metrics ({len(test_ds)} graphs)

Computed with `sklearn.metrics`. "Class 1" rows use sklearn's default binary
setting (class 1 is the positive class). "Macro" rows average the two classes
equally, which is more informative given the ~60/40 class imbalance.

| Metric | GCN | GAT |
|---|---|---|
{table}

Majority-class baseline (always predict class {majority}): accuracy
**{majority_acc:.4f}**.

### Confusion Matrices

{cm_md("GCN")}

{cm_md("GAT")}
"""

try:
    with open("README.md", encoding="utf-8") as f:
        readme = f.read()
except FileNotFoundError:
    readme = "# GNN Graph Classification on PROTEINS\n"
readme = re.sub(r"\n## Part 4: Model Evaluation.*?(?=\n## Part [^4]|\Z)",
                "", readme, flags=re.S).rstrip() + "\n\n"
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme + section)
print("\nREADME.md updated with Part 4.")
