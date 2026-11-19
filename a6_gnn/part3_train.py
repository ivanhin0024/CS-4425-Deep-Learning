"""
Part 3: Model Building and Training - GCN and GAT on PROTEINS

Run:  python part3_train.py
  - Inspects one mini-batch (Part 3.1)
  - Trains GCNClassifier (Part 3.2) and GATClassifier (Part 3.3)
  - Prints train/val loss every 10 epochs
  - Saves the lowest-validation-loss checkpoint of each model:
        gcn_best.pt, gat_best.pt
  - Saves per-epoch loss history (loss_history.json) and loss curves
    (loss_curves.png)
  - Adds a "Part 3" section to README.md

The test set is NOT used here. It is only used in Part 4 for final evaluation.
"""
import copy
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool

from part2_preprocessing import load_splits, get_loaders, RANDOM_STATE, BATCH_SIZE

# ----------------------------------------------------------------------
# Hyperparameters
# ----------------------------------------------------------------------
HIDDEN_CHANNELS = 64
HEADS = 4
DROPOUT = 0.5
LEARNING_RATE = 0.01
EPOCHS = 100
SEED = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}"
      + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))


# ======================================================================
# 3.1 Before Building the Models
# ======================================================================
# A. Prepare mini-batches (same split as Part 2, shared by GCN and GAT)
dataset, train_ds, val_ds, test_ds = load_splits()
train_loader, val_loader, test_loader = get_loaders(train_ds, val_ds, test_ds)

# B. Inspect one mini-batch to understand the model inputs
sample_batch = next(iter(train_loader))
print("=== 3.1 Mini-batch inspection ===")
print(f"Batches per epoch -> train: {len(train_loader)}, "
      f"val: {len(val_loader)}, test: {len(test_loader)}")
print(f"Sample batch: {sample_batch}")
print(f"Graphs in batch: {sample_batch.num_graphs}")
print(f"x shape          [num_nodes, num_features]: {tuple(sample_batch.x.shape)}")
print(f"edge_index shape [2, num_edges]:            {tuple(sample_batch.edge_index.shape)}")
print(f"batch shape      [num_nodes]:               {tuple(sample_batch.batch.shape)}")
print(f"y shape          [num_graphs]:              {tuple(sample_batch.y.shape)}")
print(f"batch vector (first 10): {sample_batch.batch[:10].tolist()}")
print(f"in_channels = dataset.num_features = {dataset.num_features}")
print(f"num_classes = dataset.num_classes  = {dataset.num_classes}\n")


# ======================================================================
# 3.2 A. Model 1: Graph Convolutional Network
# ======================================================================
class GCNClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=DROPOUT):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))                     # conv 1 + ReLU
        x = F.dropout(x, p=self.dropout, training=self.training)  # dropout
        x = F.relu(self.conv2(x, edge_index))                     # conv 2 + ReLU
        x = global_mean_pool(x, batch)                            # [num_graphs, hidden]
        return self.lin(x)                                        # [num_graphs, num_classes]


# ======================================================================
# 3.3 A. Model 2: Graph Attention Network
# ======================================================================
class GATClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, heads=HEADS,
                 dropout=DROPOUT):
        super().__init__()
        # heads outputs are concatenated -> hidden_channels * heads features
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels,
                             heads=1, concat=False)
        self.lin = nn.Linear(hidden_channels, num_classes)
        self.dropout = dropout

    def forward(self, x, edge_index, batch):
        x = F.elu(self.conv1(x, edge_index))                      # attn 1 + ELU
        x = F.dropout(x, p=self.dropout, training=self.training)  # dropout
        x = F.elu(self.conv2(x, edge_index))                      # attn 2 + ELU
        x = global_mean_pool(x, batch)                            # [num_graphs, hidden]
        return self.lin(x)                                        # [num_graphs, num_classes]


# ======================================================================
# Shared training / validation code (3.2 C and 3.3 C)
# ======================================================================
def train_one_epoch(model, loader, criterion, optimizer):
    """One pass over the training set. Returns average loss per graph."""
    model.train()
    total_loss, total_graphs = 0.0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)   # forward pass
        loss = criterion(out, data.y)                      # loss vs graph labels
        loss.backward()                                    # backward pass
        optimizer.step()                                   # update parameters
        total_loss += loss.item() * data.num_graphs
        total_graphs += data.num_graphs
    return total_loss / total_graphs


@torch.no_grad()
def evaluate_loss(model, loader, criterion):
    """Average loss per graph, eval mode, no gradients, no updates."""
    model.eval()
    total_loss, total_graphs = 0.0, 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        total_loss += criterion(out, data.y).item() * data.num_graphs
        total_graphs += data.num_graphs
    return total_loss / total_graphs


def fit(name, model, epochs=EPOCHS, lr=LEARNING_RATE):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val, best_epoch, best_state = float("inf"), 0, None

    print(f"=== Training {name} ===")
    print(model)
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate_loss(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Keep the checkpoint with the lowest validation loss
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 10 == 0:
            print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f}")

    ckpt_path = f"{name.lower()}_best.pt"
    torch.save({"model_state_dict": best_state, "epoch": best_epoch,
                "val_loss": best_val}, ckpt_path)
    print(f"Best {name} checkpoint: epoch {best_epoch}, "
          f"val loss {best_val:.4f} -> saved to {ckpt_path}\n")
    return history, best_epoch, best_val, model


# ======================================================================
# 3.2 B / 3.3 B. Instantiate and train
# ======================================================================
torch.manual_seed(SEED)
gcn = GCNClassifier(in_channels=dataset.num_features,
                    hidden_channels=HIDDEN_CHANNELS,
                    num_classes=dataset.num_classes)
gcn_hist, gcn_best_epoch, gcn_best_val, gcn = fit("GCN", gcn)

torch.manual_seed(SEED)
gat = GATClassifier(in_channels=dataset.num_features,
                    hidden_channels=HIDDEN_CHANNELS,
                    num_classes=dataset.num_classes,
                    heads=HEADS)
gat_hist, gat_best_epoch, gat_best_val, gat = fit("GAT", gat)

# Save loss histories (for plotting / Part 4)
with open("loss_history.json", "w") as f:
    json.dump({"GCN": {**gcn_hist, "best_epoch": gcn_best_epoch},
               "GAT": {**gat_hist, "best_epoch": gat_best_epoch}}, f, indent=2)

# Plot training vs validation loss curves
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, name, hist, best_ep in [(axes[0], "GCN", gcn_hist, gcn_best_epoch),
                                (axes[1], "GAT", gat_hist, gat_best_epoch)]:
    epochs_range = range(1, len(hist["train_loss"]) + 1)
    ax.plot(epochs_range, hist["train_loss"], label="Training loss")
    ax.plot(epochs_range, hist["val_loss"], label="Validation loss")
    ax.axvline(best_ep, color="gray", linestyle="--",
               label=f"Best val epoch ({best_ep})")
    ax.set_title(f"{name}: Training vs Validation Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("loss_curves.png", dpi=150)
print("Saved loss_curves.png and loss_history.json")


# ======================================================================
# README.md: add / replace Part 3 section
# ======================================================================
def every_10(hist):
    return "\n".join(
        f"| {e} | {hist['train_loss'][e-1]:.4f} | {hist['val_loss'][e-1]:.4f} |"
        for e in range(10, len(hist["train_loss"]) + 1, 10)
    )


n_params = lambda m: sum(p.numel() for p in m.parameters())

section = f"""## Part 3: Model Building and Training

### 3.1 Mini-Batches and Model Inputs

PyTorch Geometric's `DataLoader` creates mini-batches of {BATCH_SIZE} graphs
for each split (training shuffled; validation and test not shuffled), using
the Part 2 split (`random_state = {RANDOM_STATE}`).

| Loader | Graphs | Batches |
|---|---|---|
| Training | {len(train_ds)} | {len(train_loader)} |
| Validation | {len(val_ds)} | {len(val_loader)} |
| Test | {len(test_ds)} | {len(test_loader)} |

Example training mini-batch:

```
{sample_batch}
x shape:          {tuple(sample_batch.x.shape)}   -> [num_nodes, num_features]
edge_index shape: {tuple(sample_batch.edge_index.shape)} -> [2, num_edges]
batch shape:      {tuple(sample_batch.batch.shape)}     -> graph id of each node
y shape:          {tuple(sample_batch.y.shape)}       -> one label per graph
```

The many small graphs in a batch are combined into one large disconnected
graph. `batch` records which graph each node came from, so
`global_mean_pool(x, batch)` can average node embeddings back into one vector
per graph. `in_channels = dataset.num_features = {dataset.num_features}` and
`num_classes = dataset.num_classes = {dataset.num_classes}`.

### Shared Training Setup

| Setting | Value |
|---|---|
| Loss | `nn.CrossEntropyLoss` |
| Optimizer | `optim.Adam`, learning rate {LEARNING_RATE} |
| Epochs | {EPOCHS} |
| Batch size | {BATCH_SIZE} |
| Hidden channels | {HIDDEN_CHANNELS} |
| Dropout | {DROPOUT} |
| Random seed | {SEED} (`torch.manual_seed`) |
| Checkpoint selection | Lowest validation loss |

Training data updates the parameters. After each epoch the model is switched
to evaluation mode and the average validation loss is computed with no
gradients. The checkpoint with the lowest validation loss is kept for Part 4.
The test set is not used in this part.

### 3.2 Model 1: GCN (`GCNClassifier`)

`GCNConv({dataset.num_features}, {HIDDEN_CHANNELS})` → ReLU → Dropout →
`GCNConv({HIDDEN_CHANNELS}, {HIDDEN_CHANNELS})` → ReLU → `global_mean_pool` →
`Linear({HIDDEN_CHANNELS}, {dataset.num_classes})`

Trainable parameters: {n_params(gcn):,}

| Epoch | Train Loss | Val Loss |
|---|---|---|
{every_10(gcn_hist)}

**Selected checkpoint:** epoch **{gcn_best_epoch}**, validation loss
**{gcn_best_val:.4f}** (saved to `gcn_best.pt`).

### 3.3 Model 2: GAT (`GATClassifier`)

`GATConv({dataset.num_features}, {HIDDEN_CHANNELS}, heads={HEADS})` → ELU →
Dropout → `GATConv({HIDDEN_CHANNELS * HEADS}, {HIDDEN_CHANNELS}, heads=1,
concat=False)` → ELU → `global_mean_pool` →
`Linear({HIDDEN_CHANNELS}, {dataset.num_classes})`

The {HEADS} attention heads in the first layer are concatenated, so the second
layer takes {HIDDEN_CHANNELS} × {HEADS} = {HIDDEN_CHANNELS * HEADS} input features.

Trainable parameters: {n_params(gat):,}

| Epoch | Train Loss | Val Loss |
|---|---|---|
{every_10(gat_hist)}

**Selected checkpoint:** epoch **{gat_best_epoch}**, validation loss
**{gat_best_val:.4f}** (saved to `gat_best.pt`).

### Loss Curves

![Training and validation loss curves](loss_curves.png)
"""

try:
    with open("README.md", encoding="utf-8") as f:
        readme = f.read()
except FileNotFoundError:
    readme = "# GNN Graph Classification on PROTEINS\n"
readme = re.sub(r"\n## Part 3: Model Building and Training.*?(?=\n## Part [^3]|\Z)",
                "", readme, flags=re.S).rstrip() + "\n\n"
with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme + section)
print("README.md updated with Part 3.")
