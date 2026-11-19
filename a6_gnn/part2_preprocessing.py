"""
Part 2: Data Preprocessing - Train / Validation / Test split (70 / 10 / 20)

Run:  python part2_preprocessing.py
Prints the split sizes and appends a "Part 2" section to README.md.

Later parts (GCN and GAT) should import from this file so both models use
the SAME split:
    from part2_preprocessing import load_splits, get_loaders
"""
import re

import torch
from sklearn.model_selection import train_test_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

RANDOM_STATE = 0
TEST_SIZE = 0.20          # 20% of the full dataset
VAL_SIZE = 0.125          # 12.5% of the remaining 80% = 10% of the full dataset
BATCH_SIZE = 64


def load_splits(root="data/TUDataset"):
    """Load PROTEINS and split it at the graph level.

    Returns (dataset, train_dataset, val_dataset, test_dataset).
    Splitting graph indices guarantees no graph appears in more than one set.
    Stratifying on the label keeps the enzyme / non-enzyme ratio the same
    in all three sets.
    """
    dataset = TUDataset(root=root, name="PROTEINS")
    indices = list(range(len(dataset)))
    labels = dataset.y.tolist()

    # Step 1: reserve 20% of the graphs for testing
    train_val_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )

    # Step 2: take 12.5% of the remaining 80% for validation (10% overall)
    train_val_labels = [labels[i] for i in train_val_idx]
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=VAL_SIZE, random_state=RANDOM_STATE,
        stratify=train_val_labels,
    )

    # Sanity check: the three sets are disjoint and cover every graph
    assert not (set(train_idx) & set(val_idx))
    assert not (set(train_idx) & set(test_idx))
    assert not (set(val_idx) & set(test_idx))
    assert len(train_idx) + len(val_idx) + len(test_idx) == len(dataset)

    return (dataset,
            dataset[torch.tensor(train_idx)],
            dataset[torch.tensor(val_idx)],
            dataset[torch.tensor(test_idx)])


def get_loaders(train_dataset, val_dataset, test_dataset, batch_size=BATCH_SIZE):
    """Mini-batch loaders. Only the training loader is shuffled.

    Usage:
      train_loader -> fit model parameters
      val_loader   -> select the best checkpoint (e.g. best val accuracy)
      test_loader  -> final evaluation only, once, after training
    """
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
            DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
            DataLoader(test_dataset, batch_size=batch_size, shuffle=False))


def class_counts(ds):
    counts = torch.bincount(ds.y, minlength=2).tolist()
    return counts[0], counts[1]


if __name__ == "__main__":
    dataset, train_ds, val_ds, test_ds = load_splits()
    n = len(dataset)

    rows = []
    print(f"Random seed (random_state): {RANDOM_STATE}")
    print(f"{'Split':<12}{'Graphs':>8}{'% of total':>12}{'Class 0':>9}{'Class 1':>9}")
    for name, ds in [("Training", train_ds), ("Validation", val_ds), ("Test", test_ds)]:
        c0, c1 = class_counts(ds)
        pct = 100 * len(ds) / n
        rows.append((name, len(ds), pct, c0, c1))
        print(f"{name:<12}{len(ds):>8}{pct:>11.1f}%{c0:>9}{c1:>9}")
    print(f"{'Total':<12}{n:>8}{100.0:>11.1f}%")

    table = "\n".join(
        f"| {name} | {cnt} | {pct:.1f}% | {c0} | {c1} |"
        for name, cnt, pct, c0, c1 in rows
    )
    section = f"""## Part 2: Data Preprocessing

The 1,113 PROTEINS graphs were split **at the graph level** (by graph index),
so no graph appears in more than one set. The split was made in two steps with
scikit-learn's `train_test_split`:

1. Reserve 20% of all graphs for the **test** set.
2. From the remaining 80%, take 12.5% for the **validation** set (10% of the full dataset).

Both steps are stratified by class label so every set keeps the same
enzyme / non-enzyme ratio.

**Random seed:** `random_state = {RANDOM_STATE}`

| Split | Number of graphs | % of total | Class 0 | Class 1 |
|---|---|---|---|---|
{table}
| **Total** | **{n}** | **100%** | | |

The same split is used for both the GCN and the GAT models
(both import `load_splits()` from `part2_preprocessing.py`).

- **Training set:** used to fit the model parameters.
- **Validation set:** used to select each model's best checkpoint.
- **Test set:** used only once, for the final evaluation.
"""

    # Add (or replace) the Part 2 section in README.md
    try:
        with open("README.md", encoding="utf-8") as f:
            readme = f.read()
    except FileNotFoundError:
        readme = "# GNN Graph Classification on PROTEINS\n"
    readme = re.sub(r"\n## Part 2: Data Preprocessing.*?(?=\n## |\Z)", "",
                    readme, flags=re.S).rstrip() + "\n\n"
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme + section)
    print("\nREADME.md updated with Part 2.")
