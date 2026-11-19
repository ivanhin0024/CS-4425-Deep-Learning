"""
Part 1: Data Loading and Exploration - PROTEINS (TUDataset)

Run:  python part1_data_exploration.py
Prints all results to the console AND writes them into README.md.
"""
import io
import contextlib

import torch
from torch_geometric.datasets import TUDataset

# Show a compact preview of large tensors (first/last rows) when printing
torch.set_printoptions(threshold=50, edgeitems=5)

# ----------------------------------------------------------------------
# 1. Load the dataset
# ----------------------------------------------------------------------
dataset = TUDataset(root="data/TUDataset", name="PROTEINS")
data = dataset[0]


def section(fn):
    """Run fn, echo its printed output to console, and return it as text."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn()
    text = buf.getvalue().rstrip()
    print(text + "\n")
    return text


# ----------------------------------------------------------------------
# 2A. Basic dataset information
# ----------------------------------------------------------------------
def dataset_info():
    print(f"Dataset: {dataset}")
    print(f"Number of graphs: {len(dataset)}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"Number of node features: {dataset.num_node_features}")


# ----------------------------------------------------------------------
# 2B. Inspect the first graph, dataset[0]
# ----------------------------------------------------------------------
def first_graph():
    print(data)


def node_features():          # i. node feature matrix x
    print(data.x)
    print(f"x.shape: {data.x.shape}")


def edge_index():             # ii. edge index matrix
    print(data.edge_index)
    print(f"edge_index.shape: {data.edge_index.shape}")


def graph_label():            # iii. graph label y
    print(f"y: {data.y}")


# ----------------------------------------------------------------------
# 3. Average number of nodes per graph
# ----------------------------------------------------------------------
avg_nodes = sum(g.num_nodes for g in dataset) / len(dataset)


def avg_nodes_info():
    print(f"Average number of nodes per graph: {avg_nodes:.2f}")


print("=== 2A. Dataset Information ===")
out_info = section(dataset_info)
print("=== 2B. First Graph: dataset[0] ===")
out_graph = section(first_graph)
print("--- i. Node feature matrix x ---")
out_x = section(node_features)
print("--- ii. Edge index matrix edge_index ---")
out_edges = section(edge_index)
print("--- iii. Graph label y ---")
out_y = section(graph_label)
print("=== 3. Average Nodes per Graph ===")
out_avg = section(avg_nodes_info)

# ----------------------------------------------------------------------
# Write README.md
# ----------------------------------------------------------------------
readme = f"""# GNN Graph Classification on PROTEINS

## Part 1: Data Loading and Exploration

The `PROTEINS` dataset was loaded with `TUDataset` from PyTorch Geometric.
Each graph is a protein: nodes are secondary structure elements, edges connect
elements that are structurally or sequentially close, and each graph is
labeled as an enzyme or a non-enzyme.

### 2A. Dataset Information

```
{out_info}
```

| Property | Value |
|---|---|
| Total number of graphs | {len(dataset)} |
| Number of classes | {dataset.num_classes} |
| Number of node features | {dataset.num_node_features} |

### 2B. Inspecting the First Graph (`dataset[0]`)

```
{out_graph}
```

#### i. Node feature matrix `x`

```
{out_x}
```

`x` has shape **{tuple(data.x.shape)}**: {data.x.shape[0]} nodes, each with
{data.x.shape[1]} features.

#### ii. Edge index matrix `edge_index`

```
{out_edges}
```

`edge_index` has shape **{tuple(data.edge_index.shape)}**: row 0 holds source
nodes and row 1 holds target nodes, for {data.edge_index.shape[1]} directed
edges (each undirected edge is stored in both directions, so
{data.edge_index.shape[1] // 2} undirected edges).

#### iii. Graph label `y`

```
{out_y}
```

### 3. Average Number of Nodes per Graph

```
{out_avg}
```

On average, a protein graph in this dataset has **{avg_nodes:.2f} nodes**.
"""

with open("README.md", "w", encoding="utf-8") as f:
    f.write(readme)

print("README.md written.")
