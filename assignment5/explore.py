import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import get_raw_datasets, get_loaders

os.makedirs("figures", exist_ok=True)

# ------------------------------------------------------------ 2. Explore the dataset
train_dataset, test_dataset = get_raw_datasets()

print(f"Training samples: {len(train_dataset)}")
print(f"Testing samples:  {len(test_dataset)}")

image, (label, mask) = train_dataset[0]
print(f"First image size (width, height): {image.size}")
print(f"First image label: {label} ({'Cat' if label == 0 else 'Dog'})")
print(f"First mask size (width, height):  {mask.size}")
print(f"First mask unique values: {np.unique(np.array(mask))}")

# ------------------------------------------------------------ 3. Visualize samples
# Raw masks use values 1, 2, 3, so the color map range is fixed to 1..3 for every mask.
fig, axes = plt.subplots(3, 2, figsize=(7, 10))
for i in range(3):
    img, (lab, msk) = train_dataset[i]
    axes[i, 0].imshow(img)
    axes[i, 0].set_title(f"Image {i} ({'Cat' if lab == 0 else 'Dog'})")
    axes[i, 1].imshow(np.array(msk), cmap="viridis", vmin=1, vmax=3)
    axes[i, 1].set_title("Segmentation mask")
    axes[i, 0].axis("off")
    axes[i, 1].axis("off")
plt.tight_layout()
plt.savefig("figures/samples.png", dpi=150)
plt.close()
print("Saved figures/samples.png")

# ------------------------------------------------------------ 4. Inspect one processed batch
train_loader, test_loader = get_loaders(batch_size=16)
images, (labels, masks) = next(iter(train_loader))

print(f"images shape: {images.shape}")
print(f"labels shape: {labels.shape}")
print(f"masks shape:  {masks.shape}")
