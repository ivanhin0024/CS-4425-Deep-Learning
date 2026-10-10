import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score

from data import get_loaders
from train_cnn import SimpleCNN
from train_resnet import make_resnet
from train_unet import SmallUNet

BATCH_SIZE = 16
SEED = 0


def evaluate_classifier(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, (labels, masks) in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
    return accuracy_score(all_labels, all_preds), all_labels


def evaluate_unet(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, (labels, masks) in test_loader:
            images = images.to(device)
            outputs = model(images)
            predicted_masks = outputs.argmax(dim=1)      # [B,128,128]
            masks = masks.to(device)
            correct += (predicted_masks == masks).sum().item()
            total += masks.numel()
    return correct / total


def main():
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs("figures", exist_ok=True)

    _, test_loader = get_loaders(batch_size=BATCH_SIZE)

    # load the trained models
    cnn = SimpleCNN().to(device)
    cnn.load_state_dict(torch.load("cnn.pt", map_location=device))
    resnet = make_resnet().to(device)
    resnet.load_state_dict(torch.load("resnet.pt", map_location=device))
    unet = SmallUNet().to(device)
    unet.load_state_dict(torch.load("unet.pt", map_location=device))

    # ---------------- Part 3: evaluation ----------------
    cnn_acc, test_labels = evaluate_classifier(cnn, test_loader, device)
    resnet_acc, _ = evaluate_classifier(resnet, test_loader, device)
    baseline = sum(1 for l in test_labels if l == 1) / len(test_labels)
    unet_pixel_acc = evaluate_unet(unet, test_loader, device)

    print("\n| Model | Test accuracy |")
    print("|---|---|")
    print(f"| Regular CNN | {cnn_acc:.4f} |")
    print(f"| ResNet-18 | {resnet_acc:.4f} |")
    print(f"| Always-Dog baseline | {baseline:.4f} |")
    print(f"\nU-Net pixel accuracy: {unet_pixel_acc:.4f}")
    print(f"Test images: {len(test_labels)}")

    json.dump(
        {"cnn_acc": cnn_acc, "resnet_acc": resnet_acc,
         "baseline": baseline, "unet_pixel_acc": unet_pixel_acc},
        open("results.json", "w"), indent=2,
    )

    # ---------------- Part 4.2: training-loss plots ----------------
    cnn_losses = json.load(open("cnn_losses.json"))
    resnet_losses = json.load(open("resnet_losses.json"))
    unet_losses = json.load(open("unet_losses.json"))

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cnn_losses) + 1), cnn_losses, label="Regular CNN")
    plt.plot(range(1, len(resnet_losses) + 1), resnet_losses, label="ResNet-18")
    plt.xlabel("Epoch")
    plt.ylabel("Average training loss")
    plt.title("Classification training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/loss_classifiers.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(unet_losses) + 1), unet_losses, label="U-Net", color="tab:green")
    plt.xlabel("Epoch")
    plt.ylabel("Average training loss")
    plt.title("U-Net segmentation training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/loss_unet.png", dpi=150)
    plt.close()

    # ---------------- Part 4.3: segmentation results ----------------
    # the test loader preserves order, so the first batch holds the first test images
    images, (labels, masks) = next(iter(test_loader))
    unet.eval()
    with torch.no_grad():
        outputs = unet(images.to(device))
        predicted_masks = outputs.argmax(dim=1)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i in range(3):
        axes[i, 0].imshow(images[i].cpu().permute(1, 2, 0).numpy())
        axes[i, 0].set_title("Original image")
        axes[i, 1].imshow(masks[i].cpu().numpy(), cmap="viridis", vmin=0, vmax=2)
        axes[i, 1].set_title("Ground truth")
        axes[i, 2].imshow(predicted_masks[i].cpu().numpy(), cmap="viridis", vmin=0, vmax=2)
        axes[i, 2].set_title("U-Net prediction")
        for ax in axes[i]:
            ax.axis("off")
    plt.suptitle("Masks: 0 = pet, 1 = background, 2 = boundary")
    plt.tight_layout()
    plt.savefig("figures/unet_segmentation.png", dpi=150)
    plt.close()

    print("\nSaved figures/loss_classifiers.png, figures/loss_unet.png, "
          "figures/unet_segmentation.png, results.json")


if __name__ == "__main__":
    main()
