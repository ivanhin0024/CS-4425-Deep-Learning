import json
import time

import torch
import torch.nn as nn
from torchvision.models import resnet18

from data import get_loaders

EPOCHS = 100         # same as the CNN so the classifiers have comparable budgets
BATCH_SIZE = 16
LR = 0.001
SEED = 0


def make_resnet():
    model = resnet18(weights=None)                          # random init, no pretrained weights
    model.fc = nn.Linear(model.fc.in_features, 2)           # 2 output scores (cat, dog)
    return model                                            # all parameters stay trainable


def main():
    torch.manual_seed(SEED)
    print(f"Random seed: {SEED}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)

    # fc is replaced inside make_resnet(), i.e. before the optimizer is created
    model = make_resnet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # shape check: should be [B, 2]
    images, (labels, masks) = next(iter(train_loader))
    with torch.no_grad():
        print("ResNet-18 output shape:", model(images.to(device)).shape)

    # train (common training procedure), labels are the targets
    model.train()
    train_losses = []
    for epoch in range(EPOCHS):
        start = time.time()
        running_loss = 0.0
        n_seen = 0
        for images, (labels, masks) in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)

        epoch_loss = running_loss / n_seen
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS} - average training loss: {epoch_loss:.4f} "
              f"({time.time() - start:.1f}s)")

    torch.save(model.state_dict(), "resnet.pt")
    json.dump(train_losses, open("resnet_losses.json", "w"))
    print("Saved resnet.pt and resnet_losses.json")


if __name__ == "__main__":
    main()
