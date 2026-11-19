import json

import torch
import torch.nn as nn

from data import get_loaders

EPOCHS = 100          # raise later if training is fast enough
BATCH_SIZE = 16
LR = 0.001
SEED = 0


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # [B,16,64,64]
        x = self.pool(self.relu(self.conv2(x)))   # [B,32,32,32]
        x = self.relu(self.conv3(x))              # [B,64,32,32]
        x = self.gap(x)                           # [B,64,1,1]
        x = torch.flatten(x, start_dim=1)         # [B,64]
        return self.fc(x)                         # [B,2] raw scores (no softmax)


def main():
    torch.manual_seed(SEED)
    print(f"Random seed: {SEED}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)

    # 2. model, loss, optimizer
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # shape check: should be [B, 2]
    images, (labels, masks) = next(iter(train_loader))
    with torch.no_grad():
        print("CNN output shape:", model(images.to(device)).shape)

    # 3. train (common training procedure), labels are the targets
    model.train()
    train_losses = []
    for epoch in range(EPOCHS):
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
        print(f"Epoch {epoch + 1}/{EPOCHS} - average training loss: {epoch_loss:.4f}")

    # save so evaluation later doesn't require retraining
    torch.save(model.state_dict(), "cnn.pt")
    json.dump(train_losses, open("cnn_losses.json", "w"))
    print("Saved cnn.pt and cnn_losses.json")


if __name__ == "__main__":
    main()
