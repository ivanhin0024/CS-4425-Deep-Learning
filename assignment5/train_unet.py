import json
import time

import torch
import torch.nn as nn

from data import get_loaders

EPOCHS = 100
BATCH_SIZE = 16
LR = 0.001
SEED = 0


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder
        self.enc1 = DoubleConv(3, 16)
        self.enc2 = DoubleConv(16, 32)
        self.pool = nn.MaxPool2d(2)
        # bottleneck
        self.bottleneck = DoubleConv(32, 64)
        # decoder
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(32, 16)
        # output: 3 scores per pixel (pet, background, boundary)
        self.out = nn.Conv2d(16, 3, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)                      # E1: [B,16,128,128] (saved before downsampling)
        x = self.pool(e1)                      # [B,16,64,64]
        e2 = self.enc2(x)                      # E2: [B,32,64,64]   (saved before downsampling)
        x = self.pool(e2)                      # [B,32,32,32]
        x = self.bottleneck(x)                 # [B,64,32,32]

        x = self.up1(x)                        # [B,32,64,64]
        x = torch.cat([x, e2], dim=1)          # skip connection 1 -> [B,64,64,64]
        x = self.dec1(x)                       # [B,32,64,64]

        x = self.up2(x)                        # [B,16,128,128]
        x = torch.cat([x, e1], dim=1)          # skip connection 2 -> [B,32,128,128]
        x = self.dec2(x)                       # [B,16,128,128]

        return self.out(x)                     # [B,3,128,128] raw scores (no softmax)


def main():
    torch.manual_seed(SEED)
    print(f"Random seed: {SEED}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = get_loaders(batch_size=BATCH_SIZE)

    model = SmallUNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # shape check: should be [B, 3, 128, 128]
    images, (labels, masks) = next(iter(train_loader))
    with torch.no_grad():
        print("U-Net output shape:", model(images.to(device)).shape)

    # train (common training procedure), masks are the targets
    model.train()
    train_losses = []
    for epoch in range(EPOCHS):
        start = time.time()
        running_loss = 0.0
        n_seen = 0
        for images, (labels, masks) in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)            # [B,3,128,128]
            loss = criterion(outputs, masks)   # masks: [B,128,128]
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            n_seen += images.size(0)

        epoch_loss = running_loss / n_seen
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS} - average training loss: {epoch_loss:.4f} "
              f"({time.time() - start:.1f}s)")

    torch.save(model.state_dict(), "unet.pt")
    json.dump(train_losses, open("unet_losses.json", "w"))
    print("Saved unet.pt and unet_losses.json")


if __name__ == "__main__":
    main()
