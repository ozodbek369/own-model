from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import OWMImageDataset


# ============================================================
# 1. Configuration
# ============================================================

IMAGE_SIZE = 256
BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 20

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 2. Device
# ============================================================

device = torch.device("cpu")

print("Device:", device)


# ============================================================
# 3. Dataset
# ============================================================

dataset = OWMImageDataset(
    PROJECT_ROOT / "data" / "raw",
    IMAGE_SIZE
)


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


print("Number of images:", len(dataset))
print("Batch size:", BATCH_SIZE)


# ============================================================
# 4. Autoencoder
# ============================================================

class OWMAutoencoder(nn.Module):

    def __init__(self):
        super().__init__()

        # ----------------------------
        # Encoder
        # ----------------------------

        self.encoder = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU()
        )


        # ----------------------------
        # Decoder
        # ----------------------------

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=64,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=64,
                out_channels=32,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.ReLU(),

            nn.ConvTranspose2d(
                in_channels=32,
                out_channels=3,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            nn.Sigmoid()
        )


    # ----------------------------
    # Forward pass
    # ----------------------------

    def forward(self, x):

        latent = self.encoder(x)

        reconstructed = self.decoder(latent)

        return reconstructed


# ============================================================
# 5. Create model
# ============================================================

model = OWMAutoencoder().to(device)


# ============================================================
# 6. Loss function
# ============================================================

loss_function = nn.MSELoss()


# ============================================================
# 7. Optimizer
# ============================================================

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# 8. Training
# ============================================================

print()
print("Starting training...")
print("----------------------------")


for epoch in range(EPOCHS):

    epoch_loss = 0.0

    for batch in dataloader:

        batch = batch.to(device)

        # Forward pass
        reconstructed = model(batch)

        # Reconstruction loss
        loss = loss_function(
            reconstructed,
            batch
        )

        # Clear gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        epoch_loss += loss.item()

    average_loss = epoch_loss / len(dataloader)

    print(
        f"Epoch: {epoch + 1:2d}/{EPOCHS} | "
        f"Loss: {average_loss:.6f}"
    )


print()
print("Training finished.")
print("----------------------------")
