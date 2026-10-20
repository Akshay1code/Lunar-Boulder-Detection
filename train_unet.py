from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import LunarLandslideDataset
from unet_model import UNet


# -----------------------------
# Configuration
# -----------------------------

IMAGE_SIZE = 256
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-3

DEVICE = torch.device("cpu")

OUTPUT_DIR = Path("checkpoints")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = OUTPUT_DIR / "unet_best.pth"


def main():
    print("=" * 50)
    print("LUNAR LANDSLIDE SEGMENTATION - U-NET")
    print("=" * 50)

    print(f"Device: {DEVICE}")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")

    # Load training and validation data only.
    train_dataset = LunarLandslideDataset(
        split="train",
        image_size=IMAGE_SIZE,
    )

    valid_dataset = LunarLandslideDataset(
        split="valid",
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(valid_dataset)}")

    # Initialize model.
    model = UNet().to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_valid_loss = float("inf")

    # -----------------------------
    # Training loop
    # -----------------------------

    for epoch in range(EPOCHS):
        model.train()

        train_loss_sum = 0.0
        train_samples = 0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * images.size(0)
            train_samples += images.size(0)

        train_loss = train_loss_sum / train_samples

        # -----------------------------
        # Validation
        # -----------------------------

        model.eval()

        valid_loss_sum = 0.0
        valid_samples = 0

        with torch.no_grad():
            for images, masks in valid_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                logits = model(images)
                loss = criterion(logits, masks)

                valid_loss_sum += loss.item() * images.size(0)
                valid_samples += images.size(0)

        valid_loss = valid_loss_sum / valid_samples

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"| Train Loss: {train_loss:.4f} "
            f"| Validation Loss: {valid_loss:.4f}"
        )

        # Save the best checkpoint according to validation loss.
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "validation_loss": valid_loss,
                    "image_size": IMAGE_SIZE,
                    "annotation_status": (
                        "Unverified rectangular annotations; "
                        "not validated ground truth"
                    ),
                },
                BEST_MODEL_PATH,
            )

            print(f"Saved best checkpoint: {BEST_MODEL_PATH}")

    print("\nTraining pipeline test completed.")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Checkpoint location: {BEST_MODEL_PATH}")
    print(
        "\nIMPORTANT: These results measure fitting to the supplied "
        "masks, not verified lunar landslide segmentation accuracy."
    )


if __name__ == "__main__":
    main()