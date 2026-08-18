from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "raw"


# ============================================================
# 2. Configuration
# ============================================================

IMAGE_SIZE = 256
BATCH_SIZE = 4


# ============================================================
# 3. Image Dataset
# ============================================================

class OWMImageDataset(Dataset):

    def __init__(self, image_dir, image_size=256):

        self.image_dir = Path(image_dir)
        self.image_size = image_size

        self.image_paths = []

        # Find JPG / JPEG / PNG files
        for extension in ["*.jpg", "*.jpeg", "*.png"]:

            self.image_paths.extend(
                self.image_dir.glob(extension)
            )

        # Sort for consistent order
        self.image_paths = sorted(
            self.image_paths
        )

        # Keep only files that are really openable images
        valid_paths = []

        for image_path in self.image_paths:

            try:
                with Image.open(image_path) as image:
                    image.verify()

                valid_paths.append(image_path)

            except Exception:

                print(
                    "Skipping unreadable image:",
                    image_path.name
                )

        self.image_paths = valid_paths

        if len(self.image_paths) == 0:
            raise RuntimeError(
                f"No images found in {self.image_dir}"
            )

    def __len__(self):

        return len(self.image_paths)

    def __getitem__(self, index):

        # Get image path
        image_path = self.image_paths[index]

        # Open image
        image = Image.open(
            image_path
        ).convert("RGB")

        # Resize
        image = image.resize(
            (self.image_size, self.image_size)
        )

        # PIL → tensor
        image_tensor = torch.from_numpy(
            __import__("numpy").array(image)
        )

        # uint8 → float32
        image_tensor = image_tensor.float()

        # 0–255 → 0–1
        image_tensor = image_tensor / 255.0

        # HWC → CHW
        image_tensor = image_tensor.permute(
            2, 0, 1
        )

        return image_tensor


# ============================================================
# 4. Self test
#    Runs only with: python src\dataset.py
#    Does NOT run when imported by autoencoder.py
# ============================================================

if __name__ == "__main__":

    # ----------------------------
    # Create Dataset
    # ----------------------------

    dataset = OWMImageDataset(
        IMAGE_DIR,
        IMAGE_SIZE
    )

    print("Dataset created")
    print("----------------------------")
    print("Number of images:", len(dataset))


    # ----------------------------
    # Test one image
    # ----------------------------

    sample = dataset[0]

    print()
    print("First image")
    print("----------------------------")
    print("Tensor shape:", sample.shape)
    print("Tensor dtype:", sample.dtype)
    print(
        "Value range:",
        sample.min().item(),
        "to",
        sample.max().item()
    )


    # ----------------------------
    # Create DataLoader
    # ----------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )


    # ----------------------------
    # Test one batch
    # ----------------------------

    batch = next(iter(dataloader))

    print()
    print("First batch")
    print("----------------------------")
    print("Batch shape:", batch.shape)
    print("Batch dtype:", batch.dtype)