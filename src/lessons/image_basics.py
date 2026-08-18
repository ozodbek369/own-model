from pathlib import Path

from PIL import Image
import torch
import numpy as np


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = PROJECT_ROOT / "data" / "raw" / "test.jpg"


# ============================================================
# 2. Check image
# ============================================================

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )


# ============================================================
# 3. Open image
# ============================================================

image = Image.open(IMAGE_PATH)

print("Original image")
print("----------------------------")
print("Format:", image.format)
print("Mode:", image.mode)
print("Size:", image.size)


# ============================================================
# 4. Convert to RGB
# ============================================================

image = image.convert("RGB")


# ============================================================
# 5. Resize
# ============================================================

image = image.resize((256, 256))

print()
print("After resize")
print("----------------------------")
print("Size:", image.size)


# ============================================================
# 6. Convert image to NumPy array
# ============================================================

image_array = np.array(image)

print()
print("NumPy")
print("----------------------------")
print("Shape:", image_array.shape)
print("Dtype:", image_array.dtype)
print(
    "Pixel range:",
    image_array.min(),
    "to",
    image_array.max()
)


# ============================================================
# 7. Convert NumPy array to PyTorch tensor
# ============================================================

image_tensor = torch.from_numpy(image_array)


# ============================================================
# 8. Convert uint8 → float32
# ============================================================

image_tensor = image_tensor.float()


# ============================================================
# 9. Normalize 0–255 → 0–1
# ============================================================

image_tensor = image_tensor / 255.0


# ============================================================
# 10. Change format
#     [H, W, C] → [C, H, W]
# ============================================================

image_tensor = image_tensor.permute(2, 0, 1)


# ============================================================
# 11. Add batch dimension
#     [C, H, W] → [1, C, H, W]
# ============================================================

image_tensor = image_tensor.unsqueeze(0)


# ============================================================
# 12. Final information
# ============================================================

print()
print("Final PyTorch tensor")
print("----------------------------")
print("Shape:", image_tensor.shape)
print("Dtype:", image_tensor.dtype)
print(
    "Value range:",
    image_tensor.min().item(),
    "to",
    image_tensor.max().item()
)