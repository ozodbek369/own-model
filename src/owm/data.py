"""Ma'lumot pipeline'i: fayllarni topish, transform, train/val split."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ============================================================
# 1. Fayllarni topish va tekshirish
# ============================================================

def find_images(image_dir: str | Path, verbose: bool = True) -> list[Path]:
    """Papkadagi HAQIQATAN ochiladigan rasmlar ro'yxati.

    Kengaytmaga ishonib bo'lmaydi: `.jpg` nomli HTML fayl ham uchraydi.
    Shuning uchun har birini ochib tekshiramiz.
    """
    image_dir = Path(image_dir)

    if not image_dir.exists():
        raise FileNotFoundError(f"Papka topilmadi: {image_dir}")

    candidates = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    valid: list[Path] = []
    skipped: list[str] = []

    for path in candidates:
        try:
            with Image.open(path) as img:
                img.verify()
            valid.append(path)
        except Exception:
            skipped.append(path.name)

    if skipped and verbose:
        print(f"[data] {len(skipped)} ta buzuq fayl o'tkazib yuborildi: {skipped}")

    if not valid:
        raise RuntimeError(f"{image_dir} ichida biror ochiladigan rasm yo'q")

    return valid


# ============================================================
# 2. Transform
# ============================================================

def build_transform(image_size: int, *, train: bool, horizontal_flip: bool = True):
    """Rasm → tensor zanjiri.

    Diqqat: normalizatsiya [-1, 1] ga.
    Nega [0, 1] emas? Chunki diffusion modellar (bizning maqsadimiz)
    ham, Stable Diffusion'ning VAE'si ham [-1, 1] da ishlaydi.
    Fundamentni hozirdan to'g'ri qo'yamiz.
    """
    steps = [v2.Resize((image_size, image_size), antialias=True)]

    if train and horizontal_flip:
        steps.append(v2.RandomHorizontalFlip(p=0.5))

    steps += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),          # [0, 255] → [0, 1]
        v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),    # [0, 1]   → [-1, 1]
    ]

    return v2.Compose(steps)


# ============================================================
# 3. Dataset
# ============================================================

class OWMImageDataset(Dataset):
    """Rasm fayllari ro'yxatidan tensor beruvchi dataset."""

    def __init__(self, image_paths: list[Path], transform):
        self.image_paths = list(image_paths)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.image_paths[index]).convert("RGB")
        return self.transform(image)


# ============================================================
# 4. DataLoader'lar
# ============================================================

def build_dataloaders(config) -> tuple[DataLoader, DataLoader | None, dict]:
    """Train va validation loader'larini quradi.

    Nega validation kerak?
    Train loss pasayib, val loss ko'tarilsa — model o'rganmayapti, YODLAYAPTI.
    Buni ko'rmasdan train qilish — ko'zi yumuq haydash.
    """
    paths = find_images(config.data_path)

    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(len(paths), generator=generator).tolist()

    n_val = int(len(paths) * config.val_split)
    val_idx, train_idx = order[:n_val], order[n_val:]

    train_paths = [paths[i] for i in train_idx]
    val_paths = [paths[i] for i in val_idx]

    train_ds = OWMImageDataset(
        train_paths,
        build_transform(config.image_size, train=True,
                        horizontal_flip=config.horizontal_flip),
    )

    common = dict(
        num_workers=config.num_workers,
        pin_memory=(config.num_workers > 0),
        persistent_workers=(config.num_workers > 0),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        **common,
    )

    val_loader = None

    if val_paths:
        val_ds = OWMImageDataset(
            val_paths,
            build_transform(config.image_size, train=False),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            **common,
        )

    stats = {
        "total": len(paths),
        "train": len(train_paths),
        "val": len(val_paths),
    }

    return train_loader, val_loader, stats
