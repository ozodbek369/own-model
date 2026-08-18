"""Ma'lumot pipeline'i: fayllarni topish, transform, train/val split.

Ikki manba qo'llab-quvvatlanadi:
  - "folder"  : data/raw dagi o'z rasmlaringiz
  - standart  : flowers102 / cifar10 / celeba (torchvision avtomatik yuklaydi)

Nega standart dataset kerak?
43 rasmdan generativ model o'rgana olmaydi. VAE va diffusion uchun
kamida minglab rasm shart. O'z rasmlaringiz keyinroq — fine-tuning'da asqotadi.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.transforms import v2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# nom -> (torchvision klassi, tavsiya etilgan o'lcham, taxminiy hajm)
BUILTIN_DATASETS = {
    "flowers102": ("Flowers102", 128, 8_189),
    "cifar10":    ("CIFAR10",     32, 60_000),
    "celeba":     ("CelebA",     128, 202_599),
}


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
    """Rasm -> tensor zanjiri.

    Diqqat: normalizatsiya [-1, 1] ga.
    Nega [0, 1] emas? Chunki diffusion modellar (bizning maqsadimiz)
    ham, Stable Diffusion'ning VAE'si ham [-1, 1] da ishlaydi.
    """
    steps = [v2.Resize((image_size, image_size), antialias=True)]

    if train and horizontal_flip:
        steps.append(v2.RandomHorizontalFlip(p=0.5))

    steps += [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),          # [0, 255] -> [0, 1]
        v2.Normalize(mean=[0.5] * 3, std=[0.5] * 3),    # [0, 1]   -> [-1, 1]
    ]

    return v2.Compose(steps)


# ============================================================
# 3. Datasetlar
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


class DropLabel(Dataset):
    """torchvision datasetlari (rasm, label) qaytaradi. Bizga label kerak emas."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.dataset[index][0]


# ============================================================
# 4. Manbani qurish
# ============================================================

_PATH_CACHE: dict[str, list[Path]] = {}


def _cached_paths(config) -> list[Path]:
    """find_images ni ikki marta chaqirmaslik uchun (train + val transform)."""
    key = str(config.data_path)

    if key not in _PATH_CACHE:
        _PATH_CACHE[key] = find_images(config.data_path)

    return _PATH_CACHE[key]


def _build_source(config, transform) -> Dataset:
    """config.dataset ga qarab ma'lumot manbasini quradi."""
    name = config.dataset.lower()

    if name == "folder":
        return OWMImageDataset(_cached_paths(config), transform)

    if name not in BUILTIN_DATASETS:
        raise ValueError(
            f"Notanish dataset: {config.dataset}. "
            f"Mumkin: folder, {', '.join(BUILTIN_DATASETS)}"
        )

    import torchvision.datasets as tvd

    root = config.download_path
    root.mkdir(parents=True, exist_ok=True)

    cls_name, _, _ = BUILTIN_DATASETS[name]
    cls = getattr(tvd, cls_name)

    # Har bir datasetning bo'linishlari boshqacha nomlanadi.
    # Hammasini birlashtiramiz - bo'linishni o'zimiz qilamiz.
    if name == "cifar10":
        parts = [
            cls(root=root, train=flag, transform=transform, download=True)
            for flag in (True, False)
        ]
    elif name == "flowers102":
        parts = [
            cls(root=root, split=s, transform=transform, download=True)
            for s in ("train", "val", "test")
        ]
    else:  # celeba
        parts = [cls(root=root, split="all", transform=transform, download=True)]

    return DropLabel(ConcatDataset(parts))


# ============================================================
# 5. DataLoader'lar
# ============================================================

def build_dataloaders(config) -> tuple[DataLoader, DataLoader | None, dict]:
    """Train va validation loader'larini quradi.

    Nega validation kerak?
    Train loss pasayib, val loss ko'tarilsa - model o'rganmayapti, YODLAYAPTI.
    Buni ko'rmasdan train qilish - ko'zi yumuq haydash.

    Diqqat: bir xil ma'lumot ustidan IKKI manba quriladi - train uchun
    augmentation bilan, val uchun augmentationsiz. Keyin bir xil
    indekslar bo'yicha ajratiladi, shunda ular aralashib ketmaydi.
    """
    train_source = _build_source(
        config,
        build_transform(config.image_size, train=True,
                        horizontal_flip=config.horizontal_flip),
    )
    val_source = _build_source(
        config,
        build_transform(config.image_size, train=False),
    )

    total = len(train_source)

    generator = torch.Generator().manual_seed(config.seed)
    order = torch.randperm(total, generator=generator).tolist()

    # Katta datasetlarda 10% validation ortiqcha - cheklaymiz
    n_val = min(int(total * config.val_split), config.max_val_images)
    val_idx, train_idx = order[:n_val], order[n_val:]

    # Tez tajriba uchun train hajmini cheklash imkoniyati
    if config.max_train_images > 0:
        train_idx = train_idx[: config.max_train_images]

    train_ds = Subset(train_source, train_idx)
    val_ds = Subset(val_source, val_idx) if val_idx else None

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

    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            **common,
        )

    stats = {
        "source": config.dataset,
        "total": total,
        "train": len(train_idx),
        "val": len(val_idx),
    }

    return train_loader, val_loader, stats
