"""Har bir training uchun kerak bo'ladigan yordamchi funksiyalar."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    """Bir xil seed = bir xil natija. Tajribalarni solishtirish uchun shart."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Device
# ============================================================

def get_device(preference: str = "auto") -> torch.device:
    """Mavjud eng tez qurilmani tanlaydi.

    Shu funksiya tufayli bitta kod laptopda ham, H100'da ham ishlaydi.
    """
    if preference != "auto":
        return torch.device(preference)

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3
        return f"{device} ({name}, {total:.1f} GB)"
    return str(device)


# ============================================================
# Model
# ============================================================

def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def human_count(n: int) -> str:
    for unit, div in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if n >= div:
            return f"{n / div:.2f}{unit}"
    return str(n)


# ============================================================
# Checkpoint
# ============================================================

def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_loss: float,
    config: dict,
) -> None:
    """Modelni + optimizer holatini + qayerda to'xtaganini saqlaydi.

    Faqat model'ni saqlash yetarli emas: Adam optimizer o'z ichki
    holatini (momentum) saqlaydi. Usiz resume qilinganda training sakraydi.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "config": config,
        },
        path,
    )


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | str = "cpu",
) -> dict:
    """Checkpoint'ni yuklaydi va meta ma'lumotini qaytaradi."""
    ckpt = torch.load(path, map_location=device, weights_only=True)

    model.load_state_dict(ckpt["model"])

    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])

    return {
        "epoch": ckpt.get("epoch", 0),
        "best_val_loss": ckpt.get("best_val_loss", float("inf")),
        "config": ckpt.get("config", {}),
    }


# ============================================================
# Rasm
# ============================================================

def denormalize(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] → [0, 1]. Saqlashdan oldin har doim kerak."""
    return (x.clamp(-1, 1) + 1.0) / 2.0
