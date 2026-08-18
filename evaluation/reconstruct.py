"""Saqlangan modelni yuklab, rasmlarni qayta tiklaydi (inference).

Bu training'dan MUTLAQO ajratilgan skript — haqiqiy loyihalarda
shunday bo'ladi: train bir marta, inference ming marta.

Ishlatish:
    python evaluation/reconstruct.py
    python evaluation/reconstruct.py --checkpoint models/autoencoder/best.pt --num 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from torchvision.utils import save_image           # noqa: E402

from owm.config import TrainConfig                 # noqa: E402
from owm.data import build_transform, find_images  # noqa: E402
from owm.models import OWMAutoencoder             # noqa: E402
from owm.utils import denormalize, get_device      # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Autoencoder reconstruction")
    parser.add_argument("--checkpoint", type=str,
                        default="models/autoencoder/best.pt")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Rasm papkasi (default: checkpoint config'idagi)")
    parser.add_argument("--num", type=int, default=6, help="Nechta rasm")
    parser.add_argument("--output", type=str,
                        default="data/processed/reconstruction.png")
    args = parser.parse_args()

    ckpt_path = PROJECT_ROOT / args.checkpoint

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint topilmadi: {ckpt_path}\n"
            f"Avval training/train_autoencoder.py ni ishga tushiring."
        )

    device = get_device()

    # ----- Checkpoint'ni yuklash -----
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    saved = ckpt.get("config", {})

    config = TrainConfig(**{
        k: v for k, v in saved.items()
        if k in {f for f in TrainConfig().to_dict()}
    })

    model = OWMAutoencoder(
        base_channels=config.base_channels,
        latent_channels=config.latent_channels,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Checkpoint : {ckpt_path.name} (epoch {ckpt.get('epoch', '?')})")
    print(f"Val loss   : {ckpt.get('best_val_loss', float('nan')):.6f}")
    print(f"Device     : {device}")

    # ----- Rasmlarni yuklash -----
    data_dir = PROJECT_ROOT / (args.data_dir or config.data_dir)
    paths = find_images(data_dir)[: args.num]

    transform = build_transform(config.image_size, train=False)

    from PIL import Image
    batch = torch.stack([
        transform(Image.open(p).convert("RGB")) for p in paths
    ]).to(device)

    # ----- Reconstruction -----
    with torch.no_grad():
        latent = model.encode(batch)
        reconstructed = model.decode(latent)
        loss = torch.nn.functional.mse_loss(reconstructed, batch).item()

    print(f"Latent     : {tuple(latent.shape)}")
    print(f"Siqish     : {batch[0].numel() / latent[0].numel():.2f}x")
    print(f"MSE        : {loss:.6f}")

    # ----- Saqlash: yuqori qator original, pastki qator tiklangan -----
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    grid = torch.cat([denormalize(batch), denormalize(reconstructed)], dim=0)
    save_image(grid, output_path, nrow=len(paths))

    print(f"\nSaqlandi   : {output_path}")
    print("Yuqori qator = original, pastki qator = model tiklagani")


if __name__ == "__main__":
    main()
