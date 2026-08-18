"""Autoencoder training uchun kirish nuqtasi.

Ishlatish:
    python training/train_autoencoder.py
    python training/train_autoencoder.py --epochs 50 --batch-size 8
    python training/train_autoencoder.py --config configs/autoencoder.json
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from owm.config import TrainConfig                 # noqa: E402
from owm.data import build_dataloaders            # noqa: E402
from owm.models import OWMAutoencoder             # noqa: E402
from owm.train import Trainer                     # noqa: E402
from owm.utils import get_device, set_seed        # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Har bir config maydonini avtomatik CLI flagiga aylantiradi."""
    parser = argparse.ArgumentParser(description="OWM autoencoder training")
    parser.add_argument("--config", type=str, default=None,
                        help="JSON config fayli yo'li")

    for field in fields(TrainConfig):
        flag = "--" + field.name.replace("_", "-")

        if field.type is bool or field.type == "bool":
            parser.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes"),
                                default=None)
        else:
            parser.add_argument(flag, type=type(field.default), default=None)

    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = (
        TrainConfig.from_json(args.config) if args.config else TrainConfig()
    )

    # CLI bilan berilgan qiymatlar config'ni bosib o'tadi
    for field in fields(TrainConfig):
        value = getattr(args, field.name, None)
        if value is not None:
            setattr(config, field.name, value)

    print("Konfiguratsiya")
    print("-" * 60)
    print(config.pretty())
    print()

    set_seed(config.seed)
    device = get_device(config.device)

    train_loader, val_loader, stats = build_dataloaders(config)

    print(f"Rasmlar: jami {stats['total']} | "
          f"train {stats['train']} | val {stats['val']}")
    print(f"Batch: {config.batch_size} | "
          f"train batchlari: {len(train_loader)}")

    model = OWMAutoencoder(
        base_channels=config.base_channels,
        latent_channels=config.latent_channels,
    )

    Trainer(model, config, train_loader, val_loader, device).fit()


if __name__ == "__main__":
    main()
