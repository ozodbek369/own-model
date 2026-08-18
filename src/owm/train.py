"""Trainer — barcha keyingi modellar uchun ishlatiladigan umumiy training tsikli.

Bu tsiklda diffusion uchun ham kerak bo'ladigan hamma narsa bor:
  - train/validation ajratish
  - mixed precision (GPU'da ~2x tezlik)
  - gradient clipping
  - checkpoint saqlash va DAVOM ETTIRISH
  - CSV log (keyin grafik chizish uchun)
  - namuna rasmlar
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

from .utils import (
    count_parameters,
    denormalize,
    describe_device,
    human_count,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


class Trainer:

    def __init__(self, model, config, train_loader, val_loader=None, device=None):
        self.config = config
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn = nn.MSELoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )

        # AMP faqat CUDA'da mantiqiy. CPU'da o'zi o'chadi.
        self.use_amp = bool(config.amp) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.use_amp)

        self.start_epoch = 0
        self.best_val_loss = float("inf")

        self.ckpt_dir = Path(config.checkpoint_dir)
        self.exp_dir = Path(config.experiment_dir)
        self.sample_dir = self.exp_dir / "samples"

        for d in (self.ckpt_dir, self.exp_dir, self.sample_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.log_path = self.exp_dir / "log.csv"

        if config.resume:
            self._try_resume()

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    def _try_resume(self) -> None:
        last = self.ckpt_dir / "last.pt"

        if not last.exists():
            return

        meta = load_checkpoint(
            last, model=self.model, optimizer=self.optimizer, device=self.device
        )
        self.start_epoch = meta["epoch"]
        self.best_val_loss = meta["best_val_loss"]

        print(f"[resume] {last.name} dan davom etamiz - epoch {self.start_epoch}")

    # --------------------------------------------------------
    # Bitta epoch
    # --------------------------------------------------------

    def _train_epoch(self) -> float:
        self.model.train()
        total, count = 0.0, 0

        for batch in self.train_loader:
            batch = batch.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                reconstructed = self.model(batch)
                loss = self.loss_fn(reconstructed, batch)

            self.optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()

            if self.config.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip
                )

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total += loss.item() * batch.size(0)
            count += batch.size(0)

        return total / max(count, 1)

    @torch.no_grad()
    def _validate(self) -> float | None:
        if self.val_loader is None:
            return None

        self.model.eval()
        total, count = 0.0, 0

        for batch in self.val_loader:
            batch = batch.to(self.device, non_blocking=True)

            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                loss = self.loss_fn(self.model(batch), batch)

            total += loss.item() * batch.size(0)
            count += batch.size(0)

        return total / max(count, 1)

    # --------------------------------------------------------
    # Namuna rasm
    # --------------------------------------------------------

    @torch.no_grad()
    def _save_samples(self, epoch: int, n: int = 4) -> None:
        loader = self.val_loader or self.train_loader
        batch = next(iter(loader))[:n].to(self.device)

        self.model.eval()
        reconstructed = self.model(batch)

        # Yuqori qator: original. Pastki qator: model tiklagani.
        grid = torch.cat([denormalize(batch), denormalize(reconstructed)], dim=0)

        save_image(
            grid,
            self.sample_dir / f"epoch_{epoch:04d}.png",
            nrow=batch.size(0),
        )

    # --------------------------------------------------------
    # Log
    # --------------------------------------------------------

    def _log(self, row: dict) -> None:
        new_file = not self.log_path.exists()

        with open(self.log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if new_file:
                writer.writeheader()
            writer.writerow(row)

    # --------------------------------------------------------
    # Asosiy tsikl
    # --------------------------------------------------------

    def fit(self) -> None:
        cfg = self.config
        set_seed(cfg.seed)

        print()
        print("=" * 60)
        print(f"Run        : {cfg.run_name}")
        print(f"Device     : {describe_device(self.device)}")
        print(f"Parametrlar: {human_count(count_parameters(self.model))}")
        print(f"Latent     : {self.model.latent_shape(cfg.image_size)}")
        print(f"Siqish     : {self.model.compression_ratio(cfg.image_size):.1f}x")
        print(f"AMP        : {'yoqilgan' if self.use_amp else "o'chiq (CPU)"}")
        print("=" * 60)
        print()

        cfg.save_json(self.exp_dir / "config.json")

        for epoch in range(self.start_epoch + 1, cfg.epochs + 1):
            t0 = time.time()

            train_loss = self._train_epoch()
            val_loss = self._validate()
            elapsed = time.time() - t0

            marker = ""

            if val_loss is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                save_checkpoint(
                    self.ckpt_dir / "best.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_val_loss=self.best_val_loss,
                    config=cfg.to_dict(),
                )
                marker = "  <- eng yaxshi"

            val_text = f"val {val_loss:.6f} | " if val_loss is not None else ""

            print(
                f"Epoch {epoch:3d}/{cfg.epochs} | "
                f"train {train_loss:.6f} | "
                f"{val_text}"
                f"{elapsed:5.1f}s"
                f"{marker}"
            )

            self._log({
                "epoch": epoch,
                "train_loss": round(train_loss, 8),
                "val_loss": round(val_loss, 8) if val_loss is not None else "",
                "seconds": round(elapsed, 2),
            })

            if epoch % cfg.save_every == 0 or epoch == cfg.epochs:
                save_checkpoint(
                    self.ckpt_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_val_loss=self.best_val_loss,
                    config=cfg.to_dict(),
                )

            if epoch % cfg.sample_every == 0 or epoch == cfg.epochs:
                self._save_samples(epoch)

        print()
        print("Training tugadi.")
        print(f"Checkpointlar : {self.ckpt_dir}")
        print(f"Log va namuna : {self.exp_dir}")
