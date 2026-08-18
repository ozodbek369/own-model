"""Training konfiguratsiyasi.

Nega alohida fayl?
Chunki laptopda va bulut GPU'da BIR XIL kod ishlashi kerak.
O'zgaradigan yagona narsa — config. Kod emas.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    # ----- Ma'lumot -----
    data_dir: str = "data/raw"
    image_size: int = 256
    val_split: float = 0.1          # 10% validation uchun ajratiladi
    horizontal_flip: bool = True    # augmentation (faqat train'da)

    # ----- Model -----
    base_channels: int = 32         # birinchi conv qatlam kengligi
    latent_channels: int = 128      # latent (siqilgan) tasvir chuqurligi

    # ----- Training -----
    batch_size: int = 4
    learning_rate: float = 1e-3
    epochs: int = 20
    seed: int = 42
    num_workers: int = 0            # Windows'da 0 xavfsiz
    amp: bool = True                # mixed precision (GPU'da tezlik, CPU'da o'chadi)
    grad_clip: float = 1.0          # gradient portlashining oldini oladi

    # ----- Saqlash -----
    run_name: str = "autoencoder"
    save_every: int = 5             # har N epochda checkpoint
    sample_every: int = 5           # har N epochda namuna rasm
    resume: bool = True             # oxirgi checkpoint'dan davom etish

    # ----- Qurilma -----
    device: str = "auto"            # auto | cpu | cuda | mps

    # ------------------------------------------------------------

    @property
    def checkpoint_dir(self) -> Path:
        return PROJECT_ROOT / "models" / self.run_name

    @property
    def experiment_dir(self) -> Path:
        return PROJECT_ROOT / "experiments" / self.run_name

    @property
    def data_path(self) -> Path:
        return PROJECT_ROOT / self.data_dir

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        """configs/*.json dan yuklaydi. Notanish kalitlarni e'tiborsiz qoldiradi."""
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known

        if unknown:
            print(f"[config] Notanish kalitlar e'tiborsiz qoldirildi: {sorted(unknown)}")

        return cls(**{k: v for k, v in raw.items() if k in known})

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def pretty(self) -> str:
        width = max(len(f.name) for f in fields(self))
        lines = [f"  {k:<{width}} : {v}" for k, v in self.to_dict().items()]
        return "\n".join(lines)
