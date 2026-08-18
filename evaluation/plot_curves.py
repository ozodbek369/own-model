"""Training loglaridan loss grafigini chizadi.

Raqamlarga qarab training qilib bo'lmaydi — trendni ko'rish kerak.
Ayniqsa: train pasayib, val ko'tarilsa = overfitting.

Ishlatish:
    python evaluation/plot_curves.py
    python evaluation/plot_curves.py --run autoencoder --log-scale
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")               # oyna ochmaydi, faylga yozadi
import matplotlib.pyplot as plt     # noqa: E402


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_log(path: Path) -> dict[str, list[float]]:
    epochs, train, val = [], [], []

    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train.append(float(row["train_loss"]))
            val.append(float(row["val_loss"]) if row.get("val_loss") else float("nan"))

    return {"epoch": epochs, "train": train, "val": val}


def main() -> None:
    parser = argparse.ArgumentParser(description="Loss grafigi")
    parser.add_argument("--run", type=str, default="autoencoder")
    parser.add_argument("--log-scale", action="store_true",
                        help="Y o'qini logarifmik qilish")
    args = parser.parse_args()

    exp_dir = PROJECT_ROOT / "experiments" / args.run
    log_path = exp_dir / "log.csv"

    if not log_path.exists():
        raise FileNotFoundError(f"Log topilmadi: {log_path}")

    data = read_log(log_path)

    fig, ax = plt.subplots(figsize=(9, 5), dpi=130)

    ax.plot(data["epoch"], data["train"], label="train",
            color="#2563eb", linewidth=2)
    ax.plot(data["epoch"], data["val"], label="validation",
            color="#dc2626", linewidth=2, linestyle="--")

    # Eng yaxshi validation nuqtasi
    finite = [(e, v) for e, v in zip(data["epoch"], data["val"]) if v == v]

    if finite:
        best_epoch, best_val = min(finite, key=lambda t: t[1])
        ax.scatter([best_epoch], [best_val], color="#dc2626",
                   zorder=5, s=60, edgecolor="white", linewidth=1.5)
        ax.annotate(f"eng yaxshi: {best_val:.4f}\nepoch {best_epoch}",
                    (best_epoch, best_val), textcoords="offset points",
                    xytext=(10, 12), fontsize=9, color="#dc2626")

    if args.log_scale:
        ax.set_yscale("log")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE loss")
    ax.set_title(f"OWM — {args.run}", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, linestyle=":")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()

    output = exp_dir / "loss_curve.png"
    fig.savefig(output)

    print(f"Saqlandi: {output}")
    print(f"Epochlar: {len(data['epoch'])}")

    if finite:
        print(f"Eng yaxshi val loss: {best_val:.6f} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
