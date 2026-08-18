"""Convolutional Autoencoder.

Bu — Stable Diffusion'dagi VAE'ning soddalashtirilgan ajdodi.
Keyingi bosqichda buni VAE'ga, so'ng video VAE'ga aylantiramiz.

    Rasm  [B, 3, 256, 256]
      ↓ Encoder (3 marta 2x siqish)
    Latent [B, 128, 32, 32]
      ↓ Decoder (3 marta 2x kengaytirish)
    Qayta tiklangan rasm [B, 3, 256, 256]
"""

from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """2x kichraytiruvchi conv bloki."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


def deconv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """2x kattalashtiruvchi transposed conv bloki."""
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.SiLU(inplace=True),
    )


class OWMAutoencoder(nn.Module):
    """3 bosqichli encoder/decoder. Umumiy siqish: 8x har o'lchov bo'yicha."""

    def __init__(self, base_channels: int = 32, latent_channels: int = 128):
        super().__init__()

        c1 = base_channels           # 32
        c2 = base_channels * 2       # 64
        c3 = latent_channels         # 128

        # ----- Encoder: 256 → 128 → 64 → 32 -----
        self.encoder = nn.Sequential(
            conv_block(3, c1),
            conv_block(c1, c2),
            nn.Conv2d(c2, c3, kernel_size=4, stride=2, padding=1),
            # Oxirgi qatlamda aktivatsiya YO'Q — latent erkin qiymat olsin
        )

        # ----- Decoder: 32 → 64 → 128 → 256 -----
        self.decoder = nn.Sequential(
            deconv_block(c3, c2),
            deconv_block(c2, c1),
            nn.ConvTranspose2d(c1, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),               # chiqish [-1, 1] — kirish bilan bir xil diapazon
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    @torch.no_grad()
    def latent_shape(self, image_size: int) -> tuple[int, ...]:
        dummy = torch.zeros(1, 3, image_size, image_size,
                            device=next(self.parameters()).device)
        return tuple(self.encode(dummy).shape[1:])
