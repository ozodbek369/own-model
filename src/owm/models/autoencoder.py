"""Convolutional Autoencoder.

Bu - Stable Diffusion'dagi VAE'ning soddalashtirilgan ajdodi.
Keyingi bosqichda buni VAE'ga, so'ng video VAE'ga aylantiramiz.

    Rasm  [B, 3, H, W]
      v Encoder (3 marta 2x siqish)
    Latent [B, latent_channels, H/8, W/8]
      v Decoder (3 marta 2x kengaytirish)
    Qayta tiklangan rasm [B, 3, H, W]

Uchta muhim tanlov va sabablari:

1) GroupNorm, BatchNorm emas.
   BatchNorm statistikani batch bo'ylab hisoblaydi. batch_size=4 bo'lsa
   bu statistika juda shovqinli, va inference'da (batch=1) train paytidagidan
   boshqacha ishlaydi. GroupNorm har namunani alohida normallashtiradi -
   batch hajmiga umuman bog'liq emas. Diffusion modellarda faqat shu ishlatiladi.

2) Upsample + Conv, ConvTranspose emas.
   ConvTranspose2d kernel qadamlari bir-birining ustiga notekis tushib,
   rasmda shaxmat taxtasi naqshini qoldiradi. Avvalgi natijalarimizda
   bu naqsh aniq ko'rinardi. Upsample + Conv bu muammodan xoli.

3) Latentda aktivatsiya yo'q.
   Latent erkin qiymat olishi kerak - uni ReLU yoki Tanh bilan cheklash
   siqilgan ma'lumotni yo'qotadi.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _norm(channels: int) -> nn.GroupNorm:
    """GroupNorm - guruhlar soni kanallarga bo'linishi shart."""
    groups = min(32, channels)

    while channels % groups != 0:
        groups -= 1

    return nn.GroupNorm(groups, channels)


def down_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """2x kichraytiruvchi blok."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
        _norm(out_ch),
        nn.SiLU(),
    )


def up_block(in_ch: int, out_ch: int) -> nn.Sequential:
    """2x kattalashtiruvchi blok - shaxmat naqshisiz."""
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        _norm(out_ch),
        nn.SiLU(),
    )


class OWMAutoencoder(nn.Module):
    """3 bosqichli encoder/decoder. Umumiy siqish: 8x har o'lchov bo'yicha."""

    def __init__(self, base_channels: int = 32, latent_channels: int = 4):
        super().__init__()

        c1 = base_channels           # masalan 64
        c2 = base_channels * 2       #         128
        c3 = base_channels * 4       #         256

        self.latent_channels = latent_channels

        # ----- Encoder: H -> H/2 -> H/4 -> H/8 -----
        self.encoder = nn.Sequential(
            down_block(3, c1),
            down_block(c1, c2),
            down_block(c2, c3),
            # Latentga chiqish: aktivatsiyasiz 1x1 conv
            nn.Conv2d(c3, latent_channels, kernel_size=1),
        )

        # ----- Decoder: H/8 -> H/4 -> H/2 -> H -----
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, c3, kernel_size=1),
            _norm(c3),
            nn.SiLU(),

            up_block(c3, c2),
            up_block(c2, c1),

            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(c1, 3, kernel_size=3, padding=1),
            nn.Tanh(),               # chiqish [-1, 1] - kirish bilan bir xil
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

    @torch.no_grad()
    def compression_ratio(self, image_size: int) -> float:
        """Necha barobar siqilayotganini qaytaradi."""
        pixels = 3 * image_size * image_size
        latent = 1

        for dim in self.latent_shape(image_size):
            latent *= dim

        return pixels / latent
