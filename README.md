# OWM — Own World Model

Rasm va video generativ modellarini noldan qurish loyihasi.

## Struktura

```
configs/        JSON konfiguratsiyalar (kod emas, sozlama shu yerda)
data/raw/       xom rasmlar
data/processed/ natijalar
src/owm/        asosiy paket
  config.py       TrainConfig dataclass
  data.py         dataset, transform, train/val split
  models/         model arxitekturalari
  train.py        Trainer (loop, checkpoint, resume, log)
  utils.py        seed, device, checkpoint I/O
src/lessons/    01-04 bosqichdagi o'quv skriptlari
training/       training kirish nuqtalari
evaluation/     inference / baholash skriptlari
models/         checkpointlar (git'ga tushmaydi)
experiments/    loglar va namuna rasmlar (git'ga tushmaydi)
```

## O'rnatish

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Bulut GPU'da (CUDA 12.x):

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

## Ishlatish

```powershell
# Training
python training/train_autoencoder.py --config configs/autoencoder.json

# CLI bilan sozlamani bosib o'tish
python training/train_autoencoder.py --epochs 100 --batch-size 16

# Inference
python evaluation/reconstruct.py --num 6
```

Training uzilib qolsa, xuddi shu buyruq oxirgi checkpoint'dan davom ettiradi
(`resume: true`).

## Bosqichlar

- [x] 1. Tensor, autograd, linear regression
- [x] 2. Rasm → tensor pipeline
- [x] 3. Dataset + DataLoader
- [x] 4. Convolutional Autoencoder
- [x] 5. Professional fundament: config, checkpoint/resume, val split, logging
- [ ] 6. Loss grafiklari, kattaroq dataset
- [ ] 7. VAE (KL divergence, latent'dan sampling)
- [ ] 8. DDPM diffusion noldan
- [ ] 9. Matn sharti (CLIP + cross-attention) → text-to-image
- [ ] 10. Latent diffusion (Stable Diffusion arxitekturasi)
- [ ] 11. DiT + flow matching
- [ ] 12. Video VAE + temporal attention → text-to-video
