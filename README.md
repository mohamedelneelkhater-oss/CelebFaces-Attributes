# Methodology

This document explains how the project meets each requirement of the course brief
(**Project: Realistic Human Face Generation using GAN**).

## 1. Objective

Develop a Generative Adversarial Network that generates realistic synthetic human face images from random noise,
using as little time, internet and GPU as possible.

## 2. How a GAN works (short version)

A GAN has two networks trained against each other:

- The **Generator G** turns a random vector `z` into an image.
- The **Discriminator D** looks at an image and says whether it is real (from the dataset) or fake (from G).

D is trained to tell them apart; G is trained to fool D. As D improves, G is forced to produce more realistic faces.
After training, D is discarded and only G is used, with new random noise, to create new faces.

## 3. Requirement checklist

| Requirement from the brief | Where it is done |
|---|---|
| Load and preprocess a human face dataset (crop, resize, normalize) | Notebook section 4-5: centered square crop (82% of the shorter side), resize to 64x64, normalize to [-1, 1], random horizontal flip |
| Build Generator and Discriminator models | Notebook section 7: DCGAN-style networks |
| Combine them into a GAN and train using random latent vectors | Notebook section 9: `train_gan` alternates D and G updates with `z ~ N(0, I)` of size 100 |
| Monitor losses and save generated samples during training | Per-epoch loss history plus `samples/<model>/epoch_XXX.png` from a fixed noise batch |
| Compare results over time and present final generated faces | Notebook sections 12, 13, 15: curves, same-noise progress grid, final 64 faces |
| Evaluate generated face quality during training | Notebook section 8: quality metric every 2 epochs (see below) |
| Understand how GANs generate new realistic images | Sections 2 and 5 of this document |

## 4. Data

- **Dataset:** CelebFaces Attributes (CelebA) from Kaggle (uploaded by Jessica Li), about 202,599 aligned face images of 178x218 pixels.
- **Subset:** 20,000 random images (fixed seed 42), so loading takes about a minute and memory stays near 245 MB.
- **Preprocessing:**
  1. Crop: centered square of side `0.82 x min(width, height)` (removes most background, keeps the face).
  2. Resize: 64x64 with Lanczos resampling.
  3. Normalize: `x / 127.5 - 1`, giving [-1, 1] to match the Generator's `tanh` output.
  4. Augmentation: random horizontal flip.
- Images are stored as a `uint8` tensor and converted on the GPU per batch, so there is no slow data loader.

## 5. Architecture

### Generator (about 3.6M parameters)

| Layer | Output shape |
|---|---|
| Input noise | 100 x 1 x 1 |
| ConvTranspose 100 -> 512, BatchNorm, ReLU | 512 x 4 x 4 |
| ConvTranspose 512 -> 256, BatchNorm, ReLU | 256 x 8 x 8 |
| ConvTranspose 256 -> 128, BatchNorm, ReLU | 128 x 16 x 16 |
| ConvTranspose 128 -> 64, BatchNorm, ReLU | 64 x 32 x 32 |
| ConvTranspose 64 -> 3, Tanh | 3 x 64 x 64 |

### Discriminator

| Layer | Output shape |
|---|---|
| Input image | 3 x 64 x 64 |
| Conv 3 -> 64, LeakyReLU(0.2) | 64 x 32 x 32 |
| Conv 64 -> 128, BatchNorm, LeakyReLU | 128 x 16 x 16 |
| Conv 128 -> 256, BatchNorm, LeakyReLU | 256 x 8 x 8 |
| Conv 256 -> 512, BatchNorm, LeakyReLU | 512 x 4 x 4 |
| Conv 512 -> 1 | 1 logit |

All convolutions use kernel 4; weights are initialized from N(0, 0.02).

## 6. The two trained models

Both use the identical architecture; only the adversarial loss differs.

| Model | Loss | Details |
|---|---|---|
| **DCGAN** | Binary cross-entropy on logits | One-sided label smoothing: real label 0.9, fake label 0 |
| **LSGAN** | Least-squares (MSE) | Real label 1, fake label 0; smoother gradients, often more stable |

Training: Adam (lr 2e-4, betas 0.5 / 0.999), batch size 128, 20 epochs. Each iteration updates D on real and fake faces,
then updates G so that D labels its fakes as real.

## 7. Quality metric and best-model selection

The official FID needs a pretrained Inception network (extra download, extra GPU time). To stay light, the notebook uses a proxy:

1. Downsample images to 16x16 and flatten them (768 features).
2. Compute the mean and covariance of 3000 real and 3000 generated faces.
3. Compute the Frechet distance between the two Gaussians with NumPy.

Lower is better. **This is not the official FID**, so compare only numbers produced by this same notebook.
It is computed every 2 epochs; whenever it improves, the Generator checkpoint is saved. After both models finish,
the one with the lower best metric becomes `best_overall`.

## 8. Deployment

`app.py` loads the checkpoints from `models/`, shows a dropdown with **Best overall** plus each model, and generates
1-16 faces on CPU with an optional fixed seed. See `DEPLOYMENT.md`.

## 9. Limitations and ideas

- 64x64 resolution and 20 epochs give recognizable but imperfect faces; more epochs or images improve quality.
- The metric is a coarse proxy; a human look at the sample grids is still important.
- GANs can suffer from mode collapse or unstable training; the saved best checkpoint and the loss curves help detect this.
- Ideas: Generator EMA, spectral normalization, higher resolution (128x128), official FID or KID, truncation of the noise at inference.
