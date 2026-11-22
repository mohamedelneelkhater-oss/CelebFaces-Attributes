"""Inference-side code shared by the Gradio app and the tests.

The Generator below is the same architecture that is trained in
notebooks/gan_face_generation.ipynb on Kaggle. A checkpoint saved by that
notebook is a dictionary with these keys:

    name          model name, e.g. "DCGAN" or "LSGAN"
    loss_type     "bce" (DCGAN) or "lsgan" (LSGAN)
    state_dict    Generator weights
    latent_dim    size of the random noise vector (100)
    gen_features  base channel width of the Generator (64)
    epoch         epoch at which this checkpoint was the best
    metric        pixel-space Frechet distance at that epoch (lower is better)
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


class Generator(nn.Module):
    """DCGAN generator: noise (B, latent_dim, 1, 1) -> RGB image (B, 3, 64, 64) in [-1, 1]."""

    def __init__(self, latent_dim=100, features=64):
        super().__init__()
        self.latent_dim, self.features = latent_dim, features
        f = features
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, f * 8, 4, 1, 0, bias=False), nn.BatchNorm2d(f * 8), nn.ReLU(True),   # 4x4
            nn.ConvTranspose2d(f * 8, f * 4, 4, 2, 1, bias=False), nn.BatchNorm2d(f * 4), nn.ReLU(True),        # 8x8
            nn.ConvTranspose2d(f * 4, f * 2, 4, 2, 1, bias=False), nn.BatchNorm2d(f * 2), nn.ReLU(True),        # 16x16
            nn.ConvTranspose2d(f * 2, f, 4, 2, 1, bias=False), nn.BatchNorm2d(f), nn.ReLU(True),                # 32x32
            nn.ConvTranspose2d(f, 3, 4, 2, 1, bias=False), nn.Tanh(),                                           # 64x64
        )

    def forward(self, z):
        return self.net(z)


def denorm(x):
    """Map images from [-1, 1] back to [0, 1]."""
    return (x * 0.5 + 0.5).clamp(0, 1)


@torch.no_grad()
def generate_faces(G, n, seed=None):
    """Sample `n` faces. A given `seed` always produces the same faces; `None` gives new ones."""
    G.eval()
    gen = torch.Generator()
    gen.manual_seed(int(seed)) if seed is not None else gen.seed()
    z = torch.randn(n, G.latent_dim, 1, 1, generator=gen).to(next(G.parameters()).device)
    return G(z)


def load_generator(path, map_location="cpu"):
    """Rebuild a Generator from a saved checkpoint. Returns (generator, checkpoint_dict)."""
    ckpt = torch.load(path, map_location=map_location)
    G = Generator(ckpt["latent_dim"], ckpt["gen_features"])
    G.load_state_dict(ckpt["state_dict"])
    return G.to(map_location).eval(), ckpt


def tensor_to_pil(face, size=256):
    """Convert one (3, H, W) tensor in [-1, 1] to an upscaled PIL image for display."""
    arr = (denorm(face.detach().cpu()).permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr).resize((size, size), Image.BICUBIC)


def discover_models(models_dir):
    """Find every checkpoint in `models_dir` and return {name: info} sorted by name.

    `best_overall.pth` is skipped because it is only a copy of the winning model;
    the app finds the winner itself by comparing the saved quality metric.
    """
    found = {}
    for path in sorted(Path(models_dir).glob("*.pth")):
        if path.name == "best_overall.pth":
            continue
        ckpt = torch.load(path, map_location="cpu")
        found[ckpt["name"]] = dict(
            path=str(path),
            epoch=int(ckpt["epoch"]),
            metric=float(ckpt["metric"]),
            loss_type=ckpt.get("loss_type", "unknown"),
        )
    return found
