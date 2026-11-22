---
title: GAN Face Generator
emoji: 🙂
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
---

# Realistic Human Face Generation using GAN

Gradio demo that generates synthetic 64x64 human faces from random noise with two lightweight GANs
(DCGAN and LSGAN) trained on a CelebA subset. Pick a model in the dropdown, choose how many faces to generate,
and press **Generate faces**.

The faces are fully synthetic and do not depict real people. Inference runs on CPU.

> This file is a template. When you create a Hugging Face Space, copy this file into the Space as `README.md`.
> The block between the `---` lines is the Space configuration and must stay at the very top.
> If you want to pin a specific Gradio version, add a line `sdk_version: <version>` to that block
> (check the Hugging Face Spaces documentation for the versions that are currently supported).
