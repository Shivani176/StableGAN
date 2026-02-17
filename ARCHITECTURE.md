# Architecture — StableGAN

This document explains the components and data flow used in the StableGAN experiments.

## System Overview
StableGAN trains a DCGAN-style Generative Adversarial Network (GAN) on an image dataset. Training alternates updates between:

1) **Discriminator (D):** learns to classify real images vs generated images  
2) **Generator (G):** learns to generate realistic images from random noise

The repository currently runs primarily through Jupyter notebooks, which define the architectures, training loop, and artifact saving.

---

## Components

### 1) Dataset + DataLoader
**Input:** Pokémon images on disk  
**Typical preprocessing:**
- resize/crop to a fixed resolution (commonly 64×64)
- normalize pixel values (often to [-1, 1] when using `tanh` output)

**Output:** mini-batches of real images `x_real`

### 2) Generator (G)
**Purpose:** map a latent noise vector `z ~ N(0, 1)` to an image `x_fake`

**Typical DCGAN-style blocks**
- transposed convolutions / upsampling
- batch normalization
- activation: ReLU or LeakyReLU (experimented)
- final activation: `tanh` (common when inputs are normalized)

### 3) Discriminator (D)
**Purpose:** classify images as real or fake

**Typical DCGAN-style blocks**
- strided convolutions (downsampling)
- activation: LeakyReLU (common)
- final output: scalar probability/logit

### 4) Training Loop
For each batch:
1. Sample noise `z`, generate `x_fake = G(z)`
2. Update **D** to better distinguish `x_real` vs `x_fake`
3. Update **G** to produce `x_fake` that **D** classifies as real

**Optimizers:** commonly Adam with tuned learning rate and betas  
**Stability levers explored:** activation functions, learning-rate tuning, and architectural choices

### 5) Logging + Artifacts
- Save generated image grids every N steps/epochs
- Track generator/discriminator losses to observe stability
- Write outputs to `dcgan_images*/` folders for qualitative review

---

## Data Flow Diagram (Text)

```
Noise z  ──> Generator G ──> Fake images x_fake ─┐
                                                 ├─> Discriminator D ──> loss_D, loss_G
Real images x_real ──────────────────────────────┘

Artifacts:
- losses logged in notebook
- sample grids saved to disk periodically
```

---

## Current Limitations
- Notebook-driven training (less reproducible than script/config)
- No standardized experiment tracking (TensorBoard/W&B)
- No quantitative metric (e.g., FID) included by default

## Recommended “Production-Ready” Additions (optional)
1. `requirements.txt` / `environment.yml` for setup
2. `train.py` + `config.yaml` for scripted reproducibility
3. Checkpointing (`checkpoints/`) and consistent output folder (`outputs/`)
4. Experiment tracking (TensorBoard/W&B)
5. Quantitative evaluation metric script (e.g., FID)
