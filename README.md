# StableGAN (DCGAN Experiments on Pokémon Dataset)

StableGAN implements and compares DCGAN-style GAN variants on a Pokémon image dataset, focusing on training stability and sample quality. The project includes notebooks for baseline training and activation-function experiments (e.g., ReLU vs LeakyReLU), along with generated image snapshots across epochs.

## Key Features
- DCGAN-style **Generator** and **Discriminator**
- Experiments comparing **activation functions** (ReLU vs LeakyReLU)
- Saves generated samples across epochs for qualitative inspection
- Includes a report PDF summarizing experiments and findings

## Repository Structure
```
.
├── Part2_GAN_Task1.ipynb                 # baseline DCGAN training
├── Part2_GAN_Task2.ipynb                 # activation/stability experiment notebook
├── Part2_GAN_Report.pdf                  # write-up / results
├── dcgan_images/                         # generated samples (baseline)
├── dcgan_images_task_2/                  # generated samples (variant)
├── dcgan_images_task_2_leaky/            # generated samples (LeakyReLU variant)
└── data/                                 # dataset location (typically not committed)
```

## Setup

**Requirements**
- Python 3.9+ recommended
- PyTorch + torchvision
- NumPy, matplotlib, Pillow, tqdm, Jupyter

Create a virtual environment and install dependencies:
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install numpy matplotlib pillow tqdm jupyter
# Install PyTorch (choose CUDA/CPU based on your system)
# Example (CUDA wheels - adjust if needed):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Data

This project expects a Pokémon image dataset on disk.

Recommended folder layout:
```
data/
  pokemon/
    0001.png
    0002.png
    ...
```

If your notebooks use a different path, update the dataset path variable inside the notebook.

## How to Run

### Option A — Run Notebooks (current workflow)
1. Baseline training: open `Part2_GAN_Task1.ipynb` and run all cells.
2. Activation/stability experiments: open `Part2_GAN_Task2.ipynb` and run all cells.

```bash
jupyter notebook
```

### Outputs
Generated samples are saved into:
- `dcgan_images/`
- `dcgan_images_task_2/`
- `dcgan_images_task_2_leaky/`

## Results (High-level)
- Compared stability using generator/discriminator loss trends and visual sample progression across epochs.
- In these experiments, the **LeakyReLU** variant produced more stable training and better sample quality than the ReLU variant.



## License
Add a license if you plan to share broadly (MIT is common).
