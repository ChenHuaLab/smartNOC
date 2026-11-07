# smartNOC

A lightweight CNN that estimates the **number of contributors (NOC)** of forensic DNA mixtures **directly from raw electropherogram (EPG) signals**, with no peak calling. Built with PyTorch.

## Features

- **End-to-end:** takes the raw fluorescence signal of the six dye channels as input and predicts NOC 1-5.
- **Fast:** trains in a few minutes on a single GPU.
- **Three training setups:** simulated data, real data with a random split, and simulated pretraining followed by fine-tuning on real data with a donor-independent split.
- **Control experiment:** a CNN trained on parsed peak-height profiles instead of raw signals (`control/`), for comparison under the same data partition.
- **Pretrained weights included** in `saved_models/`.
- **Ready-made evaluation:** balanced accuracy, per-class classification report and confusion matrix, plus TensorBoard logging.

## Model

| Part | Description |
|------|-------------|
| Input | `1 x 6 x 5000` (six dye channels, 5,000 time points) |
| Backbone | Conv `4x6` + four Conv `1x6`, each followed by BatchNorm and ReLU; MaxPool `1x4` after the 2nd, 4th and 5th Conv |
| Head | Flatten (1,824) -> FC 512 -> FC 128 -> FC 32, each with BatchNorm, ReLU and Dropout (0.5) -> FC 5 |
| Output | 5 logits (NOC 1-5) |

## Repository structure

```
.
├── cfg1_train_val_test.py          # Config 1: train on simulated data
├── cfg2_train_test.py              # Config 2: train on real data (random split)
├── cfg3_pretrain_finetune_test.py  # Config 3: fine-tune on real data (donor-independent split)
├── GF_cfg2.csv                     # split file for Config 2
├── GF_cfg3.csv                     # split file for Config 3
├── utils/
│   ├── dataset.py                  # CNNOCSimDataset, CNNOCRealDataset, BalancedSampler
│   └── model.py                    # CNNOC model
├── saved_models/                   # best checkpoints: cfg1_best.pth, cfg2_best.pth, cfg3_best.pth
├── logs/                           # TensorBoard logs: cfg1/, cfg2/, cfg3/
└── control/                        # Config 2P: control experiment on parsed peak-height profiles
    ├── cfg2p_train_test.py
    ├── utils/                      # dataset and model code for the control experiment
    ├── saved_models/               # cfg2p_best.pth
    └── logs/                       # TensorBoard logs: cfg2p/
```

## Installation

```bash
git clone <this-repository-url>
cd <repository-name>
pip install torch numpy scikit-learn tensorboard
```

Tested with Python 3, PyTorch 2.8.0, CUDA 12.8 on Ubuntu 24.04. The scripts use a GPU if available and fall back to the CPU otherwise.

## Data preparation

### Real data (PROVEDIt)

1. Download the [PROVEDIt](https://lftdi.camden.rutgers.edu/provedit/) profiles generated with the GlobalFiler kit (29 cycles) on the 3500 platform (5 s, 15 s and 25 s injection), i.e. the `PROVEDIt_..._3500 <5|15|25>sec_GF29cycles.zip` files.
2. Convert the `.hid` files to `.pth` tensors with [hid2pth](https://github.com/S3nnyK/hid2pth):

   ```bash
   chmod +x hid2pth

   for i in 1 2 3 4 5; do
       ./hid2pth batch cnnoc/GF/$i cnnoc_pth/GF/$i
   done
   ```

3. Each profile is cropped to 5,000 consecutive points, using the same start point for all channels. No baseline subtraction or normalization is applied.

### Simulated data

Used by Config 1 and for the pretraining in Config 3. Each sample is a `6 x 5000` matrix, and the NOC label is read from the file name: the part before the first underscore is the number of contributors (for example `3_...` is a three-person mixture).

### Set the paths

The paths in the scripts are local paths and must be changed. Edit these variables at the top of each script:

| Variable | Used by | Meaning |
|----------|---------|---------|
| `data_dir` | all | Simulated data folder (Config 1) or converted `.pth` folder (Configs 2, 3) |
| `csv_file` | Configs 2, 3 | CSV file defining the data split |
| `pretrain_model_path` | Config 3 | Config 1 checkpoint used for initialization |

## Training and evaluation

```bash
# Config 1: train on simulated data -> saved_models/cfg1_best.pth
python cfg1_train_val_test.py

# Config 2: train on real data from scratch -> saved_models/cfg2_best.pth
python cfg2_train_test.py

# Config 3: fine-tune the Config 1 model on real data -> saved_models/cfg3_best.pth
# (requires saved_models/cfg1_best.pth)
python cfg3_pretrain_finetune_test.py

# Config 2P (control): train on parsed peak-height profiles, same partition as Config 2
cd control
python cfg2p_train_test.py
```

The control experiment has its own `utils/`, `saved_models/` and `logs/` folders, so run it from inside `control/`.

Each script trains the model, reloads the best checkpoint and evaluates it on the test set, printing balanced accuracy, a classification report (`1p`-`5p`) and the confusion matrix.

Monitor training with TensorBoard:

```bash
tensorboard --logdir logs
```

The random seed is fixed to 42.

### Configurations

| Config | Training data | Strategy | Train / Val / Test | Epochs | LR | Weight decay | Best model chosen by |
|--------|---------------|----------|--------------------|--------|----|--------------|----------------------|
| 1 | Simulated | From scratch | 14,656 / 1,832 / 1,832 | 30 | 3e-4 | 1e-4 | Validation loss |
| 2 | Real, random split | From scratch | 1,800 / - / 450 | 100 | 3e-4 | 1e-4 | Training loss |
| 3 | Real, donor-independent split | Initialize from Config 1, freeze `model.features`, train the rest | 8,130 / 561 / 1,209 | 100 | 5e-5 | 1e-3 | Validation loss |
| 2P | Real, random split (parsed peak heights) | From scratch, same partition as Config 2 | 1,785 / - / 448 | 100 | 3e-4 | 1e-4 | Training loss |

All configurations use Adam, cross-entropy loss and a batch size of 32. Config 3 draws its training data with a class-balanced sampler (`BalancedSampler`, `num_per_class=450`).

## Results

Balanced accuracy on the test sets:

| Config | BA |
|--------|----|
| 1: simulated data | 0.8131 |
| 2: real data, random split | 0.9444 |
| 2P: parsed peak heights (control), same partition as Config 2 | 0.9350 |
| 3: real data, donor-independent split | 0.5058 |

Config 2 uses a random sample-level split, so the same donors can appear in both the training and test sets. Config 3 holds out the test donors entirely, which is a much harder setting. Compare the numbers with this difference in mind.

Training time on an NVIDIA RTX A4000: 4.08 min (Config 1), 2.15 min (Config 2), 2.80 min (Config 3), 0.76 min (Config 2P).

### Pretrained weights

The trained checkpoints are included in the repository: `saved_models/cfg1_best.pth`, `saved_models/cfg2_best.pth`, `saved_models/cfg3_best.pth` and `control/saved_models/cfg2p_best.pth`. They can be loaded with:

```python
import torch
from utils.model import CNNOC

model = CNNOC()
model.load_state_dict(torch.load("saved_models/cfg3_best.pth", map_location="cpu"))
model.eval()
```

## Notes

- Evaluated on mixtures of 1-5 unrelated contributors, one kit (GlobalFiler) and one platform (Applied Biosystems 3500).
- The model is meant as decision support for estimating the number of contributors, not as a replacement for the standard forensic workflow.

## Acknowledgements

- [PROVEDIt](https://lftdi.camden.rutgers.edu/provedit/) dataset: Alfonse, L.E., Garrett, A.D., Lun, D.S., Duffy, K.R. & Grgicak, C.M. (2018). *Forensic Science International: Genetics*, 32, 62-70.
- [hid2pth](https://github.com/S3nnyK/hid2pth) for converting `.hid` files to PyTorch tensors.

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details. The PROVEDIt dataset is not covered by this license and is subject to its own terms of use.
