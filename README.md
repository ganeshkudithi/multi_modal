---
pretty_name: MicroGen3D
tags:
  - GenAI
  - LDM
  - 3d
  - microstructure
  - diffusion-model
  - materials-science
  - synthetic-data
  - voxel
license: mit
datasets:
  - microgen3D
language:
  - en
---

# microgen3D

[![Code](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/baskargroup/MicroGen3D)

## Dataset Summary

**microgen3D** is a dataset of 3D voxelized microstructures designed for training, evaluation, and benchmarking of generative models—especially Conditional Latent Diffusion Models (LDMs). It includes both synthetic (Cahn–Hilliard) and experimental microstructures with multiple phases (2 to 3). The voxel grids range from `64³` up to `128×128×64`.

The dataset consists of three microstructure types:
- **Experimental microstructures**
- **2-phase Cahn–Hilliard microstructures**
- **3-phase Cahn–Hilliard microstructures**

The two Cahn–Hilliard datasets are thresholded versions of the same simulation source.  
For each dataset type, we also provide pretrained generative model weights:
- `vae.pt` – Variational Autoencoder
- `fp.pt` – Feature Predictor
- `ddpm.pt` – Denoising Diffusion Probabilistic Model

---

## 📂 Dataset Overview

| File Name                         | Size     | Description |
|------------------------------------|----------|-------------|
| `CH_three_phase.tar.gz`            | ~5.57 GB | Full **three-phase Cahn–Hilliard** dataset with 3D microstructures and morphological descriptors. |
| `CH_two_phase.tar.gz`              | ~4.37 GB | Full **two-phase Cahn–Hilliard** dataset with 3D microstructures and morphological descriptors. |
| `experimental.tar.gz`              | ~843 MB  | **Experimental microstructure** dataset from real-world samples, voxelized for modeling. |
| `sample_CH_three_phase.tar.gz`     | ~12.2 MB | Small subset of the three-phase dataset for testing/demo purposes. |
| `sample_CH_two_phase.tar.gz`       | ~9.59 MB | Small subset of the two-phase dataset for testing/demo purposes. |

---

## 📊 Detailed Dataset Information

### **CH Two-Phase Dataset**
- **File:** `CH_two_phase.tar.gz`
- **Total Microstructures:** 47,119
- **Splits:** 10 (Train: 9, Validation: 1)
- **Microstructure Shape:** `(128, 128, 64)`
- **Attributes per Key:** 34
- **Example Attributes:**
  - `ABS_f_D`: 0.391171
  - `CT_f_D_tort1`: 0.293271
  - `phi`: 0.556
  - `chi`: 2.33
  - `source`: direct/data_chi_2.330_phi_0.556_step_235.txt


---

### **CH Three-Phase Dataset**
- **File:** `CH_three_phase.tar.gz`
- **Total Microstructures:** 45,980
- **Splits:** 10 (Train: 9, Validation: 1)
- **Microstructure Shape:** `(128, 128, 64)`
- **Attributes per Key:** 13
- **Example Attributes:**
  - `Interface_AM`: 113702.0
  - `Interface_DM`: 96692.0
  - `phi`: 0.514
  - `chi`: 2.2
  - `source`: data_chi_2.200_phi_0.514.h5____172.txt

---

### **Experimental Microstructure Dataset**
- **File:** `experimental.tar.gz`
- **Total Microstructures:** 21,421
- **Train Samples:** 19,278  
  **Validation Samples:** 2,143
- **Microstructure Shape:** `(64, 64, 64)`
- **Attributes per Key:** 23
- **Example Attributes:**
  - `ABS_f_D`: 0.591423
  - `CT_f_D_tort1`: 0.159534
  - `source`: /work/mech-ai-scratch/nirmal/generative_model_data/experimental/grid_cut/graspi/morphs/CB_120_260.txt

---

### Pretrained Weights (.pt)

We provide three pretrained weight packs aligned with the dataset families:


- `vae.pt` — Variational Autoencoder weights
- `fp.pt` — Feature Predictor weights
- `ddpm.pt` — Latent Diffusion Model weights

### Model/Weights Summary

| Pack          | Input shape     | VAE latent size | FP input (flattened) | FP output size (# predicted attrs) | Conditioning params | Manufacturing params | DDPM max features (`n_feat`) |
|--------------:|:----------------|:----------------|----------------------:|:-----------------------------------:|:-------------------:|:--------------------:|:----------------------------:|
| CH 2-Phase    | `1,128,128,64`  | `4,8,8,4`       | `1024`                | `7`                                 | `3`                 | `0`                  | `512`                        |
| CH 3-Phase    | `1,128,128,64`  | `4,8,8,4`       | `1024`                | `7`                                 | `4`                 | `3`                  | `512`                        |
| Experimental  | `64,64,64`      | `1,8,8,8`       | `512`                 | `3`                                 | `3`                 | `0`                  | `512`                        |


To learn more about the attributes and their meanings, see this [link](https://owodolab.github.io/graspi/listOfDescriptors.html).

## 📁 Repository Structure

```
microgen3D/
├── data/
│   ├── experimental.tar.gz
│   ├── ch_2phase.tar.gz
│   ├── ch_3phase.tar.gz
│   ├── sample_CH_two_phase.tar.gz
│   ├── sample_CH_three_phase.tar.gz
│   ├── experimental/                 # after extracting experimental.tar.gz
│   │   ├── dataset_info.txt
│   │   ├── train.h5
│   │   ├── val.h5
│   │   └── sample_train.h5
│   ├── ch_2phase/                    # after extracting ch_2phase.tar.gz
│   │   ├── dataset_info.txt
│   │   ├── train/                    # training split (HDF5 shards/files)
│   │   └── val/                      # validation split
│   ├── ch_3phase/                    # after extracting ch_3phase.tar.gz
│   │   ├── dataset_info.txt
│   │   ├── train/
│   │   └── val/
│   ├── ch_2phase_sample/             # after extracting sample_CH_two_phase.tar.gz
│   │   ├── dataset_info.txt
│   │   ├── train/
│   │   └── val/
│   └── ch_3phase_sample/             # after extracting sample_CH_three_phase.tar.gz
│       ├── dataset_info.txt
│       ├── train/
│       └── val/
├── models/
│   └── weights/
│       ├── experimental/
│       │   ├── vae.pt
│       │   ├── fp.pt
│       │   └── ddpm.pt
│       ├── ch_2phase/
│       │   ├── vae.pt
│       │   ├── fp.pt
│       │   └── ddpm.pt
│       └── ch_3phase/
│           ├── vae.pt
│           ├── fp.pt
│           └── ddpm.pt
└── ...

```

---

## 🚀 Quick Start

### 🔧 Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/baskargroup/MicroGen3D.git
cd MicroGen3D

# 2. Set up environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download dataset and weights (Hugging Face)
# Make sure HF CLI is installed and you're logged in: `huggingface-cli login`
```


## 📥 Download Examples

### Using Python
```python
from huggingface_hub import hf_hub_download
import os

# Download sample dataset
hf_hub_download(
    repo_id="BGLab/microgen3D",
    filename="data/experimental.tar.gz",   # correct remote path
    repo_type="dataset",
    local_dir=""
)

# Download experimental pretrained weights

for fname in ["weights/experimental/vae.pt",
              "weights/experimental/fp.pt",
              "weights/experimental/ddpm.pt"]:
    hf_hub_download(
        repo_id="BGLab/microgen3D",
        filename=fname,                # correct remote path
        repo_type="dataset",
        local_dir=""
    )
```
### 📂 Extract Dataset
```bash 
tar -xzvf data/experimental.tar.gz -C data/ 
``` 

## 🏋️ Training
For inference details refer to the GitHub repository README. [![Code](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/baskargroup/MicroGen3D)

Navigate to the training folder and run:
```bash
cd training
python training.py
```

## 🧠 Inference
For inference details refer to the GitHub repository README. [![Code](https://img.shields.io/badge/GitHub-Code-black?logo=github)](https://github.com/baskargroup/MicroGen3D)

After training, switch to the inference folder and run:
```bash
cd ../inference
python inference.py
```

---

## 📜 Citation

If you use this dataset or models, please cite:

```
@article{baishnab2025microgen3d,
  title={3D Multiphase Heterogeneous Microstructure Generation Using Conditional Latent Diffusion Models},
  author={Baishnab, Nirmal and Herron, Ethan and Balu, Aditya and Sarkar, Soumik and Krishnamurthy, Adarsh and Ganapathysubramanian, Baskar},
  journal={arXiv preprint arXiv:2503.10711},
  year={2025}
}
```

---

## ⚖️ License

This project is licensed under the **MIT License**.

---
