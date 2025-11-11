# RCPN: Towards Robust Pseudo-Labels in Unsupervised CLIP Adaptation via Consistent Prototypes and Neighborhoods

This repository contains the official PyTorch implementation of our paper [** Towards Robust Pseudo-Labels in Unsupervised CLIP Adaptation via Consistent Prototypes and Neighborhoods**].

---

# Overview
RCPN is a novel framework for unsupervised CLIP adaptation that enhances pseudo-label robustness through:

Consistent Prototypes: Leveraging stable cluster centroids for reliable guidance

Neighborhood Consistency: Exploiting local structural information for label refinement

Robust Pseudo-Labeling: Reducing noise propagation in self-training paradigms

---

## 📁 Dataset Setup

- Dataset paths are defined in [`json_files/dataset_catalog.json`](https://github.com/Externalhappy/FAIR/blob/main/json_files/dataset_catalog.json).  
  You will need to update these paths to point to your local dataset locations.

- Dataset label files are provided in [`json_files/classes.json`](https://github.com/Externalhappy/FAIR/blob/main/json_files/classes.json).

- For dataset preparation, we recommend using the scripts from the [VISSL repository](https://github.com/facebookresearch/vissl/tree/main/extra_scripts/datasets).

---

## 📦 Installation

Make sure to install the following dependencies:

- `torch >= 1.10.0`
- `timm == 0.4.12`
- `tensorboardX`
- `ftfy`

---

## 🚀 Training

To train the FAIR model, use the following command:

```bash
python train.py --dataset [name_of_dataset] --train_config ours_base --reg_batch
```

---

## 🙏 Acknowledgements

This work builds upon the codebase of [MUST](https://github.com/salesforce/MUST).  
We thank the authors for making their implementation publicly available.

If you use **RCPN** in your research, please cite our paper.

