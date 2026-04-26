# FedSAM-3D
This repository provides the official implementation of **FedSAM-3D**, a federated learning framework for efficient and transferable adaptation of large-scale medical foundation models.
## 🔍 Overview

Adapting foundation models to multi-center medical data is challenging due to data heterogeneity and data security constraints.

FedSAM-3D addresses this by enabling federated collaborative transfer of 3D medical segmentation models through parameter-efficient adaptation, achieving improved generalization on unseen datasets.

## ⚙️ Installation
We recommend using Python 3.9 with PyTorch.

```bash
conda create -n fedsam3d python=3.9  
conda activate fedsam3d  
pip install -r requirements.txt
```

The code has been tested on Linux with NVIDIA GPUs.

## 📁 Repository Structure
- `FL_core/`               Client and server training process 
- `sam_ckpt/`              Pretrained model weights  
- `segment_anything/`      Model definitions (SAM-Med3D and adapters) 
- `utils/`    Evaluation and metric computation  
- `configs/`       Configuration files for experiments  
- `utils/`         Utility functions  
