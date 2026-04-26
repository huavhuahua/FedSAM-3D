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
- `FL_core/`               Federated learning process (client and server)  
- `sam_ckpt/`              Pretrained model weights  
- `segment_anything/`      Model definitions (SAM-Med3D and adapters)  
- `utils/`                 Data preparation and utility functions  
- `FL_SAM_main.py`         Main training script  
- `validation.py`          Evaluation script  
- `requirements.txt`       Dependency list

## 🚀 Usage

The overall workflow of this project consists of three main stages:

1. **Data preparation**  
   Organize the datasets into the required format and perform preprocessing if needed.
   ```bash
   dataset/
   ├── train/
   │   ├── imagesTr/
   │   └── labelsTr/
   ├── test/
   │   ├── imagesTs/
   │   └── labelsTs/
   ```

2. **Model training**  
   The model can be trained using:
   
   ```bash
   python FL_SAM_main.py
   ```

3. **Evaluation**  
   After training, the model can be evaluated using:
   ```bash
   python validation.py
   ```
## 🧩 Notes

- This repository provides a reference implementation of the proposed framework.  
- Due to data security constraints, the datasets used in this study are not publicly available.  
- The provided scripts and configurations are intended to illustrate the overall workflow.  
- Full reproduction of the reported results may require access to the original datasets and specific experimental settings.

## 📜 Citation

If you find this work useful, please consider citing:

```bibtex
@article{wu2026fedsam3d,
  title={FedSAM-3D: A Federated Adapter-based Approach Dedicated to Enhance the Transferability of Medical Segmentation Foundation Models},
  author={Xinran Wu and Rencheng Zheng and Yuxiang Dai and Hui Zhang and Xueqin Xia and Yu Cheng and Chengyan Wang and He Wang},
  journal={Under Review},
  year={2026}
}
```
