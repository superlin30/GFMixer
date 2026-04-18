# [WWW 2026 Accepted] GFMixer: Decoupled Temporal Gradient and Fourier-Aware Attention for Time Series Forecasting

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-ACM_DL-b31b1b.svg)](https://dl.acm.org/doi/10.1145/3774904.3792345)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2+-green.svg)](https://pytorch.org/)


</div>

<div align="center">


</div>

## 📖 Overview

GFMixer is a dual-path decoupled framework for long-term multivariate time series forecasting and tackles two structural issues in frequency-domain modeling: frequency bias and spectral degradation. 

## 🚀 Quick Start

### Environment Setup

To set up the environment, install Python 3.8 with Pytorch 1.4.4. Use the following commands for convenience:

```bash
conda create -n GFMixer python=3.8
conda activate GFMixer
pip install -r requirements.txt
```

### Dataset Preparation

Download the pre-processed datasets from:
- **Google Drive**: [Download Link](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2?usp=sharing)
- **Baidu Drive**: [Download Link](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy)

Place the downloaded data in the `./dataset` folder.

### Running Experiments

Run the following scripts for different forecasting tasks:

> **⚠️ Important Notes**: 
> - Ensure you have downloaded the datasets and placed them in the correct directory
> - The default parameters provided in scripts are a good starting point, but you need to adjust them based on your specific dataset and requirements

## 📁 Project Structure

```
GFMixer/
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── run.py                    # Main entry point for training and testing
├── dataset/                  # Dataset directory
│   ├── ETT/                  # ETT datasets
│   ├── Weather/              # Weather dataset
│   ├── Electricity/          # Electricity dataset
│   ├── Traffic/              # Traffic dataset
│   └── ...
└── ...
```

## ⚙️ Training

All scripts are located in `./scripts`. For instance, to train a model using the ETTh1 dataset with an input length of 96, simply run:

```shell
bash ./scripts/GFMixer/ETTh1.sh
```

After training:

- Your trained model will be safely stored in `./checkpoints`.
- Numerical results in .npy format can be found in `./results`.
- A comprehensive summary of quantitative metrics is accessible in `./result_long_term_forecast.txt`.

## 🙏 Acknowledgement
Special thanks to the following repositories for their invaluable code and datasets:

- [PatchTST](https://github.com/yuqinie98/PatchTST)
- [Time-Series-Library](https://github.com/thuml/Time-Series-Library)
- [Times2D](https://arxiv.org/abs/2504.00118)
- [RoPE](https://arxiv.org/abs/2104.09864)
- [PDF](https://github.com/Hank0626/PDF)

## 📚 Citation

If you find this repo useful, please consider citing our paper as follows:

```bibtex
@inproceedings{Lin2026GRAFT,
author = {Zhang, Lin and Li, Qing and Zhao, Jingmei},
title = {GRAFT: Grounded Retrieval Augmentation for Time Series Forecasting},
year = {2026},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3774904.3792345},
booktitle = {Proceedings of the ACM Web Conference 2026},
pages = {7156–7166},
numpages = {11},
location = {United Arab Emirates},
series = {WWW '26}
}
```

## 📩 Contact
If you have any questions, please contact [1230202j1001@smail.swufe.edu.cn](1230202j1001@smail.swufe.edu.cn), [superlin3030@gmail.com](superlin3030@gmail.com) or submit an issue.

