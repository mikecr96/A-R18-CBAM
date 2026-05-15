# Optimizing 1,001-Class Handwritten Digit Sequence Recognition for the Mexican Electoral Process 🗳️🤖

[![Paper](https://img.shields.io/badge/Paper-MDPI--AI-blue)](https://www.mdpi.com/journal/ai)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch%202.x-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Official implementation of the paper: **"Optimizing 1,001-Class Handwritten Digit Sequence Recognition for the Mexican Electoral Process using Asymmetric ResNet-18 and CBAM"**.

## 📌 Overview
This repository presents a specialized Deep Learning framework designed for the automated transcription of handwritten electoral reports within the **Preliminary Electoral Results Programme (PREP)** in Mexico. 

### Key Contributions:
* **Asymmetric Strides:** Modified ResNet-18 backbone to preserve horizontal resolution in digit sequences.
* **CBAM Integration:** Optimized Convolutional Block Attention Modules (Stages 3 & 4) for high-cardinality classification ($1,001$ classes).
* **Large-Scale Dataset:** Trained and validated on **3.77 million** real-world handwritten images.
* **Calibration:** Achieved an **ECE of 0.0957**, ensuring high reliability for electoral auditing.

---

## 🚀 Getting Started

### Installation
```bash
git clone [https://github.com/mikecr96/A-R18-CBAM.git](https://github.com/mikecr96/A-R18-CBAM.git)
cd A-R18-CBAM
pip install -r requirements.txt