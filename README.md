# Jittor Reproduction of EViT (Alignment Verification)

This repository provides a Jittor-based reproduction of a minimal EViT model.
The goal of this project is to verify alignment with the original PyTorch implementation
under limited computational resources using a small-scale dataset.

---

##1. Environment

**System**
- OS: Windows 10 + WSL2 Ubuntu 22.04
- Python: 3.9
- Framework: Jittor
- GPU: NVIDIA RTX series (CUDA enabled)

### Create Environment

```bash
conda create -n jittor_env python=3.9 -y
conda activate jittor_env
pip install jittor matplotlib numpy


##2. Dataset

This project uses the CIFAR-10 dataset for small-scale alignment verification.

CIFAR-10 can be automatically downloaded by the training script.

If manual download is required:

Official website:
https://www.cs.toronto.edu/~kriz/cifar.html

Download (Python version):
https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

After downloading, extract and place the dataset under the project directory
according to the script configuration.

Dataset format:
Train: 50,000 images (32x32 RGB)
Test: 10,000 images


##3. Train

Run training and evaluation:python train_evit_jittor_min.py
To save terminal output as training log:python train_evit_jittor_min.py | tee logs/train_log.txt


##4. Outputs

After training, the following files will be generated:

Training Logs:
logs/train_log.txt → Full training log
logs/metrics.csv → Epoch-wise loss and accuracy records

Visualization Results:
results/loss_curve.png → Training loss curve
results/acc_curve.png → Test accuracy curve

These outputs ensure reproducibility and allow alignment comparison with the PyTorch version.


##5. Alignment Result

The Jittor implementation demonstrates:
Stable loss convergence
Consistent accuracy improvement
Final test accuracy within the same magnitude range as the PyTorch implementation

Although the training is conducted on a small-scale dataset,
the convergence trend and evaluation metrics show strong alignment
between Jittor and PyTorch frameworks.

This verifies the correctness of model migration and cross-framework reproducibility.
