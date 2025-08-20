# VCVI Replication Package

This repository contains Python packages to replicate the results in the paper **"Vector Copula Variational Inference and Dependent Block Posterior Approximations"** by Yu Fu, Michael Stanley Smith, and Anastasios Panagiotelis. ([arxiv link](https://arxiv.org/abs/2503.01072))

There are two packages implementing VI algorithms based on **PyTorch**:
- `VCVI`: All VI algorithms used in the paper, implemented on CPU
- `VCVI_GPU`: Selected VI algorithms from VCVI, implemented on GPU

## Purpose

The purpose of this repository are:
1. To provide two Python packages to replicate the results in the paper, such that the dependencies can be installed automatically
2. To provide instructions on how to replicate the paper

> **More replication files will be available soon, which will use these packages.**
> 
> **This repository is under testing.**

---

## Installation

### `VCVI`

```bash
pip install git+https://github.com/YuFuOliver/VCVI_Replication_Package.git#subdirectory=VCVI
```

### `VCVI_GPU`

```bash
pip install git+https://github.com/YuFuOliver/VCVI_Replication_Package.git#subdirectory=VCVI_GPU
```

---

## Usage
The packages are highly user-friendly. However, the purpose of this readme is to introduce how to replicate the paper but not to explain how to use the packages.

The packages support user-defined posterior distributions or any model written in **Stan**.

Training is as simple as:
```python
# mean field variational inference
from VCVI import MFVI

mf = MFVI(optimizer='Adam', sampling=False,
          stan_model=None, log_post=log_post)
          
ELBO_mf = mf.train(num_iter=40000)
```

---

## Available Methods

### `VCVI` Package (CPU)

**Main Algorithms:**
- [MFVI](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/MFVI.py): GMF (Gaussian Mean Field)
- [MFVI_anagrad](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/MFVI_anagrad.py): GMF with analytical gradients
- [FCVI](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/FCVI.py): G-Fp & GC-Fp
- [Blocked_factor](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/Blocked_factor.py): BLK & BLK-C (Section 4.1 & Section 4.2)
- [GVC_factor](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GVC_factor.py): A1 & A2 (Section 4.1 & Section 4.2)
- [GVC_orthod](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GVC_orthod.py): A3 & A4 (Section 4.1 & Section 4.2)
- [GVC_orthod_anagrad](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GVC_orthod_anagrad.py): `GVC_orthod` with analytical gradients
- [GVC_orthod_withCM](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GVC_orthod_withCM.py): A5 & A6 (Section 4.1 & Section 4.2)
- [KVC_GVC_orthod](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/KVC_GVC_orthod.py): A7 (Online Appendix)

**`SVUC` Subpackage:**
- [Blocked_SVUC](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/SVUC/Blocked_SVUC.py): BLK-C (Section 4.3)
- [GVC_factor_SVUC](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/SVUC/GVC_factor_SVUC.py): GVC-F20 (Section 4.3)
- [GVC_orthod_SVUC](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/SVUC/GVC_orthod_SVUC.py): GVC-I (Section 4.3)


**`spline` Subpackage:**
- [Blocked_spline](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/spline/Blocked_spline.py): BLK-C (Section 4.4)
- [GVC_factor_spline](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/spline/GVC_factor_spline.py): GVC-F5 & GVC-F20 (Section 4.4)
- [KVC_spline](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/spline/KVC_spline.py): KVC-G (Section 4.4)

### `VCVI_GPU` Package (GPU)

Contains GPU-optimized versions of selected algorithms from `VCVI`, plus additional GPU-specific log-posterior computations.

**Available Algorithms (same as described in `VCVI`, but on GPU):**
- [MFVI](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI_GPU/MFVI.py)
- [FCVI](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI_GPU/FCVI.py)
- [Blocked_factor](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI_GPU/Blocked_factor.py)
- [GVC_factor](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI_GPU/GVC_factor.py)
- [GVC_orthod](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI_GPU/GVC_orthod.py)
- [GVC_orthod_withCM](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI_GPU/GVC_orthod_withCM.py)
- [KVC_GVC_orthod](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI_GPU/KVC_GVC_orthod.py)

> **Note:** `VCVI_GPU` does not include analytical gradient methods (`MFVI_anagrad`, `GVC_orthod_anagrad`) or `SVUC`/`spline` subpackages as in `VCVI`.

---


## Citation

If you use this package in your research, please cite our paper:  
> [Vector Copula Variational Inference and Dependent Block Posterior Approximations](https://arxiv.org/abs/2503.01072)
