# VCVI Replication Package

This repository contains a Python package to replicate the results in the paper **"Vector Copula Variational Inference and Dependent Block Posterior Approximations"** by Yu Fu, Michael Stanley Smith, and Anastasios Panagiotelis ([arxiv link](https://arxiv.org/abs/2503.01072)). The package is developed and maintained by Yu Fu.

The `VCVI` package implements all VI algorithms from the paper based on **PyTorch**, with both CPU and GPU support:
- **CPU algorithms**: All algorithms available by default
- **GPU algorithms**: Available via `VCVI.GPU` subpackage (requires CUDA)

## Purpose

The purposes of this repository are:
1. To provide a unified Python package to replicate the results in the paper, such that the dependencies can be installed automatically
2. To provide instructions on how to replicate the results in the paper

> **More replication files will be available soon, which will use these packages.**
> 
> **This repository is under testing.**

---

## Installation

### Step 1: Install PyTorch (CPU)

**Install PyTorch CPU version first:**
```bash
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
```

### Step 2: Install VCVI Package

```bash
pip install git+https://github.com/YuFuOliver/VCVI_Replication_Package.git#subdirectory=VCVI
```

### GPU Support

GPU algorithms are available via `VCVI.GPU` but require manual PyTorch GPU setup. **GPU installation and usage will be handled separately by the package maintainer.**

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

### CPU Algorithms (`from VCVI import ...`)

**Main Algorithms:**
- [MFVI](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/MFVI.py): GMF (Gaussian Mean Field)
- [MFVI_anagrad](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/MFVI_anagrad.py): GMF with analytical gradients
- [FCVI](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/FCVI.py): G-Fp & GC-Fp
- [Blocked_factor](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/Blocked_factor.py): BLK & BLK-C (Section 4.1 & Section 4.2)
- [GVC_factor](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GVC_factor.py): A1 & A2 (Section 4.1 & Section 4.2)
- [GVC_orthod](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GVC_orthod.py): A3 & A4 (Section 4.1 & Section 4.2)
- [GVC_orthod_anagrad](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GVC_orthod_anagrad.py): GVC_orthod with analytical gradients
- [GVC_orthod_withCM](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GVC_orthod_withCM.py): A5 & A6 (Section 4.1 & Section 4.2)
- [KVC_GVC_orthod](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/KVC_GVC_orthod.py): A7 (Online Appendix)

**`SVUC` Subpackage:**
- [Blocked_SVUC](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/SVUC/Blocked_SVUC.py): BLK-C (Section 4.3)
- [GVC_factor_SVUC](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/SVUC/GVC_factor_SVUC.py): GVC-F20 (Section 4.3)
- [GVC_orthod_SVUC](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/SVUC/GVC_orthod_SVUC.py): GVC-I (Section 4.3)
- [KVC_SVUC](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/SVUC/KVC_SVUC.py): KVC-G (Section 4.3)


**`spline` Subpackage:**
- [Blocked_spline](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/spline/Blocked_spline.py): BLK-C (Section 4.4)
- [GVC_factor_spline](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/spline/GVC_factor_spline.py): GVC-F5 & GVC-F20 (Section 4.4)
- [KVC_spline](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/spline/KVC_spline.py): KVC-G (Section 4.4)

### GPU Algorithms (`from VCVI.GPU import ...`)

**Available GPU-optimized algorithms:**
- [MFVI](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GPU/MFVI.py): GMF (Gaussian Mean Field)
- [FCVI](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GPU/FCVI.py): G-Fp & GC-Fp
- [Blocked_factor](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GPU/Blocked_factor.py): BLK & BLK-C
- [GVC_factor](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GPU/GVC_factor.py): A1 & A2
- [GVC_orthod](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GPU/GVC_orthod.py): A3 & A4
- [GVC_orthod_withCM](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GPU/GVC_orthod_withCM.py): A5 & A6
- [KVC_GVC_orthod](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GPU/KVC_GVC_orthod.py): A7
- [LOGH_LOGITREG](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GPU/logh_logitreg_autodiff_GPU.py): GPU-optimized logistic regression
- [LOGH_CORRELATION](https://github.com/YuFuOliver/VCVI_Replication_Package/blob/main/VCVI/GPU/logh_correlation_lasso_GPU.py): GPU-optimized correlation model

> **Note:** GPU algorithms do not include analytical gradient methods or SVUC/spline subpackages. Use explicit imports: `from VCVI.GPU import MFVI`

---


## Citation

If you use this package in your research, please cite our paper:  
> [Vector Copula Variational Inference and Dependent Block Posterior Approximations](https://arxiv.org/abs/2503.01072)
