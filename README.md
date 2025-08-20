# VCVI Replication Package

This repository contains Python packages to replicate the results in the paper **"Vector Copula Variational Inference and Dependent Block Posterior Approximations"** by Yu Fu, Michael Stanley Smith, and Anastasios Panagiotelis. 

There are two packages implementing VI algorithms based on **PyTorch**:
- **VCVI**: All VI algorithms used in the paper, implemented on CPU
- **VCVI_GPU**: Selected VI algorithms from VCVI, implemented on GPU

## Purpose

The purpose of this repository are:
1. To provide two Python packages replicating the results in the paper, such that the dependencies can be installed automatically
2. To provide instructions on how to replicate the paper

> **More replication files will be available soon, which will use these packages.**
> 
> **This repository is under testing.**

---

## Installation

### VCVI

```bash
pip install git+https://github.com/YuFuOliver/VCVI_Replication_Package.git#subdirectory=VCVI
```

### VCVI_GPU

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

### VCVI Package (CPU)

**MFVI**: GMF (Gaussian Mean Field)  
**MFVI_anagrad**: GMF with analytical gradients  
**FCVI**: G-Fp & GC-Fp  
**Blocked_factor**: BLK & BLK-C (Section 4.1 & Section 4.2)  
**GVC_factor**: A1 & A2 (Section 4.1 & Section 4.2)  
**GVC_orthod**: A3 & A4 (Section 4.1 & Section 4.2)  
**GVC_orthod_anagrad**: GVC_orthod with analytical gradients  
**GVC_orthod_withCM**: A5 & A6 (Section 4.1 & Section 4.2)  
**KVC_GVC_orthod**: A7 (Online Appendix)  

**SVUC Subpackage:**
- **Blocked_SVUC**: BLK-C (Section 4.3)
- **GVC_factor_SVUC**: GVC-F20 (Section 4.3)
- **GVC_orthod_SVUC**: GVC-I (Section 4.3)
- **KVC_SVUC**: KVC-G (Section 4.3)

**Spline Subpackage:**
- **Blocked_spline**: BLK-C (Section 4.4)
- **GVC_factor_spline**: GVC-F5 & GVC-F20 (Section 4.4)
- **KVC_spline**: KVC-G (Section 4.4)

### VCVI_GPU Package (GPU)

Contains GPU-optimized versions of selected algorithms from VCVI package, plus additional GPU-specific log-posterior computations.

**Main Algorithms:**
 - **MFVI**  
 - **FCVI**  
 - **Blocked_factor**  
 - **GVC_factor**  
 - **GVC_orthod**  
 - **GVC_orthod_withCM**  
 - **KVC_GVC_orthod**  


> **Note:** **VCVI_GPU** does not include analytical gradient methods (`MFVI_anagrad`, `GVC_orthod_anagrad`) or SVUC/spline subpackages as in **VCVI**.