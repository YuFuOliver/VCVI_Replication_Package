# VCVI Replication Package

This repository contains two Python packages for Variational Copula Variational Inference:

- **VCVI**: CPU implementation
- **VCVI_GPU**: GPU/CUDA implementation

## Installation

### CPU Version

```bash
pip install git+https://github.com/yourusername/VCVI_Replication_Package.git#subdirectory=VCVI
```

### GPU Version

```bash
pip install git+https://github.com/yourusername/VCVI_Replication_Package.git#subdirectory=VCVI_GPU
```

## Requirements

Both packages automatically install their dependencies:

- Python >= 3.8
- torch (>= 2.4.0 for CPU, >= 2.5.1 for GPU)
- numpy >= 1.26.4
- scipy >= 1.13.1
- bridgestan
- pandas
- matplotlib
- seaborn
- scikit-learn
- cmdstanpy

**GPU version additionally requires:**
- CUDA-compatible GPU
- CUDA drivers
- PyTorch with CUDA support

## Usage

```python
# CPU version
from VCVI import MFVI, FCVI, GVC_factor, GVC_orthod
from VCVI.SVUC import Blocked_SVUC, GVC_factor_SVUC, GVC_orthod_SVUC, KVC_SVUC
from VCVI.spline import Blocked_spline, GVC_factor_spline, KVC_spline

# GPU version
from VCVI_GPU import MFVI, FCVI, GVC_factor, GVC_orthod
```

## Available Methods

### VCVI Package (CPU)

**Main algorithms for different variational approximations:**

- **MFVI**: GMF (Gaussian Mean Field)
- **MFVI_anagrad**: GMF with analytical gradients
- **FCVI**: G-Fp & GC-Fp
- **Blocked_factor**: BLK & BLK-C (Section 4.1 & Section 4.2)
- **GVC_factor**: A1 & A2 (Section 4.1 & Section 4.2)
- **GVC_orthod**: A3 & A4 (Section 4.1 & Section 4.2)
- **GVC_orthod_anagrad**: GVC_orthod with analytical gradients
- **GVC_orthod_withCM**: A5 & A6 (Section 4.1 & Section 4.2)
- **KVC_GVC_orthod**: A7 (Online Appendix)

**SVUC subpackage:**

- **Blocked_SVUC**: BLK-C (Section 4.3)
- **GVC_factor_SVUC**: GVC-F20 (Section 4.3)
- **GVC_orthod_SVUC**: GVC-I (Section 4.3)
- **KVC_SVUC**: KVC-G (Section 4.3)

**spline subpackage:**

- **Blocked_spline**: BLK-C (Section 4.4)
- **GVC_factor_spline**: GVC-F5 & GVC-F20 (Section 4.4)
- **KVC_spline**: KVC-G (Section 4.4)

### VCVI_GPU Package (GPU)

Contains GPU-optimized versions of selected algorithms from VCVI package, plus additional GPU-specific implementations:

- **LOGH_LOGITREG**: GPU-optimized logistic regression
- **LOGH_CORRELATION**: GPU-optimized correlation model

## Repository Structure

```
VCVI_Replication_Package/
├── VCVI/                 # CPU implementation
│   ├── setup.py
│   ├── __init__.py
│   ├── SVUC/            # SVUC-specific methods
│   ├── spline/          # Spline additive functions
│   └── ...
├── VCVI_GPU/            # GPU implementation
│   ├── setup.py
│   ├── __init__.py
│   └── ...
└── README.md
```

## License

[Your License Here]

## Citation

If you use this package, please cite:

```
[Your Citation Here]
``` 