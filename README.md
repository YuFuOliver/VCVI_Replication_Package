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

# GPU version
from VCVI_GPU import MFVI, FCVI, GVC_factor, GVC_orthod
```

## Available Methods

- **MFVI**: Mean Field Variational Inference
- **FCVI**: Factor Copula Variational Inference
- **GVC_factor**: Gaussian Vector Copula with factor structure
- **GVC_orthod**: Gaussian Vector Copula with orthogonal correlation
- **Blocked_factor**: Blocked factor model
- **GVC_orthod_withCM**: GVC with correlation matrix
- **KVC_GVC_orthod**: Kendall Vector Copula + GVC

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