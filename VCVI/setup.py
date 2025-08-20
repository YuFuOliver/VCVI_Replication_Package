from setuptools import setup, find_packages

setup(
    name="vcvi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.4.0",
        "numpy>=1.26.4",
        "scipy>=1.13.1",
        "bridgestan",
        "pandas",
        "matplotlib",
        "seaborn",
        "scikit-learn",
        "cmdstanpy",
    ],
    python_requires=">=3.8",
    author="Yu Fu",
    description="Variational Copula Variational Inference (CPU version)",
) 