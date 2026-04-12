from setuptools import setup, find_packages

setup(
    name             = "gfi2",
    version          = "2.0.0",
    description      = "Geomorphic Flood Index v2.0 — Python port",
    long_description = open("README.md").read(),
    long_description_content_type = "text/markdown",
    author  = "Jorge Saavedra Navarro, Cinzia Albertini, Caterina Samela, Salvatore Manfreda",
    url     = "https://github.com/YOUR_USERNAME/gfi2-python",
    license = "CC BY 4.0",
    packages = find_packages(),
    python_requires  = ">=3.9",
    install_requires = [
        "numpy>=1.23",
        "rasterio>=1.3",
        "scipy>=1.9",
        "numba>=0.57",
        "joblib>=1.2",
        "matplotlib>=3.6",
    ],
    extras_require = {
        "auto": ["pysheds>=0.3"],   # hanya MODE A
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Hydrology",
        "License :: Other/Proprietary License",
    ],
)
