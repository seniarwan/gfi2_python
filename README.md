# Geomorphic Flood Index (GFI) v2.0 — Python

Port Python dari toolbox MATLAB GFI v2.0 oleh Saavedra Navarro et al.

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18903835.svg)](https://doi.org/10.5281/zenodo.18903835)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/gfi2-python/blob/main/notebooks/GFI2_Colab.ipynb)

---

## Tentang

GFI 2.0 adalah metode delineasi daerah rawan banjir berbasis DEM yang dikembangkan oleh Manfreda et al. Formula dasar:

```
GFI = ln(hr / H)
```

- `H`  = beda elevasi antara piksel hillslope dan channel terdekat  
- `hr` = potensi muka air sungai = `a × Ariver^n`  
- `a`  = koefisien dikalibrasi via ROC terhadap peta banjir referensi  
- `n`  = 0.354429752 (rata-rata literatur, Samela et al. 2018)

**GFI 2.0** menambahkan iterasi backwater confluence: segmen sungai yang berdekatan dengan confluence mewarisi (inherit) properti hidrologi dari sungai utama downstream, menghasilkan estimasi kedalaman banjir yang lebih akurat.

## Struktur Repo

```
gfi2-python/
├── gfi2/                   ← Package utama
│   ├── __init__.py         ← Ekspor semua fungsi publik
│   ├── io.py               ← Baca/tulis raster GeoTIFF
│   ├── preprocess.py       ← Kondisi DEM (MODE A: pysheds / MODE B: manual)
│   ├── network.py          ← Ekstraksi channel & Strahler order
│   ├── tracing.py          ← Flow tracing D8 (Numba + joblib)
│   ├── gfi.py              ← Komputasi GFI v1.0 dan v2.0
│   ├── calibrate.py        ← Kalibrasi ROC threshold
│   ├── metrics.py          ← MSE, RMSE, r, KGE
│   ├── viz.py              ← Visualisasi (ROC, peta akurasi, error)
│   └── pipeline.py         ← Orchestrator run_gfi2()
├── notebooks/
│   └── GFI2_Colab.ipynb    ← Notebook siap pakai di Google Colab
├── data/
│   └── example/            ← Contoh data kecil (opsional)
├── tests/
│   └── test_gfi2.py        ← Unit test
├── requirements.txt
├── setup.py
└── README.md
```

## Instalasi

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/gfi2-python.git
cd gfi2-python

# Install dependencies
pip install -r requirements.txt

# Install sebagai package (opsional)
pip install -e .
```

## Penggunaan Cepat

### MODE A — hanya DEM mentah (pysheds generate flow dir/acc otomatis)

```python
from gfi2 import run_gfi2

results = run_gfi2(
    input_mode        = "auto",
    dem_path          = "DEM.tif",
    flood_ref_path    = "Flood_reference.tif",
    flood_depth_path  = "Water_depth_sim.tif",
    out_dir           = "output",
)
```

### MODE B — raster preprocessing sudah ada (QGIS / ArcGIS / WhiteboxTools)

```python
from gfi2 import run_gfi2

results = run_gfi2(
    input_mode        = "manual",
    dem_path          = "DEM.tif",
    demcon_path       = "DEM_filled.tif",
    flowdir_path      = "FlowDir_ESRI_D8.tif",   # encoding ESRI wajib!
    flowacc_path      = "FlowAcc.tif",
    flood_ref_path    = "Flood_reference.tif",
    flood_depth_path  = "Water_depth_sim.tif",
    out_dir           = "output",
)
```

> **Catatan encoding Flow Direction:**  
> ESRI D8: `1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE`  
> Tool yang menghasilkan ESRI D8: QGIS r.watershed, WhiteboxTools d8_pointer, ArcGIS Flow Direction.  
> Jika dari **TauDEM**: gunakan `gfi2.convert_taudem_to_esri()` dulu.

### Akses modul individual

```python
from gfi2 import (
    load_tif, save_tif, resample_to_ref,
    preprocess_dem_auto, preprocess_dem_manual,
    extract_channel_network,
    hillslope_to_river_mapping, river_to_confluence_mapping,
    compute_gfi_v1, compute_gfi_v2,
    roc_curve_maggiore,
    compute_validation_metrics,
    plot_roc_comparison,
)
```

## Google Colab

Klik badge Colab di atas, atau buka `notebooks/GFI2_Colab.ipynb`.  
Notebook sudah berisi semua sel yang diperlukan: install, clone, upload, run, download.

## Output

Fungsi `run_gfi2()` mengembalikan dictionary berisi:

| Key | Tipe | Keterangan |
|-----|------|-----------|
| `GFIv1`, `GFIv2` | float32 array | GFI index versi 1.0 dan 2.0 |
| `WDv1`, `WDv2` | float32 array | Estimasi kedalaman banjir [m] |
| `params_v1`, `params_v2` | dict | Parameter kalibrasi ROC (τ, AUC, a, FPR, TPR) |
| `metrics_v1`, `metrics_v2` | dict | MSE, RMSE, r, KGE |
| `channel`, `S_matrix` | int array | Jaringan drainase & Strahler order |
| `profile` | dict | rasterio profile untuk georeferensi |

Raster GeoTIFF (`GFI_v1.tif`, `GFI_v2.tif`, `WD_v1.tif`, `WD_v2.tif`) dan gambar PNG (ROC, akurasi spasial, error kedalaman) disimpan di `out_dir`.

## Referensi

- Saavedra Navarro et al. (2026). *GFI v2.0*. Zenodo. https://doi.org/10.5281/zenodo.18903835  
- Manfreda, S., & Samela, C. (2019). *J. Flood Risk Management.* https://doi.org/10.1111/jfr3.12541  
- Samela, C. et al. (2017). *Advances in Water Resources*, 102, 13–28.  
- Leopold, L.B., & Maddock, T. (1953). *U.S. Government Printing Office.*

## Lisensi

CC BY 4.0 — bebas digunakan dan dimodifikasi dengan atribusi ke karya asli.
