"""
gfi2.pipeline
-------------
Orchestrator pipeline GFI 2.0 — menghubungkan semua modul.
Fungsi utama: run_gfi2()
"""

import numpy as np
import os

from .io        import resample_to_ref, save_tif, get_cellsize_meters
from .preprocess import preprocess_dem_auto, preprocess_dem_manual
from .network   import extract_channel_network
from .tracing   import hillslope_to_river_mapping, river_to_confluence_mapping
from .gfi       import compute_gfi_v1, compute_gfi_v2
from .calibrate import roc_curve_maggiore
from .metrics   import compute_validation_metrics
from .viz       import (plot_roc_comparison,
                        plot_spatial_accuracy,
                        plot_water_depth_analysis)


# ---------------------------------------------------------------------------
def run_gfi2(
    # ── Input mode ────────────────────────────────────────────────────────
    input_mode:        str   = "auto",

    # ── MODE A — hanya DEM mentah ─────────────────────────────────────────
    dem_path:          str   = "Bradano.tif",

    # ── MODE B — raster sudah tersedia ────────────────────────────────────
    demcon_path:       str   = None,
    flowdir_path:      str   = None,
    flowacc_path:      str   = None,

    # ── File referensi (wajib kedua mode) ────────────────────────────────
    flood_ref_path:    str   = "Flood T = 200.tif",
    flood_depth_path:  str   = "Tiranti T = 200.tif",

    # ── Parameter ─────────────────────────────────────────────────────────
    channel_threshold: int   = 1000,
    n_exponent:        float = 0.354429752,
    roc_step:          float = 0.005,
    max_iter:          int   = 6,
    n_jobs:            int   = -1,

    # ── Output ────────────────────────────────────────────────────────────
    out_dir:           str   = ".",
    save_rasters:      bool  = True,
    show_plots:        bool  = True,
) -> dict:
    """
    Pipeline lengkap GFI 2.0.

    Parameters
    ----------
    input_mode : str
        "auto"   → hanya butuh DEM mentah (pysheds generate flow dir/acc)
        "manual" → semua raster sudah tersedia dari QGIS/ArcGIS/WBT

    dem_path : str
        Path ke DEM GeoTIFF mentah.

    demcon_path, flowdir_path, flowacc_path : str (MODE B saja)
        Path ke DEM filled, flow direction ESRI D8, flow accumulation.

    flood_ref_path : str
        Peta banjir referensi biner (0/1). Digunakan untuk kalibrasi ROC.

    flood_depth_path : str
        Peta kedalaman banjir simulasi [m]. Digunakan untuk validasi WD.

    channel_threshold : int
        Jumlah sel upstream minimum agar piksel dianggap channel.
        Panduan: DAS kecil (<100 km²) → 200–500,
                 DAS sedang (100–1000 km²) → 500–2000,
                 DAS besar (>1000 km²) → 2000–10000.
        Estimasi cepat: threshold ≈ area_min_km2 × 1e6 / cellsize²

    n_exponent : float
        Eksponen scaling Leopold & Maddock (1953). Default 0.354429752.

    roc_step : float
        Langkah sweep threshold ROC. Lebih kecil = lebih presisi.

    max_iter : int
        Batas iterasi backwater confluence GFI v2.0.

    n_jobs : int
        Jumlah core paralel untuk tracing (-1 = semua core).

    out_dir : str
        Folder untuk menyimpan output raster dan gambar.

    save_rasters : bool
        Jika True, simpan GFI_v1/v2 dan WD_v1/v2 sebagai GeoTIFF.

    show_plots : bool
        Jika True, tampilkan plot interaktif (matikan di Colab headless).

    Returns
    -------
    dict dengan kunci:
        GFIv1, GFIv2       : 2D float32 — GFI index
        WDv1, WDv2         : 2D float32 — Water Depth [m]
        params_v1, v2      : dict       — parameter kalibrasi ROC
        metrics_v1, v2     : dict       — metrik validasi
        channel, S_matrix  : 2D array   — jaringan drainase
        profile            : dict       — rasterio profile referensi

    Examples
    --------
    # MODE A:
    from gfi2 import run_gfi2
    results = run_gfi2(input_mode="auto", dem_path="DEM.tif")

    # MODE B:
    results = run_gfi2(
        input_mode   = "manual",
        dem_path     = "DEM.tif",
        demcon_path  = "DEM_filled.tif",
        flowdir_path = "FlowDir_ESRI.tif",
        flowacc_path = "FlowAcc.tif",
    )
    """
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 62)
    print("  GFI v2.0  —  Python  |  DOI: 10.5281/zenodo.18903835")
    print(f"  Input mode : {input_mode.upper()}")
    print("=" * 62)

    # ── 1. Preprocessing DEM ──────────────────────────────────────────────
    if input_mode.lower() == "auto":
        demvoid, demcon, flow_dir, flow_acc, cellsize, profile = \
            preprocess_dem_auto(dem_path)

    elif input_mode.lower() == "manual":
        missing = [n for n, v in [("demcon_path",  demcon_path),
                                   ("flowdir_path", flowdir_path),
                                   ("flowacc_path", flowacc_path)]
                   if v is None]
        if missing:
            raise ValueError(
                f"MODE 'manual' membutuhkan: {', '.join(missing)}.\n"
                "Jika belum punya raster tersebut, gunakan input_mode='auto'."
            )
        demvoid, demcon, flow_dir, flow_acc, cellsize, profile = \
            preprocess_dem_manual(dem_path, demcon_path,
                                  flowdir_path, flowacc_path)
    else:
        raise ValueError(
            f"input_mode harus 'auto' atau 'manual', bukan '{input_mode}'."
        )

    # ── 2. Ekstraksi jaringan drainase ────────────────────────────────────
    channel, S_matrix, max_order = extract_channel_network(
        flow_acc, flow_dir, demcon, cellsize,
        threshold=channel_threshold,
    )

    # ── 3. Flow tracing Part 1 & 2 ───────────────────────────────────────
    _, ROW_channel, COL_channel = hillslope_to_river_mapping(
        demcon, flow_dir, channel, cellsize, n_jobs=n_jobs,
    )
    _, ROW_confluence, COL_confluence = river_to_confluence_mapping(
        flow_dir, channel, S_matrix, max_order, cellsize, n_jobs=n_jobs,
    )

    # ── 4. GFI v1.0 ───────────────────────────────────────────────────────
    print("\nMenghitung GFI v1.0...")
    H, Ariver, hr, GFIv1 = compute_gfi_v1(
        demcon, flow_acc, channel,
        ROW_channel, COL_channel,
        cellsize, n=n_exponent,
    )

    # ── 5. Peta referensi banjir → MargArea mask ──────────────────────────
    print("Memuat peta referensi banjir...")
    flood_sim = resample_to_ref(flood_ref_path, profile) > 0

    valid    = ~np.isnan(ROW_channel)
    r_idx    = ROW_channel[valid].astype(np.int32)
    c_idx    = COL_channel[valid].astype(np.int32)
    MargArea = np.full(demcon.shape, np.nan, dtype=np.float32)
    MargArea[valid] = flood_sim[r_idx, c_idx].astype(np.float32)

    # ── 6. Kalibrasi GFI v1.0 via ROC ────────────────────────────────────
    print("Kalibrasi GFI v1.0...")
    _, fpr_v1, tpr_v1, params_v1 = roc_curve_maggiore(
        GFIv1, flood_sim, MargArea, step_size=roc_step,
    )
    a_v1 = params_v1["a_coeff"]
    WDv1 = np.maximum(0.0, (hr * a_v1) - H)
    print(f"  τ={params_v1['tau_norm']:.3f} | AUC={params_v1['auc']:.4f} | "
          f"FPR={params_v1['fpr_opt']:.3f} | TPR={params_v1['tpr_opt']:.3f} | "
          f"a={a_v1:.4f}")

    # ── 7. GFI v2.0 (confluence backwater) ───────────────────────────────
    print("\nMenghitung GFI v2.0...")
    Ariver_v2, H_v2, hr_v2, GFIv2 = compute_gfi_v2(
        demcon, flow_acc, channel,
        ROW_channel,    COL_channel,
        ROW_confluence, COL_confluence,
        a_v1, cellsize, n=n_exponent, max_iter=max_iter,
    )

    # ── 8. Kalibrasi GFI v2.0 ────────────────────────────────────────────
    print("Kalibrasi GFI v2.0...")
    _, fpr_v2, tpr_v2, params_v2 = roc_curve_maggiore(
        GFIv2, flood_sim, MargArea, step_size=roc_step,
    )
    a_v2 = params_v2["a_coeff"]
    WDv2 = np.maximum(0.0, (hr_v2 * a_v2) - H_v2)
    print(f"  τ={params_v2['tau_norm']:.3f} | AUC={params_v2['auc']:.4f} | "
          f"FPR={params_v2['fpr_opt']:.3f} | TPR={params_v2['tpr_opt']:.3f} | "
          f"a={a_v2:.4f}")

    # ── 9. Validasi kedalaman banjir ──────────────────────────────────────
    print("\nValidasi kedalaman banjir...")
    WD_sim      = resample_to_ref(flood_depth_path, profile)
    WD_sim      = np.where(WD_sim < 0, 0.0, WD_sim)
    mask        = MargArea > 0
    observed_WD = np.where(mask, WD_sim, np.nan)

    metrics_v1  = compute_validation_metrics(WDv1, observed_WD, mask)
    metrics_v2  = compute_validation_metrics(WDv2, observed_WD, mask)

    # ── 10. Cetak ringkasan ───────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  METRIK VALIDASI AKHIR")
    print("=" * 52)
    print(f"{'Metrik':<10} {'GFI v1.0':>14} {'GFI v2.0':>14}")
    print("-" * 40)
    for key in ["mse", "rmse", "r", "kge"]:
        print(f"{key.upper():<10} {metrics_v1[key]:>14.4f} {metrics_v2[key]:>14.4f}")
    print("=" * 52)
    delta_rmse = metrics_v1["rmse"] - metrics_v2["rmse"]
    delta_auc  = params_v2["auc"]   - params_v1["auc"]
    print(f"  RMSE improvement : {delta_rmse:+.4f} m")
    print(f"  AUC  improvement : {delta_auc:+.4f}")
    print("=" * 52)

    # ── 11. Simpan raster output ──────────────────────────────────────────
    if save_rasters:
        print("\nMenyimpan raster output...")
        save_tif(GFIv1, profile, os.path.join(out_dir, "GFI_v1.tif"))
        save_tif(GFIv2, profile, os.path.join(out_dir, "GFI_v2.tif"))
        save_tif(WDv1,  profile, os.path.join(out_dir, "WD_v1.tif"))
        save_tif(WDv2,  profile, os.path.join(out_dir, "WD_v2.tif"))

    # ── 12. Visualisasi ───────────────────────────────────────────────────
    print("\nMembuat visualisasi...")
    plot_roc_comparison(fpr_v1, tpr_v1, fpr_v2, tpr_v2,
                        params_v1, params_v2,
                        out_dir=out_dir, show=show_plots)
    plot_spatial_accuracy(WDv2, flood_sim, mask, WDv1=WDv1,
                          out_dir=out_dir, show=show_plots)
    plot_water_depth_analysis(WDv1, WDv2, observed_WD, mask,
                              metrics_v1, metrics_v2,
                              out_dir=out_dir, show=show_plots)

    return dict(
        GFIv1      = GFIv1,
        GFIv2      = GFIv2,
        WDv1       = WDv1,
        WDv2       = WDv2,
        params_v1  = params_v1,
        params_v2  = params_v2,
        metrics_v1 = metrics_v1,
        metrics_v2 = metrics_v2,
        channel    = channel,
        S_matrix   = S_matrix,
        profile    = profile,
    )
