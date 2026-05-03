"""
gfi2.pipeline
-------------
Orchestrator pipeline GFI 2.0 — menghubungkan semua modul.
Fungsi utama: run_gfi2()

Dua mode kalibrasi:
  calibration_mode = "roc"    → kalibrasi otomatis via ROC (butuh flood_ref_path)
  calibration_mode = "manual" → gunakan tau_gfi yang diketahui dari literatur/studi lain
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
def _apply_manual_threshold(GFI, tau_gfi, cellsize=None):
    """
    Terapkan threshold manual pada GFI (tanpa ROC sweep).

    Langkah:
      1. Normalisasi GFI ke [-1, 1] (min-max) — sama dengan ROC
      2. Hitung tau_norm dari tau_gfi (nilai asli GFI)
      3. Hitung a_coeff = 1 / exp(tau_gfi)
      4. Buat params dict yang setara dengan output roc_curve_maggiore

    Parameters
    ----------
    GFI     : 2D float32 — GFI index (nilai asli)
    tau_gfi : float      — threshold GFI dalam satuan asli (bukan ternormalisasi)

    Returns
    -------
    matrix_norm : 2D float32 — GFI ternormalisasi [-1, 1]
    params      : dict       — setara output roc_curve_maggiore
    """
    mn, mx      = float(np.nanmin(GFI)), float(np.nanmax(GFI))
    matrix_norm = (2.0 * ((GFI - mn) / (mx - mn) - 0.5)).astype(np.float32)

    # Konversi tau dari satuan asli ke ternormalisasi
    tau_norm = float(2.0 * ((tau_gfi - mn) / (mx - mn) - 0.5))
    tau_norm = float(np.clip(tau_norm, -1.0, 1.0))

    # a_coeff = 1 / exp(tau_gfi)  [Manfreda & Samela, 2019, Eq. 3]
    a_coeff  = float(1.0 / np.exp(tau_gfi))

    params = dict(
        tau_norm  = tau_norm,
        tau_real  = float(tau_gfi),
        fpr_opt   = float("nan"),   # tidak dihitung tanpa referensi
        tpr_opt   = float("nan"),
        f_optim   = float("nan"),
        auc       = float("nan"),
        a_coeff   = a_coeff,
    )
    return matrix_norm, params


# ---------------------------------------------------------------------------
def run_gfi2(
    # ── Input mode ────────────────────────────────────────────────────────
    input_mode:         str   = "auto",

    # ── MODE A — hanya DEM mentah ─────────────────────────────────────────
    dem_path:           str   = "DEM.tif",

    # ── MODE B — raster sudah tersedia ────────────────────────────────────
    demcon_path:        str   = None,
    flowdir_path:       str   = None,
    flowacc_path:       str   = None,

    # ── Mode kalibrasi ────────────────────────────────────────────────────
    calibration_mode:   str   = "roc",
    #   "roc"    → kalibrasi otomatis via ROC sweep (butuh flood_ref_path)
    #   "manual" → gunakan tau_gfi dari literatur/studi lain (tanpa data referensi)

    # ── Parameter kalibrasi ROC (hanya jika calibration_mode="roc") ──────
    flood_ref_path:     str   = None,
    flood_depth_path:   str   = None,
    roc_step:           float = 0.005,

    # ── Parameter kalibrasi manual (hanya jika calibration_mode="manual") ─
    tau_gfi_v1:         float = None,
    #   Nilai threshold GFI v1.0 dalam satuan asli (bukan ternormalisasi)
    #   Contoh dari literatur: -0.221 (Samela et al. 2017, sungai Italia)
    #   a_coeff dihitung otomatis: a = 1 / exp(tau_gfi_v1)
    tau_gfi_v2:         float = None,
    #   Nilai threshold GFI v2.0 (opsional, jika None → sama dengan tau_gfi_v1)

    # ── Parameter umum ────────────────────────────────────────────────────
    channel_threshold:  int   = 1000,
    flow_dir_encoding:  str   = "esri",
    n_exponent:         float = 0.354429752,
    max_iter:           int   = 6,
    n_jobs:             int   = -1,

    # ── Output ────────────────────────────────────────────────────────────
    out_dir:            str   = ".",
    save_rasters:       bool  = True,
    show_plots:         bool  = True,

    # ── Checkpoint (simpan/muat hasil antara) ─────────────────────────────
    save_intermediate:  bool  = False,
    # Jika True, simpan hasil Part 1, Part 2, dan GFI ke file .npz
    # sehingga tidak perlu komputasi ulang jika Colab disconnect.
    load_intermediate:  bool  = False,
    # Jika True, muat hasil antara dari file .npz yang sudah ada.
    # Gunakan ini untuk melanjutkan dari checkpoint sebelumnya.
) -> dict:
    """
    Pipeline lengkap GFI 2.0.

    Parameters
    ----------
    input_mode : str
        "auto"   → hanya butuh DEM mentah (pysheds generate flow dir/acc)
        "manual" → raster sudah tersedia dari QGIS/ArcGIS/WBT

    calibration_mode : str
        "roc"    → kalibrasi otomatis via ROC sweep
                   Membutuhkan: flood_ref_path, flood_depth_path
        "manual" → gunakan threshold yang sudah diketahui
                   Membutuhkan: tau_gfi_v1
                   Tidak membutuhkan data referensi banjir

    tau_gfi_v1 : float (wajib jika calibration_mode="manual")
        Nilai GFI threshold dalam satuan asli. Nilai ini menentukan batas
        antara flood-prone dan non-flood-prone.

        Cara mendapatkan nilai ini:
          - Dari studi sebelumnya di DAS yang sama
          - Dari literatur untuk DAS dengan karakteristik serupa
          - Dari paper GFI: kisaran umum -0.5 sampai 0.5
          - Dari output run_gfi2() dengan calibration_mode="roc" sebelumnya
            → ambil results["params_v1"]["tau_real"]

    tau_gfi_v2 : float (opsional, default = tau_gfi_v1)
        Threshold untuk GFI v2.0. Jika None, menggunakan tau_gfi_v1.

    Contoh penggunaan
    -----------------
    # Dengan data referensi (kalibrasi ROC):
    results = run_gfi2(
        calibration_mode = "roc",
        flood_ref_path   = "flood_map.tif",
        flood_depth_path = "water_depth.tif",
        ...
    )

    # Tanpa data referensi (threshold manual):
    results = run_gfi2(
        calibration_mode = "manual",
        tau_gfi_v1       = -0.221,   # dari literatur atau studi lain
        ...
    )

    Returns
    -------
    dict:
        GFIv1, GFIv2        : 2D float32 — GFI index
        WDv1, WDv2          : 2D float32 — Water Depth [m]
        flood_prone_v1/v2   : 2D bool    — area rawan banjir
        params_v1, params_v2: dict       — parameter kalibrasi
        metrics_v1, v2      : dict/None  — metrik validasi (None jika mode manual)
        channel, S_matrix   : 2D array   — jaringan drainase
        profile             : dict       — rasterio profile
    """
    os.makedirs(out_dir, exist_ok=True)

    # ── Validasi parameter ────────────────────────────────────────────────
    cal = calibration_mode.lower()
    if cal not in ("roc", "manual"):
        raise ValueError(
            f"calibration_mode harus 'roc' atau 'manual', bukan '{calibration_mode}'."
        )
    if cal == "roc" and flood_ref_path is None:
        raise ValueError(
            "calibration_mode='roc' membutuhkan flood_ref_path.\n"
            "Jika tidak punya data referensi, gunakan calibration_mode='manual' "
            "dan tentukan tau_gfi_v1."
        )
    if cal == "manual" and tau_gfi_v1 is None:
        raise ValueError(
            "calibration_mode='manual' membutuhkan tau_gfi_v1.\n"
            "Contoh: tau_gfi_v1 = -0.221  (nilai dari literatur atau studi lain)"
        )

    print("=" * 62)
    print("  GFI v2.0  —  Python  |  DOI: 10.5281/zenodo.18903835")
    print(f"  Input mode       : {input_mode.upper()}")
    print(f"  Calibration mode : {cal.upper()}")
    if cal == "manual":
        print(f"  τ GFI v1.0 (input): {tau_gfi_v1}")
        print(f"  τ GFI v2.0 (input): {tau_gfi_v2 if tau_gfi_v2 is not None else '= τ v1.0'}")
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
                f"input_mode='manual' membutuhkan: {', '.join(missing)}."
            )
        demvoid, demcon, flow_dir, flow_acc, cellsize, profile = \
            preprocess_dem_manual(dem_path, demcon_path,
                                  flowdir_path, flowacc_path)
    else:
        raise ValueError(f"input_mode harus 'auto' atau 'manual'.")

    # ── 2. Jaringan drainase ──────────────────────────────────────────────
    channel, S_matrix, max_order = extract_channel_network(
        flow_acc, flow_dir, demcon, cellsize,
        threshold=channel_threshold,
        encoding=flow_dir_encoding,
    )

    # ── 3. Flow tracing ───────────────────────────────────────────────────
    ROW_channel, COL_channel = hillslope_to_river_mapping(
        demcon, flow_dir, channel, flow_acc, cellsize,
        encoding=flow_dir_encoding, n_jobs=n_jobs,
    )
    ROW_confluence, COL_confluence = river_to_confluence_mapping(
        flow_dir, channel, S_matrix, max_order, cellsize,
        encoding=flow_dir_encoding, n_jobs=n_jobs,
    )

    # ── 3b. Checkpoint: simpan/muat hasil tracing ─────────────────────────
    ckpt_path = os.path.join(out_dir, "_checkpoint_tracing.npz")

    if load_intermediate and os.path.exists(ckpt_path):
        print(f"Memuat checkpoint tracing dari {ckpt_path} ...")
        ckpt = np.load(ckpt_path)
        ROW_channel    = ckpt["ROW_channel"]
        COL_channel    = ckpt["COL_channel"]
        ROW_confluence = ckpt["ROW_confluence"]
        COL_confluence = ckpt["COL_confluence"]
        print("  ✓ Checkpoint dimuat — Part 1 & Part 2 dilewati")
    elif save_intermediate:
        np.savez_compressed(
            ckpt_path,
            ROW_channel    = ROW_channel,
            COL_channel    = COL_channel,
            ROW_confluence = ROW_confluence,
            COL_confluence = COL_confluence,
        )
        print(f"  ✓ Checkpoint tracing disimpan: {ckpt_path}")

    # ── 4. GFI v1.0 ───────────────────────────────────────────────────────
    print("\nMenghitung GFI v1.0...")
    H, Ariver, hr, GFIv1 = compute_gfi_v1(
        demcon, flow_acc, channel,
        ROW_channel, COL_channel,
        cellsize, n=n_exponent,
    )

    # ── 4b. Checkpoint: simpan GFI v1.0 ──────────────────────────────────
    ckpt_gfi_path = os.path.join(out_dir, "_checkpoint_gfi.npz")
    if save_intermediate:
        np.savez_compressed(
            ckpt_gfi_path,
            GFIv1=GFIv1, H=H, hr=hr, Ariver=Ariver,
        )
        print(f"  ✓ Checkpoint GFI v1.0 disimpan: {ckpt_gfi_path}")

    # ── 5. Kalibrasi GFI v1.0 ────────────────────────────────────────────
    fpr_v1 = tpr_v1 = None
    MargArea = flood_sim = None

    if cal == "roc":
        print("Memuat peta referensi banjir...")
        flood_sim = resample_to_ref(flood_ref_path, profile) > 0

        valid    = ~np.isnan(ROW_channel)
        r_idx    = ROW_channel[valid].astype(np.int32)
        c_idx    = COL_channel[valid].astype(np.int32)
        MargArea = np.full(demcon.shape, np.nan, dtype=np.float32)
        MargArea[valid] = flood_sim[r_idx, c_idx].astype(np.float32)

        print("Kalibrasi GFI v1.0 via ROC...")
        _, fpr_v1, tpr_v1, params_v1 = roc_curve_maggiore(
            GFIv1, flood_sim, MargArea, step_size=roc_step,
        )
        print(f"  τ={params_v1['tau_norm']:.3f} | AUC={params_v1['auc']:.4f} | "
              f"FPR={params_v1['fpr_opt']:.3f} | TPR={params_v1['tpr_opt']:.3f} | "
              f"a={params_v1['a_coeff']:.4f}")

    else:  # manual
        print(f"Menerapkan threshold manual GFI v1.0: τ = {tau_gfi_v1}")
        _, params_v1 = _apply_manual_threshold(GFIv1, tau_gfi_v1)
        print(f"  τ_norm={params_v1['tau_norm']:.3f} | "
              f"a_coeff={params_v1['a_coeff']:.4f}")

    a_v1 = params_v1["a_coeff"]
    WDv1 = np.maximum(0.0, (hr * a_v1) - H)
    flood_prone_v1 = GFIv1 >= params_v1["tau_real"]

    # ── 6. GFI v2.0 ───────────────────────────────────────────────────────
    print("\nMenghitung GFI v2.0...")
    Ariver_v2, H_v2, hr_v2, GFIv2 = compute_gfi_v2(
        demcon, flow_acc, channel,
        ROW_channel,    COL_channel,
        ROW_confluence, COL_confluence,
        a_v1, cellsize, n=n_exponent, max_iter=max_iter,
    )

    # ── 7. Kalibrasi GFI v2.0 ────────────────────────────────────────────
    fpr_v2 = tpr_v2 = None

    if cal == "roc":
        print("Kalibrasi GFI v2.0 via ROC...")
        _, fpr_v2, tpr_v2, params_v2 = roc_curve_maggiore(
            GFIv2, flood_sim, MargArea, step_size=roc_step,
        )
        print(f"  τ={params_v2['tau_norm']:.3f} | AUC={params_v2['auc']:.4f} | "
              f"FPR={params_v2['fpr_opt']:.3f} | TPR={params_v2['tpr_opt']:.3f} | "
              f"a={params_v2['a_coeff']:.4f}")

    else:  # manual
        tau_v2 = tau_gfi_v2 if tau_gfi_v2 is not None else tau_gfi_v1
        print(f"Menerapkan threshold manual GFI v2.0: τ = {tau_v2}")
        _, params_v2 = _apply_manual_threshold(GFIv2, tau_v2)
        print(f"  τ_norm={params_v2['tau_norm']:.3f} | "
              f"a_coeff={params_v2['a_coeff']:.4f}")

    a_v2 = params_v2["a_coeff"]
    WDv2 = np.maximum(0.0, (hr_v2 * a_v2) - H_v2)
    flood_prone_v2 = GFIv2 >= params_v2["tau_real"]

    # ── 8. Validasi (hanya jika mode ROC dan ada flood_depth_path) ────────
    metrics_v1 = metrics_v2 = None
    mask = observed_WD = None

    if cal == "roc" and flood_depth_path is not None:
        print("\nValidasi kedalaman banjir...")
        WD_sim      = resample_to_ref(flood_depth_path, profile)
        WD_sim      = np.where(WD_sim < 0, 0.0, WD_sim)
        mask        = MargArea > 0
        observed_WD = np.where(mask, WD_sim, np.nan)
        metrics_v1  = compute_validation_metrics(WDv1, observed_WD, mask)
        metrics_v2  = compute_validation_metrics(WDv2, observed_WD, mask)

        print("\n" + "=" * 52)
        print("  METRIK VALIDASI AKHIR")
        print("=" * 52)
        print(f"{'Metrik':<10} {'GFI v1.0':>14} {'GFI v2.0':>14}")
        print("-" * 40)
        for key in ["mse", "rmse", "r", "kge"]:
            print(f"{key.upper():<10} {metrics_v1[key]:>14.4f} "
                  f"{metrics_v2[key]:>14.4f}")
        print("=" * 52)

    # ── 9. Ringkasan parameter ────────────────────────────────────────────
    print("\n" + "=" * 52)
    print("  PARAMETER KALIBRASI")
    print("=" * 52)
    print(f"  GFI v1.0  τ (asli)  : {params_v1['tau_real']:.4f}")
    print(f"  GFI v1.0  a_coeff   : {params_v1['a_coeff']:.4f}")
    print(f"  GFI v2.0  τ (asli)  : {params_v2['tau_real']:.4f}")
    print(f"  GFI v2.0  a_coeff   : {params_v2['a_coeff']:.4f}")
    n_prone_v1 = int(flood_prone_v1.sum())
    n_prone_v2 = int(flood_prone_v2.sum())
    total      = int(np.sum(~np.isnan(GFIv1)))
    print(f"  Flood-prone v1.0    : {n_prone_v1:,} sel "
          f"({100*n_prone_v1/max(total,1):.1f}%)")
    print(f"  Flood-prone v2.0    : {n_prone_v2:,} sel "
          f"({100*n_prone_v2/max(total,1):.1f}%)")
    print("=" * 52)

    # ── 10. Simpan raster ─────────────────────────────────────────────────
    if save_rasters:
        print("\nMenyimpan raster output...")
        save_tif(GFIv1,  profile, os.path.join(out_dir, "GFI_v1.tif"))
        save_tif(GFIv2,  profile, os.path.join(out_dir, "GFI_v2.tif"))
        save_tif(WDv1,   profile, os.path.join(out_dir, "WD_v1.tif"))
        save_tif(WDv2,   profile, os.path.join(out_dir, "WD_v2.tif"))
        save_tif(flood_prone_v1.astype(np.float32), profile,
                 os.path.join(out_dir, "FloodProne_v1.tif"))
        save_tif(flood_prone_v2.astype(np.float32), profile,
                 os.path.join(out_dir, "FloodProne_v2.tif"))

    # ── 11. Visualisasi ───────────────────────────────────────────────────
    if show_plots:
        print("\nMembuat visualisasi...")
        if cal == "roc" and fpr_v1 is not None:
            plot_roc_comparison(fpr_v1, tpr_v1, fpr_v2, tpr_v2,
                                params_v1, params_v2,
                                out_dir=out_dir, show=show_plots)
        if flood_sim is not None and mask is not None:
            plot_spatial_accuracy(WDv2, flood_sim, mask, WDv1=WDv1,
                                  out_dir=out_dir, show=show_plots)
        if metrics_v1 is not None and observed_WD is not None:
            plot_water_depth_analysis(WDv1, WDv2, observed_WD, mask,
                                      metrics_v1, metrics_v2,
                                      out_dir=out_dir, show=show_plots)

    return dict(
        GFIv1          = GFIv1,
        GFIv2          = GFIv2,
        WDv1           = WDv1,
        WDv2           = WDv2,
        flood_prone_v1 = flood_prone_v1,
        flood_prone_v2 = flood_prone_v2,
        params_v1      = params_v1,
        params_v2      = params_v2,
        metrics_v1     = metrics_v1,
        metrics_v2     = metrics_v2,
        channel        = channel,
        S_matrix       = S_matrix,
        profile        = profile,
    )
