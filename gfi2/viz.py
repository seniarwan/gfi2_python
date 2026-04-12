"""
gfi2.viz
--------
Visualisasi hasil GFI 2.0.
Setara blok figure/subplot MATLAB di GFI_v2_main.m.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


# ---------------------------------------------------------------------------
def plot_roc_comparison(
    fpr_v1:   np.ndarray,
    tpr_v1:   np.ndarray,
    fpr_v2:   np.ndarray,
    tpr_v2:   np.ndarray,
    params_v1: dict,
    params_v2: dict,
    out_dir:  str = ".",
    show:     bool = True,
) -> str:
    """
    Plot kurva ROC GFI v1.0 vs v2.0.

    Parameters
    ----------
    fpr_v1, tpr_v1 : array — FPR/TPR GFI v1.0
    fpr_v2, tpr_v2 : array — FPR/TPR GFI v2.0
    params_v1/v2   : dict  — output roc_curve_maggiore (berisi 'auc')
    out_dir        : str   — folder output
    show           : bool  — tampilkan plot di layar

    Returns
    -------
    str — path file PNG yang disimpan
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    ax.plot(fpr_v1, tpr_v1, "b--", lw=1.5,
            label=f"GFI v1.0  (AUC = {params_v1['auc']:.3f})")
    ax.plot(fpr_v2, tpr_v2, "r-",  lw=2.0,
            label=f"GFI v2.0  (AUC = {params_v2['auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k:", lw=1.0, label="Random guess")

    # Tandai titik optimal masing-masing versi
    ax.scatter([params_v1["fpr_opt"]], [params_v1["tpr_opt"]],
               color="blue", zorder=5, s=60,
               label=f"v1.0 opt  τ={params_v1['tau_norm']:.3f}")
    ax.scatter([params_v2["fpr_opt"]], [params_v2["tpr_opt"]],
               color="red",  zorder=5, s=60,
               label=f"v2.0 opt  τ={params_v2['tau_norm']:.3f}")

    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC Curve — GFI v1.0 vs GFI v2.0")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_aspect("equal"); ax.grid(True, alpha=0.4)

    plt.tight_layout()
    path = os.path.join(out_dir, "ROC_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    print(f"  Tersimpan: {path}")
    return path


# ---------------------------------------------------------------------------
def plot_spatial_accuracy(
    WDv2:      np.ndarray,
    flood_sim: np.ndarray,
    mask:      np.ndarray,
    WDv1:      np.ndarray = None,
    out_dir:   str  = ".",
    show:      bool = True,
) -> str:
    """
    Plot peta akurasi spasial klasifikasi banjir.

    Warna:
      Hijau = True Positive  (hit)
      Biru  = False Positive (overestimasi)
      Merah = False Negative (terlewat)

    Jika WDv1 disertakan, panel kedua menampilkan piksel yang
    berhasil diperbaiki oleh v2.0 (confluence correction).
    """
    cont = np.zeros(WDv2.shape, dtype=np.float32)
    cont[(WDv2 > 0) &  flood_sim & mask] = 3   # TP — hijau
    cont[(WDv2 > 0) & ~flood_sim & mask] = 2   # FP — biru
    cont[(WDv2 == 0) & flood_sim & mask] = 1   # FN — merah

    ncols = 2 if WDv1 is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    cmap = mcolors.ListedColormap(["red", "blue", "green"])
    norm = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap.N)

    im = axes[0].imshow(
        np.where(cont > 0, cont, np.nan),
        cmap=cmap, norm=norm,
    )
    cbar = plt.colorbar(im, ax=axes[0], ticks=[1, 2, 3], shrink=0.7)
    cbar.ax.set_yticklabels(["FN (miss)", "FP (over)", "TP (hit)"])
    axes[0].set_title("Akurasi Spasial — GFI v2.0")
    axes[0].axis("off")

    if WDv1 is not None:
        gain = (WDv2 > 0) & (WDv1 == 0) & flood_sim & mask
        axes[1].imshow(
            np.where(gain, 1.0, np.nan),
            cmap="Oranges", vmin=0, vmax=1,
        )
        n_gain = int(gain.sum())
        axes[1].set_title(f"Koreksi Confluence (v2.0)\n{n_gain:,} piksel baru terdeteksi")
        axes[1].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "spatial_accuracy.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    print(f"  Tersimpan: {path}")
    return path


# ---------------------------------------------------------------------------
def plot_water_depth_analysis(
    WDv1:        np.ndarray,
    WDv2:        np.ndarray,
    observed_WD: np.ndarray,
    mask:        np.ndarray,
    metrics_v1:  dict,
    metrics_v2:  dict,
    out_dir:     str  = ".",
    show:        bool = True,
) -> str:
    """
    Panel analisis kedalaman banjir: scatter korelasi + peta error absolut.
    """
    valid = mask & ~np.isnan(observed_WD)
    obs   = observed_WD[valid]
    p2    = WDv2[valid]
    err1  = np.where(mask, np.abs(WDv1 - observed_WD), np.nan)
    err2  = np.where(mask, np.abs(WDv2 - observed_WD), np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel kiri: scatter correlation ──────────────────────────────────
    lim = max(float(np.nanmax(obs)), float(np.nanmax(p2))) * 1.05
    axes[0].scatter(obs, p2, s=2, alpha=0.2, color="steelblue", rasterized=True)
    axes[0].plot([0, lim], [0, lim], "r--", lw=1)
    axes[0].set_xlabel("WD Observasi [m]")
    axes[0].set_ylabel("WD GFI v2.0 [m]")
    axes[0].set_title("Korelasi Kedalaman — v2.0")
    axes[0].set_xlim(0, lim); axes[0].set_ylim(0, lim)
    axes[0].set_aspect("equal"); axes[0].grid(True, alpha=0.4)
    axes[0].text(
        0.05, 0.82,
        f"R    = {metrics_v2['r']:.3f}\n"
        f"RMSE = {metrics_v2['rmse']:.3f} m\n"
        f"KGE  = {metrics_v2['kge']:.3f}",
        transform=axes[0].transAxes, fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # ── Panel tengah: error absolut v1.0 ────────────────────────────────
    im1 = axes[1].imshow(err1, cmap="hot_r", vmin=0, vmax=2)
    plt.colorbar(im1, ax=axes[1], label="Error [m]", shrink=0.8)
    axes[1].set_title(
        f"Error Absolut — GFI v1.0\n"
        f"RMSE={metrics_v1['rmse']:.3f} m  |  "
        f"Mean err={float(np.nanmean(err1)):.3f} m"
    )
    axes[1].axis("off")

    # ── Panel kanan: error absolut v2.0 ─────────────────────────────────
    im2 = axes[2].imshow(err2, cmap="hot_r", vmin=0, vmax=2)
    plt.colorbar(im2, ax=axes[2], label="Error [m]", shrink=0.8)
    axes[2].set_title(
        f"Error Absolut — GFI v2.0\n"
        f"RMSE={metrics_v2['rmse']:.3f} m  |  "
        f"Mean err={float(np.nanmean(err2)):.3f} m"
    )
    axes[2].axis("off")

    plt.tight_layout()
    path = os.path.join(out_dir, "water_depth_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close()
    print(f"  Tersimpan: {path}")
    return path
