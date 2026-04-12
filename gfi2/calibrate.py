"""
gfi2.calibrate
--------------
Kalibrasi threshold biner via kurva ROC.
Setara MATLAB: ROCcurve_maggiore.m + areaundercurve.m
"""

import numpy as np


# ---------------------------------------------------------------------------
def area_under_curve(fpr: np.ndarray, tpr: np.ndarray) -> float:
    """
    Hitung AUC (Area Under the ROC Curve) dengan metode trapezoid.
    Setara MATLAB: areaundercurve(FPR, TPR).

    Parameters
    ----------
    fpr : 1D np.ndarray — False Positive Rate
    tpr : 1D np.ndarray — True Positive Rate

    Returns
    -------
    float — nilai AUC [0, 1]
    """
    order = np.argsort(fpr)
    x     = np.append(fpr[order], 1.0)
    y     = np.append(tpr[order], 1.0)
    return float(np.trapz(y, x))


# ---------------------------------------------------------------------------
def roc_curve_maggiore(
    matrix:    np.ndarray,
    risk_map:  np.ndarray,
    mask:      np.ndarray,
    step_size: float = 0.005,
):
    """
    Kalibrasi ROC — setara MATLAB ROCcurve_maggiore.m.

    Langkah:
      1. Normalisasi GFI ke rentang [-1, 1] (min-max).
      2. Sweep threshold τ dari -1 ke +1.
      3. Pada setiap τ: hitung FPR dan TPR dalam area mask.
      4. Pilih τ optimal yang meminimalkan F = FPR + (1 - TPR),
         yaitu meminimalkan jarak ke titik sempurna (FPR=0, TPR=1).
      5. Denormalisasi τ ke nilai GFI asli → hitung a = exp(-τ_real).

    Parameters
    ----------
    matrix    : 2D np.ndarray — GFI index (belum ternormalisasi)
    risk_map  : 2D np.ndarray — peta referensi banjir (0/1 atau bool)
    mask      : 2D np.ndarray — area marginal hazard (1 = di dalam)
    step_size : float         — langkah sweep threshold (default 0.005)
                                Nilai kecil → lebih presisi, lebih lambat.

    Returns
    -------
    matrix_norm : 2D float32   — GFI ternormalisasi [-1, 1]
    fpr_arr     : 1D float64   — FPR untuk setiap threshold
    tpr_arr     : 1D float64   — TPR untuk setiap threshold
    params      : dict         — parameter optimal:
        tau_norm  : float — threshold ternormalisasi optimal
        tau_real  : float — threshold dalam satuan GFI asli
        fpr_opt   : float — FPR pada threshold optimal
        tpr_opt   : float — TPR pada threshold optimal
        f_optim   : float — nilai F minimum (FPR + FNR)
        auc       : float — Area Under the Curve
        a_coeff   : float — koefisien a = 1/exp(tau_real)
    """
    # Normalisasi min-max → [-1, 1]
    mn, mx      = float(np.nanmin(matrix)), float(np.nanmax(matrix))
    matrix_norm = (2.0 * ((matrix - mn) / (mx - mn) - 0.5)).astype(np.float32)

    thresholds  = np.arange(-1.0, 1.0 + step_size, step_size)
    n_steps     = len(thresholds)
    fpr_arr     = np.zeros(n_steps, dtype=np.float64)
    tpr_arr     = np.zeros(n_steps, dtype=np.float64)

    mask_bool   = (mask == 1) & ~np.isnan(matrix_norm)
    risk_bool   = risk_map.astype(bool)

    F_optim = 10.0
    params  = {}

    for i, t in enumerate(thresholds):
        R  = matrix_norm >= t

        fp = int(np.sum( R & ~risk_bool & mask_bool))
        fn = int(np.sum(~R &  risk_bool & mask_bool))
        vn = int(np.sum(~R & ~risk_bool & mask_bool))
        vp = int(np.sum( R &  risk_bool & mask_bool))

        fpr = fp / (fp + vn) if (fp + vn) > 0 else 0.0
        fnr = fn / (fn + vp) if (fn + vp) > 0 else 0.0
        tpr = 1.0 - fnr

        fpr_arr[i] = fpr
        tpr_arr[i] = tpr

        # Minimasi jarak ke titik sempurna (FPR=0, TPR=1)
        F = fpr + (1.0 - tpr)
        if F < F_optim:
            F_optim  = F
            tau_real = float(((t + 1.0) / 2.0) * (mx - mn) + mn)
            params   = dict(
                tau_norm = float(t),
                tau_real = tau_real,
                fpr_opt  = float(fpr),
                tpr_opt  = float(tpr),
                f_optim  = float(F_optim),
            )

    params["auc"]     = area_under_curve(fpr_arr, tpr_arr)
    params["a_coeff"] = float(1.0 / np.exp(params["tau_real"]))

    return matrix_norm, fpr_arr, tpr_arr, params
