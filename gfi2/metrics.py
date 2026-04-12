"""
gfi2.metrics
------------
Metrik validasi performa estimasi kedalaman banjir.
MSE, RMSE, Pearson r, Kling-Gupta Efficiency (KGE).
"""

import numpy as np
from scipy.stats import pearsonr


# ---------------------------------------------------------------------------
def compute_validation_metrics(
    WD_pred: np.ndarray,
    WD_obs:  np.ndarray,
    mask:    np.ndarray,
) -> dict:
    """
    Hitung metrik validasi antara kedalaman banjir prediksi dan observasi.

    Parameters
    ----------
    WD_pred : 2D np.ndarray — kedalaman banjir prediksi [m]
    WD_obs  : 2D np.ndarray — kedalaman banjir observasi / simulasi [m]
    mask    : 2D np.ndarray bool — area valid untuk evaluasi

    Returns
    -------
    dict dengan kunci:
        mse       : Mean Squared Error
        rmse      : Root Mean Squared Error
        r         : Pearson correlation coefficient
        kge       : Kling-Gupta Efficiency
        mean_pred : rata-rata prediksi
        mean_obs  : rata-rata observasi
        n_valid   : jumlah piksel valid yang dievaluasi
    """
    valid = mask & ~np.isnan(WD_obs) & ~np.isnan(WD_pred)
    obs   = WD_obs[valid].astype(np.float64)
    pred  = WD_pred[valid].astype(np.float64)

    if len(obs) < 2:
        return dict(mse=np.nan, rmse=np.nan, r=np.nan, kge=np.nan,
                    mean_pred=np.nan, mean_obs=np.nan, n_valid=int(len(obs)))

    mse  = float(np.mean((pred - obs) ** 2))
    rmse = float(np.sqrt(mse))
    r    = float(pearsonr(pred, obs)[0])

    # Kling-Gupta Efficiency (Gupta et al. 2009)
    mu_s,  mu_o  = pred.mean(), obs.mean()
    sig_s, sig_o = pred.std(),  obs.std()

    alpha = sig_s / sig_o if sig_o != 0 else np.nan
    beta  = mu_s  / mu_o  if mu_o  != 0 else np.nan
    kge   = float(1.0 - np.sqrt((r-1)**2 + (alpha-1)**2 + (beta-1)**2))

    return dict(
        mse       = mse,
        rmse      = rmse,
        r         = r,
        kge       = kge,
        mean_pred = float(pred.mean()),
        mean_obs  = float(obs.mean()),
        n_valid   = int(valid.sum()),
    )
