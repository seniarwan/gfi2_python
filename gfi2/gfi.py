"""
gfi2.gfi
--------
Komputasi Geomorphic Flood Index versi 1.0 dan 2.0.

GFI = ln(hr / H)
  H  = beda elevasi hillslope ke channel terdekat
  hr = a * Ariver^n   [Leopold & Maddock, 1953]
  a  = dikalibrasi via ROC (lihat gfi2.calibrate)
  n  = 0.354429752    (rata-rata literatur, Samela et al. 2018)

GFI 2.0 menambahkan iterasi backwater confluence:
  Untuk setiap segmen sungai, cek apakah WD di confluence downstream
  lebih besar dari WD saat ini. Jika iya, update properti Ariver & elevasi
  referensi. Ulang hingga max_iter atau konvergen.
  [Saavedra Navarro et al., DOI: 10.5281/zenodo.18903835]
"""

import numpy as np


# ---------------------------------------------------------------------------
def compute_gfi_v1(
    demcon:      np.ndarray,
    flow_acc:    np.ndarray,
    channel:     np.ndarray,
    ROW_channel: np.ndarray,
    COL_channel: np.ndarray,
    cellsize:    float,
    n:           float = 0.354429752,
):
    """
    Hitung GFI versi 1.0.

    GFI  = ln(hr / H)
    hr   = ((Ariver + 1) * cellsize² / 1e6)^n
    H    = demcon[piksel] - demcon[channel terdekat]
    Ariver = flow_acc[channel terdekat]

    Parameters
    ----------
    demcon      : 2D float32 — DEM terkondisi
    flow_acc    : 2D float32 — Flow Accumulation
    channel     : 2D bool    — mask piksel channel
    ROW_channel : 2D float32 — output hillslope_to_river_mapping (baris)
    COL_channel : 2D float32 — output hillslope_to_river_mapping (kolom)
    cellsize    : float      — ukuran piksel [m]
    n           : float      — eksponen scaling Leopold & Maddock

    Returns
    -------
    H      : 2D float32 — beda elevasi ke channel [m]
    Ariver : 2D float32 — contributing area channel terdekat [sel]
    hr     : 2D float32 — potensi muka air sungai [m]
    GFIv1  : 2D float32 — Geomorphic Flood Index v1.0
    """
    rows, cols = demcon.shape
    valid      = ~np.isnan(ROW_channel)

    r_idx    = ROW_channel[valid].astype(np.int32)
    c_idx    = COL_channel[valid].astype(np.int32)
    chan_lin = np.ravel_multi_index((r_idx, c_idx), (rows, cols))

    demcon_f   = demcon.ravel()
    flow_acc_f = flow_acc.ravel()
    valid_f    = valid.ravel()

    H      = np.full((rows, cols), np.nan, dtype=np.float32)
    Ariver = np.full((rows, cols), np.nan, dtype=np.float32)

    H.ravel()[valid_f]      = demcon_f[valid_f] - demcon_f[chan_lin]
    Ariver.ravel()[valid_f] = flow_acc_f[chan_lin]

    # Sel channel: H minimal, Ariver = diri sendiri
    H[channel]      = 0.001
    Ariver[channel] = flow_acc[channel]
    H[H <= 0]       = 0.001          # cegah log(negatif)

    hr    = (((Ariver + 1) * cellsize**2) / 1e6) ** n
    GFIv1 = np.real(np.log(hr / H)).astype(np.float32)

    print(f"  GFI v1.0  |  range: [{np.nanmin(GFIv1):.2f}, {np.nanmax(GFIv1):.2f}]")
    return H, Ariver, hr, GFIv1


# ---------------------------------------------------------------------------
def compute_gfi_v2(
    demcon:         np.ndarray,
    flow_acc:       np.ndarray,
    channel:        np.ndarray,
    ROW_channel:    np.ndarray,
    COL_channel:    np.ndarray,
    ROW_confluence: np.ndarray,
    COL_confluence: np.ndarray,
    a_gfi_v1:       float,
    cellsize:       float,
    n:              float = 0.354429752,
    max_iter:       int   = 6,
):
    """
    Hitung GFI versi 2.0 (confluence-aware).

    Algoritma:
      1. Inisialisasi WD tiap sel sungai dari GFI v1.0.
      2. Untuk setiap segmen sungai, hitung WD potensial jika
         menggunakan properti confluence downstream.
      3. Jika WD_confluence > WD_current → update Ariver & elevasi referensi.
      4. Ulangi hingga max_iter atau tidak ada sel yang berubah.
      5. Hillslope mewarisi properti sungai yang sudah diperbarui.

    Setara MATLAB: Sections 5.2–5.3 dalam GFI_v2_main.m

    Parameters
    ----------
    demcon, flow_acc, channel    : sama seperti compute_gfi_v1
    ROW_channel, COL_channel     : output Part 1
    ROW_confluence, COL_confluence: output Part 2
    a_gfi_v1  : float — koefisien a hasil kalibrasi GFI v1.0
    cellsize  : float — ukuran piksel [m]
    n         : float — eksponen Leopold & Maddock
    max_iter  : int   — batas iterasi backwater (default 6)

    Returns
    -------
    Ariver_v2 : 2D float32 — Ariver setelah update confluence
    H_v2      : 2D float32 — H setelah update confluence
    hr_v2     : 2D float32 — hr setelah update confluence
    GFIv2     : 2D float32 — Geomorphic Flood Index v2.0
    """
    print("GFI v2.0: Iterasi backwater confluence...")
    rows, cols    = demcon.shape
    valid         = ~np.isnan(ROW_channel)

    # ── Inisialisasi state sungai ────────────────────────────────────────
    Ariver_net    = flow_acc.copy().astype(np.float32)
    dem_river_net = demcon.copy().astype(np.float32)
    hr_net        = (((Ariver_net + 1) * cellsize**2) / 1e6) ** n
    WD_net        = np.maximum(0.0, (hr_net * a_gfi_v1) - 0.001)

    idx_river = np.flatnonzero(channel)
    curr_R    = ROW_confluence.copy()
    curr_C    = COL_confluence.copy()

    # ── Iterasi backwater ────────────────────────────────────────────────
    for k in range(max_iter):
        valid_jump = ~np.isnan(curr_R.ravel()[idx_river])
        curr_idx   = idx_river[valid_jump]
        if len(curr_idx) == 0:
            print(f"  Tidak ada sel yang bisa diperbarui. Konvergen di iterasi {k+1}.")
            break

        r_conf   = curr_R.ravel()[curr_idx].astype(np.int32)
        c_conf   = curr_C.ravel()[curr_idx].astype(np.int32)
        idx_next = np.ravel_multi_index((r_conf, c_conf), (rows, cols))

        # WD potensial menggunakan properti confluence
        A_next       = flow_acc.ravel()[idx_next]
        hr_next      = (((A_next + 1) * cellsize**2) / 1e6) ** n
        H_to_next    = demcon.ravel()[curr_idx] - demcon.ravel()[idx_next]
        H_to_next    = np.where(H_to_next <= 0, 0.001, H_to_next)
        WD_potential = np.maximum(0.0, (hr_next * a_gfi_v1) - H_to_next)

        # Update hanya jika WD confluence > WD saat ini
        improve  = WD_potential > WD_net.ravel()[curr_idx]
        idx_upd  = curr_idx[improve]
        idx_dst  = idx_next[improve]

        Ariver_net.ravel()[idx_upd]    = flow_acc.ravel()[idx_dst]
        dem_river_net.ravel()[idx_upd] = demcon.ravel()[idx_dst]
        WD_net.ravel()[idx_upd]        = WD_potential[improve]

        # Majukan pointer confluence untuk iterasi berikutnya
        curr_R.ravel()[idx_upd] = ROW_confluence.ravel()[idx_dst]
        curr_C.ravel()[idx_upd] = COL_confluence.ravel()[idx_dst]

        n_upd = int(improve.sum())
        print(f"  Iterasi {k+1}/{max_iter}: {n_upd:,} sel sungai diperbarui")
        if n_upd == 0:
            print("  Konvergen lebih awal.")
            break

    # ── Hillslope mewarisi properti sungai yang sudah diperbarui ─────────
    r_idx    = ROW_channel[valid].astype(np.int32)
    c_idx    = COL_channel[valid].astype(np.int32)
    chan_lin = np.ravel_multi_index((r_idx, c_idx), (rows, cols))
    valid_f  = np.flatnonzero(valid.ravel())

    Ariver_v2 = np.full((rows, cols), np.nan, dtype=np.float32)
    H_v2      = np.full((rows, cols), np.nan, dtype=np.float32)

    Ariver_v2.ravel()[valid_f] = Ariver_net.ravel()[chan_lin]
    H_v2.ravel()[valid_f]      = (demcon.ravel()[valid_f]
                                   - dem_river_net.ravel()[chan_lin])

    Ariver_v2[channel] = Ariver_net[channel]
    H_v2[H_v2 <= 0]   = 0.001

    hr_v2  = (((Ariver_v2 + 1) * cellsize**2) / 1e6) ** n
    GFIv2  = np.real(np.log(hr_v2 / H_v2)).astype(np.float32)

    print(f"  GFI v2.0  |  range: [{np.nanmin(GFIv2):.2f}, {np.nanmax(GFIv2):.2f}]")
    return Ariver_v2, H_v2, hr_v2, GFIv2
