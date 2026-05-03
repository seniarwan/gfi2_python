"""
gfi2.gfi
--------
Komputasi Geomorphic Flood Index versi 1.0 dan 2.0.

GFI = ln(hr / H)
  H      = beda elevasi hillslope ke channel terdekat [m]
  hr     = a * Ariver^n   [Leopold & Maddock, 1953]
  Ariver = contributing area channel terdekat [km²]
  a      = dikalibrasi via ROC (lihat gfi2.calibrate)
  n      = 0.354429752 (rata-rata literatur, Samela et al. 2018)

GFI 2.0 menambahkan iterasi backwater confluence:
  Untuk setiap segmen sungai, cek apakah WD di confluence downstream
  lebih besar dari WD saat ini. Jika iya, update Ariver & elevasi
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
    hr   = ((Ariver * cellsize²) / 1e6)^n      [Leopold & Maddock, 1953]
    H    = demcon[piksel] - demcon[channel terdekat]
    Ariver = flow_acc[channel terdekat]  (jumlah sel)

    Mengikuti struktur notebook Manfreda:
      - H  dihitung dari elevasi DEM piksel dikurangi elevasi channel
        terdekat (bukan jarak aliran, melainkan beda elevasi vertikal)
      - hr dihitung dari flow accumulation channel terdekat,
        dikonversi ke km² dengan: flow_acc × cellsize² / 1e6

    Parameters
    ----------
    demcon      : 2D float32 — DEM terkondisi
    flow_acc    : 2D float32 — Flow Accumulation (jumlah sel upstream)
    channel     : 2D bool    — mask piksel channel
    ROW_channel : 2D float32 — output Part 1: baris channel terdekat
    COL_channel : 2D float32 — output Part 1: kolom channel terdekat
    cellsize    : float      — ukuran piksel dalam meter
    n           : float      — eksponen scaling Leopold & Maddock

    Returns
    -------
    H      : 2D float32 — beda elevasi ke channel terdekat [m]
    Ariver : 2D float32 — flow acc channel terdekat [sel]
    hr     : 2D float32 — estimasi kedalaman bankfull [m]
    GFIv1  : 2D float32 — Geomorphic Flood Index v1.0
    """
    rows, cols = demcon.shape

    # Piksel yang berhasil terpetakan ke channel
    valid = ~np.isnan(ROW_channel) & ~channel

    r_idx    = ROW_channel[valid].astype(np.int32)
    c_idx    = COL_channel[valid].astype(np.int32)
    chan_lin = np.ravel_multi_index((r_idx, c_idx), (rows, cols))

    demcon_f   = demcon.ravel()
    flow_acc_f = flow_acc.ravel()
    valid_f    = valid.ravel()

    H      = np.full((rows, cols), np.nan, dtype=np.float32)
    Ariver = np.full((rows, cols), np.nan, dtype=np.float32)

    # Hillslope: ambil nilai dari channel terdekat
    H.ravel()[valid_f]      = demcon_f[valid_f] - demcon_f[chan_lin]
    Ariver.ravel()[valid_f] = flow_acc_f[chan_lin]

    # Channel: H minimal (beda elevasi dengan diri sendiri = 0 → clip ke 0.001)
    H[channel]      = 0.001
    Ariver[channel] = flow_acc[channel]

    # Clip H negatif (bisa terjadi di area yang lebih rendah dari channel,
    # misalnya di area depresi yang tidak terisi sempurna)
    H[H <= 0] = 0.001

    # hr = (Ariver × cellsize² / 1e6)^n  → Ariver dalam km²
    hr    = ((Ariver * cellsize**2) / 1e6) ** n
    GFIv1 = np.real(np.log(hr / H)).astype(np.float32)

    n_valid = int(valid.sum())
    print(f"  GFI v1.0  |  piksel valid: {n_valid:,}  |  "
          f"range: [{np.nanmin(GFIv1):.2f}, {np.nanmax(GFIv1):.2f}]")
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
    Hitung GFI versi 2.0 (confluence backwater-aware).

    Algoritma iterasi backwater confluence:
      1. Inisialisasi WD setiap sel sungai dari properti GFI v1.0.
      2. Untuk setiap segmen sungai (non-main-river), hitung WD potensial
         menggunakan properti confluence downstream-nya.
      3. Jika WD_confluence > WD_current → update Ariver & elevasi referensi
         sungai dengan nilai dari confluence.
      4. Ulangi hingga max_iter atau tidak ada sel yang berubah (konvergen).
      5. Hillslope mewarisi properti sungai yang sudah diperbarui.

    Setara MATLAB: GFI_v2_main.m Sections 5.2–5.3
    [Saavedra Navarro et al., DOI: 10.5281/zenodo.18903835]

    Parameters
    ----------
    demcon, flow_acc, channel : sama seperti compute_gfi_v1
    ROW_channel, COL_channel  : output Part 1 (hillslope_to_river_mapping)
    ROW_confluence, COL_confluence : output Part 2 (river_to_confluence_mapping)
    a_gfi_v1  : float — koefisien a dari kalibrasi GFI v1.0
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
    rows, cols = demcon.shape

    # ── State awal: properti tiap sel sungai dari flow_acc asli ──────────
    Ariver_net    = flow_acc.copy().astype(np.float32)
    dem_river_net = demcon.copy().astype(np.float32)
    hr_net        = ((Ariver_net * cellsize**2) / 1e6) ** n
    WD_net        = np.maximum(0.0, (hr_net * a_gfi_v1) - 0.001)

    idx_river = np.flatnonzero(channel)
    curr_R    = ROW_confluence.copy()
    curr_C    = COL_confluence.copy()

    # ── Iterasi ───────────────────────────────────────────────────────────
    for k in range(max_iter):
        valid_jump = ~np.isnan(curr_R.ravel()[idx_river])
        curr_idx   = idx_river[valid_jump]
        if len(curr_idx) == 0:
            print(f"  Konvergen di iterasi {k+1} (tidak ada sel tersisa).")
            break

        r_conf   = curr_R.ravel()[curr_idx].astype(np.int32)
        c_conf   = curr_C.ravel()[curr_idx].astype(np.int32)
        idx_next = np.ravel_multi_index((r_conf, c_conf), (rows, cols))

        A_next       = flow_acc.ravel()[idx_next]
        hr_next      = ((A_next * cellsize**2) / 1e6) ** n
        H_to_next    = demcon.ravel()[curr_idx] - demcon.ravel()[idx_next]
        H_to_next    = np.where(H_to_next <= 0, 0.001, H_to_next)
        WD_potential = np.maximum(0.0, (hr_next * a_gfi_v1) - H_to_next)

        improve  = WD_potential > WD_net.ravel()[curr_idx]
        idx_upd  = curr_idx[improve]
        idx_dst  = idx_next[improve]

        Ariver_net.ravel()[idx_upd]    = flow_acc.ravel()[idx_dst]
        dem_river_net.ravel()[idx_upd] = demcon.ravel()[idx_dst]
        WD_net.ravel()[idx_upd]        = WD_potential[improve]
        curr_R.ravel()[idx_upd]        = ROW_confluence.ravel()[idx_dst]
        curr_C.ravel()[idx_upd]        = COL_confluence.ravel()[idx_dst]

        n_upd = int(improve.sum())
        print(f"  Iterasi {k+1}/{max_iter}: {n_upd:,} sel sungai diperbarui")
        if n_upd == 0:
            print("  Konvergen lebih awal.")
            break

    # ── Hillslope mewarisi properti sungai yang sudah diperbarui ─────────
    valid   = ~np.isnan(ROW_channel) & ~channel
    r_idx   = ROW_channel[valid].astype(np.int32)
    c_idx   = COL_channel[valid].astype(np.int32)
    chan_lin = np.ravel_multi_index((r_idx, c_idx), (rows, cols))
    valid_f  = np.flatnonzero(valid.ravel())

    Ariver_v2 = np.full((rows, cols), np.nan, dtype=np.float32)
    H_v2      = np.full((rows, cols), np.nan, dtype=np.float32)

    Ariver_v2.ravel()[valid_f] = Ariver_net.ravel()[chan_lin]
    H_v2.ravel()[valid_f]      = (demcon.ravel()[valid_f]
                                   - dem_river_net.ravel()[chan_lin])

    Ariver_v2[channel] = Ariver_net[channel]
    H_v2[channel]      = 0.001   # piksel channel: H = 0 → clip ke 0.001
    H_v2[H_v2 <= 0]   = 0.001

    hr_v2  = ((Ariver_v2 * cellsize**2) / 1e6) ** n
    GFIv2  = np.real(np.log(hr_v2 / H_v2)).astype(np.float32)

    print(f"  GFI v2.0  |  range: [{np.nanmin(GFIv2):.2f}, {np.nanmax(GFIv2):.2f}]")
    return Ariver_v2, H_v2, hr_v2, GFIv2
