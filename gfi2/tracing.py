"""
gfi2.tracing
------------
D8 flow tracing untuk dua tujuan:
  Part 1 — Hillslope → channel terdekat   (setara MATLAB parfor Part 1)
  Part 2 — Channel   → confluence berikutnya (setara MATLAB parfor Part 2)

Fungsi trace_flow_step() dikompilasi dengan Numba (@njit) agar performanya
mendekati kecepatan C, menggantikan loop pixel-per-pixel MATLAB.
Paralelisasi baris menggunakan joblib (pengganti parfor).
"""

import numpy as np
from numba import njit
from joblib import Parallel, delayed
import time


# ---------------------------------------------------------------------------
# CORE: satu langkah tracing D8  (Numba-compiled)
# Setara MATLAB: traceFlow.m
# ---------------------------------------------------------------------------

@njit
def trace_flow_step(
    flow_dir: np.ndarray,
    x: int,
    y: int,
    Ld: float,
    cellsize: float,
):
    """
    Satu langkah D8 flow tracing (Numba JIT-compiled).
    Setara MATLAB traceFlow.m, termasuk deteksi 2-cell loop (sinkhole/flat).

    Parameters
    ----------
    flow_dir : 2D np.ndarray — Flow Direction ESRI D8
    x        : int           — baris saat ini (0-based)
    y        : int           — kolom saat ini (0-based)
    Ld       : float         — jarak kumulatif [m]
    cellsize : float         — ukuran piksel [m]

    Returns
    -------
    (x_baru, y_baru, Ld_baru) — semua NaN jika loop atau arah tidak valid
    """
    fd = int(flow_dir[x, y])

    # Lookup: direction → (drow, dcol, faktor jarak)
    if   fd == 1:   dr, dc, df =  0,  1, 1.0
    elif fd == 128: dr, dc, df = -1,  1, 1.41421356
    elif fd == 64:  dr, dc, df = -1,  0, 1.0
    elif fd == 32:  dr, dc, df = -1, -1, 1.41421356
    elif fd == 16:  dr, dc, df =  0, -1, 1.0
    elif fd == 8:   dr, dc, df =  1, -1, 1.41421356
    elif fd == 4:   dr, dc, df =  1,  0, 1.0
    elif fd == 2:   dr, dc, df =  1,  1, 1.41421356
    else:
        return np.nan, np.nan, np.nan   # arah tidak valid

    nx, ny = x + dr, y + dc

    # Deteksi 2-cell loop: cek apakah tetangga mengarah balik ke kita
    if   fd == 1:   opp = 16
    elif fd == 2:   opp = 32
    elif fd == 4:   opp = 64
    elif fd == 8:   opp = 128
    elif fd == 16:  opp = 1
    elif fd == 32:  opp = 2
    elif fd == 64:  opp = 4
    elif fd == 128: opp = 8
    else:           opp = 0

    rows, cols = flow_dir.shape
    if (0 <= nx < rows and 0 <= ny < cols and
            opp > 0 and int(flow_dir[nx, ny]) == opp):
        return np.nan, np.nan, np.nan   # 2-cell loop terdeteksi

    return float(nx), float(ny), Ld + df * cellsize


# ---------------------------------------------------------------------------
# PART 1: satu baris — hillslope → channel terdekat
# ---------------------------------------------------------------------------

def _trace_hillslope_row(
    i: int,
    demcon:   np.ndarray,
    flow_dir: np.ndarray,
    channel:  np.ndarray,
    a: int,
    b: int,
    cellsize: float,
):
    """
    Trace setiap piksel non-channel di baris i ke sel channel terdekat.
    Dipanggil secara paralel per baris oleh hillslope_to_river_mapping().
    """
    fila_D   = np.full(b, np.nan, dtype=np.float32)
    fila_ROW = np.full(b, np.nan, dtype=np.float32)
    fila_COL = np.full(b, np.nan, dtype=np.float32)

    for j in range(1, b - 1):
        if channel[i, j] or np.isnan(demcon[i, j]):
            continue

        xi, yi, Ld = i, j, 0.0

        for _ in range(100_000):          # batas aman anti-infinite loop
            if channel[xi, yi]:
                break
            fd = flow_dir[xi, yi]
            if np.isnan(fd) or fd == 0:
                xi = -1; break
            nx, ny, Ld = trace_flow_step(flow_dir, xi, yi, Ld, cellsize)
            if np.isnan(nx):
                xi = -1; break
            xi, yi = int(nx), int(ny)
            if xi <= 0 or xi >= a-1 or yi <= 0 or yi >= b-1:
                xi = -1; break

        if xi > 0 and channel[xi, yi]:
            fila_D[j]   = Ld
            fila_ROW[j] = xi
            fila_COL[j] = yi

    return fila_D, fila_ROW, fila_COL


# ---------------------------------------------------------------------------
# PART 2: satu baris — channel → confluence berikutnya
# ---------------------------------------------------------------------------

def _trace_confluence_row(
    i: int,
    flow_dir:  np.ndarray,
    channel:   np.ndarray,
    S_matrix:  np.ndarray,
    max_order: int,
    a: int,
    b: int,
    cellsize:  float,
):
    """
    Trace setiap piksel channel di baris i ke confluence berikutnya
    (titik perubahan Strahler order).
    Dipanggil secara paralel per baris oleh river_to_confluence_mapping().
    """
    fila_D   = np.full(b, np.nan, dtype=np.float32)
    fila_ROW = np.full(b, np.nan, dtype=np.float32)
    fila_COL = np.full(b, np.nan, dtype=np.float32)

    for j in range(1, b - 1):
        if not channel[i, j]:
            continue
        so = S_matrix[i, j]
        if so == 0 or so >= max_order:      # skip: non-channel atau main river
            continue

        xi, yi, Ld = i, j, 0.0

        for _ in range(200_000):
            fd = flow_dir[xi, yi]
            if np.isnan(fd) or fd == 0:
                break
            nx, ny, Ld = trace_flow_step(flow_dir, xi, yi, Ld, cellsize)
            if np.isnan(nx):
                break
            nxi, nyi = int(nx), int(ny)
            if nxi <= 0 or nxi >= a-1 or nyi <= 0 or nyi >= b-1:
                break
            if S_matrix[nxi, nyi] != so:    # order berubah = confluence
                xi, yi = nxi, nyi
                break
            xi, yi = nxi, nyi

        fila_D[j]   = Ld
        fila_ROW[j] = xi
        fila_COL[j] = yi

    return fila_D, fila_ROW, fila_COL


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def hillslope_to_river_mapping(
    demcon:   np.ndarray,
    flow_dir: np.ndarray,
    channel:  np.ndarray,
    cellsize: float,
    n_jobs:   int = -1,
):
    """
    Part 1: Untuk setiap piksel non-channel, trace aliran ke sel channel
    terdekat dan simpan jarak + koordinatnya.
    Setara MATLAB: Section 3 (parfor Part 1).

    Parameters
    ----------
    demcon   : 2D np.ndarray — DEM terkondisi
    flow_dir : 2D np.ndarray — Flow Direction ESRI D8
    channel  : 2D np.ndarray bool — mask channel
    cellsize : float         — ukuran piksel [m]
    n_jobs   : int           — jumlah core paralel (-1 = semua)

    Returns
    -------
    D_to_channel : 2D float32 — jarak ke channel terdekat [m]
    ROW_channel  : 2D float32 — baris channel terdekat
    COL_channel  : 2D float32 — kolom channel terdekat
    """
    print("Part 1: Hillslope → channel terdekat...")
    t0   = time.time()
    a, b = demcon.shape

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_trace_hillslope_row)(i, demcon, flow_dir, channel, a, b, cellsize)
        for i in range(1, a - 1)
    )

    D_rows, R_rows, C_rows = zip(*results)
    nan_row      = np.full(b, np.nan, dtype=np.float32)
    D_to_channel = np.vstack([nan_row, *D_rows, nan_row])
    ROW_channel  = np.vstack([nan_row, *R_rows, nan_row])
    COL_channel  = np.vstack([nan_row, *C_rows, nan_row])

    n_mapped = int(~np.isnan(ROW_channel).sum())
    print(f"  Selesai dalam {time.time()-t0:.1f}s  |  {n_mapped:,} piksel terpetakan")
    return D_to_channel, ROW_channel, COL_channel


def river_to_confluence_mapping(
    flow_dir:  np.ndarray,
    channel:   np.ndarray,
    S_matrix:  np.ndarray,
    max_order: int,
    cellsize:  float,
    n_jobs:    int = -1,
):
    """
    Part 2: Untuk setiap piksel channel non-main-river, trace ke confluence
    berikutnya (titik di mana Strahler order berubah).
    Setara MATLAB: Section 4 (parfor Part 2).

    Parameters
    ----------
    flow_dir  : 2D np.ndarray — Flow Direction ESRI D8
    channel   : 2D np.ndarray bool — mask channel
    S_matrix  : 2D np.ndarray int  — Strahler order
    max_order : int               — order maksimum (main river)
    cellsize  : float             — ukuran piksel [m]
    n_jobs    : int               — jumlah core paralel (-1 = semua)

    Returns
    -------
    D_to_confluence : 2D float32 — jarak ke confluence [m]
    ROW_confluence  : 2D float32 — baris confluence
    COL_confluence  : 2D float32 — kolom confluence
    """
    print("Part 2: Channel → confluence berikutnya...")
    t0   = time.time()
    a, b = flow_dir.shape

    results = Parallel(n_jobs=n_jobs, backend="loky", verbose=0)(
        delayed(_trace_confluence_row)(
            i, flow_dir, channel, S_matrix, max_order, a, b, cellsize
        )
        for i in range(1, a - 1)
    )

    D_rows, R_rows, C_rows = zip(*results)
    nan_row         = np.full(b, np.nan, dtype=np.float32)
    D_to_confluence = np.vstack([nan_row, *D_rows, nan_row])
    ROW_confluence  = np.vstack([nan_row, *R_rows, nan_row])
    COL_confluence  = np.vstack([nan_row, *C_rows, nan_row])

    n_mapped = int(~np.isnan(ROW_confluence).sum())
    print(f"  Selesai dalam {time.time()-t0:.1f}s  |  {n_mapped:,} segmen terpetakan")
    return D_to_confluence, ROW_confluence, COL_confluence
