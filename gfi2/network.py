"""
gfi2.network
------------
Ekstraksi jaringan drainase dan penghitungan Strahler stream order.
Setara MATLAB: gradient8(), streamorder(..., 'Strahler'), CHANNEL_mask.

Encoding flow direction yang didukung:
  'esri'   : 1=E, 2=SE,  4=S,  8=SW, 16=W, 32=NW, 64=N,  128=NE
  'taudem' : 1=E, 2=NE,  3=N,  4=NW,  5=W,  6=SW,  7=S,    8=SE
             (identik dengan encoding notebook Manfreda / TauDEM)
"""

import numpy as np

# ---------------------------------------------------------------------------
# Lookup tables encoding → (drow, dcol)
# ---------------------------------------------------------------------------

_DIR_MAP = {
    "esri": {
        1:   ( 0,  1),    # E
        2:   ( 1,  1),    # SE
        4:   ( 1,  0),    # S
        8:   ( 1, -1),    # SW
        16:  ( 0, -1),    # W
        32:  (-1, -1),    # NW
        64:  (-1,  0),    # N
        128: (-1,  1),    # NE
    },
    "taudem": {
        1:   ( 0,  1),    # E
        2:   (-1,  1),    # NE
        3:   (-1,  0),    # N
        4:   (-1, -1),    # NW
        5:   ( 0, -1),    # W
        6:   ( 1, -1),    # SW
        7:   ( 1,  0),    # S
        8:   ( 1,  1),    # SE
    },
}


def _resolve_encoding(encoding: str) -> dict:
    enc = encoding.lower()
    if enc in ("taudem", "manfreda"):
        return _DIR_MAP["taudem"]
    elif enc == "esri":
        return _DIR_MAP["esri"]
    else:
        raise ValueError(
            f"Encoding '{encoding}' tidak dikenal. "
            "Gunakan 'esri' atau 'taudem'."
        )


# ---------------------------------------------------------------------------
def gradient8(Z: np.ndarray, cellsize: float) -> np.ndarray:
    """
    Kemiringan maksimum 8-arah (tangen).
    Setara MATLAB gradient8(Z, cellsize).
    Tidak bergantung pada encoding flow direction.
    """
    rows, cols = Z.shape
    Zp         = np.pad(Z.astype(np.float64), 1, mode="edge")
    diag       = cellsize * np.sqrt(2)

    shifts = [(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)]
    dists  = [cellsize, diag, cellsize, diag,
              cellsize, diag, cellsize, diag]

    G      = np.zeros((rows, cols), dtype=np.float32)
    center = Zp[1:-1, 1:-1]
    for (dr, dc), dist in zip(shifts, dists):
        neigh = Zp[1+dr : rows+1+dr, 1+dc : cols+1+dc]
        G     = np.maximum(G, np.abs(center - neigh) / dist)
    return G


# ---------------------------------------------------------------------------
def compute_strahler_order(
    flow_dir:     np.ndarray,
    channel_mask: np.ndarray,
    encoding:     str = "esri",
) -> np.ndarray:
    """
    Hitung Strahler stream order untuk piksel channel via BFS downstream.

    Aturan Strahler:
      - Headwater (tanpa upstream channel) → order 1
      - Dua stream order sama bertemu → order +1
      - Stream order lebih rendah bergabung ke order lebih tinggi
        → order tetap (dipertahankan yang lebih tinggi)

    Parameters
    ----------
    flow_dir     : 2D np.ndarray — Flow Direction raster
    channel_mask : 2D np.ndarray bool — True di piksel channel
    encoding     : str — 'esri' atau 'taudem'

    Returns
    -------
    S : 2D np.ndarray int32 — Strahler order (0 = non-channel)
    """
    DIR_MAP    = _resolve_encoding(encoding)
    rows, cols = flow_dir.shape
    in_deg     = np.zeros((rows, cols), dtype=np.int32)

    # Hitung in-degree: berapa upstream channel neighbour tiap sel
    for r in range(rows):
        for c in range(cols):
            if not channel_mask[r, c]:
                continue
            fd = int(flow_dir[r, c]) if not np.isnan(flow_dir[r, c]) else 0
            if fd not in DIR_MAP:
                continue
            dr, dc = DIR_MAP[fd]
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and channel_mask[nr, nc]:
                in_deg[nr, nc] += 1

    # Inisialisasi headwater (in_deg == 0) dengan order 1
    S     = np.zeros((rows, cols), dtype=np.int32)
    queue = []
    for r in range(rows):
        for c in range(cols):
            if channel_mask[r, c] and in_deg[r, c] == 0:
                S[r, c] = 1
                queue.append((r, c))

    # BFS ke arah hilir
    while queue:
        r, c = queue.pop(0)
        fd   = int(flow_dir[r, c]) if not np.isnan(flow_dir[r, c]) else 0
        if fd not in DIR_MAP:
            continue
        dr, dc = DIR_MAP[fd]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < rows and 0 <= nc < cols):
            continue
        if not channel_mask[nr, nc]:
            continue

        in_deg[nr, nc] -= 1

        if S[nr, nc] == 0:
            S[nr, nc] = S[r, c]
        elif S[nr, nc] == S[r, c]:
            S[nr, nc] += 1
        else:
            S[nr, nc] = max(S[nr, nc], S[r, c])

        if in_deg[nr, nc] == 0:
            queue.append((nr, nc))

    S[~channel_mask] = 0
    return S


# ---------------------------------------------------------------------------
def extract_channel_network(
    flow_acc:  np.ndarray,
    flow_dir:  np.ndarray,
    demcon:    np.ndarray,
    cellsize:  float,
    threshold: int = 1000,
    encoding:  str = "esri",
):
    """
    Identifikasi piksel channel dari flow accumulation threshold sederhana:

        channel jika: FlowAcc >= threshold

    Kemudian hitung Strahler order menggunakan encoding yang benar.

    Parameters
    ----------
    flow_acc  : 2D np.ndarray — Flow Accumulation (jumlah sel upstream)
    flow_dir  : 2D np.ndarray — Flow Direction
    demcon    : 2D np.ndarray — DEM terkondisi
    cellsize  : float         — ukuran piksel [m]
    threshold : int           — jumlah sel minimum untuk channel (default 1000)
    encoding  : str           — 'esri' atau 'taudem'

    Panduan threshold (DEM dalam meter):
    ┌─────────────────────────┬──────────────┬─────────────┐
    │ DAS                     │ Resolusi DEM │ Threshold   │
    ├─────────────────────────┼──────────────┼─────────────┤
    │ Kecil  (< 100 km²)      │ 5–10 m       │  200 – 500  │
    │ Sedang (100–1000 km²)   │ 10–30 m      │  500 – 2000 │
    │ Besar  (> 1000 km²)     │ 30–90 m      │ 2000 –10000 │
    └─────────────────────────┴──────────────┴─────────────┘
    Estimasi cepat: threshold ≈ luas_min_km2 × 1e6 / cellsize²

    Returns
    -------
    channel   : 2D np.ndarray bool — mask piksel channel
    S_matrix  : 2D np.ndarray int  — Strahler order
    max_order : int                — order Strahler maksimum
    """
    print(f"Mengekstrak jaringan channel  [encoding={encoding}]...")

    channel_mask = (flow_acc >= threshold) & ~np.isnan(demcon)

    n_chan    = int(channel_mask.sum())
    area_min  = threshold * cellsize**2 / 1e6

    print(f"  Threshold        : {threshold:,} sel upstream")
    print(f"  Luas drainase min: {area_min:.2f} km²")
    print(f"  Piksel channel   : {n_chan:,}  "
          f"({100*n_chan/channel_mask.size:.2f}% DEM)")

    if n_chan == 0:
        raise ValueError(
            f"Tidak ada channel terdeteksi dengan threshold={threshold:,}.\n"
            f"Turunkan channel_threshold. Untuk DEM {cellsize:.0f}m, "
            f"coba {max(10, threshold//10):,}."
        )

    print(f"  Menghitung Strahler order  [encoding={encoding}]...")
    S_matrix  = compute_strahler_order(flow_dir, channel_mask, encoding=encoding)
    max_order = int(S_matrix.max())
    print(f"  Order Strahler maks: {max_order}")

    return channel_mask, S_matrix, max_order
