"""
gfi2.network
------------
Ekstraksi jaringan drainase dan penghitungan Strahler stream order.
Setara MATLAB: gradient8(), streamorder(..., 'Strahler'), CHANNEL_mask.
"""

import numpy as np


# ---------------------------------------------------------------------------
def gradient8(Z: np.ndarray, cellsize: float) -> np.ndarray:
    """
    Kemiringan maksimum 8-arah (tangen).
    Setara MATLAB gradient8(Z, cellsize).

    Untuk setiap sel: max(|dz| / jarak) dari 8 tetangga.

    Parameters
    ----------
    Z        : 2D np.ndarray — array elevasi
    cellsize : float         — ukuran piksel dalam satuan peta

    Returns
    -------
    G : np.ndarray float32 — slope maksimum tiap sel
    """
    rows, cols = Z.shape
    Zp         = np.pad(Z.astype(np.float64), 1, mode="edge")
    diag       = cellsize * np.sqrt(2)

    # 8 arah: E, SE, S, SW, W, NW, N, NE
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
) -> np.ndarray:
    """
    Hitung Strahler stream order untuk piksel channel via BFS downstream.

    Aturan Strahler:
      - Headwater (tanpa upstream) → order 1
      - Dua stream dengan order sama bertemu → order +1
      - Stream order lebih rendah bergabung ke order lebih tinggi
        → order tetap (dipertahankan yang lebih tinggi)

    Setara MATLAB: streamorder(FD, CHANNEL, 'Strahler')

    Parameters
    ----------
    flow_dir     : 2D np.ndarray — Flow Direction ESRI D8
    channel_mask : 2D np.ndarray bool — True di mana channel

    Returns
    -------
    S : 2D np.ndarray int32 — Strahler order (0 = non-channel)
    """
    DIR_MAP = {
        1:   ( 0,  1),   # E
        128: (-1,  1),   # NE
        64:  (-1,  0),   # N
        32:  (-1, -1),   # NW
        16:  ( 0, -1),   # W
        8:   ( 1, -1),   # SW
        4:   ( 1,  0),   # S
        2:   ( 1,  1),   # SE
    }

    rows, cols = flow_dir.shape
    in_deg     = np.zeros((rows, cols), dtype=np.int32)

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

    S     = np.zeros((rows, cols), dtype=np.int32)
    queue = []
    for r in range(rows):
        for c in range(cols):
            if channel_mask[r, c] and in_deg[r, c] == 0:
                S[r, c] = 1
                queue.append((r, c))

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
):
    """
    Identifikasi piksel channel menggunakan threshold sederhana berbasis
    flow accumulation:

        channel jika: FlowAcc >= threshold

    Nilai threshold = jumlah sel upstream minimum agar sebuah piksel
    dianggap sebagai channel. Konsisten dengan pendekatan TopoToolbox
    (MATLAB asli GFI v2.0).

    Panduan pemilihan threshold:
    ┌─────────────────────────────┬──────────────────┬───────────────┐
    │ Karakteristik DAS           │ Resolusi DEM     │ Threshold     │
    ├─────────────────────────────┼──────────────────┼───────────────┤
    │ DAS kecil  (< 100 km²)      │ 5–10 m           │ 200–500       │
    │ DAS sedang (100–1000 km²)   │ 10–30 m          │ 500–2000      │
    │ DAS besar  (> 1000 km²)     │ 30–90 m          │ 2000–10000    │
    │ Bradano (2765 km², 5 m DEM) │ 5 m              │ ~5000         │
    └─────────────────────────────┴──────────────────┴───────────────┘

    Cara estimasi cepat:
        threshold ≈ drainage_area_min_km2 × 1e6 / cellsize²
        Contoh: area min 1 km², cellsize 30 m →
                threshold = 1e6 / 30² ≈ 1111 sel → gunakan 1000

    Parameters
    ----------
    flow_acc  : 2D np.ndarray — Flow Accumulation (jumlah sel upstream)
    flow_dir  : 2D np.ndarray — Flow Direction ESRI D8
    demcon    : 2D np.ndarray — DEM terkondisi
    cellsize  : float         — ukuran piksel [m]  (hanya untuk info)
    threshold : int           — jumlah sel minimum untuk channel
                                (default 1000)

    Returns
    -------
    channel   : 2D np.ndarray bool — mask piksel channel
    S_matrix  : 2D np.ndarray int  — Strahler order (0 = non-channel)
    max_order : int                — order Strahler maksimum di DAS
    """
    print("Mengekstrak jaringan channel (metode: flow accumulation threshold)...")

    channel_mask = (flow_acc >= threshold) & ~np.isnan(demcon)

    n_chan      = int(channel_mask.sum())
    area_km2    = n_chan * cellsize**2 / 1e6
    drain_min   = threshold * cellsize**2 / 1e6

    print(f"  Threshold        : {threshold:,} sel upstream")
    print(f"  Luas drainase min: {drain_min:.2f} km²  "
          f"(= {threshold} sel × {cellsize:.0f}² m / 1e6)")
    print(f"  Piksel channel   : {n_chan:,}  "
          f"({100*n_chan/channel_mask.size:.2f}% DEM)")
    print(f"  Panjang channel  : ~{area_km2*1e6/cellsize/1000:.1f} km  "
          f"(estimasi kasar)")

    if n_chan == 0:
        raise ValueError(
            f"Tidak ada channel terdeteksi dengan threshold={threshold:,}.\n"
            f"Turunkan nilai channel_threshold. "
            f"Untuk DEM {cellsize:.0f}m, coba threshold="
            f"{max(10, int(threshold/10)):,}."
        )

    print("  Menghitung Strahler order...")
    S_matrix  = compute_strahler_order(flow_dir, channel_mask)
    max_order = int(S_matrix.max())
    print(f"  Order Strahler maks: {max_order}")

    return channel_mask, S_matrix, max_order
