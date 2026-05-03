"""
gfi2.tracing
------------
D8 flow tracing untuk dua tujuan:
  Part 1 — Hillslope → channel terdekat
  Part 2 — Channel   → confluence berikutnya (Strahler order berubah)

Algoritma mengikuti notebook GFI v1.0 resmi (tim Manfreda, Cell 9 & 11):
  - Setiap piksel mengikuti flow path sampai mencapai channel
  - PATH CACHING: piksel yang sudah dikunjungi di-cache sehingga
    piksel lain yang melewati jalur yang sama langsung ambil nilai
    tanpa trace ulang dari awal → jauh lebih cepat dari loop biasa

Encoding Flow Direction yang didukung:
  - 'esri'    : 1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N,  128=NE
  - 'taudem' : 1=E, 2=NE, 3=N, 4=NW,  5=W,  6=SW,  7=S,   8=SE
    (identik dengan encoding notebook Manfreda: dirmap=(3,2,1,8,7,6,5,4))
"""

import numpy as np
import time


# ---------------------------------------------------------------------------
# Lookup tables: nilai flow direction → (drow, dcol)
# ---------------------------------------------------------------------------

_ESRI_DR = {1: 0,  2:  1, 4:  1, 8:  1, 16: 0, 32: -1, 64: -1, 128: -1}
_ESRI_DC = {1: 1,  2:  1, 4:  0, 8: -1, 16:-1, 32: -1, 64:  0, 128:  1}

_PYSH_DR = {1: 0,  2: -1, 3: -1, 4: -1,  5: 0,  6:  1,  7:  1,   8:  1}
_PYSH_DC = {1: 1,  2:  1, 3:  0, 4: -1,  5:-1,  6: -1,  7:  0,   8:  1}


def _get_lookup(encoding: str):
    enc = encoding.lower()
    if enc == "esri":
        return _ESRI_DR, _ESRI_DC
    elif enc in ("taudem", "manfreda"):
        return _PYSH_DR, _PYSH_DC
    else:
        raise ValueError(
            f"Encoding '{encoding}' tidak dikenal. Gunakan 'esri' atau 'taudem'."
        )


def _step(flow_dir, i, j, DR, DC):
    """Satu langkah flow tracing. Return (ni, nj) atau None jika tidak valid."""
    fd = int(flow_dir[i, j])
    if fd not in DR:
        return None
    ni = i + DR[fd]
    nj = j + DC[fd]
    rows, cols = flow_dir.shape
    if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
        return None
    return ni, nj


# ---------------------------------------------------------------------------
# PART 1: Hillslope → channel terdekat  (dengan path caching)
# Mengikuti logika Cell 9 & 11 notebook Manfreda
# ---------------------------------------------------------------------------

def hillslope_to_river_mapping(
    demcon:   np.ndarray,
    flow_dir: np.ndarray,
    channel:  np.ndarray,
    flow_acc: np.ndarray,
    cellsize: float,
    encoding: str = "esri",
    n_jobs:   int = -1,
):
    """
    Part 1: Untuk setiap piksel non-channel, trace flow path ke channel
    terdekat dan simpan koordinatnya.

    PATH CACHING (mengikuti notebook Manfreda Cell 9 & 11):
      Setelah sebuah flow path ditelusuri dan channel ditemukan,
      semua piksel di sepanjang path tersebut di-cache dengan hasil
      yang sama. Piksel berikutnya yang melewati path yang sama
      langsung mengambil nilai cache tanpa trace ulang.

      Kompleksitas efektif: mendekati O(N) bukan O(N x L).

    Parameters
    ----------
    demcon   : 2D float32 — DEM terkondisi
    flow_dir : 2D float32 — Flow Direction
    channel  : 2D bool    — mask piksel channel
    flow_acc : 2D float32 — Flow Accumulation
    cellsize : float      — ukuran piksel [m]
    encoding : str        — 'esri' atau 'taudem'
    n_jobs   : int        — tidak digunakan (untuk kompatibilitas API)

    Returns
    -------
    ROW_channel : 2D float32 — baris channel terdekat (NaN = tidak terpetakan)
    COL_channel : 2D float32 — kolom channel terdekat (NaN = tidak terpetakan)
    """
    print(f"Part 1: Hillslope → channel terdekat  "
          f"[encoding={encoding}, path caching=ON]...")
    t0 = time.time()

    DR, DC     = _get_lookup(encoding)
    rows, cols = demcon.shape
    nodata     = np.isnan(demcon) | (flow_dir <= 0) | np.isnan(flow_dir)

    # Cache: -1 = belum dikunjungi, -2 = tidak ada channel downstream
    # nilai >= 0 = row/col channel yang valid
    ROW_cache = np.full((rows, cols), -1, dtype=np.int32)
    COL_cache = np.full((rows, cols), -1, dtype=np.int32)

    # Inisialisasi: piksel channel menunjuk ke dirinya sendiri
    r_ch, c_ch = np.where(channel)
    ROW_cache[r_ch, c_ch] = r_ch.astype(np.int32)
    COL_cache[r_ch, c_ch] = c_ch.astype(np.int32)

    ROW_channel = np.full((rows, cols), np.nan, dtype=np.float32)
    COL_channel = np.full((rows, cols), np.nan, dtype=np.float32)
    ROW_channel[r_ch, c_ch] = r_ch.astype(np.float32)
    COL_channel[r_ch, c_ch] = c_ch.astype(np.float32)

    visited_in_path = set()

    for i in range(rows):
        for j in range(cols):
            if channel[i, j] or nodata[i, j]:
                continue

            # Sudah di cache → langsung pakai
            cv = ROW_cache[i, j]
            if cv >= 0:
                ROW_channel[i, j] = cv
                COL_channel[i, j] = COL_cache[i, j]
                continue
            if cv == -2:
                continue

            # Trace flow path, kumpulkan semua piksel yang dilalui
            path = []
            a, b = i, j
            visited_in_path.clear()
            result = None   # (rc, cc) jika channel ditemukan

            while True:
                cv = ROW_cache[a, b]

                if cv >= 0:
                    # Ketemu cache valid
                    result = (cv, COL_cache[a, b])
                    break

                if cv == -2:
                    # Ketemu cache tidak valid
                    break

                if (a, b) in visited_in_path:
                    # Loop terdeteksi
                    break

                path.append((a, b))
                visited_in_path.add((a, b))

                nxt = _step(flow_dir, a, b, DR, DC)
                if nxt is None or nodata[nxt[0], nxt[1]]:
                    break

                a, b = nxt[0], nxt[1]

            # Isi cache untuk semua piksel di path
            if result is not None:
                rc, cc = result
                for (pi, pj) in path:
                    ROW_cache[pi, pj]   = rc
                    COL_cache[pi, pj]   = cc
                    ROW_channel[pi, pj] = float(rc)
                    COL_channel[pi, pj] = float(cc)
            else:
                for (pi, pj) in path:
                    ROW_cache[pi, pj] = -2

    n_hill   = int((~channel & ~nodata).sum())
    n_mapped = int(np.sum(~np.isnan(ROW_channel) & ~channel))
    pct      = 100.0 * n_mapped / max(n_hill, 1)
    print(f"  Selesai dalam {time.time()-t0:.1f}s")
    print(f"  Terpetakan: {n_mapped:,} / {n_hill:,} piksel hillslope ({pct:.1f}%)")
    return ROW_channel, COL_channel


# ---------------------------------------------------------------------------
# PART 2: Channel → confluence berikutnya  (dengan path caching)
# ---------------------------------------------------------------------------

def river_to_confluence_mapping(
    flow_dir:  np.ndarray,
    channel:   np.ndarray,
    S_matrix:  np.ndarray,
    max_order: int,
    cellsize:  float,
    encoding:  str = "esri",
    n_jobs:    int = -1,
):
    """
    Part 2: Untuk setiap piksel channel non-main-river, trace ke confluence
    berikutnya — titik di mana Strahler order berubah (naik).

    Juga menggunakan path caching: segmen upstream yang melewati jalur
    yang sama tidak perlu trace ulang.

    Parameters
    ----------
    flow_dir  : 2D float32 — Flow Direction
    channel   : 2D bool    — mask piksel channel
    S_matrix  : 2D int32   — Strahler order
    max_order : int        — order maksimum (main river, tidak di-trace)
    cellsize  : float      — ukuran piksel [m]
    encoding  : str        — 'esri' atau 'taudem'
    n_jobs    : int        — tidak digunakan

    Returns
    -------
    ROW_confluence : 2D float32 — baris confluence terdekat
    COL_confluence : 2D float32 — kolom confluence terdekat
    """
    print(f"Part 2: Channel → confluence berikutnya  "
          f"[encoding={encoding}, path caching=ON]...")
    t0 = time.time()

    DR, DC     = _get_lookup(encoding)
    rows, cols = flow_dir.shape

    ROW_conf_cache = np.full((rows, cols), -1, dtype=np.int32)
    COL_conf_cache = np.full((rows, cols), -1, dtype=np.int32)

    ROW_confluence = np.full((rows, cols), np.nan, dtype=np.float32)
    COL_confluence = np.full((rows, cols), np.nan, dtype=np.float32)

    visited_in_path = set()

    for i in range(rows):
        for j in range(cols):
            if not channel[i, j]:
                continue
            so = S_matrix[i, j]
            if so == 0 or so >= max_order:
                continue

            # Sudah di cache?
            cv = ROW_conf_cache[i, j]
            if cv >= 0:
                ROW_confluence[i, j] = cv
                COL_confluence[i, j] = COL_conf_cache[i, j]
                continue
            if cv == -2:
                continue

            path = []
            a, b = i, j
            visited_in_path.clear()
            result = None

            while True:
                cv = ROW_conf_cache[a, b]
                if cv >= 0:
                    result = (cv, COL_conf_cache[a, b])
                    break
                if cv == -2:
                    break
                if (a, b) in visited_in_path:
                    break

                path.append((a, b))
                visited_in_path.add((a, b))

                nxt = _step(flow_dir, a, b, DR, DC)
                if nxt is None:
                    break

                ni, nj = nxt

                # Order berubah → ini confluence
                if S_matrix[ni, nj] != so:
                    result = (ni, nj)
                    break

                a, b = ni, nj

            if result is not None:
                rc, cc = result
                for (pi, pj) in path:
                    ROW_conf_cache[pi, pj] = rc
                    COL_conf_cache[pi, pj] = cc
                    ROW_confluence[pi, pj]  = float(rc)
                    COL_confluence[pi, pj]  = float(cc)
            else:
                for (pi, pj) in path:
                    ROW_conf_cache[pi, pj] = -2

    n_target = int(((S_matrix > 0) & (S_matrix < max_order)).sum())
    n_mapped = int(np.sum(~np.isnan(ROW_confluence) & channel))
    pct      = 100.0 * n_mapped / max(n_target, 1)
    print(f"  Selesai dalam {time.time()-t0:.1f}s")
    print(f"  Terpetakan: {n_mapped:,} / {n_target:,} segmen channel ({pct:.1f}%)")
    return ROW_confluence, COL_confluence
