"""
gfi2.preprocess
---------------
Preprocessing DEM: fill sinks, flow direction D8, flow accumulation.

MODE A — auto   : menggunakan pysheds (tidak perlu raster lain)
MODE B — manual : raster preprocessing sudah tersedia dari QGIS / ArcGIS /
                  WhiteboxTools — pysheds tidak diperlukan sama sekali.

Flow direction encoding yang digunakan di seluruh paket ini: ESRI D8
  1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
"""

import numpy as np
from .io import load_tif, check_alignment

# Nilai D8 yang valid dalam encoding ESRI
_ESRI_D8_VALUES = {1, 2, 4, 8, 16, 32, 64, 128}


# ---------------------------------------------------------------------------
# MODE A — OTOMATIS (pysheds)
# ---------------------------------------------------------------------------

def preprocess_dem_auto(dem_path: str):
    """
    [MODE A] Load DEM → fill sinks → flow direction D8 (ESRI) → flow acc.
    Memerlukan pysheds (pip install pysheds).

    Parameters
    ----------
    dem_path : str — path ke DEM GeoTIFF mentah

    Returns
    -------
    demvoid  : np.ndarray float32 — DEM asli
    demcon   : np.ndarray float32 — DEM terkondisi (filled)
    flow_dir : np.ndarray float32 — Flow Direction ESRI D8
    flow_acc : np.ndarray float32 — Flow Accumulation (jumlah sel)
    cellsize : float              — ukuran piksel dalam satuan peta
    profile  : dict               — rasterio profile DEM asli
    """
    try:
        from pysheds.grid import Grid
    except ImportError:
        raise ImportError(
            "pysheds tidak terinstall.\n"
            "Jalankan: pip install pysheds\n"
            "Atau gunakan preprocess_dem_manual() (MODE B) jika sudah "
            "punya raster flow direction dan flow accumulation."
        )

    print("[MODE A] Memuat dan mengkondisi DEM dengan pysheds...")
    demvoid, profile = load_tif(dem_path)
    cellsize         = abs(profile["transform"].a)

    grid    = Grid.from_raster(dem_path)
    dem_raw = grid.read_raster(dem_path)

    print("  Mengisi pit & depresi...")
    pit_filled = grid.fill_pits(dem_raw)
    flooded    = grid.fill_depressions(pit_filled)
    inflated   = grid.resolve_flats(flooded)

    print("  Menghitung flow direction D8 (ESRI encoding)...")
    fdir = grid.flowdir(inflated)           # pysheds → ESRI D8 by default

    print("  Menghitung flow accumulation...")
    acc  = grid.accumulation(fdir)

    demcon   = np.array(inflated, dtype=np.float32)
    flow_dir = np.array(fdir,     dtype=np.float32)
    flow_acc = np.array(acc,      dtype=np.float32)

    # Nilai ≤0 (batas / nodata pysheds) → NaN
    flow_dir[flow_dir <= 0] = np.nan

    print(f"  Shape    : {demvoid.shape}")
    print(f"  Cell size: {cellsize:.2f} m  |  Max acc: {np.nanmax(flow_acc):.0f} sel")
    return demvoid, demcon, flow_dir, flow_acc, cellsize, profile


# ---------------------------------------------------------------------------
# MODE B — MANUAL (raster sudah ada)
# ---------------------------------------------------------------------------

def preprocess_dem_manual(
    dem_path:     str,
    demcon_path:  str,
    flowdir_path: str,
    flowacc_path: str,
):
    """
    [MODE B] Load raster preprocessing yang sudah ada dari luar.
    Tidak memerlukan pysheds.

    Parameters
    ----------
    dem_path     : str — DEM mentah (referensi grid & profile)
    demcon_path  : str — DEM terkondisi / filled
    flowdir_path : str — Flow Direction **ESRI D8** (1,2,4,8,16,32,64,128)
                         Tool yang menghasilkan ESRI D8 secara default:
                           • QGIS  : GRASS r.watershed  atau
                                     SAGA "D8 Flow Direction"
                           • WhiteboxTools : d8_pointer
                           • ArcGIS        : Flow Direction
                         Jika berasal dari TauDEM → gunakan
                         convert_taudem_to_esri() sebelum menyimpan.
    flowacc_path : str — Flow Accumulation (jumlah sel upstream)

    Returns
    -------
    Sama dengan preprocess_dem_auto()
    """
    print("[MODE B] Memuat raster preprocessing yang sudah ada...")

    demvoid,  profile = load_tif(dem_path)
    demcon,   _       = load_tif(demcon_path)
    flow_dir, _       = load_tif(flowdir_path)
    flow_acc, _       = load_tif(flowacc_path)
    cellsize          = abs(profile["transform"].a)

    # ── Validasi 1: semua raster harus sejajar ───────────────────────────
    check_alignment({
        "DEM":      demvoid,
        "DEM con":  demcon,
        "Flow Dir": flow_dir,
        "Flow Acc": flow_acc,
    })

    # ── Validasi 2: encoding flow direction harus ESRI D8 ────────────────
    fd_vals     = set(
        np.unique(flow_dir[~np.isnan(flow_dir)]).astype(int).tolist()
    )
    unexpected  = fd_vals - _ESRI_D8_VALUES
    if unexpected:
        raise ValueError(
            f"Flow direction mengandung nilai tidak dikenal: {unexpected}\n"
            "Pastikan encoding ESRI D8 digunakan:\n"
            "  1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE\n"
            "Jika file berasal dari TauDEM, gunakan "
            "convert_taudem_to_esri() terlebih dahulu."
        )

    # Nilai 0 / negatif (batas / nodata) → NaN
    flow_dir[flow_dir <= 0] = np.nan

    print(f"  Shape    : {demvoid.shape}")
    print(f"  Cell size: {cellsize:.2f} m  |  Max acc: {np.nanmax(flow_acc):.0f} sel")
    return demvoid, demcon, flow_dir, flow_acc, cellsize, profile


# ---------------------------------------------------------------------------
# UTILITAS: konversi encoding TauDEM → ESRI D8
# ---------------------------------------------------------------------------

def convert_taudem_to_esri(taudem_fdir: np.ndarray) -> np.ndarray:
    """
    Konversi Flow Direction dari encoding TauDEM ke ESRI D8.

    TauDEM : 1=E, 2=NE, 3=N, 4=NW, 5=W, 6=SW, 7=S, 8=SE
    ESRI D8: 1=E, 128=NE, 64=N, 32=NW, 16=W, 8=SW, 4=S, 2=SE

    Parameters
    ----------
    taudem_fdir : np.ndarray — flow direction dalam encoding TauDEM

    Returns
    -------
    np.ndarray float32 — flow direction dalam encoding ESRI D8
    """
    lut = {1: 1, 2: 128, 3: 64, 4: 32, 5: 16, 6: 8, 7: 4, 8: 2}
    out = np.full_like(taudem_fdir, np.nan, dtype=np.float32)
    for src_val, dst_val in lut.items():
        out[taudem_fdir == src_val] = dst_val
    return out
