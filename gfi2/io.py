"""
gfi2.io
-------
Utilitas baca/tulis raster GeoTIFF.
Setara MATLAB: GRIDobj, resample(), single().
"""

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject


# ---------------------------------------------------------------------------
def load_tif(path: str):
    """
    Baca GeoTIFF satu band → (array float32, rasterio profile).
    Nilai nodata diganti NaN secara otomatis.

    Parameters
    ----------
    path : str — path ke file GeoTIFF

    Returns
    -------
    data    : np.ndarray float32
    profile : dict (rasterio profile)
    """
    with rasterio.open(path) as src:
        data    = src.read(1).astype(np.float32)
        nodata  = src.nodata
        profile = src.profile.copy()
    if nodata is not None:
        data[data == nodata] = np.nan
    return data, profile


# ---------------------------------------------------------------------------
def save_tif(array: np.ndarray, profile: dict,
             out_path: str, nodata_val: float = -9999.0) -> None:
    """
    Simpan array numpy sebagai GeoTIFF float32.

    Parameters
    ----------
    array     : 2D np.ndarray — data yang akan disimpan
    profile   : dict          — rasterio profile referensi
    out_path  : str           — path output
    nodata_val: float         — nilai nodata (default -9999)
    """
    prof = profile.copy()
    prof.update(dtype="float32", count=1, nodata=nodata_val)
    arr  = np.where(np.isnan(array), nodata_val, array).astype(np.float32)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(arr, 1)
    print(f"  Tersimpan: {out_path}")


# ---------------------------------------------------------------------------
def resample_to_ref(src_path: str, ref_profile: dict) -> np.ndarray:
    """
    Resample raster src_path agar cocok resolusi/extent ref_profile.
    Menangani perbedaan CRS secara otomatis.
    Setara MATLAB: resample(raster, AUX, 'nearest').

    Parameters
    ----------
    src_path    : str  — path raster yang akan di-resample
    ref_profile : dict — rasterio profile referensi (dari load_tif)

    Returns
    -------
    data : np.ndarray float32 — raster tersampling
    """
    dst_h         = ref_profile["height"]
    dst_w         = ref_profile["width"]
    dst_crs       = ref_profile.get("crs")
    dst_transform = ref_profile.get("transform")

    with rasterio.open(src_path) as src:
        if src.crs == dst_crs or dst_crs is None:
            data = src.read(
                1,
                out_shape=(dst_h, dst_w),
                resampling=Resampling.nearest,
            ).astype(np.float32)
            nodata = src.nodata
        else:
            # CRS berbeda → reproject sekaligus
            data = np.empty((dst_h, dst_w), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1),
                destination=data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest,
            )
            nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = np.nan
    return data


# ---------------------------------------------------------------------------
def check_alignment(arrays_dict: dict) -> None:
    """
    Verifikasi semua raster punya shape yang sama.
    Lempar ValueError jika tidak sejajar.

    Parameters
    ----------
    arrays_dict : dict — {'nama': array, ...}
    """
    shapes = {k: v.shape for k, v in arrays_dict.items()}
    unique = set(shapes.values())
    if len(unique) > 1:
        detail = "\n".join(f"   {k}: {s}" for k, s in shapes.items())
        raise ValueError(
            f"Raster tidak sejajar:\n{detail}\n"
            "Gunakan resample_to_ref() atau sejajarkan di GIS terlebih dahulu."
        )
    print(f"  Semua raster sejajar: {list(unique)[0]}")


# ---------------------------------------------------------------------------
def get_cellsize_meters(profile: dict) -> float:
    """
    Ekstrak ukuran piksel dalam satuan METER dari rasterio profile.

    Menangani dua kasus:
      - CRS proyeksi (UTM, dll.) → cellsize langsung dari transform [m]
      - CRS geografis (WGS84 derajat) → konversi derajat → meter
        menggunakan aproksimasi: 1° lintang ≈ 111,320 m
        (akurat ~0.1% untuk lintang 0°–60°)

    Parameters
    ----------
    profile : dict — rasterio profile (hasil load_tif)

    Returns
    -------
    cellsize_m : float — ukuran piksel dalam meter

    Raises
    ------
    ValueError — jika CRS tidak dikenali dan tidak bisa dikonversi
    """
    from rasterio.crs import CRS

    raw_cs = abs(profile["transform"].a)   # ukuran piksel mentah
    crs    = profile.get("crs")

    if crs is None:
        # Tidak ada CRS → asumsikan sudah meter, beri peringatan
        print(
            "  PERINGATAN: DEM tidak memiliki informasi CRS.\n"
            f"  Cellsize mentah {raw_cs:.6f} diasumsikan dalam meter.\n"
            "  Jika unit sebenarnya bukan meter, hasil GFI tidak valid."
        )
        return raw_cs

    crs_obj = CRS.from_user_input(crs)

    if crs_obj.is_geographic:
        # Unit: derajat → konversi ke meter
        # Ambil lintang tengah dari transform untuk koreksi
        transform  = profile["transform"]
        center_lat = transform.f + (profile["height"] / 2) * transform.e
        center_lat = abs(center_lat)

        # Faktor konversi derajat → meter
        # 1° lintang  ≈ 111,320 m  (hampir konstan)
        # 1° bujur    ≈ 111,320 * cos(lat) m
        import math
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * math.cos(math.radians(center_lat))

        # Untuk piksel persegi dalam derajat, gunakan rata-rata lat & lon
        cellsize_m = raw_cs * (m_per_deg_lat + m_per_deg_lon) / 2.0

        print(
            f"  CRS geografis terdeteksi (unit: derajat).\n"
            f"  Lintang tengah: {center_lat:.2f}°\n"
            f"  Cellsize: {raw_cs:.6f}° → {cellsize_m:.2f} m (aproksimasi)\n"
            f"  REKOMENDASI: Proyeksikan DEM ke UTM untuk akurasi maksimal."
        )
        return cellsize_m

    else:
        # CRS proyeksi → unit sudah meter
        unit = crs_obj.axis_info[0].unit_name if crs_obj.axis_info else "unknown"
        if "metre" in unit.lower() or "meter" in unit.lower() or unit == "unknown":
            return raw_cs
        else:
            raise ValueError(
                f"Unit CRS tidak dikenal: '{unit}'.\n"
                "Hanya meter dan derajat yang didukung.\n"
                "Proyeksikan DEM ke UTM terlebih dahulu."
            )
