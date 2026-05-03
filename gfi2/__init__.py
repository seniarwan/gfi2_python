# =============================================================================
# gfi2 — Geomorphic Flood Index v2.0  (Python port)
# Original MATLAB: Saavedra Navarro et al., DOI: 10.5281/zenodo.18903835
# =============================================================================

from .io       import load_tif, save_tif, resample_to_ref, check_alignment, get_cellsize_meters
from .preprocess import preprocess_dem_auto, preprocess_dem_manual, convert_taudem_to_esri
from .network  import gradient8, compute_strahler_order, extract_channel_network
from .tracing  import hillslope_to_river_mapping, river_to_confluence_mapping
from .gfi      import compute_gfi_v1, compute_gfi_v2
from .calibrate import area_under_curve, roc_curve_maggiore
from .metrics  import compute_validation_metrics
from .viz      import plot_roc_comparison, plot_spatial_accuracy, plot_water_depth_analysis
from .pipeline import run_gfi2

__version__ = "2.0.0"
__author__  = (
    "Jorge Saavedra Navarro, Cinzia Albertini, "
    "Caterina Samela, Salvatore Manfreda"
)
__doi__     = "10.5281/zenodo.18903835"

__all__ = [
    # io
    "load_tif", "save_tif", "resample_to_ref", "check_alignment", "get_cellsize_meters",
    # preprocess
    "preprocess_dem_auto", "preprocess_dem_manual", "convert_taudem_to_esri",
    # network
    "gradient8", "compute_strahler_order", "extract_channel_network",
    # tracing
    "hillslope_to_river_mapping", "river_to_confluence_mapping",
    # gfi
    "compute_gfi_v1", "compute_gfi_v2",
    # calibrate
    "area_under_curve", "roc_curve_maggiore",
    # metrics
    "compute_validation_metrics",
    # viz
    "plot_roc_comparison", "plot_spatial_accuracy", "plot_water_depth_analysis",
    # pipeline
    "run_gfi2",
]
