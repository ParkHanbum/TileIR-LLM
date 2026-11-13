"""Experimental block-load variant; not GPU validated."""
from .api import gemm, gemm_nt, gemm_nt_fixed, linear, validate_gemm, stats, reset_stats, k_tiles, select_tile
__version__ = "0.2.0-candidate"
