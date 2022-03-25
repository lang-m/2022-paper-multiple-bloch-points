"""Simulate FeGe two-layer nanostrips that contain Bloch points.

The functionality is based on Ubermag and provides some convenience functions
and a strip class to quickly simulate and analyse a two-layer nanostrip.
"""
from .strip import Strip
from .short_init import create_pattern, pattern_ascii
from .postprocessing import collect_results
