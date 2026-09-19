"""Geometric and mathematical utilities for animgen."""

from animgen.utils.mesh import (
    center_mesh,
    compute_cotangent_laplacian,
    vertex_areas,
    triangle_areas,
    taubin_smoothing,
    duplicate_verts,
)

__all__ = [
    "center_mesh",
    "compute_cotangent_laplacian",
    "vertex_areas",
    "triangle_areas",
    "taubin_smoothing",
    "duplicate_verts",
]
