"""Mesh contraction, skeleton refinement, and skin weight diffusion engines."""

from animgen.rigging.skinning import compute_auto_skin_weights
from animgen.rigging.mesh_contraction import contract_mesh
from animgen.rigging.refine_skelaton import (
    subdivide_and_center_skeleton,
    refine_and_center_skeleton_iterative,
)
from animgen.rigging.SAM3 import SAM3Segmentation, SAM3TextEmbedder

__all__ = [
    "compute_auto_skin_weights",
    "contract_mesh",
    "subdivide_and_center_skeleton",
    "refine_and_center_skeleton_iterative",
    "SAM3Segmentation",
    "SAM3TextEmbedder",
]
