"""
animgen: Procedural Animation Generation Framework
==================================================

A high-level procedural rigging and animation framework over glTF/PyGLTFLib.
Transforms static 3D meshes into articulated, skinned, and animated biological
assets with volume-preserving continuous Bishop frame kinematics.
"""

from animgen.core.models.pipeline import Pipeline
from animgen.core.models.fish import FishModels
from animgen.core.models.serpentine import SerpentineModels
from animgen.core.models.model import BaseModelClass
from animgen.core.armature import Armature, Bone
from animgen.core.spline import Spline, Spline as CatmullRomSpline
from animgen.animation.animator import Animator
from animgen.animation.clip import AnimationClip
from animgen.animation.wave import chain_wave_generator
from animgen.animation.straight import (
    straighten,
    straighten_lateral,
    build_bishop_frame,
    compute_node_bishop_frames,
)
from animgen.animation.deformation import (
    apply_linear_blend_skinning_deformation,
    apply_dual_quaternion_skinning_deformation,
    apply_linear_blend_skinning_deformation as linear_blend_skinning,
    apply_dual_quaternion_skinning_deformation as dual_quaternion_skinning,
)
from animgen.io.model_input import load_model
from animgen.io.glb_output import export_glb

__version__ = "0.1.0"

__all__ = [
    "BaseModelClass",
    "Pipeline",
    "FishModels",
    "SerpentineModels",
    "Armature",
    "Bone",
    "Spline",
    "CatmullRomSpline",
    "Animator",
    "AnimationClip",
    "chain_wave_generator",
    "straighten",
    "straighten_lateral",
    "build_bishop_frame",
    "compute_node_bishop_frames",
    "linear_blend_skinning",
    "dual_quaternion_skinning",
    "apply_linear_blend_skinning_deformation",
    "apply_dual_quaternion_skinning_deformation",
    "load_model",
    "export_glb",
    "__version__",
]
