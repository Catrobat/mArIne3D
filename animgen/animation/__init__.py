"""Animation, deformation, and kinematics subsystems."""

from animgen.animation.animator import Animator
from animgen.animation.clip import AnimationClip
from animgen.animation.deformation import (
    apply_linear_blend_skinning_deformation,
    apply_dual_quaternion_skinning_deformation,
    apply_mesh_deformation,
    apply_linear_blend_skinning_deformation as linear_blend_skinning,
    apply_dual_quaternion_skinning_deformation as dual_quaternion_skinning,
)
from animgen.animation.kinematics import (
    compute_forward_kinematics,
    compute_forward_kinematics as forward_kinematics,
    successive_rotations,
)
from animgen.animation.straight import (
    straighten,
    straighten_lateral,
    build_bishop_frame,
    compute_node_bishop_frames,
)
from animgen.animation.wave import chain_wave_generator

__all__ = [
    "Animator",
    "AnimationClip",
    "apply_linear_blend_skinning_deformation",
    "apply_dual_quaternion_skinning_deformation",
    "apply_mesh_deformation",
    "linear_blend_skinning",
    "dual_quaternion_skinning",
    "compute_forward_kinematics",
    "forward_kinematics",
    "successive_rotations",
    "straighten",
    "straighten_lateral",
    "build_bishop_frame",
    "compute_node_bishop_frames",
    "chain_wave_generator",
]
