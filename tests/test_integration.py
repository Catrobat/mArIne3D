"""
Integration tests for the unified animgen package API and CLI interface.
"""

from pathlib import Path
import tempfile
import numpy as np
import pytest
import trimesh

import animgen
from animgen import (
    BaseModelClass,
    Pipeline,
    FishModels,
    SerpentineModels,
    Armature,
    Bone,
    CatmullRomSpline,
    Spline,
    Animator,
    AnimationClip,
    chain_wave_generator,
    straighten,
    straighten_lateral,
    build_bishop_frame,
    compute_node_bishop_frames,
    linear_blend_skinning,
    dual_quaternion_skinning,
    apply_linear_blend_skinning_deformation,
    apply_dual_quaternion_skinning_deformation,
    load_model,
    export_glb,
    __version__,
)
from animgen.cli import main as cli_main


def test_top_level_exports():
    """Verify that all key symbols are exposed at top-level animgen namespace."""
    assert __version__ == "0.1.0"
    assert issubclass(FishModels, Pipeline)
    assert issubclass(SerpentineModels, Pipeline)
    assert Spline is CatmullRomSpline
    assert linear_blend_skinning is apply_linear_blend_skinning_deformation
    assert dual_quaternion_skinning is apply_dual_quaternion_skinning_deformation
    assert callable(load_model)
    assert callable(export_glb)
    assert callable(straighten)
    assert callable(straighten_lateral)
    assert callable(build_bishop_frame)
    assert callable(compute_node_bishop_frames)
    assert callable(chain_wave_generator)
    assert issubclass(Armature, object)
    assert issubclass(Bone, object)
    assert issubclass(Animator, object)
    assert issubclass(AnimationClip, object)
    assert "BaseModelClass" in animgen.__all__


def test_subpackage_exports():
    """Verify that subpackages expose their public symbols."""
    import animgen.core
    import animgen.animation
    import animgen.io
    import animgen.rigging
    import animgen.renderer
    import animgen.utils

    assert hasattr(animgen.core, "Armature")
    assert hasattr(animgen.core, "BaseModelClass")
    assert hasattr(animgen.animation, "Animator")
    assert hasattr(animgen.animation, "straighten")
    assert hasattr(animgen.io, "load_model")
    assert hasattr(animgen.io, "export_glb")
    assert hasattr(animgen.rigging, "compute_auto_skin_weights")
    assert hasattr(animgen.renderer, "Renderer")
    assert hasattr(animgen.renderer, "visualize_skeleton")
    assert hasattr(animgen.utils, "center_mesh")
    assert hasattr(animgen.utils, "compute_cotangent_laplacian")


def test_cli_info_command(capsys):
    """Verify animgen info CLI command on an existing sample model."""
    sample_model = Path("generated_data/models/paint_mesh_Sea_Snake.glb")
    if not sample_model.exists():
        pytest.skip("Sample sea snake model not found")

    ret = cli_main(["info", "--model", str(sample_model)])
    assert ret == 0

    captured = capsys.readouterr()
    assert "animgen: Procedural Animation Generation Framework" in captured.out
    assert "Geometry Overview" in captured.out
    assert "Vertices:" in captured.out
    assert "Faces:" in captured.out


def test_cli_help(capsys):
    """Verify animgen --help outputs available subcommands."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--help"])
    assert exc_info.value.code == 0

    captured = capsys.readouterr()
    assert "process" in captured.out
    assert "autorig" in captured.out
    assert "straighten" in captured.out
    assert "info" in captured.out


def test_end_to_end_synthetic_serpentine_export():
    """Verify end-to-end autorig and animation export using top-level imports."""
    # Create a synthetic cylindrical test mesh along the X axis
    cyl = trimesh.creation.cylinder(radius=0.1, height=2.0, sections=16)
    # Rotate cylinder to align with X axis
    rot = trimesh.transformations.rotation_matrix(np.pi / 2, [0, 1, 0])
    cyl.apply_transform(rot)

    with tempfile.TemporaryDirectory() as tmpdir:
        input_glb = Path(tmpdir) / "test_cylinder.glb"
        output_glb = Path(tmpdir) / "test_cylinder_animated.glb"
        cyl.export(str(input_glb))

        # Autorig & animate via top-level API
        model = BaseModelClass(input_glb)
        pipeline = SerpentineModels(
            model=model,
            prompts=["body"],
            n_bones=6,
        )
        # Mock segmentation to avoid running SAM3 during fast unit test
        pipeline.segment = lambda: {"body": list(range(len(model.faces)))}

        result_model = pipeline.process()
        exported_path = pipeline.export(output_glb)

        assert exported_path.exists()
        assert exported_path.stat().st_size > 0
        assert result_model.armature is not None
        assert len(result_model.armature.bones_list) == 6
        assert result_model.skin_weights is not None
