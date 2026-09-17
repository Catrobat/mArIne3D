import numpy as np
from PIL import Image
import trimesh

from animgen.renderer.renderer import Renderer, render_multiview


def test_renderer_multiview():
    """
    Test that render_multiview correctly renders multi-angle perspectives of a 3D mesh:
    - Generates the expected number of view images
    - Each image has valid dimensions and contains rendered mesh geometry (non-trivial contrast)
    """
    mesh = trimesh.creation.cylinder(radius=1.0, height=4.0, sections=16)

    renderer = Renderer(viewport_width=128, viewport_height=128)
    renderer.set_object(mesh)
    renderer.set_camera()

    num_views = 8
    results = render_multiview(
        renderer,
        camera_generation_method="random_sphere",
        renderer_args={},
        sampling_args={"n": num_views, "radius": 6.0},
        verbose=False,
    )

    mattes = results.get("matte", [])
    assert len(mattes) == num_views, f"Expected {num_views} images, got {len(mattes)}"

    for idx, img_data in enumerate(mattes):
        img_arr = (
            np.array(img_data)
            if isinstance(img_data, Image.Image)
            else np.asarray(img_data)
        )
        assert img_arr.shape[:2] == (128, 128)
        assert float(np.std(img_arr)) > 1.0, f"View {idx} lacks contrast"
    renderer.delete()


def test_renderer_multiview_default_arguments():
    """
    Test that render_multiview runs without errors when called with default
    renderer_args and sampling_args.
    """
    mesh = trimesh.creation.cylinder(radius=1.0, height=4.0, sections=16)
    renderer = Renderer(viewport_width=64, viewport_height=64)
    renderer.set_object(mesh)
    renderer.set_camera()

    results = render_multiview(renderer, verbose=False)
    assert isinstance(results, dict)
    assert "poses" in results
    assert len(results["poses"]) == 12
    renderer.delete()
