from pathlib import Path
from typing import Any, Optional
import numpy as np
import trimesh

from animgen.core.armature import Armature
from animgen.renderer.renderer import Renderer, render_multiview
from animgen.animation.animator import Animator
from animgen.io.model_input import load_model
from animgen.io.glb_output import export_glb
from animgen.rigging.skinning import compute_auto_skin_weights
from animgen.utils.mesh import center_mesh


class BaseModelClass:
    def __init__(self, mesh: str | Path | trimesh.Trimesh, renderer_size=(512, 512)):
        if not isinstance(mesh, trimesh.Trimesh):
            self.mesh: trimesh.Trimesh = load_model(mesh)
        else:
            self.mesh: trimesh.Trimesh = mesh
        self.mesh = self.preprocess_mesh(self.mesh)
        self._renderer: Optional[Renderer] = None
        self._views_output: Optional[dict[str, Any]] = None
        self.armature: Optional[Armature] = None
        self.animator: Optional[Animator] = None
        self.skin_weights: Optional[dict[str, np.ndarray]] = None
        self.renderer_size: tuple[int, int] = renderer_size

    @property
    def views_output(self) -> dict[str, Any]:
        if self._views_output is None:
            self._views_output = self._get_views(
                camera_generation_method="dodecahedron",
                renderer_args={
                    "return_colored": False,
                },
                sampling_args={
                    "radius": 1.8,
                },
            )
        return self._views_output

    @views_output.setter
    def views_output(self, value: dict[str, Any]) -> None:
        """
        Setter for testing purposes.
        """
        self._views_output = value

    def clear_renders(self, delete_renderer: bool = True) -> None:
        """
        Clears cached multi-view render outputs and optionally releases the
        underlying offscreen renderer to free RAM and OpenGL VRAM.
        """
        self._views_output = None
        if delete_renderer and self._renderer is not None:
            self._renderer.delete()
            self._renderer = None

    @property
    def vertices(self) -> np.ndarray:
        return self.mesh.vertices

    @property
    def faces(self) -> np.ndarray:
        return self.mesh.faces

    def preprocess_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """
        Preprocesses the mesh:
        Centers and normalizes coordinates within a unit bounding sphere,
        preserving original vertex count, UV maps, and visual attributes.
        """
        mesh = center_mesh(mesh)
        scale = float(np.max(np.linalg.norm(mesh.vertices, axis=1)))
        if scale > 1e-8:
            mesh.vertices /= scale
        return mesh

    def _set_renderer(self) -> Renderer:
        """
        Sets up the renderer with the asset's mesh.
        """
        renderer = Renderer(
            viewport_width=self.renderer_size[0], viewport_height=self.renderer_size[1]
        )
        renderer.set_object(self.mesh)
        renderer.set_camera()
        return renderer

    def _get_views(
        self,
        camera_generation_method: str = "random_sphere",
        renderer_args: dict = {},
        sampling_args: dict = {},
        verbose: bool = True,
    ):
        """
        Generates multiple views of the asset from different camera positions.
        """
        if self._renderer is None:
            self._renderer = self._set_renderer()

        output = render_multiview(
            renderer=self._renderer,
            camera_generation_method=camera_generation_method,
            renderer_args=renderer_args,
            sampling_args=sampling_args,
            verbose=verbose,
        )
        if self._renderer is not None:
            self._renderer.delete()
            self._renderer = None
        return output

    def compute_skin_weights(self) -> dict[str, np.ndarray]:
        """
        Computes and caches skin weights for the current mesh and armature.

        Returns
        -------
        dict[str, np.ndarray]
            Mapping from bone ID to 1D numpy array of vertex skin weights.
        """
        if self.armature is None:
            raise ValueError(
                "Cannot compute skin weights: no Armature assigned to self.armature."
            )
        self.skin_weights = compute_auto_skin_weights(self.mesh, self.armature)
        if self.animator is not None:
            self.animator.skin_weights = self.skin_weights
        return self.skin_weights

    def export(
        self,
        output_path: str | Path,
        animation: Any = ...,
        armature: Armature | None = None,
        skin_weights: dict[str, np.ndarray] | None = None,
    ) -> Path:
        """
        Exports the model's mesh, armature, and optional animation to a GLB file.

        Parameters
        ----------
        output_path : str | Path
            Destination file path for the exported GLB.
        animation : Any, optional
            Animation track/clip/animator to export. If Ellipsis (...), defaults to
            self.animator.clips (or self.animator). Pass None explicitly to export a static
            rigged mesh without animations.
        armature : Armature | None, optional
            Armature to export. If None, defaults to self.armature.
        skin_weights : dict[str, np.ndarray] | None, optional
            Skin weights to export. If None, defaults to self.skin_weights.

        Returns
        -------
        Path
            The path to the exported GLB file.
        """
        export_armature = armature if armature is not None else self.armature
        if (
            export_armature is not None
            and skin_weights is None
            and self.skin_weights is None
        ):
            self.compute_skin_weights()
        export_skin_weights = (
            skin_weights if skin_weights is not None else self.skin_weights
        )

        if animation is ...:
            export_anim = self.animator.clips if self.animator is not None else None
        else:
            export_anim = animation

        return export_glb(
            mesh=self.mesh,
            output_path=output_path,
            armature=export_armature,
            skin_weights=export_skin_weights,
            animation=export_anim,
        )
