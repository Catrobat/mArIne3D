import numpy as np
import pymeshlab  # type: ignore
import trimesh
from trimesh import Trimesh


def shape_diameter_function(
    mesh: Trimesh,
    norm: bool = True,
    alpha: float = 4.0,
    rays: int = 64,
    cone_amplitude: float = 120.0,
) -> np.ndarray:
    """
    Compute the Shape Diameter Function (SDF) on a 3D surface mesh using PyMeshLab.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input surface mesh.
    norm : bool, default=True
        Whether to normalize and logarithmically scale SDF values to [0, 1].
    alpha : float, default=4.0
        Logarithmic normalization scaling factor.
    rays : int, default=64
        Number of conical rays cast inward per sample.
    cone_amplitude : float, default=120.0
        Cone opening angle in degrees.

    Returns
    -------
    sdf_values : (F,) ndarray
        Scalar Shape Diameter Function value for each face in the mesh.
    """
    ml_mesh = pymeshlab.Mesh(mesh.vertices, mesh.faces)  # type: ignore[attr-defined]
    meshset = pymeshlab.MeshSet()  # type: ignore[attr-defined]
    meshset.add_mesh(ml_mesh)
    meshset.compute_scalar_by_shape_diameter_function_per_vertex(
        rays=rays, cone_amplitude=cone_amplitude
    )

    sdf_values = meshset.current_mesh().face_scalar_array()
    sdf_values[np.isnan(sdf_values)] = 0.0
    if norm:
        val_min = sdf_values.min()
        val_max = sdf_values.max()
        if val_max > val_min:
            sdf_values = (sdf_values - val_min) / (val_max - val_min)
            sdf_values = np.log(sdf_values * alpha + 1.0) / np.log(alpha + 1.0)
        else:
            sdf_values = np.zeros_like(sdf_values)
    return sdf_values


def colormap_shape_diameter_function(mesh: Trimesh, sdf_values: np.ndarray) -> Trimesh:
    """
    Colorize mesh faces using Shape Diameter Function scalar values.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Input surface mesh.
    sdf_values : (F,) ndarray
        Per-face SDF scalar values.

    Returns
    -------
    trimesh.Trimesh
        A copy of the mesh with vertex duplicates for un-interpolated face coloring.
    """
    assert len(mesh.faces) == len(sdf_values)
    import matplotlib.pyplot as plt
    from animgen.utils.mesh import duplicate_verts

    mesh_copy = duplicate_verts(mesh)
    colors = trimesh.visual.interpolate(  # type: ignore[attr-defined]
        sdf_values, color_map=plt.get_cmap("jet")
    )
    mesh_copy.visual = trimesh.visual.ColorVisuals(  # type: ignore[attr-defined]
        mesh=mesh_copy, face_colors=colors
    )
    return mesh_copy
