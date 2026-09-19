from typing import Any, List, Union
import numpy as np
import torch
import trimesh

from animgen.core.spline import Spline


def build_bishop_frame(
    spine_pts_np: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build Bishop parallel transport frames along a 3D spline/spine path.

    Parameters
    ----------
    spine_pts_np : (N, 3) ndarray
        Dense sequence of 3D points representing the curve.

    Returns
    -------
    T_seg : (N-1, 3) ndarray
        Tangent vectors for each segment.
    N_seg : (N-1, 3) ndarray
        Normal vectors for each segment.
    B_seg : (N-1, 3) ndarray
        Binormal vectors for each segment.
    segment_lengths : (N-1,) ndarray
        Lengths of each segment.
    s : (N,) ndarray
        Cumulative arc lengths at each point along the spine.
    """
    N_points = len(spine_pts_np)
    T_seg = np.zeros((N_points - 1, 3))
    N_seg = np.zeros((N_points - 1, 3))
    B_seg = np.zeros((N_points - 1, 3))

    for i in range(N_points - 1):
        diff = spine_pts_np[i + 1] - spine_pts_np[i]
        length = np.linalg.norm(diff)
        T_seg[i] = diff / max(length, 1e-12)

    T0 = T_seg[0]
    if abs(T0[0]) < 0.9:
        V = np.array([1.0, 0.0, 0.0])
    else:
        V = np.array([0.0, 1.0, 0.0])
    N0 = V - np.dot(V, T0) * T0
    N0_norm = np.linalg.norm(N0)
    N0 = N0 / max(N0_norm, 1e-12)
    B0 = np.cross(T0, N0)

    N_seg[0] = N0
    B_seg[0] = B0

    for i in range(1, N_points - 1):
        t_prev = T_seg[i - 1]
        t_curr = T_seg[i]
        n_prev = N_seg[i - 1]

        axis_rot = np.cross(t_prev, t_curr)
        axis_norm = np.linalg.norm(axis_rot)
        if axis_norm < 1e-8:
            n_curr = n_prev - np.dot(n_prev, t_curr) * t_curr
            n_curr_norm = np.linalg.norm(n_curr)
            n_curr = n_curr / max(n_curr_norm, 1e-12)
        else:
            axis_rot = axis_rot / axis_norm
            dot_val = np.clip(np.dot(t_prev, t_curr), -1.0, 1.0)
            theta = np.arccos(dot_val)
            n_curr = (
                n_prev * np.cos(theta)
                + np.cross(axis_rot, n_prev) * np.sin(theta)
                + axis_rot * np.dot(axis_rot, n_prev) * (1.0 - np.cos(theta))
            )
            n_curr = n_curr - np.dot(n_curr, t_curr) * t_curr
            n_curr_norm = np.linalg.norm(n_curr)
            n_curr = n_curr / max(n_curr_norm, 1e-12)

        N_seg[i] = n_curr
        B_seg[i] = np.cross(t_curr, n_curr)

    diffs = np.diff(spine_pts_np, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    s = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    return T_seg, N_seg, B_seg, segment_lengths, s


def compute_node_bishop_frames(
    T_seg: np.ndarray,
    N_seg: np.ndarray,
    B_seg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute continuous, orthonormal Bishop frames at curve nodes (spine points)
    from segment frames via smooth tangent and normal interpolation.

    Parameters
    ----------
    T_seg : (N-1, 3) ndarray
        Segment unit tangents.
    N_seg : (N-1, 3) ndarray
        Segment unit normals.
    B_seg : (N-1, 3) ndarray
        Segment unit binormals.

    Returns
    -------
    T_node : (N, 3) ndarray
        Unit tangents at spine nodes.
    N_node : (N, 3) ndarray
        Unit normals at spine nodes.
    B_node : (N, 3) ndarray
        Unit binormals at spine nodes.
    """
    N_pts = len(T_seg) + 1
    T_node = np.zeros((N_pts, 3), dtype=T_seg.dtype)
    N_node = np.zeros((N_pts, 3), dtype=N_seg.dtype)
    B_node = np.zeros((N_pts, 3), dtype=B_seg.dtype)

    T_node[0] = T_seg[0]
    N_node[0] = N_seg[0]
    B_node[0] = B_seg[0]

    T_node[-1] = T_seg[-1]
    N_node[-1] = N_seg[-1]
    B_node[-1] = B_seg[-1]

    for i in range(1, N_pts - 1):
        t_avg = T_seg[i - 1] + T_seg[i]
        t_norm = np.linalg.norm(t_avg)
        t_avg = t_avg / max(t_norm, 1e-12)

        n_avg = N_seg[i - 1] + N_seg[i]
        n_avg = n_avg - np.dot(n_avg, t_avg) * t_avg
        n_norm = np.linalg.norm(n_avg)
        n_avg = n_avg / max(n_norm, 1e-12)

        T_node[i] = t_avg
        N_node[i] = n_avg
        B_node[i] = np.cross(t_avg, n_avg)

    return T_node, N_node, B_node


def deform_mesh_to_spine_numpy(
    mesh: trimesh.Trimesh,
    source_spine: np.ndarray,
    target_spine: np.ndarray,
    chunk_size: int = 10000,
    weld: bool = True,
    digits: int = 5,
) -> trimesh.Trimesh:
    """
    Deform a mesh using Bishop frame coordinate projection in pure NumPy.
    This method is geometrically exact and volume-preserving.

    When weld=True, co-located duplicate vertices (such as un-welded UV seams or
    sharp normal boundaries) are moved as a continuous, welded surface, preventing
    seams from pulling apart or tearing during deformation.
    """
    # Build Bishop parallel transport frames along both spines
    T_src, N_src, B_src, _, _ = build_bishop_frame(source_spine)
    T_tgt, N_tgt, B_tgt, _, _ = build_bishop_frame(target_spine)

    # Compute smooth orthonormal frames at curve nodes
    T_src_node, N_src_node, B_src_node = compute_node_bishop_frames(T_src, N_src, B_src)
    T_tgt_node, N_tgt_node, B_tgt_node = compute_node_bishop_frames(T_tgt, N_tgt, B_tgt)

    # Spatially weld vertices to move duplicate / seam vertices identically
    if weld and len(mesh.vertices) > 0:
        unique_idx, inverse_idx = trimesh.grouping.unique_rows(
            mesh.vertices, digits=digits
        )
        V_work = mesh.vertices[unique_idx]
    else:
        V_work = mesh.vertices
        inverse_idx = None

    # Check if spine has a primary axis with monotonic progression
    spans = np.ptp(source_spine, axis=0)
    long_axis = int(np.argmax(spans))
    diffs = np.diff(source_spine[:, long_axis])
    is_monotonic = bool(np.all(diffs > 1e-8) or np.all(diffs < -1e-8))

    A_src = source_spine[:-1]
    D_src = source_spine[1:] - source_spine[:-1]
    L2_src = np.sum(D_src**2, axis=1)
    L2_src = np.maximum(L2_src, 1e-12)

    A_tgt = target_spine[:-1]
    D_tgt = target_spine[1:] - target_spine[:-1]

    N_points = len(source_spine)
    num_vertices = len(V_work)
    V_new_work = np.zeros_like(V_work)

    # Map coordinates chunk by chunk to limit memory footprints
    for start_idx in range(0, num_vertices, chunk_size):
        end_idx = min(start_idx + chunk_size, num_vertices)
        V_chunk = V_work[start_idx:end_idx]

        if is_monotonic:
            x_spine = source_spine[:, long_axis]
            x_verts = V_chunk[:, long_axis]
            if diffs[0] < 0:
                x_rev = x_spine[::-1]
                rev_idx = np.searchsorted(x_rev, x_verts) - 1
                seg_idx = (len(x_spine) - 2) - rev_idx
            else:
                seg_idx = np.searchsorted(x_spine, x_verts) - 1
            seg_idx = np.clip(seg_idx, 0, N_points - 2)

            x0 = x_spine[seg_idx]
            x1 = x_spine[seg_idx + 1]
            dx = x1 - x0
            dx = np.where(np.abs(dx) < 1e-12, 1e-12, dx)
            t_star = np.clip((x_verts - x0) / dx, 0.0, 1.0)[:, None]
            closest_seg = seg_idx
            P_closest_src = A_src[closest_seg] + t_star * D_src[closest_seg]
        else:
            disp = V_chunk[:, None, :] - A_src[None, :, :]
            dot = np.sum(disp * D_src[None, :, :], axis=2)
            t_val = np.clip(dot / L2_src[None, :], 0.0, 1.0)
            proj = A_src[None, :, :] + t_val[:, :, None] * D_src[None, :, :]
            dist2 = np.sum((V_chunk[:, None, :] - proj) ** 2, axis=2)
            closest_seg = np.argmin(dist2, axis=1)
            row_indices = np.arange(len(V_chunk))
            t_star = t_val[row_indices, closest_seg, None]
            P_closest_src = proj[row_indices, closest_seg]

        P_closest_tgt = A_tgt[closest_seg] + t_star * D_tgt[closest_seg]

        # Continuous Bishop frame interpolation for source spine
        T_0_src = T_src_node[closest_seg]
        T_1_src = T_src_node[closest_seg + 1]
        T_v_src = (1.0 - t_star) * T_0_src + t_star * T_1_src
        T_v_src = T_v_src / np.maximum(
            np.linalg.norm(T_v_src, axis=1, keepdims=True), 1e-12
        )

        N_0_src = N_src_node[closest_seg]
        N_1_src = N_src_node[closest_seg + 1]
        N_v_src = (1.0 - t_star) * N_0_src + t_star * N_1_src
        N_v_src = N_v_src - np.sum(N_v_src * T_v_src, axis=1, keepdims=True) * T_v_src
        N_v_src = N_v_src / np.maximum(
            np.linalg.norm(N_v_src, axis=1, keepdims=True), 1e-12
        )
        B_v_src = np.cross(T_v_src, N_v_src)

        # Continuous Bishop frame interpolation for target spine
        T_0_tgt = T_tgt_node[closest_seg]
        T_1_tgt = T_tgt_node[closest_seg + 1]
        T_v_tgt = (1.0 - t_star) * T_0_tgt + t_star * T_1_tgt
        T_v_tgt = T_v_tgt / np.maximum(
            np.linalg.norm(T_v_tgt, axis=1, keepdims=True), 1e-12
        )

        N_0_tgt = N_tgt_node[closest_seg]
        N_1_tgt = N_tgt_node[closest_seg + 1]
        N_v_tgt = (1.0 - t_star) * N_0_tgt + t_star * N_1_tgt
        N_v_tgt = N_v_tgt - np.sum(N_v_tgt * T_v_tgt, axis=1, keepdims=True) * T_v_tgt
        N_v_tgt = N_v_tgt / np.maximum(
            np.linalg.norm(N_v_tgt, axis=1, keepdims=True), 1e-12
        )
        B_v_tgt = np.cross(T_v_tgt, N_v_tgt)

        # Relative coordinates in source frame
        d_vec = V_chunk - P_closest_src
        x = np.sum(d_vec * N_v_src, axis=1, keepdims=True)
        y = np.sum(d_vec * B_v_src, axis=1, keepdims=True)
        z = np.sum(d_vec * T_v_src, axis=1, keepdims=True)

        # Reconstruct coordinates in target frame
        V_new_work[start_idx:end_idx] = (
            P_closest_tgt + x * N_v_tgt + y * B_v_tgt + z * T_v_tgt
        )

    deformed_mesh = mesh.copy()
    if inverse_idx is not None:
        deformed_mesh.vertices = V_new_work[inverse_idx]
    else:
        deformed_mesh.vertices = V_new_work
    return deformed_mesh


def deform_mesh_to_spine(
    mesh: trimesh.Trimesh,
    source_spine: np.ndarray,
    target_spine: np.ndarray,
    chunk_size: int = 10000,
    weld: bool = True,
    digits: int = 5,
) -> trimesh.Trimesh:
    """
    Deforms a mesh from a source spine to a target spine using Bishop parallel transport frames.
    """
    return deform_mesh_to_spine_numpy(
        mesh, source_spine, target_spine, chunk_size, weld=weld, digits=digits
    )


def resolve_spine_points(
    spine_points: Union[np.ndarray, torch.Tensor, List[Any], Spline],
    num_segments: int = 100,
) -> np.ndarray:
    """
    Resolve spine points from various formats (Spline, Tensor, ndarray, or list of points)
    to a standardized NumPy array of shape (N, 3).
    """
    if isinstance(spine_points, Spline):
        eval_pts = spine_points.evaluate_curve(
            num_points_per_segment=max(5, num_segments // len(spine_points.points) + 1)
        )
        spine_pts_np = np.array([pt.detach().cpu().numpy() for pt in eval_pts])
    elif isinstance(spine_points, torch.Tensor):
        spine_pts_np = spine_points.detach().cpu().numpy()
    elif isinstance(spine_points, np.ndarray):
        spine_pts_np = spine_points
    else:
        resolved = []
        for pt in spine_points:
            if isinstance(pt, torch.Tensor):
                resolved.append(pt.detach().cpu().numpy())
            else:
                resolved.append(np.array(pt))
        spine_pts_np = np.stack(resolved, axis=0)

    if spine_pts_np.ndim != 2 or spine_pts_np.shape[1] != 3:
        raise ValueError(
            f"Spine points must have shape (N, 3), got {spine_pts_np.shape}"
        )
    return spine_pts_np


def straighten(
    mesh: trimesh.Trimesh,
    spine_points: Union[np.ndarray, torch.Tensor, List[Any], Spline],
    num_segments: int | None = None,
    axis: str = "z",
) -> trimesh.Trimesh:
    """
    Straighten a curved mesh along a specified axis using Bishop frame deformation.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input curved mesh.
    spine_points : array-like or Spline
        The control points or evaluated points representing the curved spine.
        - If a Spline object, it is evaluated to generate num_segments points.
        - If an array, it is used directly (or interpolated if necessary).
    num_segments : int or None
        The number of interpolation points to use along the spine.
    axis : str
        The axis along which to straighten the mesh ('x', 'y', or 'z').

    Returns
    -------
    straightened_mesh : trimesh.Trimesh
        A new mesh representing the straightened geometry.
    """
    spine_pts_np = resolve_spine_points(spine_points, num_segments or 100)

    # Calculate cumulative arc lengths to define the target straight spine
    _, _, _, _, s = build_bishop_frame(spine_pts_np)

    target_spine = np.zeros_like(spine_pts_np)
    if axis == "z":
        target_spine[:, 2] = s
    elif axis == "y":
        target_spine[:, 1] = s
    elif axis == "x":
        target_spine[:, 0] = s
    else:
        raise ValueError(f"Unknown axis: {axis}. Must be 'x', 'y', or 'z'.")

    return deform_mesh_to_spine(mesh, spine_pts_np, target_spine)


def straighten_lateral(
    mesh: trimesh.Trimesh,
    spine_points: Union[np.ndarray, torch.Tensor, List[Any], Spline],
    num_segments: int | None = None,
    straighten_axis: str = "x",
) -> trimesh.Trimesh:
    """
    Straighten a curved mesh along a specific lateral axis (e.g., left-right / X axis),
    while preserving its coordinates and shape along the other axes (e.g., Z / vertical profile).

    Parameters
    ----------
    mesh : trimesh.Trimesh
        The input curved mesh.
    spine_points : array-like or Spline
        The control points or evaluated points representing the curved spine.
        - If a Spline object, it is evaluated.
    num_segments : int or None
        The number of interpolation points to use along the spine.
    straighten_axis : str
        The lateral coordinate axis to flatten ('x', 'y', or 'z').

    Returns
    -------
    straightened_mesh : trimesh.Trimesh
        A new mesh with the lateral curvature removed.
    """
    spine_pts_np = resolve_spine_points(spine_points, num_segments or 100)

    # Create target spine by setting the specified axis to its initial value
    target_spine = spine_pts_np.copy()
    if straighten_axis == "x":
        target_spine[:, 0] = spine_pts_np[0, 0]
    elif straighten_axis == "y":
        target_spine[:, 1] = spine_pts_np[0, 1]
    elif straighten_axis == "z":
        target_spine[:, 2] = spine_pts_np[0, 2]
    else:
        raise ValueError(
            f"Unknown straighten_axis: {straighten_axis}. Must be 'x', 'y', or 'z'."
        )

    return deform_mesh_to_spine(mesh, spine_pts_np, target_spine)
