import torch

from animgen.utils.camera import (
    matrix3x4_to_4x4,
    view_matrix,
    sample_view_matrices,
    sample_view_matrices_polyhedra,
)


def test_camera_matrix_and_sampling():
    """
    Test camera transformation utilities:
    - 3x4 to 4x4 homogeneous expansion
    - View matrix lookat orientation
    - Spherical and polyhedral viewpoint sampling
    """
    # Conversion: 3x4 to 4x4
    mat_3x4 = torch.randn(2, 3, 4)
    mat_4x4 = matrix3x4_to_4x4(mat_3x4)
    assert mat_4x4.shape == (2, 4, 4)
    assert torch.allclose(mat_4x4[:, 3, 3], torch.ones(2))

    # View matrix lookat
    cam_pos = torch.tensor([[0.0, 0.0, 5.0]])
    lookat = torch.tensor([[0.0, 0.0, 0.0]])
    up = torch.tensor([0.0, 1.0, 0.0])
    poses = view_matrix(cam_pos, lookat, up)
    assert poses.shape == (1, 4, 4)
    assert torch.allclose(poses[0, :3, 3], cam_pos[0])

    # Sphere sampling
    radius = 4.0
    poses_sphere = sample_view_matrices(n=6, radius=radius)
    assert poses_sphere.shape == (6, 4, 4)

    # Polyhedral sampling
    poses_poly = sample_view_matrices_polyhedra(polygon="icosahedron", radius=radius)
    assert poses_poly.shape == (12, 4, 4)
    dists = torch.linalg.vector_norm(poses_poly[:, :3, 3], dim=-1)
    assert torch.allclose(dists, torch.tensor(radius))
