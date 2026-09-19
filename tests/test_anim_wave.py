import numpy as np
import pytest

from animgen.animation.wave import chain_wave_generator, _travelling_wave_generator
from animgen.animation.kinematics import compute_forward_kinematics
from animgen.core.armature import Armature, Bone


@pytest.mark.parametrize("wave_type", ["standing", "travelling", "pulse"])
def test_chain_wave_generator(wave_type):
    """
    Verify chain_wave_generator produces valid animations across all wave modes:
    - Generates expected number of frames
    - Produces valid orthonormal rotation matrices
    - Conserves bone lengths strictly across all animation frames via FK
    """
    # Build a connected 5-bone chain along X
    root = Bone(id="b0", head=(0.0, 0.0, 0.0), tail=(1.0, 0.0, 0.0))
    armature = Armature(root)
    prev = root
    for i in range(1, 5):
        prev = armature.add_connected_bone(prev, tail=(float(i + 1), 0.0, 0.0))
        prev.id = f"b{i}"

    index_bones = list(range(5))
    wave_duration = 2.0
    frame_rate = 10.0

    animation = chain_wave_generator(
        armature=armature,
        index_bones=index_bones,
        wave_amplitude=0.3,
        wave_duration=wave_duration,
        frame_rate=frame_rate,
        wave=wave_type,
    )

    expected_frames = int(frame_rate * wave_duration)
    assert len(animation) == expected_frames

    initial_lengths = [
        np.linalg.norm(np.array(b.tail) - np.array(b.head)) for b in armature.bones_list
    ]

    for t, frame in animation.items():
        assert len(frame) == 5
        for R in frame:
            if hasattr(R, "detach"):
                R = R.detach().cpu().numpy()
            assert R.shape == (3, 3)
            np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-5)

        # Compute FK and check bone length conservation
        _, positions = compute_forward_kinematics(armature, frame)
        for i, b in enumerate(armature.bones_list):
            head, tail = positions[b.id]
            curr_len = np.linalg.norm(np.array(tail) - np.array(head))
            assert np.isclose(curr_len, initial_lengths[i], atol=1e-4)


def test_wave_periodicity_and_growth():
    """
    Verify wave mathematical properties:
    - Periodicity: wave at t equals wave at t + wave_duration (loopability)
    - Growth factor: positive growth increases displacement amplitude towards the tail
    """
    distances = np.linspace(0.0, 5.0, 11)
    duration = 2.0
    t1 = 0.3
    t2 = t1 + duration

    # Periodicity check
    anim_periodic = _travelling_wave_generator(
        distances=distances,
        wave_amplitude=0.4,
        wave_duration=duration,
        time_stamps=[t1, t2],
        num_waves=1.5,
    )
    for R1, R2 in zip(anim_periodic[t1], anim_periodic[t2]):
        np.testing.assert_allclose(R1, R2, atol=1e-6)

    # Growth check: positive growth produces larger rotation at the end than no growth
    time_stamps = np.linspace(0.0, duration, 20)
    anim_no_growth = _travelling_wave_generator(
        distances=distances,
        wave_amplitude=0.3,
        wave_duration=duration,
        time_stamps=time_stamps,
        growth_factor=0.0,
        num_waves=1.5,
    )
    anim_growth = _travelling_wave_generator(
        distances=distances,
        wave_amplitude=0.3,
        wave_duration=duration,
        time_stamps=time_stamps,
        growth_factor=0.3,
        num_waves=1.5,
    )

    # Peak tip bone rotation deviation from identity across cycle should be larger with growth
    max_no_growth = max(
        np.linalg.norm(f[-1] - np.eye(3)) for f in anim_no_growth.values()
    )
    max_growth = max(np.linalg.norm(f[-1] - np.eye(3)) for f in anim_growth.values())
    assert max_growth > max_no_growth
