"""
Module for standing wave generation along an armature.

References
----------
https://en.wikipedia.org/wiki/Standing_wave
"""

import numpy as np

from animgen.core.armature import Armature

from numpy.typing import NDArray
from typing import Literal
from animgen.core.types import Animation
from animgen.animation.kinematics import successive_rotations


def _check_armature_chain(armature: Armature, index_bones: list[int]) -> bool:
    """
    Confirms if the index_bones corresponding to armature.bones_list make a continuous chain
    """
    if len(index_bones) < 2:
        return True  # A single bone is trivially a chain

    bones_list = armature.bones_list
    bones_chain_list = [bones_list[i] for i in index_bones]

    for i in range(len(bones_chain_list) - 1):
        parent_bone = bones_chain_list[i]
        child_bone = bones_chain_list[i + 1]
        if not child_bone.parent == parent_bone:
            return False
        if child_bone.head != parent_bone.tail:
            return False

    return True


def _compute_wave_grid(
    distances: NDArray[np.float64] | list[float],
    time_stamps: float | list[float] | NDArray[np.float64],
    num_waves: float,
    wave_duration: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Computes extended ceil distances, wave number k, and angular frequency omega
    for procedural wave animations.
    """
    d_arr = np.asarray(distances, dtype=np.float64)
    t_arr = np.asarray(time_stamps, dtype=np.float64)

    total_length = d_arr[-1]
    num_bones = len(d_arr) - 1
    bone_length = total_length / num_bones

    if num_waves <= 0.0:
        ceil_num_bones = num_bones
    else:
        ceil_waves = float(np.ceil(num_waves))
        ceil_length_target = total_length * (ceil_waves / num_waves)
        ceil_num_bones = int(np.ceil(ceil_length_target / bone_length))

    ceil_distances = np.arange(ceil_num_bones + 1) * bone_length
    wave_number = 2 * np.pi * num_waves / total_length
    angular_frequency = 2 * np.pi / wave_duration

    return d_arr, t_arr, ceil_distances, wave_number, angular_frequency


def _integrate_tangents_to_rotations(
    distances: np.ndarray,
    time_stamps: np.ndarray,
    tangent: np.ndarray,
    ceil_distances: np.ndarray,
) -> Animation:
    """
    Integrates spatial tangent vectors into 3D bone positions, applies mean-Y centering
    for spatial equilibrium, truncates to target armature length, and converts into
    successive local rotation matrices via inverse kinematics.
    """
    if time_stamps.ndim == 0 and tangent.ndim == 2:
        tangent = tangent[None, ...]

    bind_positions = np.stack(
        (distances, np.zeros_like(distances), np.zeros_like(distances)),
        axis=-1,
    )

    animation: Animation = {}
    seg_lens = np.diff(ceil_distances)

    for time, frame_tangent in zip(np.atleast_1d(time_stamps), tangent):
        frame = [(0.0, 0.0, 0.0)]
        for index in range(1, len(ceil_distances)):
            seg_length = seg_lens[index - 1]
            previous_position = np.asarray(frame[-1], dtype=np.float64)
            direction = frame_tangent[index - 1]
            position = previous_position + seg_length * direction
            frame.append(tuple(position.tolist()))

        y_coords = np.array([p[1] for p in frame])
        mean_y_ceil = float(np.mean(y_coords))
        shifted_frame = [(p[0], p[1] - mean_y_ceil, p[2]) for p in frame]
        truncated_frame = np.array(shifted_frame[: len(distances)])

        animation[float(time)] = successive_rotations(
            bind_positions,
            truncated_frame,
            is_positions=True,
        )

    return animation


def _travelling_wave_generator(
    distances: list[float] | NDArray[np.float64],
    wave_amplitude: float,
    wave_duration: float,
    time_stamps: float | list[float] | NDArray[np.float64],
    growth_factor: float = 0,
    num_waves: float = 2.6,
    phi_s: float = 0.0,
    phi_t: float = 0.0,
) -> Animation:
    """
    Generate a travelling-wave animation from spatial distances and timestamps.

    The travelling wave is modeled as:

        u(s, t) = A exp(g s)
                sin(k s - omega t + phi_s + phi_t)

    where the spatial wave number ``k`` and temporal angular frequency
    ``omega`` are defined as:

        k = 2 pi N / L

        omega = 2 pi / T

    Here, ``N`` is the number of spatial waves, ``L`` is the total
    spatial length, and ``T`` is the temporal period of the wave.

    The resulting displacement is applied along the y-axis to a chain whose
    rest configuration lies along the x-axis. Consequently, each generated
    frame contains the 3D positions of the chain at a given timestamp.

    Parameters
    ----------
    distances : list[float] | NDArray[np.float64]
        Cumulative spatial distances ``s`` at which to evaluate the wave.
        The distances are measured from the root of the armature and define
        the x-coordinate of each point in the generated chain.

    wave_amplitude : float
        Base amplitude ``A`` of the wave at ``s = 0``.

    wave_duration : float
        Temporal period ``T`` of the wave in seconds. The wave completes
        one full temporal oscillation every ``wave_duration`` seconds.

    time_stamps : float | list[float] | NDArray[np.float64]
        Timestamp or timestamps ``t`` at which to evaluate the wave,
        in seconds.

    growth_factor : float, default=0.0
        Exponential spatial growth rate ``g`` of the wave amplitude.

        The amplitude at distance ``s`` is:

            A(s) = A * exp(g * s)

        Positive values increase the amplitude with distance, zero
        produces a constant amplitude, and negative values produce
        exponential damping.

    num_waves : float, default=2.6
        Number of complete spatial wavelengths across the total spatial
        length ``L``.

    phi_s : float, default=0.0
        Spatial phase offset in radians.

    phi_t : float, default=0.0
        Temporal phase offset in radians.

    Returns
    -------
    Animation
        Mapping from timestamps to animation frames containing local bone rotations.

        Each timestamp maps to a list of rotation matrices (shape ``(num_bones, 3, 3)``)
        representing local bone rotations that deform the armature chain from its bind pose
        to the wave shape.
    """
    d_arr, t_arr, ceil_distances, wave_number, angular_frequency = _compute_wave_grid(
        distances, time_stamps, num_waves, wave_duration
    )

    spatial_amplitude = wave_amplitude * np.exp(growth_factor * ceil_distances)
    phase = (
        wave_number * ceil_distances
        - angular_frequency * t_arr[..., None]
        + phi_s
        + phi_t
    )

    spatial_derivative = spatial_amplitude * (
        growth_factor * np.sin(phase) + wave_number * np.cos(phase)
    )

    tangent = np.stack(
        (
            np.ones_like(spatial_derivative),
            spatial_derivative,
            np.zeros_like(spatial_derivative),
        ),
        axis=-1,
    )
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)

    return _integrate_tangents_to_rotations(d_arr, t_arr, tangent, ceil_distances)


def _standing_wave_generator(
    distances: list[float] | NDArray[np.float64],
    wave_amplitude: float,
    wave_duration: float,
    time_stamps: float | list[float] | NDArray[np.float64],
    growth_factor: float = 0,
    num_waves: float = 2.6,
    phi_s: float = 0.0,
    phi_t: float = 0.0,
) -> Animation:
    """
    Generate a standing-wave animation from spatial distances and timestamps.

    The standing wave is modeled as:

        u(s, t) = 2 A exp(g s)
                sin(k s + phi_s)
                sin(omega t + phi_t)

    where the spatial wave number ``k`` and temporal angular frequency
    ``omega`` are defined as:

        k = 2 pi N / L

        omega = 2 pi / T

    Here, ``N`` is the number of spatial waves, ``L`` is the total
    spatial length, and ``T`` is the temporal period of the wave.

    The resulting displacement is applied along the y-axis to a chain whose
    rest configuration lies along the x-axis. Consequently, each generated
    frame contains the 3D positions of the chain at a given timestamp.

    Parameters
    ----------
    distances : list[float] | NDArray[np.float64]
        Cumulative spatial distances ``s`` at which to evaluate the wave.
        The distances are measured from the root of the armature and define
        the x-coordinate of each point in the generated chain.

    wave_amplitude : float
        Base amplitude ``A`` of the wave at ``s = 0``.

    wave_duration : float
        Temporal period ``T`` of the wave in seconds. The wave completes
        one full temporal oscillation every ``wave_duration`` seconds.

    time_stamps : float | list[float] | NDArray[np.float64]
        Timestamp or timestamps ``t`` at which to evaluate the wave,
        in seconds.

    growth_factor : float, default=0.0
        Exponential spatial growth rate ``g`` of the wave amplitude.

        The amplitude at distance ``s`` is:

            A(s) = A * exp(g * s)

        Positive values increase the amplitude with distance, zero
        produces a constant amplitude, and negative values produce
        exponential damping.

    num_waves : float, default=2.6
        Number of complete spatial wavelengths across the total spatial
        length ``L``.

    phi_s : float, default=0.0
        Spatial phase offset in radians.

    phi_t : float, default=0.0
        Temporal phase offset in radians.

    Returns
    -------
    Animation
        Mapping from timestamps to animation frames containing local bone rotations.

        Each timestamp maps to a list of rotation matrices (shape ``(num_bones, 3, 3)``)
        representing local bone rotations that deform the armature chain from its bind pose
        to the wave shape.
    """
    d_arr, t_arr, ceil_distances, wave_number, angular_frequency = _compute_wave_grid(
        distances, time_stamps, num_waves, wave_duration
    )

    spatial_amplitude = 2 * wave_amplitude * np.exp(growth_factor * ceil_distances)
    cos_part = np.cos(wave_number * ceil_distances + phi_s)
    sin_part = np.sin(wave_number * ceil_distances + phi_s)
    spatial_deriv = growth_factor * sin_part + wave_number * cos_part

    t_1d = np.atleast_1d(t_arr)
    temporal_part = np.sin(angular_frequency * t_1d[..., None] + phi_t)
    spatial_combined = spatial_amplitude * spatial_deriv
    derivative = temporal_part * spatial_combined[None, :]

    tangent = np.stack(
        (
            np.ones_like(derivative),
            derivative,
            np.zeros_like(derivative),
        ),
        axis=-1,
    )
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)

    return _integrate_tangents_to_rotations(d_arr, t_arr, tangent, ceil_distances)


def _pulse_wave_generator(
    distances: list[float] | NDArray[np.float64],
    wave_amplitude: float,
    wave_duration: float,
    time_stamps: float | list[float] | NDArray[np.float64],
    growth_factor: float = 0,
    num_waves: float = 2.6,
    phi_s: float = 0.0,
    phi_t: float = 0.0,
    pulse_width: float = 1.0,
    pulse_center: float = 0.0,
) -> Animation:
    """
    Generate a pulse-wave (wave packet) animation from spatial distances and timestamps.

    The pulse wave is modeled as a travelling sinusoidal carrier modulated by a Gaussian envelope:

        u(s, t) = A exp(g s) * exp(-0.5 * (phase / sigma)^2) * cos(phase)

    where the carrier phase is defined as:

        phase = k * (s - s_0) - omega * t + phi_s + phi_t

    with ``sigma = pulse_width`` controlling the spatial width of the pulse packet and
    ``s_0 = pulse_center`` defining its spatial center at t=0.

    Parameters
    ----------
    distances : list[float] | NDArray[np.float64]
        Cumulative spatial distances ``s`` at which to evaluate the wave.
    wave_amplitude : float
        Peak amplitude ``A`` of the wave at the pulse center.
    wave_duration : float
        Temporal period ``T`` of the wave in seconds.
    time_stamps : float | list[float] | NDArray[np.float64]
        Timestamp or timestamps ``t`` at which to evaluate the wave, in seconds.
    growth_factor : float, default=0.0
        Exponential spatial growth rate ``g`` of the wave amplitude.
    num_waves : float, default=2.6
        Carrier spatial wavenumber parameter.
    phi_s : float, default=0.0
        Spatial phase offset in radians.
    phi_t : float, default=0.0
        Temporal phase offset in radians.
    pulse_width : float, default=1.0
        Standard deviation ``sigma`` of the Gaussian envelope in spatial units.
    pulse_center : float, default=0.0
        Spatial center position ``s_0`` of the pulse at t=0.

    Returns
    -------
    Animation
        Mapping from timestamps to animation frames containing local bone rotations.
    """
    d_arr, t_arr, ceil_distances, wave_number, angular_frequency = _compute_wave_grid(
        distances, time_stamps, num_waves, wave_duration
    )
    t_1d = np.atleast_1d(t_arr)

    spatial_amplitude = wave_amplitude * np.exp(growth_factor * ceil_distances)
    phase = (
        wave_number * (ceil_distances[None, :] - pulse_center)
        - angular_frequency * t_1d[:, None]
        + phi_s
        + phi_t
    )

    sigma = float(pulse_width)
    envelope = np.exp(-0.5 * (phase / sigma) ** 2)

    spatial_deriv = (
        spatial_amplitude[None, :]
        * envelope
        * (
            growth_factor * np.cos(phase)
            - wave_number * (np.sin(phase) + (phase / (sigma**2)) * np.cos(phase))
        )
    )

    tangent = np.stack(
        (
            np.ones_like(spatial_deriv),
            spatial_deriv,
            np.zeros_like(spatial_deriv),
        ),
        axis=-1,
    )
    tangent /= np.linalg.norm(tangent, axis=-1, keepdims=True)

    return _integrate_tangents_to_rotations(d_arr, t_arr, tangent, ceil_distances)


def chain_wave_generator(
    armature: Armature,
    index_bones: list[int],
    wave_amplitude: float,
    wave_duration: float,
    frame_rate: float,
    growth_factor: float = 0,
    num_waves: float = 2.6,
    phi_s: float = 0.0,
    phi_t: float = 0.0,
    pulse_width: float = 1.0,
    pulse_center: float = 0.0,
    axis=2,
    wave: Literal["standing", "travelling", "pulse"] = "travelling",
) -> Animation:
    """
    Generate a sequence of armature positions representing a wave animation (standing, travelling, or pulse).

    The wave is evaluated at discrete time steps determined by
    ``frame_rate`` and ``wave_duration``. The resulting sequence can be
    used to create a looping animation of the armature.

    Parameters
    ----------
    armature : Armature
        Armature whose bones are used to generate the wave.

    index_bones : list[int]
        Indices of the bones in ``armature`` to which the wave is applied.
        The bones must form a continuous chain.

    wave_amplitude : float
        Base amplitude ``A`` of the wave at the root of the armature.
        The maximum displacement at the root is ``2 * wave_amplitude`` for standing waves,
        and ``wave_amplitude`` for travelling waves.

    wave_duration : float
        Period ``T`` of the wave in seconds. After one wave duration,
        the temporal component completes one full oscillation, allowing
        the generated animation to loop seamlessly.

    frame_rate : float
        Number of animation frames generated per second.

    growth_factor : float, default=0
        Exponential spatial growth rate ``g`` of the wave amplitude.

        The amplitude at arc length ``s`` is given by::

            A(s) = A * exp(g * s)

        Positive values increase the amplitude toward the end of the
        armature, zero produces a constant amplitude, and negative
        values produce exponential damping.

    num_waves : float, default=2.6
        Number of complete spatial wavelengths across the total armature
        length ``L``. The corresponding wave number is::

            k = 2 * pi * num_waves / L

    phi_s : float, default=0.0
        Spatial phase offset in radians. This shifts the wave along the
        armature without changing its wavelength.

    phi_t : float, default=0.0
        Temporal phase offset in radians. This shifts the starting point
        of the oscillation in time.

    pulse_width : float, default=1.0
        Spatial width parameter of the pulse envelope (used when wave='pulse').

    pulse_center : float, default=0.0
        Initial spatial location of the pulse peak at t = 0 (used when wave='pulse').

    axis : int, default=2
        Spatial axis along which the displacement is applied.
        ``0`` corresponds to X, ``1`` corresponds to Y, and ``2``
        corresponds to Z.

    wave : str, literal, default='travelling'
        Type of wave to generate.
        ``standing`` corresponds to a standing wave.
        ``travelling`` corresponds to a travelling wave.
        ``pulse`` corresponds to a wave pulse propagating in positive x.

    Returns
    -------
    Animation
        Mapping from timestamps to animation frames containing local bone rotations.

    Raises
    ------
    ValueError
        If the bones specified by ``index_bones`` do not form a
        continuous chain.

    Notes
    -----
    For travelling waves, the displacement for each bone is calculated using::

        u(s, t) = A exp(g s) sin(k s - omega t + phi_s + phi_t)

    For standing waves, the displacement is calculated using::

        u(s, t) = 2 A exp(g s) sin(k s + phi_s) sin(omega t + phi_t)

    For pulse waves, the displacement is calculated using::

        u(s, t) = A exp(g s) exp( - (k (s - pulse_center) - omega t + phi_s + phi_t)^2 / (2 pulse_width^2) ) cos(k (s - pulse_center) - omega t + phi_s + phi_t)

    where the spatial wave number ``k`` and temporal angular frequency
    ``omega`` are defined as::

        k = 2 * pi * N / L

        omega = 2 * pi / T

    ``N`` is the number of spatial waves, ``L`` is the total armature
    length, and ``T`` is ``wave_duration``.
    """

    if not _check_armature_chain(armature, index_bones):
        raise ValueError("Bones in index_bones do not form a continuous chain.")

    distances = [0.0]
    for bone_idx in index_bones:
        bone = armature.bones_list[bone_idx]
        bone_len = float(np.linalg.norm(np.array(bone.tail) - np.array(bone.head)))
        distances.append(distances[-1] + bone_len)

    total_timestamps = int(frame_rate * wave_duration)
    time_stamps = np.linspace(0, wave_duration, num=total_timestamps, endpoint=False)

    if wave == "standing":
        return _standing_wave_generator(
            distances=distances,
            wave_amplitude=wave_amplitude,
            wave_duration=wave_duration,
            time_stamps=time_stamps,
            growth_factor=growth_factor,
            num_waves=num_waves,
            phi_s=phi_s,
            phi_t=phi_t,
        )
    elif wave == "travelling":
        return _travelling_wave_generator(
            distances=distances,
            wave_amplitude=wave_amplitude,
            wave_duration=wave_duration,
            time_stamps=time_stamps,
            growth_factor=growth_factor,
            num_waves=num_waves,
            phi_s=phi_s,
            phi_t=phi_t,
        )
    elif wave == "pulse":
        return _pulse_wave_generator(
            distances=distances,
            wave_amplitude=wave_amplitude,
            wave_duration=wave_duration,
            time_stamps=time_stamps,
            growth_factor=growth_factor,
            num_waves=num_waves,
            phi_s=phi_s,
            phi_t=phi_t,
            pulse_width=pulse_width,
            pulse_center=pulse_center,
        )
    else:
        raise ValueError(
            f"Invalid wave type: {wave}, Permitted types: {{'standing', 'travelling', 'pulse'}}"
        )
