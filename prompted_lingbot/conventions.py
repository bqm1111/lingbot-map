"""LingbotMap geometric conventions -- empirically verified, not assumed.

Every statement below was established by measurement in the Phase-0 audit
(see ``docs/prompted_lingbot_feasibility.md`` for the numbers).  Two of them
CONTRADICT the docstrings/comments in the upstream repository, so read this
module before touching any geometry.

Verified facts
--------------
1. ``GCTStream.inference_streaming`` returns ``pose_enc [B, S, 9]``,
   ``depth [B, S, H, W, 1]``, ``depth_conf [B, S, H, W]``.
   ``world_points`` is NOT produced (the point head is disabled by default).

2. ``pose_enc`` is ``absT_quaR_FoV``: ``[t(3), quat(4), fov_h, fov_w]``.

3. ``pose_encoding_to_extri_intri(pose_enc, (H, W))`` returns a ``[B, S, 3, 4]``
   matrix that is **CAMERA-TO-WORLD**, i.e. ``x_world = R @ x_cam + t`` and the
   camera centre in world coordinates is simply ``t``.

   The upstream docstring claims "camera from world" (world-to-camera).  That is
   WRONG.  Measured on TartanAir ForestEnv/Data_hard/P000 (64 frames), aligning
   the predicted trajectory to ground truth with Umeyama Sim(3):

       reading      ATE(Sim3)   median rotation error
       c2w          0.173 m     5.09 deg
       w2c          0.940 m     172.63 deg

   A 172 deg median rotation error is a sign flip, not a near miss.

4. Consequently ``demo.postprocess`` -- which inverts the matrix before saving --
   writes a **WORLD-TO-CAMERA** ``extrinsic`` into the ``--save_predictions``
   NPZs, despite its docstring saying "c2w".  This module never reads that field;
   it works from ``pose_enc`` directly.

5. ``lingbot_map.utils.geometry.depth_to_world_coords_points`` and
   ``GCTBase._unproject_depth_to_world`` both internally invert their
   ``extrinsic`` argument, so they expect world-to-camera input.  Feeding them
   the raw ``pose_encoding_to_extri_intri`` output produces wrong world points.
   (They agree with each other to 4.1e-6, so the bug is consistent, not random.)
   The deployed model never calls them, because the point head is disabled.

6. ``depth`` is **Z-depth** in the camera frame (the z component of the camera-frame
   point), not ray distance.  Measured multi-view relative depth reprojection error:
   z-depth 0.038 vs ray-depth 0.128.

7. Camera frame is OpenCV: +x right, +y down, +z forward.
   World frame is the first frame's camera frame.  Scale is arbitrary (monocular);
   on the audit sequence the predicted-to-metric scale was ~25.3.

Sim(3) correction
-----------------
A correction ``C = (s, R, t)`` is applied as::

    x_metric = R @ (s * x_pred) + t          # world points
    c_metric = R @ (s * c_pred) + t          # camera centres
    R_c2w_metric = R @ R_c2w_pred            # camera orientations
    D_metric = s * D_pred                    # depth, scale only

Depth takes the scale factor alone because it is a camera-frame coordinate: a
rigid motion of the world does not change what the camera sees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# OpenCV camera axes, world frame = first camera frame.
CAMERA_CONVENTION = "opencv_x_right_y_down_z_forward"
EXTRINSIC_FROM_POSE_ENC = "camera_to_world"
DEPTH_REPRESENTATION = "z_depth_camera_frame"
POSE_ENCODING_TYPE = "absT_quaR_FoV"


# --------------------------------------------------------------------------- #
# Sim(3)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Sim3:
    """A similarity transform ``x -> R @ (s * x) + t``."""

    s: float
    R: np.ndarray  # (3, 3)
    t: np.ndarray  # (3,)

    @staticmethod
    def identity() -> "Sim3":
        return Sim3(1.0, np.eye(3), np.zeros(3))

    def __post_init__(self) -> None:
        if self.R.shape != (3, 3):
            raise ValueError(f"R must be (3, 3), got {self.R.shape}")
        if self.t.shape != (3,):
            raise ValueError(f"t must be (3,), got {self.t.shape}")

    # -- application ------------------------------------------------------- #
    def apply_points(self, x: np.ndarray) -> np.ndarray:
        """Transform ``(..., 3)`` world points."""
        return (self.s * x) @ self.R.T + self.t

    def apply_centres(self, c: np.ndarray) -> np.ndarray:
        """Transform ``(..., 3)`` camera centres (identical to points)."""
        return self.apply_points(c)

    def apply_rotations(self, R_c2w: np.ndarray) -> np.ndarray:
        """Transform ``(..., 3, 3)`` camera-to-world rotations."""
        return np.einsum("ij,...jk->...ik", self.R, R_c2w)

    def apply_depth(self, depth: np.ndarray) -> np.ndarray:
        """Scale depth.  Rotation and translation do not affect camera-frame Z."""
        return self.s * depth

    def apply_pose_c2w(self, extrinsic_c2w: np.ndarray) -> np.ndarray:
        """Transform a ``(..., 3, 4)`` camera-to-world extrinsic."""
        R = self.apply_rotations(extrinsic_c2w[..., :3, :3])
        t = self.apply_centres(extrinsic_c2w[..., :3, 3])
        return np.concatenate([R, t[..., None]], axis=-1)

    # -- algebra ----------------------------------------------------------- #
    def compose(self, other: "Sim3") -> "Sim3":
        """``self ∘ other``: apply ``other`` first, then ``self``."""
        return Sim3(
            self.s * other.s,
            self.R @ other.R,
            self.s * (self.R @ other.t) + self.t,
        )

    def inverse(self) -> "Sim3":
        s_inv = 1.0 / self.s
        R_inv = self.R.T
        return Sim3(s_inv, R_inv, -s_inv * (R_inv @ self.t))

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return {"s": float(self.s), "R": self.R.tolist(), "t": self.t.tolist()}

    @staticmethod
    def from_dict(d: dict) -> "Sim3":
        return Sim3(float(d["s"]), np.asarray(d["R"], float), np.asarray(d["t"], float))


def umeyama_sim3(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> Sim3:
    """Least-squares Sim(3) mapping ``src`` onto ``dst``.

    Args:
        src: ``(N, 3)`` source points.
        dst: ``(N, 3)`` target points.
        with_scale: if False, the scale is fixed at 1 (SE(3) alignment).

    Returns:
        The ``Sim3`` minimising ``||dst - (R (s src) + t)||^2``.

    Raises:
        ValueError: if fewer than 3 correspondences are given.
    """
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(f"src/dst must both be (N, 3); got {src.shape} and {dst.shape}")
    n = src.shape[0]
    if n < 3:
        raise ValueError(f"Umeyama needs at least 3 correspondences, got {n}")

    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    var_s = (sc ** 2).sum() / n
    s = float(np.trace(np.diag(D) @ S) / var_s) if (with_scale and var_s > 1e-12) else 1.0
    t = mu_d - s * (R @ mu_s)
    return Sim3(s, R, t)


def rotation_conditioning(src: np.ndarray, residual_rms: float = 0.0,
                          median_scene_depth: float = 0.0) -> dict:
    """Spatial conditioning of a rotation estimated from point correspondences.

    A singular-value *ratio* is not a usable degeneracy test for camera-centre
    correspondences: a vehicle driving down a straight road produces a window
    whose ratio sits around 0.03-0.08 -- far above any threshold tuned for exact
    collinearity -- while the rotation about the direction of travel is in
    practice unconstrained.  What matters is the **absolute extent** of the
    correspondence cloud in its least-observed direction, measured against the
    noise that has to be resolved, and the **baseline** measured against the
    depth of the scene the rotation will be applied to.

    Args:
        src: ``(N, 3)`` correspondences, in the units the rotation will act on.
        residual_rms: RMS fit residual, i.e. the noise the extent must beat.
        median_scene_depth: typical scene depth, for the baseline test.

    Returns:
        ``extent`` (the three singular values, largest first, as lengths),
        ``baseline``, ``lateral``, ``vertical``, ``extent_ratio``,
        ``baseline_over_depth`` and ``rotation_sigma_rad`` -- a first-order
        estimate of the angular uncertainty, ``residual_rms / smallest extent``.
    """
    src = np.asarray(src, float)
    n = src.shape[0]
    out = {"n": n, "extent": (0.0, 0.0, 0.0), "baseline": 0.0, "lateral": 0.0,
           "vertical": 0.0, "extent_ratio": 0.0, "baseline_over_depth": 0.0,
           "rotation_sigma_rad": np.inf}
    if n < 3:
        return out
    c = src - src.mean(0)
    # Singular values of the centred cloud scaled to RMS lengths, so that
    # "extent" is in metres and comparable with residual_rms.
    sv = np.linalg.svd(c, compute_uv=False) / np.sqrt(n)
    out["extent"] = tuple(float(v) for v in sv)
    out["baseline"], out["lateral"], out["vertical"] = (float(sv[0]), float(sv[1]), float(sv[2]))
    out["extent_ratio"] = float(sv[2] / sv[0]) if sv[0] > 1e-12 else 0.0
    if median_scene_depth > 1e-9:
        out["baseline_over_depth"] = float(sv[0] / median_scene_depth)
    if sv[2] > 1e-12:
        out["rotation_sigma_rad"] = float(residual_rms / sv[2])
    elif residual_rms <= 1e-12:
        out["rotation_sigma_rad"] = 0.0
    return out


def sim3_degeneracy(src: np.ndarray) -> float:
    """Ratio of smallest to largest singular value of the centred source points.

    Near zero means the correspondences are collinear (or coincident) and the
    rotation about the line is unconstrained.  Callers should refuse to update
    a rotation when this falls below a threshold.
    """
    src = np.asarray(src, float)
    if src.shape[0] < 3:
        return 0.0
    c = src - src.mean(0)
    sv = np.linalg.svd(c, compute_uv=False)
    if sv[0] < 1e-12:
        return 0.0
    return float(sv[-1] / sv[0])


# --------------------------------------------------------------------------- #
# Projection helpers (Z-depth, OpenCV camera, camera-to-world extrinsic)
# --------------------------------------------------------------------------- #
def pixel_rays(H: int, W: int, K: np.ndarray) -> np.ndarray:
    """``(H, W, 3)`` normalised-plane directions with unit z."""
    u, v = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    return np.stack(
        [(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], axis=-1
    )


def depth_to_camera_points(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """``(H, W)`` Z-depth -> ``(H, W, 3)`` camera-frame points."""
    H, W = depth.shape
    return pixel_rays(H, W, K) * depth[..., None]


def camera_to_world(points_cam: np.ndarray, extrinsic_c2w: np.ndarray) -> np.ndarray:
    """Apply a ``(3, 4)`` CAMERA-TO-WORLD extrinsic to ``(..., 3)`` camera points."""
    R, t = extrinsic_c2w[:3, :3], extrinsic_c2w[:3, 3]
    return points_cam @ R.T + t


def world_to_camera(points_world: np.ndarray, extrinsic_c2w: np.ndarray) -> np.ndarray:
    """Inverse of :func:`camera_to_world`."""
    R, t = extrinsic_c2w[:3, :3], extrinsic_c2w[:3, 3]
    return (points_world - t) @ R


def depth_to_world_points(depth: np.ndarray, K: np.ndarray, extrinsic_c2w: np.ndarray) -> np.ndarray:
    """``(H, W)`` Z-depth -> ``(H, W, 3)`` world points."""
    return camera_to_world(depth_to_camera_points(depth, K), extrinsic_c2w)


def camera_centre(extrinsic_c2w: np.ndarray) -> np.ndarray:
    """Camera centre in world coordinates: the translation of a c2w extrinsic."""
    return np.asarray(extrinsic_c2w)[..., :3, 3]


def project(points_cam: np.ndarray, K: np.ndarray):
    """``(..., 3)`` camera points -> ``(u, v, z)``."""
    z = points_cam[..., 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = points_cam[..., 0] / zs * K[0, 0] + K[0, 2]
    v = points_cam[..., 1] / zs * K[1, 1] + K[1, 2]
    return u, v, z


def pose_enc_to_c2w_and_K(pose_enc, image_hw):
    """``pose_enc [.., S, 9]`` -> ``(extrinsic_c2w [.., S, 3, 4], K [.., S, 3, 3])``.

    Thin wrapper over the upstream converter that exists purely so the
    camera-to-world fact recorded in this module is applied in exactly one place.
    """
    import torch
    from lingbot_map.utils.pose_enc import pose_encoding_to_extri_intri

    if not isinstance(pose_enc, torch.Tensor):
        pose_enc = torch.as_tensor(np.asarray(pose_enc, np.float32))
    squeeze = pose_enc.dim() == 2
    if squeeze:
        pose_enc = pose_enc[None]
    extrinsic, K = pose_encoding_to_extri_intri(pose_enc.float(), tuple(image_hw))
    if squeeze:
        extrinsic, K = extrinsic[0], K[0]
    return extrinsic, K


# --------------------------------------------------------------------------- #
# Rotation helpers
# --------------------------------------------------------------------------- #
def rotation_geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> np.ndarray:
    """Geodesic angle in degrees between ``(..., 3, 3)`` rotation matrices."""
    rel = np.einsum("...ij,...kj->...ik", R_a, R_b)
    tr = np.trace(rel, axis1=-2, axis2=-1)
    return np.degrees(np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0)))


def axis_angle_to_matrix(w: np.ndarray) -> np.ndarray:
    """Rodrigues: ``(..., 3)`` axis-angle -> ``(..., 3, 3)``."""
    w = np.asarray(w, float)
    theta = np.linalg.norm(w, axis=-1, keepdims=True)
    small = theta < 1e-8
    axis = np.where(small, 0.0, w / np.where(small, 1.0, theta))
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    zero = np.zeros_like(x)
    Kx = np.stack(
        [np.stack([zero, -z, y], -1), np.stack([z, zero, -x], -1), np.stack([-y, x, zero], -1)],
        axis=-2,
    )
    th = theta[..., None]
    eye = np.broadcast_to(np.eye(3), Kx.shape)
    return eye + np.sin(th) * Kx + (1.0 - np.cos(th)) * (Kx @ Kx)


def matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` -> ``(..., 3)`` axis-angle."""
    R = np.asarray(R, float)
    tr = np.trace(R, axis1=-2, axis2=-1)
    theta = np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))
    v = np.stack(
        [R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]],
        axis=-1,
    )
    denom = 2.0 * np.sin(theta)
    safe = np.abs(denom) > 1e-8
    out = np.where(safe[..., None], v * (theta / np.where(safe, denom, 1.0))[..., None], v * 0.5)
    return out
