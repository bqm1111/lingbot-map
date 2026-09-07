"""Offscreen 3D voxel rendering, for looking at occupancy rather than scoring it.

Open3D's EGL headless renderer draws one merged ``TriangleMesh`` of shrunken cubes, which
is far cheaper than one geometry per voxel and keeps the voxel grid legible: the small gap
between cubes is what lets you see that a wall is a wall and not a solid block.

Nothing here is part of the evaluation path; it exists so a person can see the difference
between two occupancy predictions that a scalar IoU only summarises.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

_CUBE_V = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                    [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
_CUBE_F = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4],
                    [1, 2, 6], [1, 6, 5], [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]],
                   np.int32)


def cube_mesh(idx: np.ndarray, colors: np.ndarray, voxel: float, origin: Sequence[float],
              shrink: float = 0.88):
    """Merged cube mesh at integer voxel indices, one colour per voxel."""
    import open3d as o3d
    idx = np.asarray(idx)
    if len(idx) == 0:
        return None
    n = len(idx)
    c = (idx.astype(float) + 0.5) * voxel + np.asarray(origin, float)
    V = (c[:, None, :] + (_CUBE_V[None] - 0.5) * voxel * shrink).reshape(-1, 3)
    F = (_CUBE_F[None] + (np.arange(n) * 8)[:, None, None]).reshape(-1, 3)
    C = np.repeat(np.asarray(colors, float), 8, axis=0)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    m.vertex_colors = o3d.utility.Vector3dVector(np.clip(C, 0, 1))
    m.compute_vertex_normals()
    return m


def height_colors(idx: np.ndarray, voxel: float, z0: float, nz: int, cmap: str = "viridis"):
    import matplotlib.cm as cm
    z = (np.asarray(idx)[:, 2].astype(float) + 0.5) * voxel + z0
    t = np.clip((z - z0) / max(nz * voxel, 1e-6), 0, 1)
    return cm.get_cmap(cmap)(t)[:, :3]


def render(meshes: Sequence, size: Tuple[int, int] = (900, 640),
           eye_offset: Sequence[float] = (-26.0, -20.0, 26.0),
           look_at: Optional[Sequence[float]] = None, fov: float = 48.0,
           bg: Sequence[float] = (1, 1, 1, 1)) -> np.ndarray:
    """Render a list of meshes from one oblique viewpoint; returns an RGB array."""
    import open3d as o3d
    r = o3d.visualization.rendering.OffscreenRenderer(int(size[0]), int(size[1]))
    r.scene.set_background(list(bg))
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    allpts = []
    for i, m in enumerate(meshes):
        if m is None:
            continue
        r.scene.add_geometry(f"g{i}", m, mat)
        allpts.append(np.asarray(m.vertices))
    if not allpts:
        return np.full((int(size[1]), int(size[0]), 3), 255, np.uint8)
    P = np.concatenate(allpts, 0)
    ctr = np.asarray(look_at, float) if look_at is not None else P.mean(0)
    eye = ctr + np.asarray(eye_offset, float)
    r.setup_camera(float(fov), ctr.astype(float), eye.astype(float),
                   np.array([0.0, 0.0, 1.0]))
    r.scene.scene.set_sun_light([-0.4, -0.5, -0.8], [1.0, 1.0, 1.0], 78000)
    r.scene.scene.enable_sun_light(True)
    img = np.asarray(r.render_to_image())
    del r
    return img


def _segment_box(p0, p1, r):
    """One line segment as a thin oriented box -- Open3D's lit shader ignores LineSets."""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    L = float(np.linalg.norm(d))
    if L < 1e-9:
        return None, None
    z = d / L
    a = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    V = (_CUBE_V - np.array([0.5, 0.5, 0.0])) * np.array([2 * r, 2 * r, L])
    return V @ R.T + p0, _CUBE_F


def frustum_mesh(T_cam_to_grid, fov_x_deg, fov_y_deg, depth, radius,
                 colour=(0.05, 0.05, 0.05)):
    """Wireframe camera frustum in grid coordinates: apex at the optical centre.

    ``T_cam_to_grid`` is the usual pinhole convention -- columns right / down / forward --
    so the frustum shows both where the camera sits and which way it looks, which a voxel
    grid alone cannot convey.
    """
    import open3d as o3d
    T = np.asarray(T_cam_to_grid, float)
    C, R = T[:3, 3], T[:3, :3]
    tx = np.tan(np.radians(fov_x_deg) / 2.0)
    ty = np.tan(np.radians(fov_y_deg) / 2.0)
    far = [C + R @ (np.array([sx * tx, sy * ty, 1.0]) * depth)
           for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    segs = [(C, f) for f in far] + [(far[i], far[(i + 1) % 4]) for i in range(4)]
    V, F, n = [], [], 0
    for p0, p1 in segs:
        v, f = _segment_box(p0, p1, radius)
        if v is None:
            continue
        V.append(v); F.append(f + n); n += len(v)
    if not V:
        return None
    V = np.concatenate(V, 0); F = np.concatenate(F, 0)
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V),
                                  o3d.utility.Vector3iVector(F))
    m.vertex_colors = o3d.utility.Vector3dVector(np.tile(colour, (len(V), 1)))
    m.compute_vertex_normals()
    return m


__all__ = ["cube_mesh", "height_colors", "render", "frustum_mesh"]
