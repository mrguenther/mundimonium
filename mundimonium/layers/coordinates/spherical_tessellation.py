from __future__ import annotations

from mundimonium.layers.coordinates.tessellation import Tessellation
from mundimonium.layers.coordinates.isometric import IsometricPoint

from typing import override
import numpy as np

class SphericalTessellation(Tessellation):
  """
  Specialized Tessellation for meshes projected onto a sphere S^2.
  Overrides distance and path queries with exact O(1) Great-Circle / Haversine
  calculations and Slerp paths.
  """

  def __init__(self, radius: float = 1.0):
    super().__init__()
    self.radius = float(radius)

  def _point_to_3d(self, pt: IsometricPoint) -> np.ndarray:
    face = pt.grid
    wb, ws, wd = pt.barycentric
    v_b, v_s, v_d = face.vertex_b, face.vertex_s, face.vertex_d

    p3d = (
        wb * np.array(v_b.projection_coordinates, dtype=np.float64) +
        ws * np.array(v_s.projection_coordinates, dtype=np.float64) +
        wd * np.array(v_d.projection_coordinates, dtype=np.float64)
    )
    norm = np.linalg.norm(p3d)
    if norm < 1e-12:
      return p3d
    return p3d / norm

  @override
  def geodesic_distance(self, p1: IsometricPoint, p2: IsometricPoint) -> float:
    v1 = self._point_to_3d(p1)
    v2 = self._point_to_3d(p2)
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(self.radius * np.arccos(cos_theta))

  @override
  def shortest_path(
      self,
      p1: IsometricPoint,
      p2: IsometricPoint,
      num_samples: int = 20
  ) -> list[np.ndarray]:
    """
    Returns 3D Euclidean points sampled along the great-circle arc.
    """
    v1 = self._point_to_3d(p1)
    v2 = self._point_to_3d(p2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    omega = np.arccos(dot)

    if omega < 1e-10:
      return [self.radius * v1]

    sin_omega = np.sin(omega)
    path_3d = []
    for t in np.linspace(0.0, 1.0, num_samples):
      p = (np.sin((1.0 - t) * omega) / sin_omega) * v1 + (np.sin(t * omega) / sin_omega) * v2
      path_3d.append(self.radius * p)
    return path_3d