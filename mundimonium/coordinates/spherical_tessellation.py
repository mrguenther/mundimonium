from __future__ import annotations

from mundimonium.coordinates.tessellation import (
    Tessellation, TessellationFace, TessellationVertex
)
from mundimonium.coordinates.isometric import IsometricPoint

import math
from typing import Self, override
import numpy as np


class SphericalTessellation(Tessellation):
  """
  Specialized Tessellation for meshes projected onto a sphere S^2.
  Overrides distance and path queries with exact O(1) Great-Circle / Haversine
  calculations and Slerp paths.
  """

  def __init__(
      self,
      radius: float = 1.0,
      frequency: int | None = None,
      subdivisions: int | None = None,
      center: tuple[float, float, float] = (0.0, 0.0, 0.0),
      *,
      vertex_type: type[TessellationVertex] | None = None,
      face_type: type[TessellationFace] | None = None):
    super().__init__(vertex_type=vertex_type, face_type=face_type)
    self.radius = float(radius)
    self.center = tuple(float(c) for c in center)

    effective_freq = frequency
    if effective_freq is None and subdivisions is not None:
      effective_freq = 2 ** subdivisions

    if effective_freq is not None:
      self._generate_tessellation(frequency=effective_freq, center=self.center)

  @override
  def _generate_tessellation(
      self,
      frequency: int = 1,
      center: tuple[float, float, float] = (0.0, 0.0, 0.0)
  ) -> None:
    """
    Populates this tessellation as a Class I geodesic sphere of given frequency.
    Populates the 3D rectangular spatial coordinates of each vertex into its
    projection_coordinates.
    """
    if frequency < 1:
      raise ValueError(f"Frequency must be an integer >= 1, got {frequency}.")

    self.center = tuple(float(c) for c in center)
    cx, cy, cz = self.center

    # 1. Base regular icosahedron vertices (normalized to unit sphere)
    phi = (1.0 + math.sqrt(5.0)) / 2.0
    base_raw = [
        [-1.0,  phi,  0.0], [ 1.0,  phi,  0.0], [-1.0, -phi,  0.0], [ 1.0, -phi,  0.0],
        [ 0.0, -1.0,  phi], [ 0.0,  1.0,  phi], [ 0.0, -1.0, -phi], [ 0.0,  1.0, -phi],
        [ phi,  0.0, -1.0], [ phi,  0.0,  1.0], [-phi,  0.0, -1.0], [-phi,  0.0,  1.0]
    ]
    norm_factor = math.sqrt(1.0 + phi ** 2)
    base_verts = [np.array(coord, dtype=np.float64) / norm_factor for coord in base_raw]

    # 20 base equilateral faces (oriented CCW when viewed from outside)
    base_faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1]
    ]

    # Vertex cache to ensure shared vertices across adjacent faces
    vertex_map: dict[tuple[float, float, float], TessellationVertex] = {}

    def get_or_create_vertex(pos_unit: np.ndarray) -> TessellationVertex:
      norm = np.linalg.norm(pos_unit)
      p_sphere = pos_unit / norm if norm > 1e-12 else pos_unit

      # Spatial (rectangular) coordinates on the sphere
      x = float(cx + self.radius * p_sphere[0])
      y = float(cy + self.radius * p_sphere[1])
      z = float(cz + self.radius * p_sphere[2])

      # Key rounded to 8 decimal places for floating-point tolerance
      key = (round(x, 8), round(y, 8), round(z, 8))
      if key not in vertex_map:
        v = self.vertex_type([x, y, z])
        self.add_vertex(v)
        vertex_map[key] = v
      return vertex_map[key]

    # 2. Subdivide each of the 20 icosahedral faces
    for v0_idx, v1_idx, v2_idx in base_faces:
      v0 = base_verts[v0_idx]
      v1 = base_verts[v1_idx]
      v2 = base_verts[v2_idx]

      # Grid of vertex points on this face: (i, j) where i + j <= frequency
      grid: dict[tuple[int, int], TessellationVertex] = {}
      for i in range(frequency + 1):
        for j in range(frequency + 1 - i):
          k = frequency - i - j
          pos = (k * v0 + i * v1 + j * v2) / frequency
          grid[(i, j)] = get_or_create_vertex(pos)

      # Triangulate the subgrid
      for i in range(frequency):
        for j in range(frequency - i):
          # Upward-pointing triangle: (i, j) -> (i + 1, j) -> (i, j + 1)
          v_bl = grid[(i, j)]
          v_br = grid[(i + 1, j)]
          v_top = grid[(i, j + 1)]
          self.add_face([v_bl, v_br, v_top])

          # Downward-pointing triangle: (i + 1, j) -> (i + 1, j + 1) -> (i, j + 1)
          if i + j + 1 < frequency:
            v_tr = grid[(i + 1, j + 1)]
            self.add_face([v_br, v_tr, v_top])

  @classmethod
  def create_geodesic_sphere(
      cls,
      radius: float = 1.0,
      frequency: int | None = None,
      subdivisions: int | None = None,
      center: tuple[float, float, float] = (0.0, 0.0, 0.0)
  ) -> Self:
    """
    Factory constructor to build and return a fully populated SphericalTessellation
    geodesic sphere.
    """
    if frequency is None and subdivisions is None:
      frequency = 1
    return cls(
        radius=radius,
        frequency=frequency,
        subdivisions=subdivisions,
        center=center
    )

  def _point_to_3d_unit(self, pt: IsometricPoint) -> np.ndarray:
    """Maps an IsometricPoint to a unit direction vector from the sphere's center."""
    face = pt.grid
    wb, ws, wd = pt.barycentric
    v_b, v_s, v_d = face.vertex_b, face.vertex_s, face.vertex_d

    # Interpolate rectangular coordinates from vertices
    p3d = (
        wb * np.array(v_b.projection_coordinates, dtype=np.float64) +
        ws * np.array(v_s.projection_coordinates, dtype=np.float64) +
        wd * np.array(v_d.projection_coordinates, dtype=np.float64)
    )
    # Shift relative to center
    rel = p3d - np.array(self.center, dtype=np.float64)
    norm = np.linalg.norm(rel)
    if norm < 1e-12:
      return rel
    return rel / norm

  @override
  def geodesic_distance(self, p1: IsometricPoint, p2: IsometricPoint) -> float:
    """Computes exact O(1) Great-Circle distance between two points on the sphere."""
    v1 = self._point_to_3d_unit(p1)
    v2 = self._point_to_3d_unit(p2)
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
    Returns 3D Euclidean points sampled along the great-circle arc on the sphere.
    """
    v1 = self._point_to_3d_unit(p1)
    v2 = self._point_to_3d_unit(p2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    omega = np.arccos(dot)

    c = np.array(self.center, dtype=np.float64)
    if omega < 1e-10:
      return [c + self.radius * v1]

    sin_omega = np.sin(omega)
    path_3d = []
    for t in np.linspace(0.0, 1.0, num_samples):
      p_dir = (np.sin((1.0 - t) * omega) / sin_omega) * v1 + (np.sin(t * omega) / sin_omega) * v2
      path_3d.append(c + self.radius * p_dir)
    return path_3d


def build_geodesic_sphere(
    radius: float = 1.0,
    frequency: int | None = None,
    subdivisions: int | None = None,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
) -> SphericalTessellation:
  """
  Builds a SphericalTessellation given standard parameters of a geodesic sphere,
  populating the 3D rectangular spatial coordinates of each vertex into its
  projection coordinates.

  Parameters:
    radius: Radius of the geodesic sphere.
    frequency: Class I subdivision frequency nu >= 1 (1 = base icosahedron, 2 = 80 faces, 3 = 180 faces, etc.).
    subdivisions: Optional power-of-two subdivision level (k -> frequency = 2^k).
    center: 3D (x, y, z) rectangular coordinates of the sphere center.

  Returns:
    A fully populated SphericalTessellation instance.
  """
  return SphericalTessellation.create_geodesic_sphere(
      radius=radius,
      frequency=frequency,
      subdivisions=subdivisions,
      center=center
  )