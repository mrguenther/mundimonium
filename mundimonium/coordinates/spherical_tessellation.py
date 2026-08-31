from __future__ import annotations

from mundimonium.coordinates.tessellation import (
    Tessellation, TessellationFace, TessellationVertex
)
from mundimonium.coordinates.isometric import IsometricPoint

from typing import Self, override
import math
import numpy as np


class SphericalTessellation(Tessellation):
  """
  Geodesic sphere tessellation.
  """

  def __init__(
      self,
      radius: float = 1.0,
      frequency: int = 1,
      center: tuple[float, float, float] = (0.0, 0.0, 0.0),
      **kwargs):
    """Constructs a spherical mesh.

    Exactly one of the arguments ('frequency', 'subdivisions') must be provided.

    Args:
      radius:       Radius of the sphere.
      frequency:    Class I subdivision frequency nu >= 1 (1 = base icosahedron,
                    2 = 80 faces, 3 = 180 faces, etc.).
      center:       3D (x, y, z) rectangular coordinates of the sphere center.
      **kwargs:     Forwarded up the method resolution order.
    """
    ideal_surface_area = 4.0 * math.pi * radius * radius
    face_count = 20 * frequency * frequency
    ideal_face_area = ideal_surface_area / face_count
    ideal_face_side_length = math.sqrt(ideal_face_area * 4 / math.sqrt(3))

    super().__init__(face_side_length=ideal_face_side_length, **kwargs)

    self._radius = float(radius)
    self._center = tuple(float(c) for c in center)

    self._generate_tessellation(frequency=frequency)

  @property
  def radius(self) -> float:
    return self._radius

  @property
  def center(self) -> tuple[float, float, float]:
    return self._radius

  def _spherical_to_3d(
      self, colatitude: float, longitude: float,
  ) -> np.ndarray:
    """Convert spherical coordinates to a 3D position vector.

    Args:
      colatitude: Polar angle from +Z axis in radians [0, pi].
      longitude:  Azimuthal angle from +X axis toward +Y in radians [0, 2*pi).

    Returns:
      Vector (x, y, z) of length `self.radius` relative to the sphere's center.
    """
    sin_colatitude = math.sin(colatitude)
    return self._radius * np.array([
        sin_colatitude * math.cos(longitude),
        sin_colatitude * math.sin(longitude),
        math.cos(colatitude),
    ], dtype=np.float64)

  def _locate_face_and_barycentric(
      self,
      direction: np.ndarray,
  ) -> tuple[TessellationFace, tuple[float, float, float]]:
    """Find the face containing a direction from the sphere's center, in O(1).

    Exploits face-transitivity of the icosahedron: every face plane is
    equidistant from the center, so the face plane through which a ray exits
    through is the one whose centroid direction has the largest dot product with
    the ray (a 20-way argmax). The specific sub-face within that face plane is
    then found by inverting the affine grid parameterization used when the mesh
    was generated.

    Returns:
      The containing TessellationFace and the barycentric weights (wb, ws, wd)
      of `direction` within it.
    """
    unit_dir = direction / self._radius

    base_idx = int(np.argmax(self._base_face_normals @ unit_dir))
    (_, _, v0, e1, e2, face_n, plane_d,
     d00, d01, d11, cramer_inv, subgrid) = self._spatial_index[base_idx]
    frequency = self._frequency

    # Gnomonic projection of the direction onto the base face's plane.
    denom = np.dot(face_n, unit_dir)
    if abs(denom) < 1e-15:
      w1, w2 = 1 / 3, 1 / 3
    else:
      t = plane_d / denom
      p = t * unit_dir - v0
      d20 = np.dot(p, e1)
      d21 = np.dot(p, e2)
      w1 = (d11 * d20 - d01 * d21) * cramer_inv
      w2 = (d00 * d21 - d01 * d20) * cramer_inv

    # Invert the grid parameterization pos = (k*v0 + i*v1 + j*v2) / frequency
    # to find the continuous (i, j) grid coordinates, then the sub-triangle.
    i_f = w1 * frequency
    j_f = w2 * frequency
    i0 = min(max(int(math.floor(i_f)), 0), frequency - 1)
    j0 = min(max(int(math.floor(j_f)), 0), frequency - 1 - i0)
    fi = i_f - i0
    fj = j_f - j0

    is_downward = (fi + fj > 1.0) and (i0 + j0 < frequency - 1)
    face = subgrid[(i0, j0, is_downward)]

    # (fi, fj) only located the sub-triangle: subdivision vertices are
    # renormalized onto the sphere (a nonlinear step), so they don't sit at
    # the exact affine (fi, fj) split of the flat macro-face grid. Get exact
    # weights by projecting directly onto this one face's actual plane --
    # still O(1), since it's a single face rather than a search.
    bary = self._barycentric_on_face(face, unit_dir)

    return face, bary

  def _barycentric_on_face(
      self,
      face: TessellationFace,
      unit_dir: np.ndarray,
  ) -> tuple[float, float, float]:
    """Exact barycentric weights of a direction within a given face's plane."""
    center = np.array(self._center, dtype=np.float64)
    vb = np.array(face.vertex_b.projection_coordinates, dtype=np.float64) - center
    vs = np.array(face.vertex_s.projection_coordinates, dtype=np.float64) - center
    vd = np.array(face.vertex_d.projection_coordinates, dtype=np.float64) - center

    e1 = vs - vb
    e2 = vd - vb
    face_normal = np.cross(e1, e2)

    denom = np.dot(face_normal, unit_dir)
    if abs(denom) < 1e-15:
      return (1 / 3, 1 / 3, 1 / 3)

    t = np.dot(face_normal, vb) / denom
    p = t * unit_dir - vb

    d00 = np.dot(e1, e1)
    d01 = np.dot(e1, e2)
    d11 = np.dot(e2, e2)
    d20 = np.dot(p, e1)
    d21 = np.dot(p, e2)

    inv_denom = 1.0 / (d00 * d11 - d01 * d01)
    ws = float((d11 * d20 - d01 * d21) * inv_denom)
    wd = float((d00 * d21 - d01 * d20) * inv_denom)
    wb = float(1.0 - ws - wd)

    return (wb, ws, wd)

  @override
  def new_point_at_coords(
      self, *coords: tuple[float, float]) -> IsometricPoint | None:
    """Returns an IsometricPoint at the given spherical coordinates.

    Args:
      colatitude: Polar angle from +Z axis in radians [0, pi].
      longitude:  Azimuthal angle from +X axis toward +Y in radians [0, 2*pi).

    Returns:
      An IsometricPoint on the containing face.
    """
    colatitude, longitude = coords
    direction = self._spherical_to_3d(colatitude, longitude)
    face, (wb, ws, wd) = self._locate_face_and_barycentric(direction)
    return IsometricPoint.from_barycentric(face, wb, ws, wd)

  @override
  def get_face_at_coords(
      self, *coords: tuple[float, float]) -> TessellationFace | None:
    """Returns the face containing the given spherical coordinates.

    Args:
      colatitude: Polar angle from +Z axis in radians [0, pi].
      longitude:  Azimuthal angle from +X axis toward +Y in radians [0, 2*pi).

    Returns:
      The TessellationFace containing the point.
    """
    colatitude, longitude = coords
    direction = self._spherical_to_3d(colatitude, longitude)
    face, _ = self._locate_face_and_barycentric(direction)
    return face

  @override
  def coords_at_point(self, point: IsometricPoint) -> tuple[float, float]:
    """Returns the spherical coordinates `(colatitude, longitude)` of `point`.

    Inverse of `new_point_at_coords`.

    Returns:
      colatitude: Polar angle from +Z axis in radians [0, pi].
      longitude:  Azimuthal angle from +X axis toward +Y in radians [0, 2*pi).
    """
    x, y, z = self._point_to_3d_unit(point)
    colatitude = math.acos(np.clip(z, -1.0, 1.0))
    longitude = math.atan2(y, x) % (2.0 * math.pi)
    return colatitude, longitude

  def _generate_tessellation(
      self,
      frequency: int = 1,
  ) -> None:
    """
    Populates this tessellation as a Class I geodesic sphere of given frequency.
    """
    if frequency < 1:
      raise ValueError(f"Frequency must be an integer >= 1, got {frequency}.")

    cx, cy, cz = self._center

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
      """Projects `pos_unit` onto the sphere and returns its (shared) vertex."""
      norm = np.linalg.norm(pos_unit)
      p_sphere = pos_unit / norm if norm > 1e-12 else pos_unit

      # Spatial (rectangular) coordinates on the sphere
      x = float(cx + self._radius * p_sphere[0])
      y = float(cy + self._radius * p_sphere[1])
      z = float(cz + self._radius * p_sphere[2])

      # Key rounded to 8 decimal places for floating-point tolerance
      key = (round(x, 8), round(y, 8), round(z, 8))
      if key not in vertex_map:
        v = self.vertex_type([x, y, z])
        self.add_vertex(v)
        vertex_map[key] = v
      return vertex_map[key]

    # 2. Subdivide each of the 20 icosahedral faces and build spatial index
    self._frequency = frequency
    self._spatial_index: list[tuple] = []
    base_face_normals: list[np.ndarray] = []

    for v0_idx, v1_idx, v2_idx in base_faces:
      v0 = base_verts[v0_idx]
      v1 = base_verts[v1_idx]
      v2 = base_verts[v2_idx]
      base_face_normals.append((v0 + v1 + v2) / np.linalg.norm(v0 + v1 + v2))

      # Grid of vertex points on this face: (i, j) where i + j <= frequency
      grid: dict[tuple[int, int], TessellationVertex] = {}
      for i in range(frequency + 1):
        for j in range(frequency + 1 - i):
          k = frequency - i - j
          pos = (k * v0 + i * v1 + j * v2) / frequency
          grid[(i, j)] = get_or_create_vertex(pos)

      # Triangulate the subgrid, recording face references
      subgrid: dict[tuple[int, int, bool], TessellationFace] = {}
      for i in range(frequency):
        for j in range(frequency - i):
          # Upward-pointing triangle: (i, j) -> (i + 1, j) -> (i, j + 1)
          v_bl = grid[(i, j)]
          v_br = grid[(i + 1, j)]
          v_top = grid[(i, j + 1)]
          subgrid[(i, j, False)] = self.add_face([v_bl, v_br, v_top])

          # Downward-pointing triangle: (i + 1, j) -> (i + 1, j + 1) -> (i, j + 1)
          if i + j + 1 < frequency:
            v_tr = grid[(i + 1, j + 1)]
            subgrid[(i, j, True)] = self.add_face([v_br, v_tr, v_top])

      # Precompute spatial index data for O(1) face lookup.
      # Edge normals and opposite-vertex dot products for containment test:
      edge_normals = [
          np.cross(v0, v1), np.cross(v1, v2), np.cross(v2, v0),
      ]
      opp_dots = [
          float(np.dot(edge_normals[0], v2)),
          float(np.dot(edge_normals[1], v0)),
          float(np.dot(edge_normals[2], v1)),
      ]
      # Gnomonic projection constants for barycentric sub-face indexing.
      # Base face parameterization: point = w0*v0 + w1*v1 + w2*v2 (on plane).
      # w1 and w2 are solved via Cramer's rule on edges e1=v1-v0, e2=v2-v0.
      e1 = v1 - v0
      e2 = v2 - v0
      face_n = np.cross(e1, e2)
      plane_d = float(np.dot(face_n, v0))
      d00 = float(np.dot(e1, e1))
      d01 = float(np.dot(e1, e2))
      d11 = float(np.dot(e2, e2))
      cramer_inv = 1.0 / (d00 * d11 - d01 * d01)

      # Tuple: (edge_normals, opp_dots, e1, e2, face_n, plane_d,
      #         d00, d01, d11, cramer_inv, subgrid)
      self._spatial_index.append((
          edge_normals, opp_dots, v0, e1, e2, face_n, plane_d,
          d00, d01, d11, cramer_inv, subgrid,
      ))

    self._base_face_normals = np.array(base_face_normals, dtype=np.float64)

  def _point_to_3d_unit(self, pt: IsometricPoint) -> np.ndarray:
    """Returns a unit vector pointing from the sphere's center toward `point`.
    """
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
    rel = p3d - np.array(self._center, dtype=np.float64)
    norm = np.linalg.norm(rel)
    if norm < 1e-12:
      return rel
    return rel / norm

  @override
  def geodesic_distance(self, p1: IsometricPoint, p2: IsometricPoint) -> float:
    """Computes great-circle distance between two points on the sphere."""
    v1 = self._point_to_3d_unit(p1)
    v2 = self._point_to_3d_unit(p2)
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(self._radius * np.arccos(cos_theta))

  @override
  def shortest_path(
      self,
      p1: IsometricPoint,
      p2: IsometricPoint,
      num_samples: int = 20
  ) -> list[np.ndarray]:
    """
    Returns 3D Euclidean points sampled along a great-circle arc of the sphere.
    """
    v1 = self._point_to_3d_unit(p1)
    v2 = self._point_to_3d_unit(p2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    omega = np.arccos(dot)

    c = np.array(self._center, dtype=np.float64)
    if omega < 1e-10:
      return [c + self._radius * v1]

    sin_omega = np.sin(omega)
    path_3d = []
    for t in np.linspace(0.0, 1.0, num_samples):
      p_dir = (
          (np.sin((1.0 - t) * omega) / sin_omega) * v1 +
          (np.sin(t * omega) / sin_omega) * v2
      )
      path_3d.append(c + self._radius * p_dir)
    return path_3d

  @override
  def geodesically_canonicalize_point(
      self, point: IsometricPoint) -> IsometricPoint:
    """Moves `point` to a new grid if located outside its current grid's bounds.

    Exploits the sphere's exact embedding: `point`'s (possibly out-of-range)
    barycentric coordinates on its current face extrapolate, via an affine
    combination of that face's vertices, to a point on the face's plane --
    and gnomonic projection is a bijection between a face's plane and 3D ray
    directions from the sphere's center, so normalizing that extrapolated
    point (`_point_to_3d_unit` already does exactly this) recovers the exact
    direction `point` represents. Feeding that into the O(1)
    `_locate_face_and_barycentric` then finds the correct face directly, with
    no face-by-face walking needed (unlike the generic, non-embedded case
    handled by `GenericTessellation.geodesically_canonicalize_point`).

    Mutates and returns `point`, not a copy.
    """
    unit_dir = self._point_to_3d_unit(point)
    face, (wb, ws, wd) = self._locate_face_and_barycentric(
        self._radius * unit_dir)
    alt = face.altitude
    return point.update(grid=face, b=wb * alt, s=ws * alt)
