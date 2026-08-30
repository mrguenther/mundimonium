from __future__ import annotations

from mundimonium.coordinates.tessellation import (
    Tessellation, TessellationFace, TessellationVertex
)
from mundimonium.coordinates.isometric import (
    IsometricDirection, IsometricPoint, isometric_distance
)

from numbers import Number
from typing import override
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class GenericTessellation(Tessellation):
  """
  Generic equilateral triangular mesh with geodesic pathfinding and distance
  queries powered by the Heat Method (Crane, Weischedel, Wardetzky).
  """

  def __init__(self,
               *,
               vertex_type: type[TessellationVertex] | None = None,
               face_type: type[TessellationFace] | None = None):
    super().__init__(vertex_type=vertex_type, face_type=face_type)
    self._matrices_built: bool = False
    self._heat_solver = None
    self._poisson_solver = None

  @override
  def new_point_at_coords(
      self, *coords: tuple[Number, ...]) -> IsometricPoint | None:
    """Returns a new `IsometricPoint` at the specified coordinates."""
    raise NotImplementedError()

  @override
  def get_face_at_coords(
      self, *coords: tuple[Number, ...]) -> TessellationFace | None:
    """Returns the face containing the specified coordinates."""
    raise NotImplementedError()

  @override
  def coords_at_point(self, point: IsometricPoint) -> tuple[Number, ...]:
    raise NotImplementedError()

  @override
  def invalidate_solvers(self) -> None:
    """Invalidates cached matrix factorizations when topology changes."""
    self._matrices_built = False
    self._heat_solver = None
    self._poisson_solver = None

  def _build_solvers(self) -> None:
    if not self._faces:
      raise ValueError("Cannot build solvers on a Tessellation with no faces.")

    # Ensure all vertices referenced by faces are indexed
    for face in self._faces:
      for v in face._adjacent_vertices:
        if v not in self._vertex_index_map:
          self.add_vertex(v)

    num_faces = len(self._faces)
    num_vertices = len(self._vertices)

    self._face_indices = np.zeros((num_faces, 3), dtype=np.int32)
    for f_idx, face in enumerate(self._faces):
      self._face_indices[f_idx] = [
          self._vertex_index_map[face.vertex_b],
          self._vertex_index_map[face.vertex_s],
          self._vertex_index_map[face.vertex_d],
      ]

    # Geometry parameters from face 0
    a = float(self._faces[0].side_length)
    self._a = a
    self._face_area = (np.sqrt(3.0) / 4.0) * (a ** 2)
    self._cot_alpha = 1.0 / np.sqrt(3.0)  # cot(60 deg)
    self._t = a ** 2  # Mean edge length squared

    # Canonical 2D frame: p0=(0,0), p1=(a,0), p2=(a/2, a*sqrt(3)/2)
    self._p_local = np.array([
        [0.0, 0.0],
        [self._a, 0.0],
        [0.5 * self._a, (np.sqrt(3.0) / 2.0) * self._a]
    ], dtype=np.float64)

    # Outward rotated edge vectors for gradient:
    # e0 = p2 - p1 -> rotated 90 deg CW = (p2_y - p1_y, p1_x - p2_x)
    # e1 = p0 - p2 -> rotated 90 deg CW = (p0_y - p2_y, p2_x - p0_x)
    # e2 = p1 - p0 -> rotated 90 deg CW = (p1_y - p0_y, p0_x - p1_x)
    self._grad_coeffs = np.array([
        [self._p_local[2, 1] - self._p_local[1, 1], self._p_local[1, 0] - self._p_local[2, 0]],
        [self._p_local[0, 1] - self._p_local[2, 1], self._p_local[2, 0] - self._p_local[0, 0]],
        [self._p_local[1, 1] - self._p_local[0, 1], self._p_local[0, 0] - self._p_local[1, 0]],
    ]) / (2.0 * self._face_area)

    # Sum of outgoing edges from each face vertex for divergence:
    # v0: (p1 - p0) + (p2 - p0)
    # v1: (p0 - p1) + (p2 - p1)
    # v2: (p0 - p2) + (p1 - p2)
    self._div_vecs = np.array([
        (self._p_local[1] - self._p_local[0]) + (self._p_local[2] - self._p_local[0]),
        (self._p_local[0] - self._p_local[1]) + (self._p_local[2] - self._p_local[1]),
        (self._p_local[0] - self._p_local[2]) + (self._p_local[1] - self._p_local[2]),
    ])

    # 1. Lumped Diagonal Mass Matrix M
    vert_valence = np.zeros(num_vertices, dtype=np.float64)
    for i in range(3):
      np.add.at(vert_valence, self._face_indices[:, i], 1)
    m_diag = vert_valence * (self._face_area / 3.0)
    self._M = sp.diags(m_diag, 0, format="csc")

    # 2. Cotangent Laplacian Matrix L
    rows, cols, data = [], [], []
    edges = [(0, 1), (1, 2), (2, 0)]
    for v_i, v_j in edges:
      i_idx = self._face_indices[:, v_i]
      j_idx = self._face_indices[:, v_j]
      w = 0.5 * self._cot_alpha

      rows.extend([i_idx, j_idx, i_idx, j_idx])
      cols.extend([j_idx, i_idx, i_idx, j_idx])
      data.extend([
          np.full(num_faces, w), np.full(num_faces, w),
          np.full(num_faces, -w), np.full(num_faces, -w)
      ])

    self._L = sp.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(num_vertices, num_vertices)
    ).tocsc()

    # 3. Factorize Linear Operators
    heat_op = (self._M - self._t * self._L).tocsc()
    self._heat_solver = spla.factorized(heat_op)

    poisson_op = (-self._L + sp.eye(num_vertices, format="csc") * 1e-8).tocsc()
    self._poisson_solver = spla.factorized(poisson_op)
    self._matrices_built = True

  def _barycentric_to_local_2d(self, bary: tuple[float, float, float]) -> np.ndarray:
    w0, w1, w2 = bary
    return w0 * self._p_local[0] + w1 * self._p_local[1] + w2 * self._p_local[2]

  def _local_2d_to_barycentric(self, p: np.ndarray) -> tuple[float, float, float]:
    x, y = p[0], p[1]
    h = (np.sqrt(3.0) / 2.0) * self._a
    w2 = y / h
    w1 = (x - 0.5 * y / np.sqrt(3.0)) / self._a
    w0 = 1.0 - w1 - w2
    return (float(w0), float(w1), float(w2))

  def compute_distance_field(self, src_point: IsometricPoint) -> np.ndarray:
    """
    Solves for the scalar distance field phi at all vertices from an arbitrary
    source IsometricPoint.
    """
    if not self._matrices_built:
      self._build_solvers()

    src_face = src_point.grid
    if src_face not in self._face_index_map:
      self.register_face(src_face)
      self._build_solvers()

    src_f_idx = self._face_index_map[src_face]
    wb, ws, wd = src_point.barycentric

    # Step 1: Initialize Dirac delta source vector
    delta = np.zeros(len(self._vertices), dtype=np.float64)
    for v_idx, w in zip(self._face_indices[src_f_idx], [wb, ws, wd]):
      delta[v_idx] += float(w)

    # Step 2: Solve heat diffusion (M - tL)u = delta
    u = self._heat_solver(delta)
    u_faces = u[self._face_indices]

    # Step 3: Compute normalized vector field X = -grad(u) / ||grad(u)||
    grad_u = (
        u_faces[:, 0:1] * self._grad_coeffs[0] +
        u_faces[:, 1:2] * self._grad_coeffs[1] +
        u_faces[:, 2:3] * self._grad_coeffs[2]
    )
    grad_norm = np.linalg.norm(grad_u, axis=1, keepdims=True)
    grad_norm[grad_norm < 1e-12] = 1.0
    X = -grad_u / grad_norm

    # Step 4: Compute integrated divergence at each vertex
    div_b = np.zeros(len(self._vertices), dtype=np.float64)
    for k in range(3):
      contrib = 0.5 * self._cot_alpha * np.sum(self._div_vecs[k] * X, axis=1)
      np.add.at(div_b, self._face_indices[:, k], contrib)

    # Step 5: Solve Poisson system (-L) phi = div_b
    phi = self._poisson_solver(div_b)

    # Shift distance field so interpolated source equals 0
    source_dist = sum(
        float(w) * phi[v_idx]
        for v_idx, w in zip(self._face_indices[src_f_idx], [wb, ws, wd])
    )
    phi -= source_dist
    return phi

  @override
  def geodesic_distance(self, p1: IsometricPoint, p2: IsometricPoint) -> float:
    """
    Computes geodesic distance between two arbitrary points on the mesh.
    """
    if p1.grid is p2.grid:
      b_comp = p2.b - p1.b
      s_comp = p2.s - p1.s
      return float(isometric_distance(b_comp - 0.5 * s_comp, s_comp - 0.5 * b_comp))

    if hasattr(p1.grid, "is_adjacent_to_face") and p1.grid.is_adjacent_to_face(p2.grid):
      return float(p1.project_onto_adjacent_grid(p2.grid).distance_from(p2))

    phi = self.compute_distance_field(p1)
    tgt_face = p2.grid
    if tgt_face not in self._face_index_map:
      self.register_face(tgt_face)
      self._build_solvers()
      phi = self.compute_distance_field(p1)

    tgt_f_idx = self._face_index_map[tgt_face]
    wb, ws, wd = p2.barycentric
    tgt_dist = sum(
        float(w) * phi[v_idx]
        for v_idx, w in zip(self._face_indices[tgt_f_idx], [wb, ws, wd])
    )
    return float(max(0.0, tgt_dist))

  @override
  def shortest_path(
      self,
      p1: IsometricPoint,
      p2: IsometricPoint,
      max_steps: int = 500
  ) -> list[IsometricPoint]:
    """
    Traces the geodesic polyline from p1 to p2 across faces,
    returning a continuous sequence of IsometricPoint instances.
    """
    if p1.grid is p2.grid:
      return [p1, p2]

    phi = self.compute_distance_field(p1)
    src_face = p1.grid
    tgt_face = p2.grid

    # Trace backward from p2 to p1 along -grad(phi)
    path = [p2]
    curr_face = tgt_face
    curr_pos = self._barycentric_to_local_2d(p2.barycentric)

    for _ in range(max_steps):
      if curr_face is src_face:
        path.append(p1)
        break

      f_idx = self._face_index_map[curr_face]
      phi_f = phi[self._face_indices[f_idx]]
      grad_phi = (
          phi_f[0] * self._grad_coeffs[0] +
          phi_f[1] * self._grad_coeffs[1] +
          phi_f[2] * self._grad_coeffs[2]
      )

      direction = -grad_phi
      norm = np.linalg.norm(direction)
      if norm < 1e-10:
        path.append(p1)
        break
      direction /= norm

      min_t = float('inf')
      hit_edge = -1
      hit_alpha = 0.0

      for k in range(3):
        pA = self._p_local[(k + 1) % 3]
        pB = self._p_local[(k + 2) % 3]
        edge_dir = pB - pA

        det = direction[0] * (-edge_dir[1]) - direction[1] * (-edge_dir[0])
        if abs(det) > 1e-12:
          dx = pA[0] - curr_pos[0]
          dy = pA[1] - curr_pos[1]
          t = (dx * (-edge_dir[1]) - dy * (-edge_dir[0])) / det
          alpha = (direction[0] * dy - direction[1] * dx) / det
          if t > 1e-7 and 0.0 <= alpha <= 1.0:
            if t < min_t:
              min_t = t
              hit_edge = k
              hit_alpha = alpha

      if hit_edge == -1:
        path.append(p1)
        break

      pA = self._p_local[(hit_edge + 1) % 3]
      pB = self._p_local[(hit_edge + 2) % 3]
      curr_pos = pA + hit_alpha * (pB - pA)
      bary = self._local_2d_to_barycentric(curr_pos)
      path.append(IsometricPoint.from_barycentric(curr_face, *bary))

      # Step into adjacent face
      edge_dir_enum = IsometricDirection(hit_edge)
      next_face = curr_face.face_on_edge(edge_dir_enum)
      if next_face is None:
        break

      next_edge_dir = next_face.direction_away_from_face(curr_face)
      next_k = next_edge_dir.value
      pA_next = self._p_local[(next_k + 1) % 3]
      pB_next = self._p_local[(next_k + 2) % 3]
      curr_pos = pA_next + (1.0 - hit_alpha) * (pB_next - pA_next)
      curr_face = next_face

    # Reverse so path is ordered from p1 -> p2
    path.reverse()
    return path
