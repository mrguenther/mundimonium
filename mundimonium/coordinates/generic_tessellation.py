from __future__ import annotations

from mundimonium.coordinates.exceptions import EndOfMeshSurfaceException
from mundimonium.coordinates.tessellation import (
    Tessellation, TessellationFace, TessellationVertex
)
from mundimonium.coordinates.isometric import (
    IsometricDirection, IsometricPoint
)

from collections.abc import Sequence
from numbers import Number
from typing import override
import heapq
import itertools
import math
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# A tiny, fixed rotation applied to the ray direction in
# `geodesically_canonicalize_point` before tracing. Its only purpose is to
# generically avoid the ray passing *exactly* through a mesh vertex (see that
# method's docstring) -- it's small enough to leave every real result
# unaffected (~1e-7 radians), and being a fixed constant rather than
# query-dependent, it doesn't affect how smoothly the result varies with the
# query direction: two nearby queries stay exactly as close after the
# rotation as before it.
_DEGENERATE_RAY_NUDGE_RADIANS = 1e-7


class GenericTessellation(Tessellation):
  """
  Generic equilateral triangular mesh with exact geodesic pathfinding and
  distance queries.

  `geodesic_distance`, `shortest_path`, and `shortest_path_by_segment` are
  powered by an A* search over the face-adjacency graph (to find a
  "corridor" of faces to travel through) followed by an exact taut-string
  ("funnel") refinement within that corridor -- see `shortest_path_by_segment`
  for the full explanation. `compute_distance_field` is a separate,
  independent utility that instead computes an approximate scalar distance
  field over the whole mesh via the Heat Method (Crane, Weischedel,
  Wardetzky); it isn't used by the three methods above.
  """

  def __init__(self, *, euclidean: bool = True, **kwargs):
    """Constructs an empty tessellation of arbitrary geometry.

    Heat/Poisson solvers (used only by `compute_distance_field`) are built
    lazily on first use.

    Args:
      euclidean: Whether `TessellationVertex.projection_coordinates` form an
                 accurate Euclidean embedding (e.g. a 3D model used for
                 physical distances), as opposed to a visualization of a
                 non-Euclidean space where straight-line 3D distance isn't a
                 meaningful lower bound on surface distance. When `True`,
                 `geodesic_distance`/`shortest_path`/`shortest_path_by_segment`
                 use straight-line 3D distance as an A* heuristic to speed up
                 their corridor search; when `False`, no heuristic is used
                 (plain Dijkstra) -- slower, but still exact.
      **kwargs:  Forwarded up the method resolution order.
    """
    super().__init__(**kwargs)
    self._euclidean: bool = euclidean
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
    """Returns the coordinates of the specified point."""
    raise NotImplementedError()

  @override
  def flatten_region(
      self, center: IsometricPoint, targets: Sequence[IsometricPoint],
  ) -> list[tuple[Number, Number]]:
    """Not supported: `GenericTessellation` has no closed-form embedding
    or persistent neighborhood-flattening mechanism to build this from.
    """
    raise NotImplementedError()

  @override
  def on_vertex_added(self, vertex: TessellationVertex) -> None:
    """Hook for updating internal state when mesh topology changes."""
    super().on_vertex_added(vertex)
    self.invalidate_solvers()

  @override
  def on_face_added(self, face: TessellationFace) -> None:
    """Hook for updating internal state when mesh topology changes."""
    super().on_face_added(face)
    self.invalidate_solvers()


  def invalidate_solvers(self) -> None:
    """Invalidates cached matrix factorizations when topology changes."""
    self._matrices_built = False
    self._heat_solver = None
    self._poisson_solver = None

  def _build_solvers(self) -> None:
    """Builds solvers for heat-based distance calculations.

    Builds and factorizes the mass, cotangent-Laplacian, heat, and Poisson
    matrices used by the Heat Method, caching the factorizations for reuse.
    """
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
        [self._p_local[2, 1] - self._p_local[1, 1],
         self._p_local[1, 0] - self._p_local[2, 0]],
        [self._p_local[0, 1] - self._p_local[2, 1],
         self._p_local[2, 0] - self._p_local[0, 0]],
        [self._p_local[1, 1] - self._p_local[0, 1],
         self._p_local[0, 0] - self._p_local[1, 0]],
    ]) / (2.0 * self._face_area)

    # Sum of outgoing edges from each face vertex for divergence:
    # v0: (p1 - p0) + (p2 - p0)
    # v1: (p0 - p1) + (p2 - p1)
    # v2: (p0 - p2) + (p1 - p2)
    self._div_vecs = np.array([
        (self._p_local[1] - self._p_local[0]) +
        (self._p_local[2] - self._p_local[0]),
        (self._p_local[0] - self._p_local[1]) +
        (self._p_local[2] - self._p_local[1]),
        (self._p_local[0] - self._p_local[2]) +
        (self._p_local[1] - self._p_local[2]),
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

  def _barycentric_to_local_2d(
      self, bary: tuple[float, float, float]) -> np.ndarray:
    """Maps barycentric weights to a 2D position in the canonical face frame.

    Inverse of `_local_2d_to_barycentric`.
    """
    w0, w1, w2 = bary
    return w0 * self._p_local[0] + w1 * self._p_local[1] + w2 * self._p_local[2]

  def _local_2d_to_barycentric(
      self, p: np.ndarray) -> tuple[float, float, float]:
    """Maps a 2D position in the canonical face frame to barycentric weights.

    Inverse of `_barycentric_to_local_2d`.
    """
    x, y = p[0], p[1]
    h = (np.sqrt(3.0) / 2.0) * self._a
    w2 = y / h
    w1 = (x - y / np.sqrt(3.0)) / self._a
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

  @staticmethod
  def _point_at_vertex(
      face: TessellationFace, direction: IsometricDirection) -> IsometricPoint:
    """The exact `IsometricPoint` at `face`'s vertex in the given direction."""
    weights = [0.0, 0.0, 0.0]
    weights[direction.value] = 1.0
    return IsometricPoint.from_barycentric(face, *weights)

  def _fan_around_vertex(
      self, face: TessellationFace, vertex: TessellationVertex,
  ) -> list[TessellationFace] | None:
    """Faces touching `vertex`, in cyclic order around it, starting at `face`.

    Returns `None` if `vertex` lies on an open mesh boundary (the fan of
    faces around it doesn't close back into a cycle).
    """
    vertex_dir = face.direction_toward_vertex(vertex)
    edges_at_vertex = [d for d in IsometricDirection if d != vertex_dir]
    fan = [face]
    curr_face = face
    walk_dir = edges_at_vertex[0]
    for _ in range(len(self._faces) + 1):
      next_face = curr_face.face_on_edge(walk_dir)
      if next_face is None:
        return None
      if next_face is face:
        return fan
      arrived_via = next_face.direction_away_from_face(curr_face)
      next_vertex_dir = next_face.direction_toward_vertex(vertex)
      next_edges_at_vertex = [
          d for d in IsometricDirection if d != next_vertex_dir]
      walk_dir = (
          next_edges_at_vertex[0] if next_edges_at_vertex[0] != arrived_via
          else next_edges_at_vertex[1])
      fan.append(next_face)
      curr_face = next_face
    raise RuntimeError(
        "_fan_around_vertex: the fan around a vertex did not close after "
        "visiting every face in the mesh; the mesh may be non-manifold.")

  def _face_moves(
      self, face: TessellationFace,
  ) -> list[tuple[TessellationFace, float]]:
    """Candidate A* moves from `face`.

    Two kinds:
    - To each edge-adjacent face, cost = local centroid-to-centroid distance.
    - A direct "saddle-vertex" crossing to the face sharing a vertex of
      valence >= 6 with `face`, exactly three fan-hops away, cost =
      centroid-to-vertex-to-centroid distance (exactly twice a normal move's
      cost for congruent faces). Valence >= 6 is exactly the threshold where
      such a shortcut beats the fan's own hop-by-hop path (that path costs
      `floor(valence / 2)` at the fan's "opposite" face, which first exceeds
      the shortcut's cost of 2 at valence 6); three hops away is exactly
      when that comparison favors the shortcut over tying it.

      Only the nearest valid target is offered, even though every fan-hop
      distance up to `n - 3` ties on cost (centroid-to-vertex distance is
      the same constant for every face touching the vertex on a uniform
      mesh). A farther, equally-"cheap" target still takes more actual mesh
      hops to reconstruct via `_expand_corridor`, so leaving every tied
      target in the move pool let A*'s heap tie-breaking nondeterministically
      pick a farther target in one query direction and a nearer one in the
      other, producing wildly asymmetric `geodesic_distance` results. A
      shortcut further around a high-valence fan is still reachable by
      chaining multiple three-hop jumps around the same vertex.
    """
    moves: list[tuple[TessellationFace, float]] = []
    centroid = face.centroid_local_coords

    for direction in IsometricDirection:
      neighbor = face.face_on_edge(direction)
      if neighbor is not None:
        cost = centroid.project_onto_adjacent_grid(neighbor).distance_from(
            neighbor.centroid_local_coords)
        moves.append((neighbor, float(cost)))

    for direction in IsometricDirection:
      vertex = face.vertex_at(direction)
      fan = self._fan_around_vertex(face, vertex)
      if fan is None:
        continue
      if len(fan) < 6:
        continue
      dist_to_vertex = centroid.distance_from(
          self._point_at_vertex(face, direction))
      target_face = fan[3]
      target_dir = target_face.direction_toward_vertex(vertex)
      target_dist = self._point_at_vertex(
          target_face, target_dir).distance_from(
              target_face.centroid_local_coords)
      moves.append((target_face, float(dist_to_vertex + target_dist)))

    return moves

  def _find_face_corridor(
      self, src_face: TessellationFace, tgt_face: TessellationFace,
  ) -> list[TessellationFace]:
    """A* search over the face-adjacency graph (see `_face_moves`) for a
    face "corridor" from `src_face` to `tgt_face`.

    This corridor's own cost is only an estimate (straight centroid-to-
    centroid distances, and a heuristic stand-in for saddle-vertex
    shortcuts) -- `shortest_path_by_segment` separately computes the true
    shortest path *within* whatever corridor this returns, via an exact
    taut-string refinement.
    """
    if src_face is tgt_face:
      return [src_face]

    tgt_3d = np.array(tgt_face.centroid_projection_coords, dtype=np.float64)

    def heuristic(face: TessellationFace) -> float:
      if not self._euclidean:
        return 0.0
      face_3d = np.array(face.centroid_projection_coords, dtype=np.float64)
      return float(np.linalg.norm(face_3d - tgt_3d))

    counter = itertools.count()
    g_score: dict[TessellationFace, float] = {src_face: 0.0}
    came_from: dict[TessellationFace, TessellationFace] = {}
    frontier = [(heuristic(src_face), next(counter), src_face)]
    closed: set[TessellationFace] = set()

    while frontier:
      _, _, face = heapq.heappop(frontier)
      if face in closed:
        continue
      closed.add(face)
      if face is tgt_face:
        corridor = [face]
        while corridor[-1] is not src_face:
          corridor.append(came_from[corridor[-1]])
        corridor.reverse()
        return corridor

      for neighbor, cost in self._face_moves(face):
        tentative = g_score[face] + cost
        if tentative < g_score.get(neighbor, float('inf')):
          g_score[neighbor] = tentative
          came_from[neighbor] = face
          heapq.heappush(
              frontier,
              (tentative + heuristic(neighbor), next(counter), neighbor))

    raise EndOfMeshSurfaceException(
        "shortest_path_by_segment: no path exists between the two points' "
        "faces on this mesh.")

  def _expand_corridor(
      self, corridor: list[TessellationFace],
  ) -> list[TessellationFace]:
    """Replaces every saddle-vertex jump in `corridor` with the actual
    sequence of edge-adjacent faces around that vertex's fan (the shorter
    way around), so the result is a corridor where every consecutive pair
    of faces is genuinely edge-adjacent -- required for the funnel
    refinement in `shortest_path_by_segment`, which unfolds one shared edge
    at a time.
    """
    expanded = [corridor[0]]
    for face, next_face in zip(corridor, corridor[1:]):
      if face.is_adjacent_to_face(next_face):
        expanded.append(next_face)
        continue

      fan = None
      for direction in IsometricDirection:
        vertex = face.vertex_at(direction)
        candidate = self._fan_around_vertex(face, vertex)
        if candidate is not None and next_face in candidate:
          idx = candidate.index(next_face)
          n = len(candidate)
          if min(idx, n - idx) > 2:
            fan = candidate
            break
      if fan is None:
        raise RuntimeError(
            "_expand_corridor: found no valid saddle-vertex connection "
            "between two non-adjacent corridor faces.")

      idx = fan.index(next_face)
      n = len(fan)
      # Go the shorter way around the fan. NOTE: when the two faces are
      # exactly opposite each other around an even-valence fan (idx == n -
      # idx), both directions tie in hop count, but we always break the tie
      # by going forward, without checking whether the other side would
      # unfold into a shorter (or just different) exact path -- a known,
      # currently-unaddressed source of a possibly-suboptimal (but still
      # valid) path in that specific tie case.
      if idx <= n - idx:
        expanded.extend(fan[1:idx + 1])
      else:
        expanded.extend(reversed(fan[idx:n]))
    return expanded

  @staticmethod
  def _funnel(
      left: list[np.ndarray], right: list[np.ndarray],
  ) -> list[tuple[str, int]]:
    """The "Simple Stupid Funnel Algorithm" (string-pulling).

    Given parallel lists of "left" and "right" channel-boundary points
    (`left[0] is right[0]` must be the start point, `left[-1] is right[-1]`
    the end point -- same object, for exact identity comparisons below),
    returns the taut (shortest) path through the channel they define, as a
    list of `(side, index)` pairs identifying each bend point (including
    the start and end, both labeled `'L'` arbitrarily).
    """
    def triarea2(a, b, c):
      return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

    n = len(left)
    apex, left_pt, right_pt = left[0], left[0], right[0]
    apex_idx = left_idx = right_idx = 0
    path: list[tuple[str, int]] = [('L', 0)]

    # The final portal is degenerate (left[-1] is right[-1] -- a single
    # target point, not a widening edge), so it's excluded from the main
    # loop and handled separately below: running it through the same dual
    # widen-or-pop checks as a real portal can trigger a spurious pop (the
    # "widen right" check can set right_pt to the target itself, which then
    # makes the immediately-following "widen left" check's fallback
    # condition -- comparing the target against itself -- degenerate,
    # forcing an unwanted pop).
    i = 1
    while i < n - 1:
      left_candidate, right_candidate = left[i], right[i]

      # Try to tighten the right side. A vertex held fixed across several
      # consecutive portals (e.g. a saddle-vertex fan expansion) makes
      # `right_candidate` the exact same object as `right_pt`, so `triarea2`
      # below is exactly zero and this still falls through as a same-point
      # "widen" that only refreshes `right_idx` -- it must NOT be
      # special-cased into a no-op, since the pop-or-widen decision also
      # depends on `left_pt`, which can have changed since `right_pt` last
      # held this value.
      if triarea2(apex, right_pt, right_candidate) <= 0.0:
        if apex is right_pt or triarea2(apex, left_pt, right_candidate) > 0.0:
          right_pt, right_idx = right_candidate, i
        else:
          path.append(('L', left_idx))
          apex, apex_idx = left_pt, left_idx
          left_pt = right_pt = apex
          left_idx = right_idx = apex_idx
          i = apex_idx + 1
          continue

      # Try to tighten the left side (same reasoning as the right side).
      if triarea2(apex, left_pt, left_candidate) >= 0.0:
        if apex is left_pt or triarea2(apex, right_pt, left_candidate) < 0.0:
          left_pt, left_idx = left_candidate, i
        else:
          path.append(('R', right_idx))
          apex, apex_idx = right_pt, right_idx
          left_pt = right_pt = apex
          left_idx = right_idx = apex_idx
          i = apex_idx + 1
          continue

      i += 1

    # Close the funnel onto the final target point: since it's a single
    # point rather than an edge, it can pop through at most one side (never
    # both), whichever (if either) it currently falls outside of.
    target = left[n - 1]
    if apex is not right_pt and triarea2(apex, right_pt, target) < 0.0:
      path.append(('R', right_idx))
    elif apex is not left_pt and triarea2(apex, left_pt, target) > 0.0:
      path.append(('L', left_idx))

    path.append(('L', n - 1))
    return path

  @override
  def shortest_path_by_segment(
      self, p1: IsometricPoint, p2: IsometricPoint,
  ) -> list[tuple[IsometricPoint, IsometricPoint]]:
    """Traces the exact geodesic path from p1 to p2, split into per-face
    segments.

    Two-phase algorithm:
    1. A* over the face-adjacency graph (`_find_face_corridor`) finds a
       "corridor" of faces to travel through, using straight-line
       centroid-to-centroid distances (and a similar heuristic for
       saddle-vertex shortcuts through high-valence vertices -- see
       `_face_moves`) as an approximate cost. When `self._euclidean` is
       `True`, straight-line 3D distance to the target guides the search;
       when `False`, no heuristic is used (plain Dijkstra), since 3D
       straight-line distance isn't a meaningful lower bound on surface
       distance for a mesh whose `projection_coordinates` are only a
       visualization of a non-Euclidean space.
    2. The corridor (after `_expand_corridor` replaces any saddle-vertex
       jumps with their actual constituent faces) is refined into the
       *exact* shortest path via the funnel/string-pulling algorithm
       (`_funnel`): its faces are unfolded flat into one shared 2D frame,
       one shared edge at a time (the same rotation + vertex-identity
       matching `geodesically_canonicalize_point` uses), and the taut path
       through the resulting flat channel is computed exactly. This step is
       always exact, regardless of `self._euclidean`.

    A corridor is only ever an approximately-good *choice* of faces to route
    through; the funnel step then finds the true shortest path *through that
    specific corridor*. A different corridor could in principle yield an
    even shorter path -- see `_expand_corridor`'s note on the one known case
    where this can happen (an exactly-opposite saddle-vertex fan tie).
    """
    src_face = p1.grid
    tgt_face = p2.grid
    if src_face is tgt_face:
      return [(p1, p2)]

    corridor = self._expand_corridor(
        self._find_face_corridor(src_face, tgt_face))

    # Unfold the corridor's faces into one common flat 2D frame, one shared
    # edge at a time. `frames[0]` is `_canonical_local_frame(corridor[0])`
    # itself (no transform); `frames[i]` is `corridor[i]`'s own vertices,
    # each mapped into that same common frame.
    frames = [self._canonical_local_frame(corridor[0])]
    portal_vertices: list[tuple[TessellationVertex, TessellationVertex]] = []
    portal_positions: list[tuple[np.ndarray, np.ndarray]] = []

    for prev_face, curr_face in zip(corridor, corridor[1:]):
      prev_frame = frames[-1]
      edge_dir = prev_face.direction_away_from_face(curr_face)
      vA = prev_face.vertex_at(IsometricDirection((edge_dir.value + 1) % 3))
      vB = prev_face.vertex_at(IsometricDirection((edge_dir.value + 2) % 3))
      pA = prev_frame[(edge_dir.value + 1) % 3]
      pB = prev_frame[(edge_dir.value + 2) % 3]

      curr_local = self._canonical_local_frame(curr_face)
      dir_A = curr_face.direction_toward_vertex(vA)
      dir_B = curr_face.direction_toward_vertex(vB)
      pA_local = curr_local[dir_A.value]
      pB_local = curr_local[dir_B.value]

      # Rotation mapping `curr_face`'s own local edge vector onto its
      # already-unfolded (common-frame) counterpart -- i.e. "unfolding"
      # `curr_face` flat, hinged along the shared edge.
      u = pB_local - pA_local
      v = pB - pA
      inv_len_sq = 1.0 / (u[0] ** 2 + u[1] ** 2)
      cos_t = (u[0] * v[0] + u[1] * v[1]) * inv_len_sq
      sin_t = (u[0] * v[1] - u[1] * v[0]) * inv_len_sq
      rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

      curr_frame = np.array([
          pA + rotation @ (curr_local[k] - pA_local) for k in range(3)])
      frames.append(curr_frame)
      portal_vertices.append((vA, vB))
      portal_positions.append((pA, pB))

    # Assign a consistent left/right side to each portal. The funnel
    # algorithm's two checks use opposite-direction inequalities, so this
    # isn't just a self-consistent labeling exercise: the first portal's
    # side must match actual geometric handedness relative to the apex
    # (chosen here so that apex->left->right turns clockwise, i.e.
    # triarea2(apex, left, right) <= 0 -- the convention the checks below
    # assume); every later portal then keeps whichever side its vertex
    # shared with the previous portal already had.
    def triarea2(a, b, c):
      return (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])

    wb1, ws1, wd1 = p1.barycentric
    p1_pos = wb1 * frames[0][0] + ws1 * frames[0][1] + wd1 * frames[0][2]
    wb2, ws2, wd2 = p2.barycentric
    p2_pos = wb2 * frames[-1][0] + ws2 * frames[-1][1] + wd2 * frames[-1][2]

    left_positions = [p1_pos]
    right_positions = [p1_pos]
    left_vertices: list[TessellationVertex | None] = [None]
    right_vertices: list[TessellationVertex | None] = [None]
    prev_left_v: TessellationVertex | None = None
    prev_right_v: TessellationVertex | None = None
    prev_left_pos: np.ndarray | None = None
    prev_right_pos: np.ndarray | None = None

    for (vA, vB), (pA, pB) in zip(portal_vertices, portal_positions):
      if prev_left_v is None:
        if triarea2(p1_pos, pA, pB) > 0.0:
          left_v, left_p, right_v, right_p = vB, pB, vA, pA
        else:
          left_v, left_p, right_v, right_p = vA, pA, vB, pB
      else:
        # Reuse the *exact* previous-portal position object (not a freshly
        # computed one, even though it'd be numerically near-identical) for
        # whichever vertex is shared with the previous portal: `_funnel`
        # relies on `is`-identity to recognize "this candidate is the same
        # point as the current boundary". Every portal in a triangle
        # corridor shares exactly one vertex with the previous one, so
        # exactly one of these four cases always applies.
        if vA is prev_left_v:
          left_v, left_p, right_v, right_p = vA, prev_left_pos, vB, pB
        elif vA is prev_right_v:
          right_v, right_p, left_v, left_p = vA, prev_right_pos, vB, pB
        elif vB is prev_left_v:
          left_v, left_p, right_v, right_p = vB, prev_left_pos, vA, pA
        else:
          right_v, right_p, left_v, left_p = vB, prev_right_pos, vA, pA
      left_positions.append(left_p)
      right_positions.append(right_p)
      left_vertices.append(left_v)
      right_vertices.append(right_v)
      prev_left_v, prev_right_v = left_v, right_v
      prev_left_pos, prev_right_pos = left_p, right_p

    left_positions.append(p2_pos)
    right_positions.append(p2_pos)
    left_vertices.append(None)
    right_vertices.append(None)

    apex_path = self._funnel(left_positions, right_positions)

    # Walk each straight bend-to-bend stretch of the taut path back through
    # the (known) corridor faces it crosses, recording one (entry, exit)
    # segment per face -- each stretch spans corridor[idx : next_idx]
    # (portal index `idx` sits exactly at the boundary between
    # corridor[idx - 1] and corridor[idx]; idx 0 is p1 on corridor[0],
    # the last index is p2 on corridor[-1]).
    segments: list[tuple[IsometricPoint, IsometricPoint]] = []
    for (side, idx), (next_side, next_idx) in zip(apex_path, apex_path[1:]):
      pos = (left_positions if side == 'L' else right_positions)[idx]
      next_pos = (
          left_positions if next_side == 'L' else right_positions)[next_idx]
      direction = next_pos - pos

      if idx == 0:
        entry = p1
      else:
        vertex = (left_vertices if side == 'L' else right_vertices)[idx]
        entry = self._point_at_vertex(
            corridor[idx], corridor[idx].direction_toward_vertex(vertex))

      for j in range(idx, next_idx):
        face = corridor[j]
        next_entry = None  # only set (and used) when j is not the last face
        if j == next_idx - 1:
          if next_idx == len(left_positions) - 1:
            exit_point = p2
          else:
            vertex = (
                left_vertices if next_side == 'L' else right_vertices
            )[next_idx]
            exit_point = self._point_at_vertex(
                face, face.direction_toward_vertex(vertex))
        else:
          # A plain pass-through crossing (not a funnel bend): the taut
          # path crosses straight through this portal without needing to
          # touch either endpoint vertex.
          pA, pB = portal_positions[j]
          e = pB - pA
          denom = direction[0] * (-e[1]) - direction[1] * (-e[0])
          if abs(denom) < 1e-12:
            # `direction` runs exactly along this portal's own edge (common
            # on this mesh's regular, highly symmetric geometry): there's no
            # single well-defined crossing point, so treat it as happening
            # immediately at `pos` -- a zero-length segment for this face,
            # the same "genuine zero-length middle segment" case already
            # tolerated elsewhere.
            t = 0.0
          else:
            dx = pA[0] - pos[0]
            dy = pA[1] - pos[1]
            t = (dx * (-e[1]) - dy * (-e[0])) / denom
          cross_pos = pos + t * direction
          bary = self._local_2d_to_barycentric_in_frame(cross_pos, frames[j])
          exit_point = IsometricPoint(
              face, bary[0] * face.altitude, bary[1] * face.altitude)
          # `exit_point` is on `face` (corridor[j]); the entry point for the
          # *next* face is the same physical (common-frame) location, but
          # expressed on that face's own grid.
          next_face = corridor[j + 1]
          next_bary = self._local_2d_to_barycentric_in_frame(
              cross_pos, frames[j + 1])
          next_entry = IsometricPoint(
              next_face,
              next_bary[0] * next_face.altitude,
              next_bary[1] * next_face.altitude,
          )
        segments.append((entry, exit_point))
        entry = exit_point if j == next_idx - 1 else next_entry

    if not segments:
      return [(p1, p2)]

    # Drop precision-artifact segments -- e.g. an endpoint that started
    # exactly on an edge/vertex and immediately left that grid -- but only
    # at the start or end of the path. A minuscule segment in the *middle*
    # is a straight stretch that merely grazes a vertex (common for a taut
    # funnel path); dropping it would splice its two neighbors together,
    # but they generally sit on faces that aren't themselves adjacent,
    # breaking the per-grid-segment contract.
    def is_minuscule(segment: tuple[IsometricPoint, IsometricPoint]) -> bool:
      a, b = segment
      return a.distance_from(b) <= 1e-6 * a.grid.altitude

    start = 0
    end = len(segments)
    while start < end - 1 and is_minuscule(segments[start]):
      start += 1
    while end > start + 1 and is_minuscule(segments[end - 1]):
      end -= 1
    return segments[start:end]

  @override
  def shortest_path(
      self, p1: IsometricPoint, p2: IsometricPoint,
  ) -> list[IsometricPoint]:
    """Traces the exact geodesic polyline from p1 to p2 across faces,
    returning a continuous sequence of IsometricPoint instances.

    Derived from `shortest_path_by_segment`: each segment's start point
    already matches the previous segment's end point (up to floating-point
    precision), so the flattened sequence is `[p1, then each segment's end
    point in order]`.
    """
    segments = self.shortest_path_by_segment(p1, p2)
    points = [segments[0][0]]
    points.extend(b for _, b in segments)
    return points

  @override
  def geodesic_distance(self, p1: IsometricPoint, p2: IsometricPoint) -> float:
    """Computes the exact geodesic distance between two arbitrary points on
    the mesh, as the sum of `shortest_path_by_segment`'s segment lengths.
    """
    segments = self.shortest_path_by_segment(p1, p2)
    return float(sum(a.distance_from(b) for a, b in segments))

  @staticmethod
  def _canonical_local_frame(face: TessellationFace) -> np.ndarray:
    """The canonical local 2D layout of `face`'s own (b, s, d) system.

    Vertices B, S, D sit at (0, 0), (a, 0), (a/2, h), where `a` is
    `face.side_length` and `h` is `face.altitude`. Computed per-face
    (unlike the mesh-wide `self._p_local`, which assumes a uniform side
    length across every face) so it stays correct for distorted faces too.
    """
    a = float(face.side_length)
    h = float(face.altitude)
    return np.array([[0.0, 0.0], [a, 0.0], [0.5 * a, h]], dtype=np.float64)

  @staticmethod
  def _local_2d_to_barycentric_in_frame(
      p: np.ndarray, frame: np.ndarray) -> tuple[float, float, float]:
    """Like `_local_2d_to_barycentric`, but for an arbitrary `frame` (three
    2D vertex positions, in `(B, S, D)` order) rather than the mesh-wide
    `self._p_local`.

    Uses the general barycentric-coordinate formula for an arbitrary
    triangle (not just one with vertex 0 at the origin, like the raw output
    of `_canonical_local_frame`) -- `frame` may equally be one that's been
    rotated and translated into a shared/common frame (e.g. by
    `shortest_path_by_segment`'s corridor unfolding), where vertex 0 is
    generally *not* at the origin.
    """
    v0, v1, v2 = frame[0], frame[1], frame[2]
    x, y = p[0], p[1]
    denom = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
    w0 = (
        (v1[1] - v2[1]) * (x - v2[0]) + (v2[0] - v1[0]) * (y - v2[1])
    ) / denom
    w1 = (
        (v2[1] - v0[1]) * (x - v2[0]) + (v0[0] - v2[0]) * (y - v2[1])
    ) / denom
    w2 = 1.0 - w0 - w1
    return (float(w0), float(w1), float(w2))

  @override
  def geodesically_canonicalize_point(
      self, point: IsometricPoint, max_steps: int = 500,
  ) -> IsometricPoint:
    """Moves `point` to a new grid if located outside its current grid's
    bounds, walking across as many faces as necessary.

    `point`'s out-of-range barycentric coordinates on its current face are
    treated as a straight-line displacement from that face's centroid, in
    the face's own flat local frame. This ray is walked face by face: each
    shared edge crossing "unfolds" the two faces flat via a rotation+scale
    transform (rotation alone if the two faces' side lengths match) applied
    to the ray's remaining displacement -- the same edge-crossing technique
    `shortest_path` uses per step, generalized here to also carry a
    direction vector rather than just a traced position. This makes the
    result the mesh's "straightest geodesic" from the centroid: identical to
    the true shortest path everywhere except where that path would need to
    bend around a mesh vertex with nonzero angle defect.

    A ray that passes *exactly* through a mesh vertex is a genuine
    degenerate case (every edge touching that vertex reports the same
    crossing distance, and after landing there every edge reports distance
    0, so there's no well-defined "next edge" to cross), so the ray
    direction is nudged by a fixed, tiny rotation before tracing to avoid it
    generically -- this doesn't materially affect any result, and being a
    fixed offset rather than a query-dependent one, doesn't affect how
    smoothly the result varies with the query direction either.

    (A ray that merely *grazes* a vertex closely, without hitting it
    exactly, is a different matter: which side it passes on is a genuine,
    unavoidable discontinuity for straightest geodesics on a curved
    polyhedral mesh -- not something this nudge, or any tie-breaking rule,
    can smooth over. That case is left as-is.)

    Mutates and returns `point`, not a copy.

    Raises:
      EndOfMeshSurfaceException: If the ray walks off the edge of an open
        mesh before landing inside a face.
      RuntimeError: If `max_steps` is exceeded without landing inside a
        face (e.g. a topology bug causing the walk to cycle).
    """
    curr_face: TessellationFace = point.grid
    p_local = self._canonical_local_frame(curr_face)
    wb, ws, wd = point.barycentric
    target_2d = wb * p_local[0] + ws * p_local[1] + wd * p_local[2]
    curr_pos = (p_local[0] + p_local[1] + p_local[2]) / 3.0
    direction = target_2d - curr_pos

    nudge_cos = math.cos(_DEGENERATE_RAY_NUDGE_RADIANS)
    nudge_sin = math.sin(_DEGENERATE_RAY_NUDGE_RADIANS)
    direction = np.array([
        nudge_cos * direction[0] - nudge_sin * direction[1],
        nudge_sin * direction[0] + nudge_cos * direction[1],
    ])

    for _ in range(max_steps):
      p_local = self._canonical_local_frame(curr_face)
      min_t = float('inf')
      hit_edge = -1
      hit_alpha = 0.0

      for k in range(3):
        pA = p_local[(k + 1) % 3]
        pB = p_local[(k + 2) % 3]
        edge_dir = pB - pA

        det = direction[0] * (-edge_dir[1]) - direction[1] * (-edge_dir[0])
        if abs(det) > 1e-12:
          dx = pA[0] - curr_pos[0]
          dy = pA[1] - curr_pos[1]
          t = (dx * (-edge_dir[1]) - dy * (-edge_dir[0])) / det
          alpha = (direction[0] * dy - direction[1] * dx) / det
          if 1e-9 < t <= 1.0 + 1e-9 and -1e-9 <= alpha <= 1.0 + 1e-9:
            if t < min_t:
              min_t = t
              hit_edge = k
              hit_alpha = alpha

      if hit_edge == -1:
        # The (remaining) target lies within this face: done.
        final_pos = curr_pos + direction
        fwb, fws, fwd = self._local_2d_to_barycentric_in_frame(
            final_pos, p_local)
        alt = curr_face.altitude
        return point.update(grid=curr_face, b=fwb * alt, s=fws * alt)

      pA = p_local[(hit_edge + 1) % 3]
      pB = p_local[(hit_edge + 2) % 3]
      edge_point = pA + hit_alpha * (pB - pA)

      edge_dir_enum = IsometricDirection(hit_edge)
      next_face = curr_face.face_on_edge(edge_dir_enum)
      if next_face is None:
        raise EndOfMeshSurfaceException(
            "Cannot canonicalize point located outside of mesh-surface "
            "boundary.")

      # Match by actual shared-vertex identity, not by index pattern: each
      # face labels its own vertices independently, so which of `next_face`'s
      # three local directions corresponds to `pA`/`pB` isn't fixed -- it has
      # to be looked up per vertex.
      vertex_at_pA = curr_face.vertex_at(IsometricDirection((hit_edge + 1) % 3))
      vertex_at_pB = curr_face.vertex_at(IsometricDirection((hit_edge + 2) % 3))
      next_p_local = self._canonical_local_frame(next_face)
      pA_next = next_p_local[next_face.direction_toward_vertex(vertex_at_pA).value]
      pB_next = next_p_local[next_face.direction_toward_vertex(vertex_at_pB).value]

      # Rotation+scale transform taking this shared edge (as seen from
      # `curr_face`) onto the same edge as seen from `next_face` -- i.e.
      # "unfolding" the two faces flat, hinged along the shared edge.
      u = pB - pA
      v = pB_next - pA_next
      inv_len_sq = 1.0 / (u[0] ** 2 + u[1] ** 2)
      cos_t = (u[0] * v[0] + u[1] * v[1]) * inv_len_sq
      sin_t = (u[0] * v[1] - u[1] * v[0]) * inv_len_sq
      rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

      curr_pos = pA_next + rotation @ (edge_point - pA)
      direction = rotation @ (direction * (1.0 - min_t))
      curr_face = next_face

    raise RuntimeError(
        f"geodesically_canonicalize_point exceeded {max_steps} steps; the "
        "mesh may contain a cycle preventing convergence.")
