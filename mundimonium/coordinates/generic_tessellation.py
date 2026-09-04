from __future__ import annotations

from mundimonium.coordinates.exceptions import EndOfMeshSurfaceException
from mundimonium.coordinates.tessellation import (
    Tessellation, TessellationFace, TessellationVertex
)
from mundimonium.coordinates.isometric import (
    IsometricDirection, IsometricPoint
)

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Number
from typing import override
import collections
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

# How finely `RelaxableFace` samples each face for `GenericTessellation.
# flatten_region`'s relaxation algorithm -- each face contributes
# `(resolution + 1) * (resolution + 2) / 2` points. A starting, empirically-
# tunable placeholder (see the plan).
_LATTICE_RESOLUTION = 3

# The radius (in this tessellation's own geodesic-distance units) `flatten_
# region` relaxes out to from its anchor point. Also a starting placeholder.
RELAXATION_RADIUS = 3.0

# Position-based-dynamics relaxation passes per `flatten_region` call: more
# when starting from a fresh hinge-unfolded guess (a new/distant anchor),
# fewer when warm-started from a previous call's already-relaxed positions
# (the same or an adjacent anchor -- see the plan). Empirically chosen: at
# `_SOR_FACTOR` below, 30 cold-start passes bring per-pass point movement
# (the convergence signal that actually matters -- see `_SOR_FACTOR`'s own
# comment) down to roughly 0.5% of the average edge length on a regular
# mesh and roughly 2-3% on an irregular-valence one (the latter never
# reaches exactly zero -- flattening a non-valence-6 vertex is inherently
# lossy, so some residual settling is expected, not a bug); 4 warm-start
# passes bring even a deliberately large (10% of edge length) perturbation
# back under 1% within 4-5 passes, and real anchor-to-anchor jumps between
# adjacent lattice points move far less than that. Still starting
# placeholders in the sense that nothing has tuned them against an actual
# rendered frame budget yet -- see the plan.
_COLD_START_PASSES = 30
_WARM_START_PASSES = 4

# Successive-*under*-relaxation factor applied to each PBD correction in
# `_relaxation_pass`. `1.0` is the "textbook" undamped Gauss-Seidel PBD
# update, but empirically (a fixed-anchor benchmark sweeping this factor
# against per-pass point movement over many passes, on both a regular and
# an irregular-valence mesh -- not part of the test suite, since it's a
# one-time tuning question, not an ongoing correctness one) `1.0` doesn't
# actually converge here: this constraint graph has enough interlocking
# short cycles (six edges meeting at most lattice points) that a full,
# undamped correction overshoots and settles into a small stable
# oscillation instead of decaying to zero. *Under*-relaxing damps that
# overshoot: values from about `0.5` to `0.65` converge cleanly and
# fastest (geometric decay, not a plateau); below that range convergence
# slows down again, and above roughly `0.7` the same oscillation reappears
# on the irregular mesh. `0.6` was the best-converging value in that
# stable band on both meshes tested.
_SOR_FACTOR = 0.6


def _canonical_local_frame(face: TessellationFace) -> np.ndarray:
  """The canonical local 2D layout of `face`'s own (b, s, d) system.

  Vertices B, S, D sit at (0, 0), (a, 0), (a/2, h), where `a` is
  `face.side_length` and `h` is `face.altitude`. Computed per-face (rather
  than assuming a uniform side length mesh-wide) so it stays correct for
  distorted faces too.

  Module-level (not a method) since both `GenericTessellation` (corridor/
  ray unfolding) and `RelaxableFace` (lattice rest-length geometry) need
  it, with no other dependency on either class.
  """
  a = float(face.side_length)
  h = float(face.altitude)
  return np.array([[0.0, 0.0], [a, 0.0], [0.5 * a, h]], dtype=np.float64)


def _lattice_ijk(resolution: int) -> list[tuple[int, int, int]]:
  """Every barycentric-index triple `(i, j, k)` with `i + j + k ==
  resolution`, `i, j, k >= 0` -- `RelaxableFace`'s sample pattern, the
  same relative shape on every face regardless of size.
  """
  return [
      (i, j, resolution - i - j)
      for i in range(resolution + 1)
      for j in range(resolution + 1 - i)
  ]


def _lattice_adjacency(resolution: int) -> list[list[int]]:
  """For each index into `_lattice_ijk(resolution)`, the indices of its
  immediate lattice neighbors (one step along either barycentric axis,
  within the same face) -- up to 6, fewer on the pattern's own boundary.
  """
  triples = _lattice_ijk(resolution)
  index_of = {triple: index for index, triple in enumerate(triples)}
  steps = ((-1, 1, 0), (1, -1, 0), (-1, 0, 1), (1, 0, -1), (0, -1, 1), (0, 1, -1))
  adjacency = []
  for i, j, k in triples:
    neighbors = []
    for di, dj, dk in steps:
      candidate = (i + di, j + dj, k + dk)
      if candidate in index_of:
        neighbors.append(index_of[candidate])
    adjacency.append(neighbors)
  return adjacency


_LATTICE_TRIPLES = _lattice_ijk(_LATTICE_RESOLUTION)
_LATTICE_ADJACENCY = _lattice_adjacency(_LATTICE_RESOLUTION)


def _edge_key(
    vertex_a: TessellationVertex, vertex_b: TessellationVertex,
    numerator_from_a: int,
) -> tuple:
  """A canonical, face-independent identity for the relaxation point
  `numerator_from_a / _LATTICE_RESOLUTION` of the way from `vertex_a` to
  `vertex_b` along their shared edge.

  Canonicalized (by an arbitrary but stable `id()` tie-break) so the two
  faces on either side of a shared edge -- which generally label
  `vertex_a`/`vertex_b` differently -- agree on the same key for the same
  physical point.
  """
  if id(vertex_a) > id(vertex_b):
    vertex_a, vertex_b = vertex_b, vertex_a
    numerator_from_a = _LATTICE_RESOLUTION - numerator_from_a
  return ('edge', vertex_a, vertex_b, numerator_from_a)


def _interior_key(face: TessellationFace, i: int, j: int) -> tuple:
  """A relaxation point strictly inside `face` -- never shared with any
  other face, unlike a corner (`TessellationVertex` identity) or edge
  point (`_edge_key`).
  """
  return ('interior', face, i, j)


@dataclass(frozen=True)
class _RelaxationNeighborhood:
  """Everything within `RELAXATION_RADIUS` of one anchor relaxation point
  -- computed once (by `GenericTessellation._relaxation_neighborhood`) and
  cached per anchor key, since it depends only on mesh topology/geometry,
  never on which `flatten_region` call asked for it.
  """

  # Every key in range, mapped to its distance from the anchor and one
  # concrete `(face, local_index)` location for it (arbitrary, if the key
  # has more than one -- e.g. a shared vertex -- since they're all
  # geometrically coincident).
  locations: dict[object, tuple[float, TessellationFace, int]]

  # Every face touched by `locations`, in discovery order (deterministic,
  # for reproducible relaxation -- see `edges`).
  faces: tuple[TessellationFace, ...]

  # Every lattice edge with both endpoints in `locations`, as `(key_a,
  # key_b, rest_length)`. Not deduplicated -- an edge between two shared
  # (multi-face) keys can appear more than once, from different faces'
  # own copies of it, which is harmless for the position-based-dynamics
  # relaxation that consumes this (each pass just applies the same
  # correction more than once) but not worth the bookkeeping to avoid.
  edges: tuple[tuple[object, object, float], ...]


class RelaxableFace(TessellationFace):
  """A `TessellationFace` that additionally exposes a fixed lattice of
  interpolated sample points, used by `GenericTessellation.flatten_
  region`'s relaxation algorithm (see the plan).

  Used via `GenericTessellation(..., face_type=RelaxableFace)`,
  mirroring how `LodMeshFace` opts a `SphericalTessellation` into
  LOD-tree awareness.
  """

  def __init__(
      self,
      *,
      vertex_b: TessellationVertex,
      vertex_s: TessellationVertex,
      vertex_d: TessellationVertex,
      **kwargs):
    super().__init__(
        vertex_b=vertex_b, vertex_s=vertex_s, vertex_d=vertex_d, **kwargs)
    self._relaxation_points: tuple[IsometricPoint, ...] | None = None
    self._relaxation_dedup_keys: tuple[object, ...] | None = None
    self._relaxation_key_by_point: dict[IsometricPoint, object] | None = None

  @property
  def relaxation_points(self) -> tuple[IsometricPoint, ...]:
    """A fixed lattice of `_LATTICE_RESOLUTION`-based interpolated points
    on this face -- the same relative pattern as every other
    `RelaxableFace`, in the same order as `dedup_keys()`. Computed once,
    lazily, and cached.
    """
    if self._relaxation_points is None:
      self._build_relaxation_lattice()
    return self._relaxation_points

  def dedup_keys(self) -> tuple[object, ...]:
    """`relaxation_points[i]`'s cross-face identity, for the same index
    `i` -- a `TessellationVertex` (corner), an `_edge_key(...)` tuple
    (edge point), or an `_interior_key(...)` tuple (interior point).
    Computed once, lazily, and cached.
    """
    if self._relaxation_dedup_keys is None:
      self._build_relaxation_lattice()
    return self._relaxation_dedup_keys

  def dedup_key_for_point(self, point: IsometricPoint) -> object | None:
    """The `dedup_keys()` entry for `point`, if `point` happens to
    coincide exactly with one of this face's own `relaxation_points`
    (e.g. it was constructed the same way, such as a face corner) --
    `None` otherwise. Used by `GenericTessellation._resolve_target` to
    read a lattice-point target's already-relaxed position directly,
    rather than only ever approximating it via barycentric interpolation
    of the face's 3 corners.
    """
    if self._relaxation_key_by_point is None:
      self._build_relaxation_lattice()
    return self._relaxation_key_by_point.get(point)

  def _build_relaxation_lattice(self) -> None:
    """Computes and caches `relaxation_points`/`dedup_keys` together."""
    n = _LATTICE_RESOLUTION
    altitude = self.altitude
    points = []
    keys = []
    for i, j, k in _LATTICE_TRIPLES:
      points.append(
          IsometricPoint(self, (i / n) * altitude, (j / n) * altitude))
      keys.append(self._relaxation_dedup_key(i, j, k))
    self._relaxation_points = tuple(points)
    self._relaxation_dedup_keys = tuple(keys)
    self._relaxation_key_by_point = dict(zip(points, keys))

  def _relaxation_dedup_key(self, i: int, j: int, k: int) -> object:
    """The cross-face identity of this face's own lattice point `(i, j,
    k)` -- see `_edge_key`/`_interior_key`'s docstrings for the
    classification and canonicalization this relies on.
    """
    n = _LATTICE_RESOLUTION
    if i == n:
      return self.vertex_b
    if j == n:
      return self.vertex_s
    if k == n:
      return self.vertex_d
    if i == 0:
      return _edge_key(self.vertex_s, self.vertex_d, k)
    if j == 0:
      return _edge_key(self.vertex_d, self.vertex_b, i)
    if k == 0:
      return _edge_key(self.vertex_b, self.vertex_s, j)
    return _interior_key(self, i, j)


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

    # `flatten_region`'s caches (see the plan). `_relaxation_neighborhoods`
    # maps an anchor point's dedup key to its precomputed neighborhood:
    # every other key within RELAXATION_RADIUS, each mapped to `(distance,
    # face, local_index)` -- computed once per anchor, lazily, and reused
    # by every later call that snaps to the same anchor.
    # `_flatten_position_cache` holds the last solve's relaxed positions,
    # keyed by dedup key, for warm-starting the next call (see `_relax`).
    self._relaxation_neighborhoods: dict[
        object, dict[object, tuple[float, TessellationFace, int]]] = {}
    self._flatten_position_cache: dict[object, np.ndarray] = {}

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
  ) -> list[tuple[float, float]]:
    """Maps `targets` into a locally flat 2D coordinate system, relaxed
    outward from `center`'s nearest `RelaxableFace` sample point.

    Unlike `SphericalTessellation`/`HyperbolicTessellation` (Phase 4),
    `GenericTessellation` has no closed-form embedding to compute an
    exact log map from, so this instead relaxes a fixed lattice of
    sample points (`RelaxableFace.relaxation_points`) via position-based
    dynamics -- pinned exactly at an anchor point snapped to `center`,
    with progressively more freedom out to `RELAXATION_RADIUS`. See the
    plan for the full algorithm and why it's structured this way.

    Requires `self.face_type` to be (a subclass of) `RelaxableFace`.

    Args:
      center: The point the flattened region is centered on. Snapped to
        whichever of its own face's `relaxation_points` is closest.
      targets: The points to flatten, in any order.

    Returns:
      One `(x, y)` pair per point in `targets`, in the same order.

    Raises:
      TypeError: If `self.face_type` isn't a `RelaxableFace` subclass.
      ValueError: If a target's face lies outside `RELAXATION_RADIUS` of
        `center`.
    """
    if not isinstance(center.grid, RelaxableFace):
      raise TypeError(
          "GenericTessellation.flatten_region: requires self.face_type to "
          "be a RelaxableFace subclass, got "
          f"{type(center.grid).__name__}.")

    if not targets:
      return []

    anchor_face = center.grid
    anchor_index = self._closest_relaxation_point(anchor_face, center)
    anchor_key = anchor_face.dedup_keys()[anchor_index]

    neighborhood = self._relaxation_neighborhood(
        anchor_key, anchor_face, anchor_index)
    positions = self._relax(anchor_key, neighborhood)

    return [self._resolve_target(target, positions) for target in targets]

  @staticmethod
  def _closest_relaxation_point(
      face: TessellationFace, point: IsometricPoint) -> int:
    """The index into `face.relaxation_points` closest to `point` (also
    on `face`), by true local distance.
    """
    points = face.relaxation_points
    return min(
        range(len(points)), key=lambda index: point.distance_from(points[index]))

  def _relaxation_neighborhood(
      self, anchor_key: object, anchor_face: TessellationFace,
      anchor_index: int,
  ) -> _RelaxationNeighborhood:
    """Returns (computing and caching if needed) everything within
    `RELAXATION_RADIUS` of `anchor_key` -- a single-source Dijkstra over
    the lattice-point graph, capped at `RELAXATION_RADIUS`. See
    `_RelaxationNeighborhood`.
    """
    cached = self._relaxation_neighborhoods.get(anchor_key)
    if cached is not None:
      return cached

    locations: dict[object, tuple[float, TessellationFace, int]] = {
        anchor_key: (0.0, anchor_face, anchor_index),
    }
    faces: list[TessellationFace] = [anchor_face]
    faces_seen: set[TessellationFace] = {anchor_face}
    counter = itertools.count()
    frontier = [(0.0, next(counter), anchor_key, anchor_face, anchor_index)]

    while frontier:
      dist, _, key, arrival_face, arrival_index = heapq.heappop(frontier)
      if dist > locations[key][0]:
        continue
      for occurrence_face, occurrence_index in self._relaxation_occurrences(
          key, arrival_face, arrival_index):
        if occurrence_face not in faces_seen:
          faces_seen.add(occurrence_face)
          faces.append(occurrence_face)
        for neighbor_key, neighbor_index, rest_length in (
            self._relaxation_local_edges(occurrence_face, occurrence_index)):
          new_dist = dist + rest_length
          if new_dist > RELAXATION_RADIUS:
            continue
          existing = locations.get(neighbor_key)
          if existing is not None and existing[0] <= new_dist:
            continue
          locations[neighbor_key] = (
              new_dist, occurrence_face, neighbor_index)
          heapq.heappush(
              frontier,
              (new_dist, next(counter), neighbor_key, occurrence_face,
               neighbor_index))

    edges = self._relaxation_edges(faces, locations)
    neighborhood = _RelaxationNeighborhood(
        locations=locations, faces=tuple(faces), edges=edges)
    self._relaxation_neighborhoods[anchor_key] = neighborhood
    return neighborhood

  @staticmethod
  def _relaxation_occurrences(
      key: object, arrival_face: TessellationFace, arrival_index: int,
  ) -> list[tuple[TessellationFace, int]]:
    """Every `(face, local_index)` where `key` appears as a relaxation
    point -- every face whose own lattice must be locally expanded to
    find all of `key`'s neighbors (a shared vertex can touch many faces;
    a shared edge point, at most two; an interior point, only its own).
    """
    if isinstance(key, TessellationVertex):
      return [
          (face, face.dedup_keys().index(key))
          for face in key.adjacent_faces()
      ]

    if isinstance(key, tuple) and key[0] == 'edge':
      i, j, _k = _LATTICE_TRIPLES[arrival_index]
      if i == 0:
        direction = IsometricDirection.B
      elif j == 0:
        direction = IsometricDirection.S
      else:
        direction = IsometricDirection.D
      occurrences = [(arrival_face, arrival_index)]
      other_face = arrival_face.face_on_edge(direction)
      if other_face is not None:
        occurrences.append((other_face, other_face.dedup_keys().index(key)))
      return occurrences

    return [(arrival_face, arrival_index)]  # interior point

  @staticmethod
  def _relaxation_local_edges(
      face: TessellationFace, local_index: int,
  ) -> list[tuple[object, int, float]]:
    """Every immediate lattice-neighbor of `face`'s own point at
    `local_index`, as `(neighbor_key, neighbor_local_index, rest_length)`
    -- rest length from the flat 2D distance in `face`'s own canonical
    frame, exact since that frame *is* the flat truth within `face`.
    """
    keys = face.dedup_keys()
    frame = _canonical_local_frame(face)
    n = _LATTICE_RESOLUTION
    i0, j0, k0 = _LATTICE_TRIPLES[local_index]
    pos0 = (i0 / n) * frame[0] + (j0 / n) * frame[1] + (k0 / n) * frame[2]
    results = []
    for neighbor_index in _LATTICE_ADJACENCY[local_index]:
      i1, j1, k1 = _LATTICE_TRIPLES[neighbor_index]
      pos1 = (i1 / n) * frame[0] + (j1 / n) * frame[1] + (k1 / n) * frame[2]
      rest_length = float(np.linalg.norm(pos1 - pos0))
      results.append((keys[neighbor_index], neighbor_index, rest_length))
    return results

  @staticmethod
  def _relaxation_edges(
      faces: Sequence[TessellationFace],
      locations: dict[object, tuple[float, TessellationFace, int]],
  ) -> tuple[tuple[object, object, float], ...]:
    """Every lattice edge, from every face in `faces`, whose both
    endpoints are in `locations` -- see `_RelaxationNeighborhood.edges`.
    """
    edges = []
    for face in faces:
      for index_a, key_a in enumerate(face.dedup_keys()):
        if key_a not in locations:
          continue
        for key_b, _index_b, rest_length in (
            GenericTessellation._relaxation_local_edges(face, index_a)):
          if key_b in locations:
            edges.append((key_a, key_b, rest_length))
    return tuple(edges)

  def _relax(
      self, anchor_key: object, neighborhood: _RelaxationNeighborhood,
  ) -> dict[object, np.ndarray]:
    """Solves for every neighborhood point's 2D position, warm-starting
    from the previous call's positions whenever `anchor_key` was itself
    among them -- true not just when the anchor is exactly unchanged, but
    also when it's moved to a nearby lattice point still within the old
    neighborhood (the common case as a camera pans continuously: most of
    the neighborhood persists between one snapped anchor and the next).
    """
    old_positions = self._flatten_position_cache
    # The new anchor's own position under the *old* frame -- re-centering
    # every reused position against this (rather than reusing them as-is)
    # is what makes them valid seeds in the new frame, where `anchor_key`
    # itself must land at exactly `(0, 0)`. `None` when the new anchor
    # wasn't part of the previous call's neighborhood at all (a distant
    # jump, or the very first call) -- nothing to meaningfully reuse.
    old_anchor_position = old_positions.get(anchor_key)

    positions: dict[object, np.ndarray] = {}
    if old_anchor_position is not None:
      for key in neighborhood.locations:
        cached = old_positions.get(key)
        if cached is not None:
          positions[key] = cached - old_anchor_position

    missing = [key for key in neighborhood.locations if key not in positions]
    if missing:
      self._seed_initial_positions(anchor_key, neighborhood, positions, missing)
    positions[anchor_key] = np.zeros(2, dtype=np.float64)

    mobility = {
        key: min(1.0, distance / RELAXATION_RADIUS)
        for key, (distance, _face, _index) in neighborhood.locations.items()
    }
    mobility[anchor_key] = 0.0

    warm = old_anchor_position is not None
    passes = _WARM_START_PASSES if warm else _COLD_START_PASSES
    for _ in range(passes):
      self._relaxation_pass(positions, neighborhood.edges, mobility)

    self._flatten_position_cache = positions
    return positions

  @staticmethod
  def _seed_initial_positions(
      anchor_key: object,
      neighborhood: _RelaxationNeighborhood,
      positions: dict[object, np.ndarray],
      missing: Sequence[object],
  ) -> None:
    """Fills in `positions[key]` for every `key` in `missing`, via
    hinge-unfolding -- an initial guess for `_relax`'s relaxation passes
    to refine, re-centered so `anchor_key` lands at exactly `(0, 0)`.
    """
    frames = GenericTessellation._hinge_unfold_faces(neighborhood.faces)
    n = _LATTICE_RESOLUTION

    def unfolded_position(face: TessellationFace, index: int) -> np.ndarray:
      i, j, k = _LATTICE_TRIPLES[index]
      frame = frames[face]
      return (i / n) * frame[0] + (j / n) * frame[1] + (k / n) * frame[2]

    _, anchor_face, anchor_index = neighborhood.locations[anchor_key]
    anchor_origin = unfolded_position(anchor_face, anchor_index)

    for key in missing:
      _, face, index = neighborhood.locations[key]
      positions[key] = unfolded_position(face, index) - anchor_origin

  @staticmethod
  def _hinge_unfold_faces(
      faces: Sequence[TessellationFace],
  ) -> dict[TessellationFace, np.ndarray]:
    """Hinge-unfolds every face in `faces` into `faces[0]`'s own
    (untransformed) canonical frame, via face adjacency -- the same
    rotation transform `shortest_path_by_segment`/`geodesically_
    canonicalize_point` use to unfold one shared edge at a time,
    generalized from a single corridor to a whole BFS tree. Every face in
    `faces` must be reachable from `faces[0]` via `face_on_edge` without
    leaving the set (true of `_relaxation_neighborhood`'s own `faces`,
    since it discovers them by exactly that kind of traversal).
    """
    relevant = set(faces)
    root = faces[0]
    unfolded: dict[TessellationFace, np.ndarray] = {
        root: _canonical_local_frame(root),
    }
    queue = collections.deque([root])
    while queue:
      face = queue.popleft()
      face_frame = unfolded[face]
      for direction in IsometricDirection:
        neighbor = face.face_on_edge(direction)
        if (neighbor is None or neighbor not in relevant
            or neighbor in unfolded):
          continue

        vertex_a = face.vertex_at(
            IsometricDirection((direction.value + 1) % 3))
        vertex_b = face.vertex_at(
            IsometricDirection((direction.value + 2) % 3))
        pos_a = face_frame[(direction.value + 1) % 3]
        pos_b = face_frame[(direction.value + 2) % 3]

        neighbor_local = _canonical_local_frame(neighbor)
        local_a = neighbor_local[neighbor.direction_toward_vertex(vertex_a).value]
        local_b = neighbor_local[neighbor.direction_toward_vertex(vertex_b).value]

        # Rotation mapping `neighbor`'s own local edge vector onto its
        # already-unfolded counterpart -- "unfolding" `neighbor` flat,
        # hinged along the shared edge.
        u = local_b - local_a
        v = pos_b - pos_a
        inv_len_sq = 1.0 / (u[0] ** 2 + u[1] ** 2)
        cos_t = (u[0] * v[0] + u[1] * v[1]) * inv_len_sq
        sin_t = (u[0] * v[1] - u[1] * v[0]) * inv_len_sq
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

        unfolded[neighbor] = np.array([
            pos_a + rotation @ (neighbor_local[k] - local_a)
            for k in range(3)
        ])
        queue.append(neighbor)
    return unfolded

  @staticmethod
  def _relaxation_pass(
      positions: dict[object, np.ndarray],
      edges: Sequence[tuple[object, object, float]],
      mobility: dict[object, float],
  ) -> None:
    """One position-based-dynamics sweep: for each edge, nudges both
    endpoints toward satisfying its rest length, weighted by mobility
    (`0` pinned, `1` fully free) -- see the plan.
    """
    for key_a, key_b, rest_length in edges:
      pos_a = positions[key_a]
      pos_b = positions[key_b]
      delta = pos_a - pos_b
      current_length = float(np.linalg.norm(delta))
      if current_length < 1e-12:
        continue
      weight_a = mobility[key_a]
      weight_b = mobility[key_b]
      total_weight = weight_a + weight_b
      if total_weight < 1e-12:
        continue
      direction = delta / current_length
      correction = _SOR_FACTOR * (current_length - rest_length)
      positions[key_a] = (
          pos_a - direction * (weight_a / total_weight) * correction)
      positions[key_b] = (
          pos_b + direction * (weight_b / total_weight) * correction)

  @staticmethod
  def _resolve_target(
      target: IsometricPoint, positions: dict[object, np.ndarray],
  ) -> tuple[float, float]:
    """`target`'s relaxed 2D position.

    If `target` coincides exactly with one of its own face's lattice
    sample points (the common case: mesh vertices and other lattice-
    aligned points always are one), its own already-relaxed position is
    used directly -- exact, not an approximation. Otherwise, falls back
    to barycentric interpolation of its face's 3 corner nodes' relaxed
    positions, treating the relaxed face as still locally affine, the
    same simplification other callers in this project already accept for
    reading a position off a parameterized triangle (e.g.
    `flat_mesh_export.py`).

    Raises:
      ValueError: If `target`'s face lies outside `RELAXATION_RADIUS` of
        `center` (i.e. wasn't part of the relaxed neighborhood).
    """
    face = target.grid
    if isinstance(face, RelaxableFace):
      key = face.dedup_key_for_point(target)
      if key is not None:
        position = positions.get(key)
        if position is None:
          raise ValueError(
              "GenericTessellation.flatten_region: a target's face lies "
              "outside RELAXATION_RADIUS of center.")
        return (float(position[0]), float(position[1]))

    wb, ws, wd = target.barycentric
    try:
      position = (
          wb * positions[face.vertex_b]
          + ws * positions[face.vertex_s]
          + wd * positions[face.vertex_d])
    except KeyError as error:
      raise ValueError(
          "GenericTessellation.flatten_region: a target's face lies "
          "outside RELAXATION_RADIUS of center.") from error
    return (float(position[0]), float(position[1]))

  @override
  def on_vertex_added(self, vertex: TessellationVertex) -> None:
    """Hook for updating internal state when mesh topology changes."""
    super().on_vertex_added(vertex)
    self.invalidate_solvers()
    self.invalidate_relaxation_caches()

  @override
  def on_face_added(self, face: TessellationFace) -> None:
    """Hook for updating internal state when mesh topology changes."""
    super().on_face_added(face)
    self.invalidate_solvers()
    self.invalidate_relaxation_caches()

  def invalidate_solvers(self) -> None:
    """Invalidates cached matrix factorizations when topology changes."""
    self._matrices_built = False
    self._heat_solver = None
    self._poisson_solver = None

  def invalidate_relaxation_caches(self) -> None:
    """Invalidates `flatten_region`'s caches when topology changes.

    Not selective (clears every cached neighborhood/position, not just
    ones a specific new face/vertex could actually affect) -- simple and
    correct, at the cost of some rework if the mesh is mutated frequently
    after caches are already warm (not the primary use case -- see the
    plan).
    """
    self._relaxation_neighborhoods = {}
    self._flatten_position_cache = {}

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
    frames = [_canonical_local_frame(corridor[0])]
    portal_vertices: list[tuple[TessellationVertex, TessellationVertex]] = []
    portal_positions: list[tuple[np.ndarray, np.ndarray]] = []

    for prev_face, curr_face in zip(corridor, corridor[1:]):
      prev_frame = frames[-1]
      edge_dir = prev_face.direction_away_from_face(curr_face)
      vA = prev_face.vertex_at(IsometricDirection((edge_dir.value + 1) % 3))
      vB = prev_face.vertex_at(IsometricDirection((edge_dir.value + 2) % 3))
      pA = prev_frame[(edge_dir.value + 1) % 3]
      pB = prev_frame[(edge_dir.value + 2) % 3]

      curr_local = _canonical_local_frame(curr_face)
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
    p_local = _canonical_local_frame(curr_face)
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
      p_local = _canonical_local_frame(curr_face)
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
      next_p_local = _canonical_local_frame(next_face)
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
