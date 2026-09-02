from __future__ import annotations

from mundimonium.coordinates.exceptions import EndOfMeshSurfaceException
from mundimonium.coordinates.isometric import IsometricDirection, IsometricPoint
from mundimonium.coordinates.tessellation import (
    Tessellation, TessellationFace, TessellationVertex
)

from numbers import Number
from typing import override
import cmath
import math
import numpy as np


_MAX_WALK_STEPS = 10_000
_MAX_EXTEND_TO_INCLUDE_ITERATIONS = 1_000

# A tiny fixed rotation applied to `geodesically_canonicalize_point`'s
# target direction before exponentiating, purely to generically avoid it
# passing *exactly* through a mesh vertex -- where every edge touching that
# vertex reports the same crossing, leaving no well-defined "next edge" for
# `_walk_klein_line` to pick (the same degenerate-ray problem
# `GenericTessellation.geodesically_canonicalize_point` guards against with
# its own `_DEGENERATE_RAY_NUDGE_RADIANS`). Small enough to leave every real
# result unaffected.
_DEGENERATE_RAY_NUDGE_RADIANS = 1e-7

# Tolerance for treating barycentric weights as "in [0, 1]" (a point has
# landed inside a face) or as "the same vertex" (mesh-growth deduplication),
# both in units of the tessellation's own edge length / hyperbolic distance.
_BARYCENTRIC_EPSILON = 1e-9
# Looser than most other tolerances in this file: composing several
# Poincare-disk Mobius transformations during mesh growth (see
# `_complete_fan`) accumulates floating-point error up to roughly 1e-7 by
# the time two independent derivations of "the same" vertex are compared,
# well above naive double-precision epsilon.
_VERTEX_MATCH_EPSILON = 1e-6

# Side length, in Poincare-disk coordinates, of the square cells used to
# spatially index already-placed vertices during mesh growth (see
# `_find_existing_vertex_near`) -- must be comfortably larger than any
# expected floating-point drift between two derivations of "the same"
# vertex position, and comfortably smaller than the tessellation's own
# edge length (in Poincare-disk terms) so that genuinely distinct vertices
# essentially never collide into the same cell.
_VERTEX_CELL_SIZE = 0.01


def _edge_length_for_order(order: int) -> float:
  """Edge length of the regular {3, `order`} hyperbolic tiling's equilateral
  triangle.

  Every face has the same angle `alpha = 2*pi/order` at each of its three
  corners (so `order` of them tile exactly around each vertex). The
  hyperbolic law of cosines for angles, `cos(C) = -cos(A)*cos(B) +
  sin(A)*sin(B)*cosh(c)`, specialized to an equilateral triangle
  (`A = B = C = alpha`, `a = b = c = L`), solves to:
  `cosh(L) = cos(alpha) / (1 - cos(alpha))`.
  """
  alpha = 2.0 * math.pi / order
  cos_alpha = math.cos(alpha)
  return math.acosh(cos_alpha / (1.0 - cos_alpha))


def _to_origin_poincare(z: complex, center: complex) -> complex:
  """The disk automorphism of the Poincare disk sending `center` to the
  origin, applied to `z`."""
  return (z - center) / (1.0 - center.conjugate() * z)


def _from_origin_poincare(w: complex, center: complex) -> complex:
  """Inverse of `_to_origin_poincare`."""
  return (w + center) / (1.0 + center.conjugate() * w)


def _rotate_around_poincare(
    z: complex, center: complex, theta: float) -> complex:
  """Rotates `z` by `theta` about `center`, as a hyperbolic isometry of the
  Poincare disk: conjugates an origin-centered rotation through the disk
  automorphism that sends `center` to the origin (and its inverse, back).
  """
  w = _to_origin_poincare(z, center) * cmath.exp(1j * theta)
  return _from_origin_poincare(w, center)


def _minkowski_of_poincare(z: complex) -> tuple[float, float, float]:
  """Converts a Poincare-disk point to Minkowski (hyperboloid) `(X, Y, T)`
  coordinates."""
  r2 = z.real * z.real + z.imag * z.imag
  denom = 1.0 - r2
  return (2.0 * z.real / denom, 2.0 * z.imag / denom, (1.0 + r2) / denom)


def _poincare_of_minkowski(p: tuple[float, float, float]) -> complex:
  """Converts Minkowski `(X, Y, T)` coordinates to a Poincare-disk point."""
  x, y, t = p
  denom = 1.0 + t
  return complex(x / denom, y / denom)


def _klein_of_minkowski(p: tuple[float, float, float]) -> tuple[float, float]:
  """Converts Minkowski `(X, Y, T)` coordinates to Klein-disk `(u, v)`."""
  x, y, t = p
  return (x / t, y / t)


def _minkowski_of_klein(u: float, v: float) -> tuple[float, float, float]:
  """Converts Klein-disk `(u, v)` coordinates to Minkowski `(X, Y, T)`."""
  denom = math.sqrt(max(1e-15, 1.0 - u * u - v * v))
  return (u / denom, v / denom, 1.0 / denom)


def _poincare_of_klein(u: float, v: float) -> complex:
  """Converts Klein-disk `(u, v)` coordinates to a Poincare-disk point."""
  denom = 1.0 + math.sqrt(max(0.0, 1.0 - u * u - v * v))
  return complex(u / denom, v / denom)


def _klein_of_poincare(z: complex) -> tuple[float, float]:
  """Converts a Poincare-disk point to Klein-disk `(u, v)` coordinates."""
  denom = 1.0 + z.real * z.real + z.imag * z.imag
  return (2.0 * z.real / denom, 2.0 * z.imag / denom)


def _minkowski_of_vertex(
    vertex: TessellationVertex) -> tuple[float, float, float]:
  """A vertex's Minkowski `(X, Y, T)` position, from its stored
  `projection_coordinates = (X, Y, -T)`."""
  x, y, z = vertex.projection_coordinates
  return (x, y, -z)


def _klein_of_vertex(vertex: TessellationVertex) -> tuple[float, float]:
  return _klein_of_minkowski(_minkowski_of_vertex(vertex))


def _poincare_of_vertex(vertex: TessellationVertex) -> complex:
  return _poincare_of_minkowski(_minkowski_of_vertex(vertex))


def _minkowski_inner(
    p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
  """The Minkowski (Lorentzian) inner product of two `(X, Y, T)` triples."""
  return p[0] * q[0] + p[1] * q[1] - p[2] * q[2]


def _hyperbolic_distance(
    p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
  """Exact hyperbolic distance between two Minkowski-model points."""
  return math.acosh(max(1.0, -_minkowski_inner(p, q)))


def _barycentric_in_bounds(bary: tuple[float, float, float]) -> bool:
  return all(-_BARYCENTRIC_EPSILON <= w <= 1.0 + _BARYCENTRIC_EPSILON
             for w in bary)


def _clamp_barycentric(
    bary: tuple[float, float, float]) -> tuple[float, float, float]:
  """Clamps near-boundary floating-point noise back into `[0, 1]`."""
  wb = min(1.0, max(0.0, bary[0]))
  ws = min(1.0, max(0.0, bary[1]))
  return (wb, ws, 1.0 - wb - ws)


def _vertex_cell_key(z: complex) -> tuple[int, int]:
  return (math.floor(z.real / _VERTEX_CELL_SIZE),
          math.floor(z.imag / _VERTEX_CELL_SIZE))


def _local_frame(face: TessellationFace) -> np.ndarray:
  """The canonical local flat 2D layout of `face`'s own (b, s, d) system:
  vertices B, S, D at `(0, 0)`, `(a, 0)`, `(a/2, h)`, where `a` is
  `face.side_length` and `h` is `face.altitude`.

  Used only as a reference frame for interpreting an out-of-`[0,1]`-weight
  point's *displacement direction and magnitude* from its face's centroid
  (see `geodesically_canonicalize_point`) -- not, as in `GenericTessellation`,
  as a chart to be patched face-to-face by unfolding: the flat frame's own
  notion of distance only matches true hyperbolic distance for the 3
  vertices themselves (calibrated exactly to the tiling's edge length),
  not for arbitrary interior/exterior points, since a hyperbolic triangle's
  interior is not isometric to a flat one.
  """
  a = float(face.side_length)
  h = float(face.altitude)
  return np.array([[0.0, 0.0], [a, 0.0], [0.5 * a, h]], dtype=np.float64)


class HyperbolicTessellation(Tessellation):
  """A growable patch of the hyperbolic plane, tiled entirely by equilateral
  triangles: the regular {3, `order`} tiling (`order` triangles meeting at
  each vertex; `order >= 7` is required for the tiling to actually be
  hyperbolic).

  Unlike `GenericTessellation`, this class doesn't answer geodesic queries
  via a graph search over face adjacency -- a hyperbolic ball of radius r
  contains exponentially many faces in r (unlike a sphere, where it's
  bounded), so a search touching a meaningful fraction of the mesh becomes
  catastrophically expensive even for modest distances. Instead, every
  vertex's exact position is tracked in the Minkowski (hyperboloid) model,
  whose **Klein disk** projection maps every hyperbolic geodesic to a
  Euclidean straight line in one shared global 2D frame (the exact
  hyperbolic analogue of gnomonic projection on the sphere, generalized
  from "straight locally, per face" to "straight globally"). A geodesic
  query is therefore a direct walk of that already-known straight line
  through the faces it crosses (O(1) work per face, via ordinary 2D
  line/edge intersection), extending the mesh on demand if the walk runs
  past its currently-built extent -- no search, no per-face unfolding.

  Three coordinate systems describe the same points:
  - Minkowski/hyperboloid `(X, Y, T)`, `X^2+Y^2-T^2=-1, T>0`: the *stored*
    representation, as `TessellationVertex.projection_coordinates =
    (X, Y, -T)`. Used directly for the exact distance formula.
  - Klein disk `(u, v) = (X, Y) / T`: this tessellation's mesh/geodesic
    coordinate system (`coords_at_point`/`new_point_at_coords`/
    `get_face_at_coords`) -- the model where face-walking is a straight
    line.
  - Poincare disk `(x, y) = (X, Y) / (1 + T)`: not stored anywhere; a cheap
    on-demand conversion (`to_poincare`), used internally as scratch math
    during mesh growth (see `extend`) and in
    `geodesically_canonicalize_point`, both of which rely on conformal
    (angle-preserving) rotation about an arbitrary point having a clean
    closed form there.

  Every face is a plain equilateral `TessellationFace` (uniform
  `side_length`), so `IsometricPoint.distance_from`/`IsometricGrid.
  local_distance` still work -- but deliberately do *not* agree with
  `geodesic_distance`, even for two points on the same face: that flat
  local-frame formula is only exact for the 3 vertices themselves
  (calibrated, by construction, to the tiling's true edge length), not for
  arbitrary interior points, since a hyperbolic triangle's interior isn't
  isometric to a flat one. Use this tessellation's own `geodesic_distance`/
  `shortest_path_by_segment`, not `distance_from`, wherever exactness
  matters.
  """

  def __init__(self, *, order: int = 7, rings: int = 2, **kwargs):
    """Constructs an initial patch of the {3, `order`} hyperbolic tiling.

    Args:
      order: Number of equilateral-triangle faces meeting at each vertex.
             Must be >= 7 (the threshold below which {3, order} is
             spherical or Euclidean, not hyperbolic).
      rings: How many "rings" of faces to build outward from the seed
             vertex initially (ring 1 is just the seed's own fan of
             `order` faces; each further ring completes the fan of every
             vertex discovered by the previous ring). More can be added
             later via `extend`/`extend_to_include`, including implicitly,
             on demand, by any geodesic query that reaches the mesh's
             current edge.
      **kwargs: Forwarded up the method resolution order.
    """
    if order < 7:
      raise ValueError(
          f"'order' must be >= 7 for a hyperbolic {{3, order}} tiling "
          f"(got {order}).")
    if rings < 1:
      raise ValueError(f"'rings' must be >= 1 (got {rings}).")
    super().__init__(**kwargs)
    self._order: int = order
    self._edge_length: float = _edge_length_for_order(order)
    self._face_by_vertex_triple: dict[frozenset, TessellationFace] = {}
    self._vertex_cells: dict[tuple[int, int], list[TessellationVertex]] = {}
    self._frontier: list[tuple[TessellationVertex, TessellationVertex]] = []
    self._seed_vertex: TessellationVertex | None = None
    self._seed_face: TessellationFace | None = None
    self._build_seed()
    if rings > 1:
      self.extend(rings=rings - 1)

  @property
  def order(self) -> int:
    """The tiling's `{3, order}` Schlafli parameter."""
    return self._order

  @property
  def faces(self) -> list[TessellationFace]:
    """Every face built so far."""
    return list(self._faces)

  @property
  def vertices(self) -> list[TessellationVertex]:
    """Every vertex built so far."""
    return list(self._vertices)

  @staticmethod
  def to_poincare(vertex: TessellationVertex) -> tuple[float, float]:
    """`vertex`'s Poincare-disk `(x, y)` position, derived on demand from
    its stored Minkowski `projection_coordinates`."""
    z = _poincare_of_vertex(vertex)
    return (z.real, z.imag)

  # ---------------------------------------------------------------------
  # Mesh construction / growth
  # ---------------------------------------------------------------------

  def _new_vertex(self, poincare_z: complex) -> TessellationVertex:
    """Creates and spatially-indexes a new vertex at Poincare position
    `poincare_z`, storing its Minkowski `(X, Y, -T)` projection."""
    x, y, t = _minkowski_of_poincare(poincare_z)
    vertex = self.vertex_type([x, y, -t])
    self._vertex_cells.setdefault(_vertex_cell_key(poincare_z), []).append(
        vertex)
    return vertex

  def _find_existing_vertex_near(
      self, poincare_z: complex) -> TessellationVertex | None:
    """An already-created vertex within `_VERTEX_MATCH_EPSILON` hyperbolic
    distance of `poincare_z`, if any."""
    cx, cy = _vertex_cell_key(poincare_z)
    candidate_minkowski = _minkowski_of_poincare(poincare_z)
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        for candidate in self._vertex_cells.get((cx + dx, cy + dy), ()):
          if _hyperbolic_distance(
              candidate_minkowski, _minkowski_of_vertex(candidate)
          ) < _VERTEX_MATCH_EPSILON:
            return candidate
    return None

  def _add_face_if_new(
      self,
      v_b: TessellationVertex, v_s: TessellationVertex,
      v_d: TessellationVertex,
  ) -> TessellationFace:
    """Registers a new face for `(v_b, v_s, v_d)`, or returns the existing
    one if this triangle was already built (from the "other side" of a
    shared edge)."""
    key = frozenset((v_b, v_s, v_d))
    existing = self._face_by_vertex_triple.get(key)
    if existing is not None:
      return existing
    face = self.add_face([v_b, v_s, v_d], side_length=self._edge_length)
    self._face_by_vertex_triple[key] = face
    return face

  def _build_seed(self) -> None:
    """Builds the seed vertex (at the Minkowski/Klein/Poincare origin) and
    its full fan of `order` faces."""
    order = self._order
    v0 = self._new_vertex(0.0j)
    radius = math.tanh(self._edge_length / 2.0)
    ring = [
        self._new_vertex(radius * cmath.exp(1j * 2.0 * math.pi * k / order))
        for k in range(order)
    ]
    for k in range(order):
      self._add_face_if_new(v0, ring[k], ring[(k + 1) % order])
    self._seed_vertex = v0
    self._seed_face = self._face_by_vertex_triple[
        frozenset((v0, ring[0], ring[1]))]
    # An interior reference point to start raw-coordinate walks from (see
    # `get_face_at_coords`/`new_point_at_coords`) -- deliberately *not*
    # `v0` itself: starting exactly at a shared vertex is degenerate (the
    # ray toward a target outside that one face's angular wedge at `v0`
    # never enters the face at all, since it starts on the boundary
    # between wedges rather than inside one), whereas any interior point
    # correctly steps into whichever neighboring wedge the target actually
    # requires.
    self._seed_start_klein = self._klein_of_barycentric(
        self._seed_face, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    self._frontier = [(v, v0) for v in ring]
    self._conformal_scale = self._compute_conformal_scale(self._seed_face)

  def _compute_conformal_scale(self, face: TessellationFace) -> float:
    """The ratio between true hyperbolic distance and local flat-frame
    distance, from any face's centroid to its own vertices -- the same
    constant for every face and every direction (see
    `geodesically_canonicalize_point`'s docstring for why), so computed
    once from an arbitrary reference face and reused everywhere.
    """
    p_local = _local_frame(face)
    centroid_2d = (p_local[0] + p_local[1] + p_local[2]) / 3.0
    local_dist = float(np.hypot(*(p_local[0] - centroid_2d)))
    centroid_klein = self._klein_of_barycentric(
        face, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    true_dist = _hyperbolic_distance(
        _minkowski_of_klein(*centroid_klein),
        _minkowski_of_vertex(face.vertex_b))
    return true_dist / local_dist

  def _complete_fan(
      self,
      vertex: TessellationVertex, known_neighbor: TessellationVertex,
      next_frontier: list[tuple[TessellationVertex, TessellationVertex]],
  ) -> None:
    """Discovers the rest of `vertex`'s `order` neighbors (it has exactly
    one known so far, `known_neighbor`, from the edge that discovered it)
    by rotating `known_neighbor` around `vertex` in increments of
    `2*pi/order`, then builds the `order` faces around `vertex`.

    Newly-discovered vertices are pushed onto `next_frontier` (paired with
    `vertex`, the neighbor that discovered them) for a later ring to
    complete their own fans in turn.
    """
    order = self._order
    vertex_z = _poincare_of_vertex(vertex)
    neighbor_z = _poincare_of_vertex(known_neighbor)
    ring = [known_neighbor]
    for k in range(1, order):
      theta = 2.0 * math.pi * k / order
      candidate_z = _rotate_around_poincare(neighbor_z, vertex_z, theta)
      match = self._find_existing_vertex_near(candidate_z)
      if match is None:
        match = self._new_vertex(candidate_z)
        next_frontier.append((match, vertex))
      ring.append(match)
    for k in range(order):
      self._add_face_if_new(vertex, ring[k], ring[(k + 1) % order])

  def extend(self, rings: int = 1) -> None:
    """Grows the mesh outward by `rings` more rings from its current
    frontier, in place. Never touches or recreates any existing vertex or
    face.

    A "ring" completes the fan of every vertex on the current frontier
    (each with exactly one known neighbor so far); any newly-discovered
    vertices become the next ring's frontier.
    """
    for _ in range(rings):
      if not self._frontier:
        return
      current, self._frontier = self._frontier, []
      for vertex, known_neighbor in current:
        self._complete_fan(vertex, known_neighbor, self._frontier)

  def extend_to_include(self, point: tuple[float, float, float]) -> None:
    """Grows the mesh (via repeated `extend(rings=1)`) until `point` (a
    Minkowski `(X, Y, T)` triple) is guaranteed to lie within the built
    extent.

    Coverage is checked via the true hyperbolic metric (never a Euclidean
    approximation): the mesh is grown until the closest current-frontier
    vertex is farther from the seed than `point` is, which is a
    conservative (possibly-more-than-necessary, but always sufficient)
    bound, since the frontier's own distance from the seed only grows with
    each ring.
    """
    seed_minkowski = _minkowski_of_vertex(self._seed_vertex)
    target_distance = _hyperbolic_distance(seed_minkowski, point)
    for _ in range(_MAX_EXTEND_TO_INCLUDE_ITERATIONS):
      if not self._frontier:
        return
      frontier_distance = min(
          _hyperbolic_distance(seed_minkowski, _minkowski_of_vertex(v))
          for v, _ in self._frontier)
      if frontier_distance > target_distance:
        return
      self.extend(rings=1)
    raise RuntimeError(
        f"extend_to_include exceeded {_MAX_EXTEND_TO_INCLUDE_ITERATIONS} "
        "growth iterations without covering the target point.")

  # ---------------------------------------------------------------------
  # Klein-coordinate geometry (barycentric conversion, face-walking)
  # ---------------------------------------------------------------------

  def _barycentric_of_klein(
      self, face: TessellationFace, klein_point: tuple[float, float],
  ) -> tuple[float, float, float]:
    """The barycentric `(wb, ws, wd)` weights of `klein_point` (a global
    Klein-disk position) relative to `face`'s 3 vertices' own Klein
    positions.

    This is the exact analogue of `SphericalTessellation`'s gnomonic
    unprojection: because a face's vertices' Klein positions are already
    an exact "flattening" of the hyperbolic triangle (the defining
    property of the Klein model), ordinary planar barycentric coordinates
    here correspond exactly to the face's own intrinsic (b, s, d)
    coordinates -- unlike, say, naively averaging 3D hyperboloid points,
    which would not.
    """
    kb = _klein_of_vertex(face.vertex_b)
    ks = _klein_of_vertex(face.vertex_s)
    kd = _klein_of_vertex(face.vertex_d)
    x, y = klein_point
    denom = (
        (ks[1] - kd[1]) * (kb[0] - kd[0]) + (kd[0] - ks[0]) * (kb[1] - kd[1]))
    wb = (
        (ks[1] - kd[1]) * (x - kd[0]) + (kd[0] - ks[0]) * (y - kd[1])
    ) / denom
    ws = (
        (kd[1] - kb[1]) * (x - kd[0]) + (kb[0] - kd[0]) * (y - kd[1])
    ) / denom
    return (wb, ws, 1.0 - wb - ws)

  def _klein_of_barycentric(
      self, face: TessellationFace, bary: tuple[float, float, float],
  ) -> tuple[float, float]:
    """Inverse of `_barycentric_of_klein`: the global Klein-disk position
    of a point with barycentric weights `bary` on `face`. Well-defined
    (as a straight-line extrapolation) even for weights outside `[0, 1]`,
    as long as they sum to 1."""
    wb, ws, wd = bary
    kb = _klein_of_vertex(face.vertex_b)
    ks = _klein_of_vertex(face.vertex_s)
    kd = _klein_of_vertex(face.vertex_d)
    return (
        wb * kb[0] + ws * ks[0] + wd * kd[0],
        wb * kb[1] + ws * ks[1] + wd * kd[1],
    )

  def _find_exit(
      self,
      face: TessellationFace, pos: tuple[float, float],
      target: tuple[float, float],
  ) -> tuple[IsometricDirection, tuple[float, float]]:
    """The edge of `face` (and the exact position on it) where the straight
    segment from `pos` (inside `face`, in global Klein coordinates) toward
    `target` first exits `face`.

    Same style of 2D line/edge-intersection math as
    `GenericTessellation.shortest_path_by_segment`'s portal crossings and
    `geodesically_canonicalize_point`'s `min_t`/`hit_edge` loop -- except
    here every face's vertices are already in one shared global frame, so
    no per-face unfolding is needed first.
    """
    direction = (target[0] - pos[0], target[1] - pos[1])
    vertex_klein = {
        IsometricDirection.B: _klein_of_vertex(face.vertex_b),
        IsometricDirection.S: _klein_of_vertex(face.vertex_s),
        IsometricDirection.D: _klein_of_vertex(face.vertex_d),
    }
    min_t = float('inf')
    hit_dir: IsometricDirection | None = None
    hit_pos: tuple[float, float] | None = None
    for opposite in IsometricDirection:
      p_a = vertex_klein[IsometricDirection((opposite.value + 1) % 3)]
      p_b = vertex_klein[IsometricDirection((opposite.value + 2) % 3)]
      e_x, e_y = p_b[0] - p_a[0], p_b[1] - p_a[1]
      denom = direction[0] * (-e_y) - direction[1] * (-e_x)
      if abs(denom) < 1e-14:
        continue
      dx, dy = p_a[0] - pos[0], p_a[1] - pos[1]
      t = (dx * (-e_y) - dy * (-e_x)) / denom
      alpha = (direction[0] * dy - direction[1] * dx) / denom
      if 1e-9 < t < min_t and -1e-9 <= alpha <= 1.0 + 1e-9:
        min_t = t
        hit_dir = opposite
        hit_pos = (pos[0] + t * direction[0], pos[1] + t * direction[1])
    if hit_dir is None:
      raise RuntimeError(
          "HyperbolicTessellation: no edge crossing found walking toward "
          "the target point -- the mesh's adjacency may be corrupt.")
    return hit_dir, hit_pos

  def _walk_klein_line(
      self,
      start_face: TessellationFace, start_klein: tuple[float, float],
      target_klein: tuple[float, float],
  ) -> list[tuple[TessellationFace, tuple[float, float], tuple | None]]:
    """Walks the straight (global Klein-coordinate) line from `start_klein`
    (known to lie in `start_face`) to `target_klein`, one face at a time,
    extending the mesh on demand whenever the walk reaches its current
    edge.

    Returns one `(face, entry_klein, exit_klein)` tuple per face crossed;
    the last tuple has `exit_klein = None`, meaning `target_klein` itself
    lies within that face.
    """
    steps: list[tuple[TessellationFace, tuple[float, float], tuple | None]] \
        = []
    curr_face = start_face
    pos = start_klein
    for _ in range(_MAX_WALK_STEPS):
      if _barycentric_in_bounds(
          self._barycentric_of_klein(curr_face, target_klein)):
        steps.append((curr_face, pos, None))
        return steps
      exit_dir, exit_pos = self._find_exit(curr_face, pos, target_klein)
      steps.append((curr_face, pos, exit_pos))
      next_face = curr_face.face_on_edge(exit_dir)
      if next_face is None:
        self.extend(rings=1)
        next_face = curr_face.face_on_edge(exit_dir)
        if next_face is None:
          raise EndOfMeshSurfaceException(
              "HyperbolicTessellation: target point lies beyond the mesh's "
              "reachable extent even after extending.")
      curr_face = next_face
      pos = exit_pos
    raise RuntimeError(
        f"_walk_klein_line exceeded {_MAX_WALK_STEPS} steps; the walk may "
        "be stuck (a mesh topology bug) or the target is extremely far "
        "away.")

  # ---------------------------------------------------------------------
  # Tessellation interface
  # ---------------------------------------------------------------------

  @override
  def coords_at_point(self, point: IsometricPoint) -> tuple[Number, ...]:
    """The global Klein-disk `(u, v)` position of `point`."""
    return self._klein_of_barycentric(point.grid, point.barycentric)

  @override
  def new_point_at_coords(self, *coords: tuple[Number, ...]) -> IsometricPoint:
    """The `IsometricPoint` at Klein-disk coordinates `(u, v)`, extending
    the mesh as needed to reach it."""
    u, v = coords
    steps = self._walk_klein_line(
        self._seed_face, self._seed_start_klein, (u, v))
    face = steps[-1][0]
    bary = _clamp_barycentric(self._barycentric_of_klein(face, (u, v)))
    return IsometricPoint.from_barycentric(face, *bary)

  @override
  def get_face_at_coords(
      self, *coords: tuple[Number, ...]) -> TessellationFace:
    """The face containing Klein-disk coordinates `(u, v)`, extending the
    mesh as needed to reach it."""
    u, v = coords
    steps = self._walk_klein_line(
        self._seed_face, self._seed_start_klein, (u, v))
    return steps[-1][0]

  @override
  def geodesic_distance(self, p1: IsometricPoint, p2: IsometricPoint) -> float:
    """The exact hyperbolic distance between `p1` and `p2`, in O(1) --
    always via the direct Minkowski-model formula, regardless of how far
    apart the two points are (no search, no face-walking needed for the
    distance alone)."""
    p1_minkowski = _minkowski_of_klein(*self.coords_at_point(p1))
    p2_minkowski = _minkowski_of_klein(*self.coords_at_point(p2))
    return _hyperbolic_distance(p1_minkowski, p2_minkowski)

  @override
  def shortest_path_by_segment(
      self, p1: IsometricPoint, p2: IsometricPoint,
  ) -> list[tuple[IsometricPoint, IsometricPoint]]:
    """Traces the exact geodesic from `p1` to `p2`, split into per-face
    segments, by walking the straight Klein-coordinate line between them
    (see `_walk_klein_line`) -- extending the mesh as needed."""
    if p1.grid is p2.grid:
      return [(p1, p2)]

    target_klein = self.coords_at_point(p2)
    steps = self._walk_klein_line(
        p1.grid, self.coords_at_point(p1), target_klein)

    segments: list[tuple[IsometricPoint, IsometricPoint]] = []
    entry = p1
    for i, (face, _, exit_klein) in enumerate(steps):
      if exit_klein is None:
        segments.append((entry, p2))
        break
      exit_bary = _clamp_barycentric(
          self._barycentric_of_klein(face, exit_klein))
      exit_point = IsometricPoint.from_barycentric(face, *exit_bary)
      segments.append((entry, exit_point))
      next_face = steps[i + 1][0]
      next_bary = _clamp_barycentric(
          self._barycentric_of_klein(next_face, exit_klein))
      entry = IsometricPoint.from_barycentric(next_face, *next_bary)
    return segments

  @override
  def shortest_path(
      self, p1: IsometricPoint, p2: IsometricPoint,
  ) -> list[IsometricPoint]:
    """Traces the exact geodesic polyline from `p1` to `p2` across faces.

    Derived from `shortest_path_by_segment`: each segment's start point
    already matches the previous segment's end point, so the flattened
    sequence is `[p1, then each segment's end point in order]`.
    """
    segments = self.shortest_path_by_segment(p1, p2)
    points = [segments[0][0]]
    points.extend(b for _, b in segments)
    return points

  @override
  def geodesically_canonicalize_point(
      self, point: IsometricPoint) -> IsometricPoint:
    """Moves `point` to a new grid if located outside its current grid's
    bounds, extending the mesh as needed.

    Unlike `GenericTessellation` (which patches its local flat frame across
    faces by unfolding), this stays O(1)-per-step by using the global
    embedding directly, via the hyperbolic exponential map:

    1. `point`'s out-of-`[0, 1]` barycentric weights place a 2D position in
       `point.grid`'s local flat frame (`_local_frame`); its displacement
       from the face's centroid is treated as a *tangent vector at the
       centroid* -- a direction plus a magnitude -- rather than as a
       literal flat-frame position to reach (the local frame's own flat
       distances only match true hyperbolic distances between the 3
       vertices themselves, which is what it's calibrated to; nothing
       guarantees that for an arbitrary interior/exterior point).
    2. That tangent vector is mapped into the Poincare disk's tangent space
       at the centroid's own (already-known) position via a similarity
       transform (rotation + uniform scale, no shear) -- justified by the
       Poincare disk being conformal, so its differential at any point is
       exactly a rotation+scale, with no directional dependence. Both the
       rotation and the scale factor are calibrated once, from an
       arbitrary reference face's centroid-to-vertex direction and true
       hyperbolic distance (`_compute_conformal_scale`) -- the same
       constants apply to every face and every direction, since the
       hyperbolic plane looks identical from every point (homogeneity) and
       every direction at a point (isotropy).
    3. The resulting Poincare-disk tangent vector is exponentiated exactly
       (`tanh(distance / 2)` gives the Euclidean radius, in a frame with
       the centroid moved to the origin, of the point at that hyperbolic
       distance in that direction) -- always landing at a valid point,
       however large the original displacement, unlike naively
       extrapolating an affine (Klein-coordinate) barycentric combination,
       which can just as easily overshoot the disk's boundary into
       "beyond infinity".
    4. `_walk_klein_line` (this engine's one shared face-walking primitive)
       finds which face the resulting point actually lands in.

    Mutates and returns `point`, not a copy.
    """
    bary = point.barycentric
    if _barycentric_in_bounds(bary):
      return point

    face = point.grid
    p_local = _local_frame(face)
    centroid_2d = (p_local[0] + p_local[1] + p_local[2]) / 3.0
    local_target = (
        bary[0] * p_local[0] + bary[1] * p_local[1] + bary[2] * p_local[2])
    local_vec = local_target - centroid_2d
    local_magnitude = float(np.hypot(*local_vec))

    centroid_klein = self._klein_of_barycentric(
        face, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    centroid_poincare = _poincare_of_klein(*centroid_klein)
    vertex_b_local_vec = p_local[0] - centroid_2d
    vertex_b_at_centroid_origin = _to_origin_poincare(
        _poincare_of_vertex(face.vertex_b), centroid_poincare)
    rotation_offset = (
        cmath.phase(vertex_b_at_centroid_origin)
        - math.atan2(vertex_b_local_vec[1], vertex_b_local_vec[0]))

    base_angle = math.atan2(local_vec[1], local_vec[0]) + rotation_offset
    hyperbolic_distance = self._conformal_scale * local_magnitude
    euclidean_radius = math.tanh(hyperbolic_distance / 2.0)

    # A fixed nudge avoids the ray passing *exactly* through a vertex in
    # the common case, but for a long enough ray (crossing many faces),
    # any single fixed nudge can still happen to realign with some *other*
    # vertex farther out -- so escalate to a few different nudges before
    # giving up, rather than fixing on just one. (A `RuntimeError` here can
    # also mean the target is simply too far: Klein-disk floating-point
    # precision degrades near the disk's boundary, an inherent limitation
    # of any disk model at large hyperbolic distances -- no amount of
    # nudging fixes that, but the two failure modes aren't easy to tell
    # apart from here, hence trying a few nudges regardless.)
    steps = None
    for nudge in (
        _DEGENERATE_RAY_NUDGE_RADIANS, -_DEGENERATE_RAY_NUDGE_RADIANS,
        1e3 * _DEGENERATE_RAY_NUDGE_RADIANS,
        -1e3 * _DEGENERATE_RAY_NUDGE_RADIANS,
    ):
      disk_angle = base_angle + nudge
      target_at_centroid_origin = euclidean_radius * cmath.exp(1j * disk_angle)
      target_poincare = _from_origin_poincare(
          target_at_centroid_origin, centroid_poincare)
      target_klein = _klein_of_poincare(target_poincare)
      try:
        steps = self._walk_klein_line(face, centroid_klein, target_klein)
        break
      except RuntimeError:
        continue
    if steps is None:
      raise RuntimeError(
          "HyperbolicTessellation.geodesically_canonicalize_point: could "
          "not locate the target point even after several nudges -- "
          "either the ray kept passing exactly through a mesh vertex, or "
          "the displacement is large enough to hit this disk model's "
          "floating-point precision limit near its boundary.")
    final_face = steps[-1][0]
    final_bary = _clamp_barycentric(
        self._barycentric_of_klein(final_face, target_klein))
    alt = final_face.altitude
    return point.update(
        grid=final_face, b=final_bary[0] * alt, s=final_bary[1] * alt)
