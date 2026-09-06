from __future__ import annotations

from mundimonium.coordinates.exceptions import EndOfMeshSurfaceException
from mundimonium.coordinates.isometric import IsometricDirection, IsometricPoint
from mundimonium.coordinates.tessellation import (
    Tessellation, TessellationFace, TessellationVertex
)

from collections.abc import Sequence
from numbers import Number
from typing import override
import cmath
import collections
import functools
import math
import numpy as np


_MAX_WALK_STEPS = 10_000
_DEFAULT_MAX_STABLE_HOPS = 3

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
# Fallback tolerance for `_find_existing_vertex_near`, in Poincare-disk
# Euclidean distance, used only when the exact combinatorial check
# (`_closure_neighbor`) can't yet determine whether a candidate position
# is an already-existing vertex (see `HyperbolicTessellation._complete_fan`'s
# docstring for why that's sometimes unavoidable). Both compared positions
# are now O(log depth) accurate (see `_seed_ward_frame`), so this can stay
# close to double-precision epsilon rather than growing with mesh depth.
_VERTEX_MATCH_EPSILON = 1e-9

# Side length, in Poincare-disk coordinates, of the square cells used to
# spatially index already-placed vertices during mesh growth (see
# `_find_existing_vertex_near`) -- must be comfortably larger than any
# expected floating-point drift between two derivations of "the same"
# vertex position, and comfortably smaller than the tessellation's own
# edge length (in Poincare-disk terms) so that genuinely distinct vertices
# essentially never collide into the same cell.
_VERTEX_CELL_SIZE = 0.01


def _edge_length_for_order(order: int) -> float:
  """Edge length of the regular {3, `order`} hyperbolic tiling's
  equilateral triangle.

  Every face has the same angle `alpha = 2*pi/order` at each of its three
  corners, so `order` of them tile exactly around each vertex.

  The hyperbolic law of cosines for angles,
  `cos(C) = -cos(A)*cos(B) + sin(A)*sin(B)*cosh(c)`, specialized to an
  equilateral triangle (`A = B = C = alpha`, `a = b = c = L`), solves to
  `cosh(L) = cos(alpha) / (1 - cos(alpha))`.

  Args:
    order: Number of faces meeting at each vertex of the {3, order} tiling.

  Returns:
    The hyperbolic length of one edge.
  """
  alpha = 2.0 * math.pi / order
  cos_alpha = math.cos(alpha)
  return math.acosh(cos_alpha / (1.0 - cos_alpha))


def _to_origin_poincare(z: complex, center: complex) -> complex:
  """The disk automorphism of the Poincare disk sending `center` to the
  origin, applied to `z`.

  Args:
    z: The point to transform.
    center: The point to send to the origin.

  Returns:
    The transformed point.
  """
  return (z - center) / (1.0 - center.conjugate() * z)


def _from_origin_poincare(w: complex, center: complex) -> complex:
  """Inverse of `_to_origin_poincare`.

  Args:
    w: The point to transform.
    center: The point `_to_origin_poincare` sent to the origin.

  Returns:
    The transformed point.
  """
  return (w + center) / (1.0 + center.conjugate() * w)


# A Poincare-disk hyperbolic isometry as a normalized 2x2 complex matrix
# [[a, b], [conj(b), conj(a)]], |a|^2 - |b|^2 = 1 (an element of SU(1,1)),
# acting on a point `z` as `(a*z + b) / (conj(b)*z + conj(a))`. Composing
# two such matrices is a single matrix multiplication, which is what makes
# the O(log depth) binary-lifting scheme below (`_build_ancestor_jumps`,
# `_seed_ward_frame`) cheap: `_mobius_compose(A, B)` means "A after B"
# (apply B's transform first, then A's), matching ordinary matrix product
# `A @ B`. The plain-`complex` scalar helpers above stay in use elsewhere
# (query-time/recentering code); this matrix form is only needed where
# transforms get composed many times over, i.e. mesh construction.
def _mobius_identity() -> np.ndarray:
  """The identity Mobius transform.

  Returns:
    The 2x2 identity matrix, in SU(1,1) form.
  """
  return np.array([[1.0 + 0j, 0j], [0j, 1.0 + 0j]], dtype=np.complex128)


def _mobius_rotation(theta: float) -> np.ndarray:
  """The matrix form of an origin-centered rotation by `theta`.

  Args:
    theta: The rotation angle, in radians.

  Returns:
    The corresponding Mobius transform matrix.
  """
  half = 0.5 * theta
  return np.array(
      [[cmath.exp(1j * half), 0j], [0j, cmath.exp(-1j * half)]],
      dtype=np.complex128)


def _mobius_from_origin(c: complex) -> np.ndarray:
  """The matrix form of `_from_origin_poincare(., c)`.

  Args:
    c: The point `_from_origin_poincare` sends the origin to.

  Returns:
    The corresponding Mobius transform matrix.
  """
  norm = 1.0 / math.sqrt(max(1e-300, 1.0 - abs(c) ** 2))
  return norm * np.array(
      [[1.0 + 0j, c], [c.conjugate(), 1.0 + 0j]], dtype=np.complex128)


def _mobius_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  """Composes two Mobius transforms as `a` after `b`.

  Computed as `a @ b`, in SU(1,1)'s `[[a, b], [conj(b), conj(a)]]` layout,
  renormalized -- reconstructed from just the resulting top row, so
  floating-point drift in the (redundant, conjugate-determined) bottom row
  can never accumulate.

  Args:
    a: The transform to apply second.
    b: The transform to apply first.

  Returns:
    The composed transform.
  """
  product = a @ b
  pa, pb = product[0, 0], product[0, 1]
  norm = 1.0 / math.sqrt(
      max(1e-300, (pa * pa.conjugate()).real - (pb * pb.conjugate()).real))
  return norm * np.array(
      [[pa, pb], [pb.conjugate(), pa.conjugate()]], dtype=np.complex128)


def _mobius_apply(a: np.ndarray, z: complex) -> complex:
  """Applies Mobius transform `a` to `z`.

  Args:
    a: The transform to apply.
    z: The point to transform.

  Returns:
    The transformed point.
  """
  return (a[0, 0] * z + a[0, 1]) / (a[1, 0] * z + a[1, 1])


def _minkowski_of_poincare(z: complex) -> tuple[float, float, float]:
  """Converts a Poincare-disk point to Minkowski (hyperboloid) `(X, Y, T)`
  coordinates.

  Args:
    z: A point in the Poincare disk.

  Returns:
    The corresponding `(X, Y, T)` Minkowski coordinates.
  """
  r2 = z.real * z.real + z.imag * z.imag
  denom = 1.0 - r2
  return (2.0 * z.real / denom, 2.0 * z.imag / denom, (1.0 + r2) / denom)


def _poincare_of_minkowski(p: tuple[float, float, float]) -> complex:
  """Converts Minkowski `(X, Y, T)` coordinates to a Poincare-disk point.

  Args:
    p: The `(X, Y, T)` Minkowski coordinates.

  Returns:
    The corresponding Poincare-disk point.
  """
  x, y, t = p
  denom = 1.0 + t
  return complex(x / denom, y / denom)


def _klein_of_minkowski(p: tuple[float, float, float]) -> tuple[float, float]:
  """Converts Minkowski `(X, Y, T)` coordinates to Klein-disk `(u, v)`.

  Args:
    p: The `(X, Y, T)` Minkowski coordinates.

  Returns:
    The corresponding Klein-disk `(u, v)` coordinates.
  """
  x, y, t = p
  return (x / t, y / t)


def _minkowski_of_klein(u: float, v: float) -> tuple[float, float, float]:
  """Converts Klein-disk `(u, v)` coordinates to Minkowski `(X, Y, T)`.

  Args:
    u: Klein-disk `u` coordinate.
    v: Klein-disk `v` coordinate.

  Returns:
    The corresponding `(X, Y, T)` Minkowski coordinates.
  """
  denom = math.sqrt(max(1e-15, 1.0 - u * u - v * v))
  return (u / denom, v / denom, 1.0 / denom)


def _poincare_of_klein(u: float, v: float) -> complex:
  """Converts Klein-disk `(u, v)` coordinates to a Poincare-disk point.

  Args:
    u: Klein-disk `u` coordinate.
    v: Klein-disk `v` coordinate.

  Returns:
    The corresponding Poincare-disk point.
  """
  denom = 1.0 + math.sqrt(max(0.0, 1.0 - u * u - v * v))
  return complex(u / denom, v / denom)


def _klein_of_poincare(z: complex) -> tuple[float, float]:
  """Converts a Poincare-disk point to Klein-disk `(u, v)` coordinates.

  Args:
    z: A point in the Poincare disk.

  Returns:
    The corresponding Klein-disk `(u, v)` coordinates.
  """
  denom = 1.0 + z.real * z.real + z.imag * z.imag
  return (2.0 * z.real / denom, 2.0 * z.imag / denom)


def _poincare_of_vertex(vertex: TessellationVertex) -> complex:
  """`vertex`'s Poincare-disk position, from its stored (seed-anchored)
  Minkowski `projection_coordinates`.

  Args:
    vertex: The vertex to convert.

  Returns:
    The vertex's Poincare-disk position.
  """
  x, y, z = vertex.projection_coordinates
  return _poincare_of_minkowski((x, y, -z))


def _global_klein_of_vertex(vertex: TessellationVertex) -> tuple[float, float]:
  """`vertex`'s Klein-disk position in the fixed, seed-anchored global
  frame.

  Used only for coarse, best-available distance judgments (recentering,
  the stable-region BFS radius) and to seed the reference frame itself --
  never for precision-critical face-walk math (see
  `HyperbolicTessellation._relative_klein_of_vertex`).

  Args:
    vertex: The vertex to convert.

  Returns:
    The vertex's global Klein-disk position.
  """
  return _klein_of_poincare(_poincare_of_vertex(vertex))


def _minkowski_inner(
    p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
  """The Minkowski (Lorentzian) inner product of two `(X, Y, T)` triples.

  Args:
    p: The first Minkowski triple.
    q: The second Minkowski triple.

  Returns:
    The Lorentzian inner product of `p` and `q`.
  """
  return p[0] * q[0] + p[1] * q[1] - p[2] * q[2]


def _hyperbolic_distance(
    p: tuple[float, float, float], q: tuple[float, float, float]) -> float:
  """Exact hyperbolic distance between two Minkowski-model points.

  Args:
    p: The first point, in Minkowski coordinates.
    q: The second point, in Minkowski coordinates.

  Returns:
    The hyperbolic distance between `p` and `q`.
  """
  return math.acosh(max(1.0, -_minkowski_inner(p, q)))


def _minkowski_midpoint(
    p: tuple[float, float, float], q: tuple[float, float, float],
) -> tuple[float, float, float]:
  """The hyperbolic midpoint of two Minkowski-model points.

  Computed as their raw sum, renormalized back onto the hyperboloid -- the
  hyperboloid-model analogue of normalizing the sum of two unit vectors to
  get a great-circle arc's midpoint on a sphere.

  Args:
    p: The first point, in Minkowski coordinates.
    q: The second point, in Minkowski coordinates.

  Returns:
    The midpoint, in Minkowski coordinates.
  """
  mx, my, mt = p[0] + q[0], p[1] + q[1], p[2] + q[2]
  scale = 1.0 / math.sqrt(max(1e-300, mt * mt - mx * mx - my * my))
  return (mx * scale, my * scale, mt * scale)


def _barycentric_in_bounds(bary: tuple[float, float, float]) -> bool:
  """Whether barycentric weights `bary` lie within `[0, 1]`, up to
  `_BARYCENTRIC_EPSILON`.

  Args:
    bary: The barycentric weights to check.

  Returns:
    True if `bary` represents a point inside its triangle.
  """
  return all(-_BARYCENTRIC_EPSILON <= w <= 1.0 + _BARYCENTRIC_EPSILON
             for w in bary)


def _clamp_barycentric(
    bary: tuple[float, float, float]) -> tuple[float, float, float]:
  """Clamps near-boundary floating-point noise back into `[0, 1]`.

  Args:
    bary: The barycentric weights to clamp.

  Returns:
    The clamped weights, still summing to 1.
  """
  wb = min(1.0, max(0.0, bary[0]))
  ws = min(1.0, max(0.0, bary[1]))
  return (wb, ws, 1.0 - wb - ws)


def _vertex_cell_key(z: complex) -> tuple[int, int]:
  """The spatial-hash cell containing Poincare-disk point `z` (see
  `_VERTEX_CELL_SIZE`).

  Args:
    z: A point in the Poincare disk.

  Returns:
    The `(x, y)` cell index containing `z`.
  """
  return (math.floor(z.real / _VERTEX_CELL_SIZE),
          math.floor(z.imag / _VERTEX_CELL_SIZE))


def _local_frame(face: TessellationFace) -> np.ndarray:
  """The canonical local flat 2D layout of `face`'s own (b, s, d) system.

  Vertices B, S, D sit at `(0, 0)`, `(a, 0)`, `(a/2, h)`, where `a` is
  `face.side_length` and `h` is `face.altitude`.

  Used only as a reference frame for interpreting an out-of-`[0, 1]`-weight
  point's *displacement direction and magnitude* from its face's centroid
  (see `geodesically_canonicalize_point`) -- not, as in `GenericTessellation`,
  as a chart to be patched face-to-face by unfolding. The flat frame's own
  notion of distance only matches true hyperbolic distance for the 3
  vertices themselves (calibrated exactly to the tiling's edge length), not
  for arbitrary interior/exterior points, since a hyperbolic triangle's
  interior is not isometric to a flat one.

  Args:
    face: The face to lay out.

  Returns:
    A `3x2` array of the `(B, S, D)` vertices' local `(x, y)` positions.
  """
  a = float(face.side_length)
  h = float(face.altitude)
  return np.array([[0.0, 0.0], [a, 0.0], [0.5 * a, h]], dtype=np.float64)


def _child_step_matrices(order: int, edge_length: float) -> list[np.ndarray]:
  """The `order` fixed Mobius transforms `S_k` used to place each vertex's
  `k`-th neighbor.

  For any vertex `V` whose own `frame` maps the origin to `V`'s position
  and maps `r_edge` (on the positive real axis) to `V`'s discovering
  parent's position, `V.frame @ S_k` is the corresponding frame for `V`'s
  `k`-th neighbor. This convention continues recursively: the child's own
  slot 0 -- real axis, distance `r_edge` -- points back to `V`.

  `S_k` is the unique disk automorphism with `S_k(0) = target_k` and
  `S_k(r_edge) = 0`, where `target_k = r_edge * exp(i*2*pi*k/order)`:
  `S_k = _mobius_from_origin(target_k) @ _mobius_rotation(theta_k + pi)`.
  (`_mobius_rotation(theta_k + pi)` sends `0` to `0` and `r_edge` to
  `-target_k`; `_mobius_from_origin(target_k)` then sends those to
  `target_k` and `0` respectively.)

  Args:
    order: Number of faces meeting at each vertex of the {3, order} tiling.
    edge_length: The tiling's edge length.

  Returns:
    The `order` step matrices, indexed by `k`.
  """
  r_edge = math.tanh(edge_length / 2.0)
  steps = []
  for k in range(order):
    theta_k = 2.0 * math.pi * k / order
    target_k = r_edge * cmath.exp(1j * theta_k)
    steps.append(_mobius_compose(
        _mobius_from_origin(target_k), _mobius_rotation(theta_k + math.pi)))
  return steps


def _build_ancestor_jumps(
    parent: HyperbolicTessellationVertex, step: np.ndarray,
) -> list[tuple[HyperbolicTessellationVertex, np.ndarray]]:
  """Binary-lifting ("2^j-ancestor") table for a new vertex one hop past
  `parent` via transform `step`.

  `jumps[j] = (ancestor 2^j hops back, transform from that ancestor's
  frame to this new vertex's frame)`, each built by combining two of
  `parent`'s own, already-built, half-as-deep entries -- the same
  incremental doubling used for LCA binary lifting, generalized to
  accumulate a composed transform alongside the jump itself.

  See `_seed_ward_frame` for what this buys.

  Args:
    parent: The new vertex's discovering parent.
    step: The transform from `parent`'s frame to the new vertex's frame.

  Returns:
    The new vertex's binary-lifting table.
  """
  jumps: list[tuple[HyperbolicTessellationVertex, np.ndarray]] = [
      (parent, step)]
  j = 0
  while True:
    ancestor, transform = jumps[j]
    if j >= len(ancestor.ancestor_jumps):
      break
    grand_ancestor, grand_transform = ancestor.ancestor_jumps[j]
    jumps.append((grand_ancestor, _mobius_compose(grand_transform, transform)))
    j += 1
  return jumps


def _seed_ward_frame(
    jumps: list[tuple[HyperbolicTessellationVertex, np.ndarray]],
    depth: int,
) -> np.ndarray:
  """Composes a newly-built `ancestor_jumps` table into the seed-to-vertex
  transform (this vertex's `frame`).

  Repeatedly takes the largest available jump toward the seed and
  accumulates its transform: O(log depth) steps (one per set bit of
  `depth`) instead of one per generation.

  This, not a single hop off `parent.frame`, is why deep vertices stay
  accurate: caching a vertex's frame as `parent.frame @ step` would still,
  by induction, chain one rounding per generation all the way back to the
  seed -- the same accumulation as naive one-hop-at-a-time construction,
  just reorganized into matrix form. Each `ancestor_jumps` entry is itself
  built from two half-as-deep entries (`_build_ancestor_jumps`), so
  composing O(log depth) of them here keeps the total rounding error
  logarithmic in depth instead of linear.

  Args:
    jumps: The new vertex's binary-lifting table (`_build_ancestor_jumps`).
    depth: The new vertex's discovery depth.

  Returns:
    The seed-to-vertex transform.
  """
  transform = _mobius_identity()
  remaining = depth
  while remaining > 0:
    j = remaining.bit_length() - 1
    ancestor, step = jumps[j]
    transform = _mobius_compose(step, transform)
    remaining -= (1 << j)
    jumps = ancestor.ancestor_jumps
  return transform


def _closure_neighbor(
    vertex: TessellationVertex, prev_neighbor: TessellationVertex,
    prev_face: TessellationFace,
) -> TessellationVertex | None:
  """Exact, floating-point-free lookup for the next vertex around
  `vertex`'s fan, continuing past `prev_neighbor`.

  `prev_face` (the face that placed `prev_neighbor`) shares edge
  `(vertex, prev_neighbor)` with exactly one other face. If that face has
  already been registered -- from any direction whatsoever, not
  necessarily via `prev_neighbor`'s own fan being complete -- its third
  vertex is `vertex`'s next neighbor.

  Can never be wrong when it fires: `TessellationFace.face_on_edge` only
  reports adjacency once `recalculate_adjacency_to` has verified two faces
  genuinely share 2 vertices -- this reads an already-established fact, it
  never infers one. It's also not *complete* -- see
  `HyperbolicTessellation._complete_fan`'s docstring -- so callers must
  still fall back to something else when this returns `None`.

  Args:
    vertex: The vertex whose fan is being walked.
    prev_neighbor: The neighbor of `vertex` just placed.
    prev_face: The face that placed `prev_neighbor`.

  Returns:
    `vertex`'s next neighbor past `prev_neighbor`, or `None` if the
    relevant face hasn't been built yet.
  """
  prev_prev = next(
      v for v in (prev_face.vertex_b, prev_face.vertex_s, prev_face.vertex_d)
      if v is not vertex and v is not prev_neighbor)
  candidate_face = prev_face.face_on_edge(
      prev_face.direction_toward_vertex(prev_prev))
  if candidate_face is None:
    return None
  return next(
      v for v in (candidate_face.vertex_b, candidate_face.vertex_s,
                  candidate_face.vertex_d)
      if v is not vertex and v is not prev_neighbor)


class HyperbolicTessellationVertex(TessellationVertex):
  """A `TessellationVertex` with discovery-tree bookkeeping.

  Lets `HyperbolicTessellation` construction derive a deep vertex's
  position from O(log depth) composed Mobius transforms instead of
  chaining one hop per generation (see `_build_ancestor_jumps`/
  `_seed_ward_frame` and `HyperbolicTessellation._new_child_vertex`).

  `projection_coordinates` keeps its usual meaning -- this vertex's fixed,
  seed-anchored Minkowski position -- just computed differently; see the
  tessellation class's own docstring for what this construction scheme
  does and doesn't fix.

  Attributes:
    discovery_depth: This vertex's distance, in fan-completion hops, from
      the seed vertex.
    discovery_parent: The neighbor whose fan first discovered this vertex,
      or `None` for the seed vertex itself.
    frame: The Mobius transform mapping the origin to this vertex's
      position, and `r_edge` (on the positive real axis) to
      `discovery_parent`'s position.
    ancestor_jumps: This vertex's binary-lifting table (see
      `_build_ancestor_jumps`).
  """

  def __init__(
      self, projection_coordinates: list[Number], *,
      discovery_depth: int = 0,
      discovery_parent: HyperbolicTessellationVertex | None = None,
      frame: np.ndarray | None = None):
    """Constructs a vertex with the given discovery-tree bookkeeping.

    Args:
      projection_coordinates: The vertex's `(X, Y, Z)` Minkowski
        projection coordinates (`Z = -T`).
      discovery_depth: This vertex's distance, in fan-completion hops,
        from the seed vertex.
      discovery_parent: The neighbor whose fan discovered this vertex, or
        `None` for the seed vertex itself.
      frame: The Mobius transform mapping the origin to this vertex's
        position, or the identity transform if omitted.
    """
    super().__init__(projection_coordinates)
    self.discovery_depth = discovery_depth
    self.discovery_parent = discovery_parent
    self.frame = frame if frame is not None else _mobius_identity()
    self.ancestor_jumps: list[
        tuple[HyperbolicTessellationVertex, np.ndarray]] = []


class HyperbolicTessellation(Tessellation):
  """A growable patch of the hyperbolic plane, tiled entirely by equilateral
  triangles: the regular {3, `order`} tiling (`order` triangles meeting at
  each vertex; `order >= 7` is required for the tiling to actually be
  hyperbolic).

  Unlike `GenericTessellation`, this class doesn't answer geodesic queries
  via a graph search over face adjacency -- a hyperbolic ball of radius r
  contains exponentially many faces in r (unlike a sphere, where it's
  bounded), so a search touching a meaningful fraction of the mesh becomes
  catastrophically expensive even for modest distances.

  Instead, every vertex's exact position is tracked in the Minkowski
  (hyperboloid) model, whose **Klein disk** projection maps every
  hyperbolic geodesic to a Euclidean straight line in one shared 2D frame
  (the exact hyperbolic analogue of gnomonic projection on the sphere,
  generalized from "straight locally, per face" to "straight throughout
  one frame"). A geodesic query is therefore a direct walk of that
  already-known straight line through the faces it crosses (O(1) work per
  face, via ordinary 2D line/edge intersection), extending the mesh on
  demand if the walk runs past its currently-built extent -- no search, no
  per-face unfolding.

  **Floating reference frame**: Klein/Poincare/Minkowski coordinates all
  compress unbounded hyperbolic distance into a bounded range, so precision
  degrades the farther a point is from wherever those coordinates are
  anchored. Rather than anchoring everything permanently at the mesh's
  original seed vertex, this class keeps a movable `_reference_point`
  (see `recenter`) and expresses every face-walk computation relative to
  *that*, re-centered on demand (see `_ensure_in_range`) whenever a query
  touches a point too far from the current reference -- keeping precision
  good for whatever region is currently "in view," independent of how far
  that region has drifted from the original seed over a long session (see
  `recenter`'s docstring for the numerical reasoning and its limits).

  Mesh vertices' *stored* positions (`TessellationVertex.projection_coordinates`)
  remain fixed, seed-anchored Minkowski coordinates regardless of the
  current reference frame -- only the mesh's own internal geometry
  computations (and the `coords_at_point`/`new_point_at_coords`/
  `get_face_at_coords` coordinate system, which is reference-frame-relative
  by design) are affected by recentering.

  **Construction** (as opposed to querying) doesn't use the reference
  frame above -- vertex positions are computed once, at creation, always
  relative to the seed.

  Two mechanisms keep that accurate far beyond what one-hop-at-a-time
  chaining could reach: an exact, floating-point-free rule
  (`_closure_neighbor`) recognizes most already-built adjacencies from
  bookkeeping alone, and genuinely new vertices get their position from an
  O(log depth) composition of Mobius transforms
  (`HyperbolicTessellationVertex`'s `frame`/`ancestor_jumps`) rather than a
  chain that degrades with depth (see `_complete_fan`'s docstring for the
  full mechanism, including the one case `_closure_neighbor` can't resolve
  on its own).

  This fixes error growing *with discovery-tree depth*; it does not fix
  Poincare-disk floating-point precision loss near the disk's boundary at
  very large absolute hyperbolic distance from the seed, since vertex
  positions stay permanently seed-anchored by design -- a real, separate
  ceiling, just one that construction depth is no longer the binding
  constraint on.

  Three coordinate systems describe the same points:
  - Minkowski/hyperboloid `(X, Y, T)`, `X^2+Y^2-T^2=-1, T>0`: the *stored*
    representation, as `TessellationVertex.projection_coordinates =
    (X, Y, -T)`, always seed-anchored regardless of the reference frame.
  - Klein disk `(u, v) = (X, Y) / T`: this tessellation's mesh/geodesic
    coordinate system (`coords_at_point`/`new_point_at_coords`/
    `get_face_at_coords`) -- the model where face-walking is a straight
    line. *Relative to the current reference frame* -- `(0, 0)` is
    wherever `_reference_point` currently is, not the original seed.
  - Poincare disk `(x, y) = (X, Y) / (1 + T)`: not stored anywhere; a cheap
    on-demand conversion (`to_poincare`, always seed-anchored), used
    internally as scratch math during mesh growth and recentering, both of
    which rely on conformal (angle-preserving) rotation about an arbitrary
    point having a clean closed form there.

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

  def __init__(
      self, *, order: int = 7, rings: int = 2,
      max_stable_hops: int = _DEFAULT_MAX_STABLE_HOPS, **kwargs):
    """Constructs an initial patch of the {3, `order`} hyperbolic tiling.

    Args:
      order: Number of equilateral-triangle faces meeting at each vertex.
        Must be >= 7 (the threshold below which {3, order} is spherical or
        Euclidean, not hyperbolic).
      rings: How many "rings" of faces to build outward from the seed
        vertex initially (ring 1 is just the seed's own fan of `order`
        faces; each further ring completes the fan of every vertex
        discovered by the previous ring). More can be added later via
        `extend`/`extend_to_include`, including implicitly, on demand, by
        any geodesic query that reaches the mesh's current edge.
      max_stable_hops: How many combinatorial hops out from the current
        reference frame (see `recenter`) are trusted as numerically
        reliable, rebuilt on every recenter. Larger values cover more of
        the mesh per reference frame (fewer recenters needed) at the cost
        of precision near the edge of that region; the default is a
        conservative guess intended to be safe across order 7-10, not a
        precisely derived value -- tune per `order` if needed.
      **kwargs: Forwarded up the method resolution order.
    """
    if order < 7:
      raise ValueError(
          f"'order' must be >= 7 for a hyperbolic {{3, order}} tiling "
          f"(got {order}).")
    if rings < 1:
      raise ValueError(f"'rings' must be >= 1 (got {rings}).")
    if max_stable_hops < 1:
      raise ValueError(f"'max_stable_hops' must be >= 1 (got {max_stable_hops}).")
    vertex_type = kwargs.get('vertex_type')
    if vertex_type is None:
      kwargs['vertex_type'] = HyperbolicTessellationVertex
    elif not issubclass(vertex_type, HyperbolicTessellationVertex):
      raise TypeError(
          "'vertex_type' must be 'HyperbolicTessellationVertex' or a "
          "subclass -- construction relies on its discovery-tree fields.")
    super().__init__(**kwargs)
    self._order: int = order
    self._edge_length: float = _edge_length_for_order(order)
    self._max_stable_hops: int = max_stable_hops
    self._child_steps: list[np.ndarray] = _child_step_matrices(
        order, self._edge_length)
    self._face_by_vertex_triple: dict[frozenset, TessellationFace] = {}
    self._vertex_cells: dict[tuple[int, int], list[TessellationVertex]] = {}
    self._frontier: dict[
        TessellationVertex, tuple[TessellationVertex, TessellationFace]] = {}
    self._seed_vertex: TessellationVertex | None = None
    self._reference_point: IsometricPoint | None = None
    self._reference_poincare: complex | None = None
    self._reference_global_minkowski: tuple[float, float, float] | None = None
    self._stable_faces: dict[TessellationFace, int] = {}
    self._stable_radius: float = 0.0
    seed_face = self._build_seed()
    if rings > 1:
      self.extend(rings=rings - 1)
    self._recenter_to(seed_face.centroid_local_coords)

  @property
  def order(self) -> int:
    """The tiling's `{3, order}` Schlafli parameter."""
    return self._order

  @property
  def reference_point(self) -> IsometricPoint:
    """The current reference frame's anchor point (see `recenter`)."""
    return self._reference_point

  @property
  def stable_faces(self) -> frozenset[TessellationFace]:
    """Every face in the current numerically-stable region (see
    `_rebuild_stable_region`) -- within `max_stable_hops` combinatorial
    hops of `reference_point`, and therefore both already built and safe
    to run precision-sensitive Klein/Poincare math against."""
    return frozenset(self._stable_faces)

  @property
  def stable_radius(self) -> float:
    """The hyperbolic distance from `reference_point` to the farthest
    face in `stable_faces` (see `_rebuild_stable_region`) -- a
    conservative bound on how far a query can stray from `reference_point`
    before it falls outside the numerically-stable region and triggers an
    automatic recenter (see `_ensure_in_range`)."""
    return self._stable_radius

  @staticmethod
  def to_poincare(vertex: TessellationVertex) -> tuple[float, float]:
    """`vertex`'s Poincare-disk `(x, y)` position, derived on demand from
    its stored (seed-anchored) Minkowski `projection_coordinates`.

    Args:
      vertex: The vertex to convert.

    Returns:
      The vertex's Poincare-disk `(x, y)` position.
    """
    z = _poincare_of_vertex(vertex)
    return (z.real, z.imag)

  # ---------------------------------------------------------------------
  # Mesh construction / growth
  # ---------------------------------------------------------------------

  def _new_root_vertex(self) -> HyperbolicTessellationVertex:
    """Creates the seed vertex: depth 0, identity frame, at the Poincare
    origin.

    Returns:
      The new seed vertex.
    """
    vertex = self.vertex_type(
        [0.0, 0.0, -1.0], discovery_depth=0, discovery_parent=None,
        frame=_mobius_identity())
    self._vertex_cells.setdefault(_vertex_cell_key(0j), []).append(vertex)
    return vertex

  def _new_child_vertex(
      self, parent: HyperbolicTessellationVertex, k: int,
  ) -> HyperbolicTessellationVertex:
    """Creates `parent`'s `k`-th neighbor.

    Extends `parent`'s binary-lifting table (`_build_ancestor_jumps`) and
    derives the new vertex's frame via the O(log depth) seed-ward walk
    (`_seed_ward_frame`) -- never by a single hop off `parent.frame`, which
    would silently reintroduce the same O(depth) accumulation this whole
    scheme exists to avoid (see `_seed_ward_frame`'s docstring). Then
    computes its Poincare/Minkowski position and spatially indexes it.

    Args:
      parent: The new vertex's discovering parent.
      k: The new vertex's position in `parent`'s fan.

    Returns:
      The new vertex.
    """
    depth = parent.discovery_depth + 1
    jumps = _build_ancestor_jumps(parent, self._child_steps[k])
    frame = _seed_ward_frame(jumps, depth)
    z = _mobius_apply(frame, 0j)
    x, y, t = _minkowski_of_poincare(z)
    vertex = self.vertex_type(
        [x, y, -t], discovery_depth=depth, discovery_parent=parent,
        frame=frame)
    vertex.ancestor_jumps = jumps
    self._vertex_cells.setdefault(_vertex_cell_key(z), []).append(vertex)
    return vertex

  def _find_existing_vertex_near(
      self, poincare_z: complex) -> TessellationVertex | None:
    """An already-created vertex within `_VERTEX_MATCH_EPSILON`
    Poincare-disk Euclidean distance of `poincare_z`, if any.

    Args:
      poincare_z: The Poincare-disk position to look up.

    Returns:
      The matching vertex, or `None` if none is within tolerance.
    """
    # Compared directly in Poincare-disk coordinates, not via
    # `_hyperbolic_distance` (`acosh`): `acosh`'s derivative blows up as
    # its argument approaches 1 (zero distance), amplifying ordinary
    # ~1e-16 floating-point coordinate noise between two accurate
    # derivations of "the same" vertex into an apparent distance orders of
    # magnitude larger -- exactly the near-zero regime this comparison
    # lives in. Plain Euclidean distance in the (bounded, Poincare-disk)
    # coordinates being compared has no such blowup, and is consistent
    # with the Euclidean spatial-hash cells (`_vertex_cell_key`) already
    # used to find candidates.
    cx, cy = _vertex_cell_key(poincare_z)
    for dx in (-1, 0, 1):
      for dy in (-1, 0, 1):
        for candidate in self._vertex_cells.get((cx + dx, cy + dy), ()):
          if abs(_poincare_of_vertex(candidate) - poincare_z) \
              < _VERTEX_MATCH_EPSILON:
            return candidate
    return None

  def _add_face_if_new(
      self,
      v_b: TessellationVertex, v_s: TessellationVertex,
      v_d: TessellationVertex,
  ) -> TessellationFace:
    """Registers a new face for `(v_b, v_s, v_d)`, or returns the existing
    one if this triangle was already built (from the "other side" of a
    shared edge).

    Args:
      v_b: The face's B vertex.
      v_s: The face's S vertex.
      v_d: The face's D vertex.

    Returns:
      The new or already-existing face.
    """
    key = frozenset((v_b, v_s, v_d))
    existing = self._face_by_vertex_triple.get(key)
    if existing is not None:
      return existing
    face = self.add_face([v_b, v_s, v_d], side_length=self._edge_length)
    self._face_by_vertex_triple[key] = face
    return face

  def _build_seed(self) -> TessellationFace:
    """Builds the seed vertex and its full fan of `order` faces.

    Returns the seed face, so `__init__` can establish the reference frame
    there *after* building the requested initial `rings` -- not here,
    since `recenter` itself now grows the mesh (see
    `_rebuild_stable_region`), and doing that before `rings` is fully
    built would make the two growth mechanisms interact confusingly (the
    stable region's own reach would silently override/interleave with the
    `rings` request).

    Returns:
      The seed vertex's first face.
    """
    order = self._order
    v0 = self._new_root_vertex()
    ring = [self._new_child_vertex(v0, k) for k in range(order)]
    for k in range(order):
      self._add_face_if_new(v0, ring[k], ring[(k + 1) % order])
    self._seed_vertex = v0
    # `discovering_face` is deliberately the face on ring[k]'s (k+1)-side,
    # not its (k-1)-side: `_closure_neighbor` walks toward whichever
    # vertex is on the *other* side of the (v0, ring[k]) edge from
    # `discovering_face`, and that direction has to agree with which way
    # `_child_step_matrices` actually rotates. The (k-1)-side face is just
    # as valid and already built, but sends the first closure check of
    # every ring vertex's own fan around in the wrong direction, colliding
    # with a vertex placed earlier in the same fan.
    self._frontier = {
        ring[k]: (v0, self._face_by_vertex_triple[
            frozenset((v0, ring[(k + 1) % order], ring[k]))])
        for k in range(order)
    }
    return self._face_by_vertex_triple[frozenset((v0, ring[0], ring[1]))]

  def _complete_fan(
      self,
      vertex: TessellationVertex, known_neighbor: TessellationVertex,
      discovering_face: TessellationFace,
      next_frontier: dict[
          TessellationVertex, tuple[TessellationVertex, TessellationFace]],
  ) -> None:
    """Discovers the rest of `vertex`'s `order` neighbors (it has exactly
    one known so far, `known_neighbor`, from `discovering_face`, an
    already-registered face containing both). Faces are built one at a
    time, interleaved with discovery, not in a second pass after all
    `order` positions are known -- required, not cosmetic: it's what lets
    `_closure_neighbor` see this fan's own newly-built faces as early as
    possible, including its final wrap-around edge.

    For each next vertex: try `_closure_neighbor` first (exact, no
    floating point) -- it can't be *wrong* when it fires, but it's
    provably not *complete* (see its docstring: two vertices discovered
    independently, as children of two different parents, can turn out to
    be mutually adjacent with neither side having built a face touching
    the other yet -- a "first contact" no bounded local lookup can
    detect). When it returns `None`, fall back to an epsilon-based spatial
    match (`_find_existing_vertex_near`) against a one-hop candidate
    position computed from `vertex.frame`, accurate to `vertex.frame`'s
    own O(log depth) precision plus one hop -- fine here since it's only
    ever used as a lookup key, never stored as any vertex's permanent
    frame -- creating a genuinely new vertex (`_new_child_vertex`, with
    its own proper O(log depth) frame) only if that also finds nothing.

    Newly-discovered vertices are added to `next_frontier` (mapped to
    `(vertex, the face that discovered them)`) for a later step to
    complete their own fans in turn; reused vertices, from either dedup
    path, are not, since they're already scheduled or already complete
    via their own discovery.

    One subtlety in *which* face gets recorded: `_closure_neighbor`'s
    first check for a vertex at fan position `j` must see the face on
    `j`'s *own* "next position" side (`j+1`, wrapping), not the "previous
    position" (`j-1`) side that's actually already built when position
    `j` is first placed -- the `j-1`-side face sends that first check the
    wrong way around the new fan, eventually colliding with an earlier
    position (the same issue `_build_seed`'s own frontier setup has to
    avoid, see its comment). The `j+1`-side face doesn't exist until one
    iteration later, when position `j+1` itself is placed -- so each
    position's `next_frontier` entry is recorded one iteration delayed,
    once the face that actually belongs to it exists.

    Args:
      vertex: The vertex whose fan to complete.
      known_neighbor: `vertex`'s one already-known neighbor.
      discovering_face: The already-registered face containing both
        `vertex` and `known_neighbor`.
      next_frontier: Dict to add newly-discovered vertices to, mapped to
        `(vertex, the face that discovered them)`.
    """
    order = self._order
    ring = [known_neighbor]
    prev_face = discovering_face
    pending_vertex = None  # awaiting its own "next position"-side face
    for k in range(1, order):
      next_vertex = _closure_neighbor(vertex, ring[-1], prev_face)
      if next_vertex is None:
        candidate_z = _mobius_apply(
            _mobius_compose(vertex.frame, self._child_steps[k]), 0j)
        next_vertex = self._find_existing_vertex_near(candidate_z)
        if next_vertex is None:
          next_vertex = self._new_child_vertex(vertex, k)
      face = self._add_face_if_new(vertex, ring[-1], next_vertex)
      if pending_vertex is not None:
        next_frontier[pending_vertex] = (vertex, face)
      pending_vertex = (
          next_vertex if next_vertex.discovery_parent is vertex else None)
      ring.append(next_vertex)
      prev_face = face
    final_face = self._add_face_if_new(vertex, ring[-1], ring[0])
    if pending_vertex is not None:
      next_frontier[pending_vertex] = (vertex, final_face)

  def _complete_fan_for(self, vertex: TessellationVertex) -> None:
    """Completes just this one frontier vertex's fan.

    A no-op if `vertex` isn't on the frontier (already complete).
    Otherwise removes it from `self._frontier` and adds any
    newly-discovered vertices to it.

    Args:
      vertex: The frontier vertex whose fan to complete.
    """
    known_neighbor, discovering_face = self._frontier.pop(vertex, (None, None))
    if known_neighbor is None:
      return
    self._complete_fan(vertex, known_neighbor, discovering_face, self._frontier)

  def extend(self, rings: int = 1) -> None:
    """Grows the mesh outward by `rings` more rings from its current
    frontier, in place. Never touches or recreates any existing vertex or
    face.

    A "ring" completes the fan of every vertex on the current frontier
    (each with exactly one known neighbor so far); any newly-discovered
    vertices become the next ring's frontier.

    Args:
      rings: How many rings to add.
    """
    for _ in range(rings):
      if not self._frontier:
        return
      for vertex in list(self._frontier):
        self._complete_fan_for(vertex)

  def extend_to_include(self, point: IsometricPoint) -> None:
    """Grows the mesh -- and recenters the reference frame first, if
    needed -- so that `point` is reachable, without building anything
    outside the path from the current reference frame to `point`.

    Args:
      point: The point to make reachable.
    """
    self._ensure_in_range(point)

  # ---------------------------------------------------------------------
  # Reference frame
  # ---------------------------------------------------------------------

  @functools.singledispatchmethod
  def recenter(self, anchor) -> None:
    """Moves the reference frame (see this class's docstring) to `anchor`.

    Accepts an `IsometricPoint` (recenters exactly there), a
    `TessellationFace` (recenters on its centroid), or a `tuple[float,
    float]` (current-frame-relative Klein `(u, v)` coordinates, resolved
    via `new_point_at_coords` -- so this still works correctly even though
    it's specified in terms of the frame being replaced).

    Rebuilds the "numerically stable region" (see `_ensure_in_range`)
    around the new reference; queries touching faces outside the new
    region will themselves trigger further automatic recentering as
    needed.

    Args:
      anchor: Where to recenter: an `IsometricPoint`, a `TessellationFace`,
        or a `tuple[float, float]`.

    Raises:
      TypeError: If `anchor` isn't one of the supported types.
      ValueError: If `anchor` isn't registered to this tessellation.
    """
    raise TypeError(f"Unsupported anchor type: {type(anchor).__name__}")

  @recenter.register
  def _(self, anchor: IsometricPoint) -> None:
    if anchor.grid.tessellation is not self:
      raise ValueError(
          "'anchor' is not on a face registered to this tessellation.")
    self._recenter_to(anchor)

  @recenter.register
  def _(self, anchor: TessellationFace) -> None:
    if anchor.tessellation is not self:
      raise ValueError("'anchor' is not registered to this tessellation.")
    self._recenter_to(anchor.centroid_local_coords)

  @recenter.register
  def _(self, anchor: tuple) -> None:
    self._recenter_to(self.new_point_at_coords(*anchor))

  def _recenter_to(self, point: IsometricPoint) -> None:
    """Moves the reference frame to `point` and rebuilds the stable
    region around it.

    Internal counterpart of `recenter` that skips registration checks
    (always called with an already-valid point).

    Args:
      point: The new reference point.
    """
    self._reference_point = point
    ref_klein_global = self._global_klein_of_point(point)
    self._reference_global_minkowski = _minkowski_of_klein(*ref_klein_global)
    self._reference_poincare = _poincare_of_klein(*ref_klein_global)
    self._rebuild_stable_region()

  def _rebuild_stable_region(self) -> None:
    """BFS from the current reference frame's face, out to
    `self._max_stable_hops`.

    Populates `self._stable_faces` (face -> hop count) and
    `self._stable_radius` (the farthest hyperbolic distance, via
    best-available global coordinates, of any face reached). Pure
    combinatorial graph traversal via `_ensure_face_on_edge` (growing the
    mesh, path-limited, as needed to actually reach `max_stable_hops`, not
    just checking whatever already happens to be built) -- no
    Klein/Poincare math beyond the one distance computation per face.
    """
    self._stable_faces = {}
    self._stable_radius = 0.0
    start_face = self._reference_point.grid
    queue = collections.deque([(start_face, 0)])
    visited = {start_face}
    while queue:
      face, hops = queue.popleft()
      self._stable_faces[face] = hops
      face_distance = _hyperbolic_distance(
          self._reference_global_minkowski,
          self._global_minkowski_of_point(face.centroid_local_coords))
      self._stable_radius = max(self._stable_radius, face_distance)
      if hops >= self._max_stable_hops:
        continue
      for direction in IsometricDirection:
        neighbor = self._ensure_face_on_edge(face, direction)
        if neighbor not in visited:
          visited.add(neighbor)
          queue.append((neighbor, hops + 1))

  def _ensure_in_range(self, *points: IsometricPoint) -> None:
    """Recenters the reference frame as needed so that every one of
    `points` is within the current stable region (see
    `_rebuild_stable_region`) afterward.

    Args:
      *points: The points that must end up in range.
    """
    for point in points:
      if point.grid not in self._stable_faces:
        self._auto_recenter_toward(point)

  def _auto_recenter_toward(self, point: IsometricPoint) -> None:
    """Moves the reference frame partway toward `point`.

    Recenters on the hyperbolic midpoint between the current reference and
    `point`, or raises if `point` is too far to bridge in one step.

    Uses only best-available *global* (seed-anchored) coordinates for the
    coarse "how far, and is it reachable at all" distance judgment --
    exactness isn't needed here, only for the geometry computed *after*
    the frame lands somewhere point is actually reachable from.

    Args:
      point: The point to recenter toward.

    Raises:
      ValueError: If `point` is at least twice the current stable radius
        away, too far to bridge in one recenter.
    """
    target_global = self._global_minkowski_of_point(point)
    distance = _hyperbolic_distance(
        self._reference_global_minkowski, target_global)
    if distance >= 2.0 * self._stable_radius:
      raise ValueError(
          "HyperbolicTessellation: target point is too far from the "
          "current reference frame to recenter toward directly (at least "
          "twice the numerically stable radius away) -- move in smaller "
          "steps, recentering partway there first.")
    midpoint_global = _minkowski_midpoint(
        self._reference_global_minkowski, target_global)
    midpoint_point = self._locate_in_current_frame(midpoint_global)
    self._recenter_to(midpoint_point)

  def _locate_in_current_frame(
      self, target_global: tuple[float, float, float]) -> IsometricPoint:
    """The `IsometricPoint` at Minkowski position `target_global`, found
    by walking from the *current* reference frame (before it's replaced).

    Used only while computing a new reference point, extending the mesh
    as needed along the way.

    Args:
      target_global: The target position, in (seed-anchored) global
        Minkowski coordinates.

    Returns:
      The point at `target_global`.
    """
    target_klein = _klein_of_poincare(
        _to_origin_poincare(
            _poincare_of_minkowski(target_global), self._reference_poincare))
    face, bary = self._walk_from_reference(target_klein)
    return IsometricPoint.from_barycentric(face, *bary)

  # ---------------------------------------------------------------------
  # Klein-coordinate geometry (barycentric conversion, face-walking)
  # ---------------------------------------------------------------------

  def _relative_klein_of_vertex(
      self, vertex: TessellationVertex) -> tuple[float, float]:
    """`vertex`'s Klein-disk position relative to the *current reference
    frame*.

    This is the coordinate system every precision-critical geometry
    computation in this class uses, as opposed to `_global_klein_of_vertex`
    (the fixed seed-anchored frame, used only for coarse recentering
    decisions).

    Args:
      vertex: The vertex to convert.

    Returns:
      The vertex's reference-frame-relative Klein-disk position.
    """
    return _klein_of_poincare(
        _to_origin_poincare(_poincare_of_vertex(vertex), self._reference_poincare))

  def _relative_klein_and_weight_of_vertex(
      self, vertex: TessellationVertex,
  ) -> tuple[tuple[float, float], float]:
    """`vertex`'s relative Klein position, paired with a weight for
    combining it with other vertices' positions.

    The weight is the correction factor `T_relative / T_global` needed to
    combine several vertices' relative positions into a barycentric
    combination correctly (see `_barycentric_of_klein`'s docstring for why
    this is needed at all). `T_global` is read directly off
    `vertex.projection_coordinates` (cheap, exact, and -- unlike a
    *subtraction* of two large/near-boundary values -- safe to use as a
    plain multiplicative weight even when the vertex is far from the
    seed).

    Args:
      vertex: The vertex to convert.

    Returns:
      A `(relative_klein_position, weight)` pair.
    """
    k_rel = self._relative_klein_of_vertex(vertex)
    t_rel = 1.0 / math.sqrt(max(1e-15, 1.0 - k_rel[0] ** 2 - k_rel[1] ** 2))
    t_global = -vertex.projection_coordinates[2]
    return k_rel, t_rel / t_global

  def _global_klein_of_point(
      self, point: IsometricPoint) -> tuple[float, float]:
    """`point`'s Klein-disk position in the fixed, seed-anchored global
    frame.

    Used only for coarse distance judgments, never face-walk math (see
    `_relative_klein_of_vertex`).

    Args:
      point: The point to convert.

    Returns:
      The point's global Klein-disk position.
    """
    wb, ws, wd = point.barycentric
    face = point.grid
    kb = _global_klein_of_vertex(face.vertex_b)
    ks = _global_klein_of_vertex(face.vertex_s)
    kd = _global_klein_of_vertex(face.vertex_d)
    return (
        wb * kb[0] + ws * ks[0] + wd * kd[0],
        wb * kb[1] + ws * ks[1] + wd * kd[1],
    )

  def _global_minkowski_of_point(
      self, point: IsometricPoint) -> tuple[float, float, float]:
    """`point`'s Minkowski position in the fixed, seed-anchored global
    frame.

    Args:
      point: The point to convert.

    Returns:
      The point's global Minkowski position.
    """
    return _minkowski_of_klein(*self._global_klein_of_point(point))

  def _barycentric_of_klein(
      self, face: TessellationFace, klein_point: tuple[float, float],
  ) -> tuple[float, float, float]:
    """The barycentric `(wb, ws, wd)` weights of `klein_point` (a
    reference-frame-relative Klein-disk position) relative to `face`'s 3
    vertices.

    A hyperbolic isometry, viewed in Klein coordinates, is a *projective*
    transform (Klein is the Beltrami/projectivized hyperboloid model), and
    -- unlike an affine one -- a projective transform does not commute
    with plain affine barycentric combination: naively treating a face's
    vertices' individually-correct relative Klein positions as an ordinary
    flat triangle gives a measurably wrong interior point. The fix is the
    standard one for projective transforms of weighted/rational
    combinations (as in rational Bezier curves): weight each vertex's
    relative position by `T_relative / T_global` (see
    `_relative_klein_and_weight_of_vertex`) before combining -- this
    exactly restores the correct result, and reduces to plain unweighted
    barycentric combination when the reference frame *is* the seed (every
    weight is then exactly 1).

    Ordinary (unweighted) barycentric extraction from `klein_point`
    relative to the vertices' own *relative* Klein positions gives
    `(wb', ws', wd')` satisfying `wi' = wi * weight_i / sum(wj * weight_j)`
    -- so recovering the true `(wb, ws, wd)` just needs dividing back out
    by each vertex's weight and renormalizing.

    Args:
      face: The face `klein_point` lies on.
      klein_point: A reference-frame-relative Klein-disk position.

    Returns:
      The corresponding `(wb, ws, wd)` barycentric weights.
    """
    (kb, cb), (ks, cs), (kd, cd) = (
        self._relative_klein_and_weight_of_vertex(face.vertex_b),
        self._relative_klein_and_weight_of_vertex(face.vertex_s),
        self._relative_klein_and_weight_of_vertex(face.vertex_d))
    x, y = klein_point
    denom = (
        (ks[1] - kd[1]) * (kb[0] - kd[0]) + (kd[0] - ks[0]) * (kb[1] - kd[1]))
    wb_prime = (
        (ks[1] - kd[1]) * (x - kd[0]) + (kd[0] - ks[0]) * (y - kd[1])
    ) / denom
    ws_prime = (
        (kd[1] - kb[1]) * (x - kd[0]) + (kb[0] - kd[0]) * (y - kd[1])
    ) / denom
    wd_prime = 1.0 - wb_prime - ws_prime
    raw_b, raw_s, raw_d = wb_prime / cb, ws_prime / cs, wd_prime / cd
    total = raw_b + raw_s + raw_d
    return (raw_b / total, raw_s / total, raw_d / total)

  def _klein_of_barycentric(
      self, face: TessellationFace, bary: tuple[float, float, float],
  ) -> tuple[float, float]:
    """Inverse of `_barycentric_of_klein`.

    The reference-frame-relative Klein-disk position of a point with
    barycentric weights `bary` on `face`. Well-defined (as a straight-line
    extrapolation) even for weights outside `[0, 1]`, as long as they sum
    to 1. See `_barycentric_of_klein`'s docstring for why this weights
    each vertex's contribution rather than combining plainly.

    Args:
      face: The face `bary` is relative to.
      bary: The `(wb, ws, wd)` barycentric weights, summing to 1.

    Returns:
      The corresponding reference-frame-relative Klein-disk position.
    """
    wb, ws, wd = bary
    (kb, cb), (ks, cs), (kd, cd) = (
        self._relative_klein_and_weight_of_vertex(face.vertex_b),
        self._relative_klein_and_weight_of_vertex(face.vertex_s),
        self._relative_klein_and_weight_of_vertex(face.vertex_d))
    weight_b, weight_s, weight_d = wb * cb, ws * cs, wd * cd
    total = weight_b + weight_s + weight_d
    return (
        (weight_b * kb[0] + weight_s * ks[0] + weight_d * kd[0]) / total,
        (weight_b * kb[1] + weight_s * ks[1] + weight_d * kd[1]) / total,
    )

  def _find_exit(
      self,
      face: TessellationFace, pos: tuple[float, float],
      target: tuple[float, float],
  ) -> tuple[IsometricDirection, tuple[float, float]]:
    """The edge of `face` (and the exact position on it) where the straight
    segment from `pos` toward `target` first exits `face`.

    Same style of 2D line/edge-intersection math as
    `GenericTessellation.shortest_path_by_segment`'s portal crossings and
    `geodesically_canonicalize_point`'s `min_t`/`hit_edge` loop -- except
    here every face's vertices are already in one shared frame, so no
    per-face unfolding is needed first.

    Args:
      face: The face `pos` lies inside.
      pos: The segment's start, in reference-frame-relative Klein
        coordinates.
      target: The segment's end, in reference-frame-relative Klein
        coordinates.

    Returns:
      A `(direction, position)` pair: the edge direction crossed, and the
      exact Klein-coordinate position of the crossing.

    Raises:
      RuntimeError: If no edge crossing is found, indicating corrupt mesh
        adjacency.
    """
    direction = (target[0] - pos[0], target[1] - pos[1])
    vertex_klein = {
        IsometricDirection.B: self._relative_klein_of_vertex(face.vertex_b),
        IsometricDirection.S: self._relative_klein_of_vertex(face.vertex_s),
        IsometricDirection.D: self._relative_klein_of_vertex(face.vertex_d),
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

  def _ensure_face_on_edge(
      self, face: TessellationFace, direction: IsometricDirection,
  ) -> TessellationFace:
    """`face.face_on_edge(direction)`, building just the one or two
    frontier vertices needed to complete it if it doesn't exist yet.

    Grows only the specific faces needed (rather than the whole current
    frontier), so any walk only ever builds faces actually on its path.

    Args:
      face: The face whose neighbor to find.
      direction: Which edge of `face` to cross.

    Returns:
      The neighboring face across `direction`.

    Raises:
      RuntimeError: If neither of the crossed edge's two vertices needed
        its fan completed, indicating corrupt mesh adjacency.
      EndOfMeshSurfaceException: If the target lies beyond the mesh's
        reachable extent even after extending.
    """
    next_face = face.face_on_edge(direction)
    if next_face is not None:
      return next_face
    v_a = face.vertex_at(IsometricDirection((direction.value + 1) % 3))
    v_b = face.vertex_at(IsometricDirection((direction.value + 2) % 3))
    completed_any = False
    for vertex in (v_a, v_b):
      if next_face is not None:
        break
      if vertex in self._frontier:
        self._complete_fan_for(vertex)
        completed_any = True
        next_face = face.face_on_edge(direction)
    if next_face is None:
      if not completed_any:
        raise RuntimeError(
            "HyperbolicTessellation: expected at least one of the "
            "crossed edge's two vertices to still need its fan "
            "completed, but neither did -- the mesh's adjacency may be "
            "corrupt.")
      raise EndOfMeshSurfaceException(
          "HyperbolicTessellation: target point lies beyond the mesh's "
          "reachable extent even after extending.")
    # This new face is exactly one hop past `face`, so it's part of the
    # currently-trusted region too if `face` was -- keeps the stable-region
    # cache accurate as the mesh grows, without needing a full rebuild.
    if face in self._stable_faces:
      hops = self._stable_faces[face] + 1
      if hops <= self._max_stable_hops:
        self._stable_faces[next_face] = min(
            self._stable_faces.get(next_face, hops), hops)
    return next_face

  def _walk_klein_line(
      self,
      start_face: TessellationFace, start_klein: tuple[float, float],
      target_klein: tuple[float, float],
  ) -> list[tuple[TessellationFace, tuple[float, float], tuple | None]]:
    """Walks the straight (reference-frame-relative Klein-coordinate) line
    from `start_klein` to `target_klein`, one face at a time, extending
    the mesh on demand (see `_ensure_face_on_edge`) whenever the walk
    reaches its current edge.

    Args:
      start_face: The face `start_klein` lies in.
      start_klein: The walk's start, in reference-frame-relative Klein
        coordinates.
      target_klein: The walk's end, in reference-frame-relative Klein
        coordinates.

    Returns:
      One `(face, entry_klein, exit_klein)` tuple per face crossed; the
      last tuple has `exit_klein = None`, meaning `target_klein` itself
      lies within that face.

    Raises:
      RuntimeError: If the walk exceeds `_MAX_WALK_STEPS`, indicating
        either a mesh topology bug or an extremely distant target.
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
      curr_face = self._ensure_face_on_edge(curr_face, exit_dir)
      pos = exit_pos
    raise RuntimeError(
        f"_walk_klein_line exceeded {_MAX_WALK_STEPS} steps; the walk may "
        "be stuck (a mesh topology bug) or the target is extremely far "
        "away.")

  def _walk_from_reference(
      self, target_klein: tuple[float, float],
  ) -> tuple[TessellationFace, tuple[float, float, float]]:
    """The `(face, barycentric)` at reference-frame-relative Klein
    coordinates `target_klein`, walking from the current reference point
    (whose own relative Klein position is always `(0, 0)`, by
    construction).

    Args:
      target_klein: The target, in reference-frame-relative Klein
        coordinates.

    Returns:
      The `(face, barycentric)` at `target_klein`.
    """
    steps = self._walk_klein_line(
        self._reference_point.grid, (0.0, 0.0), target_klein)
    face = steps[-1][0]
    bary = _clamp_barycentric(self._barycentric_of_klein(face, target_klein))
    return face, bary

  # ---------------------------------------------------------------------
  # Tessellation interface
  # ---------------------------------------------------------------------

  @override
  def coords_at_point(self, point: IsometricPoint) -> tuple[Number, ...]:
    """The current-reference-frame-relative Klein-disk `(u, v)` position
    of `point`, recentering the reference frame first if `point` is
    outside the current stable region.

    Args:
      point: The point to convert.

    Returns:
      `point`'s current-reference-frame-relative Klein-disk `(u, v)`
      position.
    """
    self._ensure_in_range(point)
    return self._klein_of_barycentric(point.grid, point.barycentric)

  @override
  def new_point_at_coords(self, *coords: tuple[Number, ...]) -> IsometricPoint:
    """The `IsometricPoint` at current-reference-frame-relative Klein-disk
    coordinates `(u, v)`, extending the mesh as needed to reach it.

    Args:
      *coords: The target `(u, v)` Klein-disk coordinates.

    Returns:
      The point at `coords`.
    """
    face, bary = self._walk_from_reference((coords[0], coords[1]))
    return IsometricPoint.from_barycentric(face, *bary)

  @override
  def get_face_at_coords(
      self, *coords: tuple[Number, ...]) -> TessellationFace:
    """The face containing current-reference-frame-relative Klein-disk
    coordinates `(u, v)`, extending the mesh as needed to reach it.

    Args:
      *coords: The target `(u, v)` Klein-disk coordinates.

    Returns:
      The face containing `coords`.
    """
    face, _ = self._walk_from_reference((coords[0], coords[1]))
    return face

  @override
  def geodesic_distance(self, p1: IsometricPoint, p2: IsometricPoint) -> float:
    """The exact hyperbolic distance between `p1` and `p2`, in O(1).

    Always computed via the direct Minkowski-model formula, regardless of
    how far apart the two points are (no search, no face-walking needed
    for the distance alone), after ensuring both are within the current
    (possibly newly-recentered) reference frame's stable region.

    Args:
      p1: The first point.
      p2: The second point.

    Returns:
      The hyperbolic distance between `p1` and `p2`.
    """
    self._ensure_in_range(p1, p2)
    p1_minkowski = _minkowski_of_klein(
        *self._klein_of_barycentric(p1.grid, p1.barycentric))
    p2_minkowski = _minkowski_of_klein(
        *self._klein_of_barycentric(p2.grid, p2.barycentric))
    return _hyperbolic_distance(p1_minkowski, p2_minkowski)

  @override
  def flatten_region(
      self, center: IsometricPoint, targets: Sequence[IsometricPoint],
  ) -> list[tuple[float, float]]:
    """Maps `targets` into a locally flat 2D coordinate system at `center`
    -- the hyperbolic exponential map, in closed form.

    Exact, unlike `geodesically_canonicalize_point`'s centroid-only
    version of this same idea: that one needs `_compute_conformal_scale`'s
    calibration because it starts from a *flat local frame* displacement,
    only an approximation of a face's true hyperbolic shape. This method
    starts and ends with exact Klein/Poincare coordinates on both `center`
    and each target, so no such approximation -- or its calibration -- is
    needed.

    The Mobius transform sending `center` to the Poincare disk's origin
    (`_to_origin_poincare`) is an isometry, so `target`'s position in that
    recentered frame gives its exact hyperbolic distance and direction
    from `center` directly: distance via the Poincare exponential map's
    inverse (`2 * arctanh(radius)`), direction via its angle.

    The *distance* half of that (`hypot(x, y)`) is exact and reference-
    frame-independent, matching the base class's own contract. The
    *angle* is not independently absolute, though: both `center` and each
    `target` are read via `_klein_of_barycentric`'s reference-frame-
    relative coordinates before the Mobius transform is applied, and
    composing that transform with an unrelated prior reference-to-
    reference hop is not a pure translation -- it carries a rotation too.
    So the angle is only meaningful relative to whichever reference frame
    is active *at the time of this call*; it will differ between two
    calls with the same `center`/`target` if something recentered the
    tessellation in between (see `unflatten_point`, which depends on this
    directly).

    Args:
      center: The point the flattened region is centered on.
      targets: The points to flatten, in any order, anywhere on the mesh.

    Returns:
      One `(x, y)` pair per point in `targets`, in the same order.
    """
    if not targets:
      return []
    self._ensure_in_range(center, *targets)

    center_klein = self._klein_of_barycentric(
        center.grid, center.barycentric)
    center_poincare = _poincare_of_klein(*center_klein)

    target_poincare = np.array(
        [
            _poincare_of_klein(*self._klein_of_barycentric(
                target.grid, target.barycentric))
            for target in targets
        ],
        dtype=np.complex128)

    recentered = _to_origin_poincare(target_poincare, center_poincare)
    radius = np.clip(np.abs(recentered), 0.0, 1.0 - 1e-12)
    distance = 2.0 * np.arctanh(radius)
    angle = np.angle(recentered)

    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    return list(zip(x.tolist(), y.tolist()))

  def unflatten_point(
      self, center: IsometricPoint, x: Number, y: Number,
  ) -> IsometricPoint:
    """The mesh point `flatten_region(center, [that point])` would map to
    `(x, y)` -- the exact inverse of `flatten_region`, for real-time
    re-centering while panning a flat-mode view (see `server.py`'s own
    `get_flat_mesh` handler).

    Requires `center` to be the tessellation's *current* reference point
    (see `recenter`) -- enforced here by recentering to it unconditionally
    before doing anything else. This isn't just a precision nice-to-have:
    `flatten_region`'s own `(x, y)` output is expressed relative to
    whichever reference frame happens to be active *at the time of that
    call* (its Mobius recentering is exact, but composing it with an
    unrelated prior reference-to-reference hop is not a pure translation
    -- it also carries a rotation, empirically confirmed by calling
    `flatten_region` for the same `center`/target both with and without an
    intervening `recenter` to an unrelated point and comparing the two
    results: distances matched exactly, angles did not). So `(x, y)` is
    only interpretable correctly by whichever call -- forward or backward
    -- shares `flatten_region`'s notion of `center` being the reference at
    the time.

    No transported-orientation state needs to be carried forward across
    calls the way `GenericTessellation.unflatten_point_and_transport_
    orientation` does, though: as long as every caller recenters to its
    own `center` immediately before flattening around it (which
    `server.py`'s own request handling always does), each request
    re-establishes the correct convention from scratch.

    Args:
      center: The point flat-mode panning was last centered on. Must
        already be (or become, via the unconditional `recenter` below)
        the tessellation's current reference point.
      x: The panned-to view center's flattened x coordinate, relative to
        `center`.
      y: Same, for y.

    Returns:
      The mesh point at `(x, y)`.
    """
    self.recenter(center)
    distance = math.hypot(x, y)
    if distance < _BARYCENTRIC_EPSILON:
      return center

    angle = math.atan2(y, x)
    poincare_radius = math.tanh(distance / 2.0)
    recentered = cmath.rect(poincare_radius, angle)

    center_klein = self._klein_of_barycentric(center.grid, center.barycentric)
    center_poincare = _poincare_of_klein(*center_klein)
    target_poincare = _from_origin_poincare(recentered, center_poincare)

    u, v = _klein_of_poincare(target_poincare)
    return self.new_point_at_coords(u, v)

  @override
  def point_to_3d_position(self, point: IsometricPoint) -> np.ndarray:
    """Not supported: `TessellationVertex.projection_coordinates` here are
    Minkowski `(X, Y, Z)` coordinates, not a Euclidean 3D embedding (see
    `mesh_export.tessellation_to_buffers`'s docstring for the same
    caveat) -- this tessellation isn't wired into the rendering pipeline
    at all yet.
    """
    raise NotImplementedError()

  @override
  def shortest_path_by_segment(
      self, p1: IsometricPoint, p2: IsometricPoint,
  ) -> list[tuple[IsometricPoint, IsometricPoint]]:
    """Traces the exact geodesic from `p1` to `p2`, split into per-face
    segments, by walking the straight Klein-coordinate line between them
    (see `_walk_klein_line`) -- recentering the reference frame and/or
    extending the mesh as needed.

    Args:
      p1: The path's start.
      p2: The path's end.

    Returns:
      One `(entry, exit)` point pair per face crossed.
    """
    if p1.grid is p2.grid:
      return [(p1, p2)]

    self._ensure_in_range(p1, p2)
    target_klein = self._klein_of_barycentric(p2.grid, p2.barycentric)
    steps = self._walk_klein_line(
        p1.grid, self._klein_of_barycentric(p1.grid, p1.barycentric),
        target_klein)

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

    Args:
      p1: The path's start.
      p2: The path's end.

    Returns:
      The polyline from `p1` to `p2`, one point per face boundary crossed
      plus the two endpoints.
    """
    segments = self.shortest_path_by_segment(p1, p2)
    points = [segments[0][0]]
    points.extend(b for _, b in segments)
    return points

  @override
  def geodesically_canonicalize_point(
      self, point: IsometricPoint) -> IsometricPoint:
    """Moves `point` to a new grid if located outside its current grid's
    bounds, recentering the reference frame and/or extending the mesh as
    needed.

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

    Args:
      point: The point to canonicalize.

    Returns:
      `point`, mutated in place.

    Raises:
      RuntimeError: If the target still can't be located after several
        nudges (see the nudging loop below) -- either the ray kept passing
        exactly through a mesh vertex, or the displacement is large enough
        to hit this disk model's floating-point precision limit near its
        boundary.
    """
    bary = point.barycentric
    if _barycentric_in_bounds(bary):
      return point
    self._ensure_in_range(point)

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
        _poincare_of_klein(*self._relative_klein_of_vertex(face.vertex_b)),
        centroid_poincare)
    rotation_offset = (
        cmath.phase(vertex_b_at_centroid_origin)
        - math.atan2(vertex_b_local_vec[1], vertex_b_local_vec[0]))

    base_angle = math.atan2(local_vec[1], local_vec[0]) + rotation_offset
    hyperbolic_distance = self._compute_conformal_scale(face) * local_magnitude
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

  def _compute_conformal_scale(self, face: TessellationFace) -> float:
    """The ratio between true hyperbolic distance and local flat-frame
    distance, from `face`'s centroid to its own vertices.

    The same constant for every face and every direction (see
    `geodesically_canonicalize_point`'s docstring for why). Recomputed per
    call since it depends on the current reference frame (through
    `_relative_klein_of_vertex`) -- cheap, just a few conversions.

    Args:
      face: The face to compute the scale for.

    Returns:
      The ratio of true hyperbolic distance to local flat-frame distance.
    """
    p_local = _local_frame(face)
    centroid_2d = (p_local[0] + p_local[1] + p_local[2]) / 3.0
    local_dist = float(np.hypot(*(p_local[0] - centroid_2d)))
    centroid_klein = self._klein_of_barycentric(
        face, (1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0))
    true_dist = _hyperbolic_distance(
        _minkowski_of_klein(*centroid_klein),
        _minkowski_of_klein(*self._relative_klein_of_vertex(face.vertex_b)))
    return true_dist / local_dist
