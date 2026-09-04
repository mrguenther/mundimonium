import math
import random

import pytest

from mundimonium.coordinates.hyperbolic_tessellation import (
    HyperbolicTessellation, _closure_neighbor, _hyperbolic_distance,
    _minkowski_of_klein, _minkowski_of_poincare,
)
from mundimonium.coordinates.isometric import IsometricDirection, IsometricPoint
from mundimonium.coordinates.tessellation import TessellationFace


@pytest.fixture(scope="module")
def tess():
  # A generous `max_stable_hops` keeps essentially all of the mesh within
  # one stable region, so tests can pick any two random faces without
  # incidentally exercising auto-recentering (that gets its own dedicated
  # tests below) or, worse, the "too far to bridge" exception for a
  # pair that happens to land near-opposite sides of the built extent.
  return HyperbolicTessellation(order=7, rings=1, max_stable_hops=8)


@pytest.fixture(scope="module")
def tess8():
  return HyperbolicTessellation(order=8, rings=1, max_stable_hops=4)


def _random_point(face, rng):
  """A uniformly random point inside `face`.

  Args:
    face: The face to place the point on.
    rng: The random number generator to use.

  Returns:
    A random point on `face`.
  """
  weights = [rng.random() for _ in range(3)]
  total = sum(weights)
  wb, ws, _ = [w / total for w in weights]
  return IsometricPoint(face, wb * face.altitude, ws * face.altitude)


def _random_stable_face(tess, rng) -> TessellationFace:
  """A random face guaranteed to be within `tess`'s current stable
  region -- see the `tess`/`tess8` fixtures' comment.

  Args:
    tess: The tessellation to pick a face from.
    rng: The random number generator to use.

  Returns:
    A random face from `tess`'s current stable region.
  """
  stable = list(tess._stable_faces)
  return stable[rng.randrange(len(stable))]


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_order_below_seven_is_rejected():
  with pytest.raises(ValueError):
    HyperbolicTessellation(order=6)


def test_every_face_has_the_tilings_edge_length(tess):
  for face in tess.faces:
    assert face.side_length == pytest.approx(tess._edge_length)


def test_interior_vertices_have_valence_equal_to_order(tess8):
  # A vertex is fully "interior" (its fan of `order` faces is complete)
  # once every face around it has been built -- true of every vertex
  # discovered at least one ring before the mesh's current edge.
  interior = [v for v in tess8.vertices if len(v.adjacent_faces()) == 8]
  assert len(interior) > 0
  boundary_valences = {
      len(v.adjacent_faces()) for v in tess8.vertices
      if len(v.adjacent_faces()) != 8
  }
  # Partially-completed frontier vertices only ever have 2 or 3 faces
  # (from the one or two already-built faces touching the edge that
  # discovered them) before their own fan is completed.
  assert boundary_valences <= {2, 3}


def test_every_vertex_lies_on_the_hyperboloid(tess):
  for v in tess.vertices:
    x, y, z = v.projection_coordinates
    assert x * x + y * y - z * z == pytest.approx(-1.0, abs=1e-6)
    assert z <= -1.0


# ---------------------------------------------------------------------------
# Mesh coordinates (Klein disk, relative to the current reference frame)
# ---------------------------------------------------------------------------

def test_coords_at_point_round_trips_through_get_face_at_coords(tess):
  rng = random.Random(0)
  for _ in range(30):
    face = _random_stable_face(tess, rng)
    p = _random_point(face, rng)
    u, v = tess.coords_at_point(p)
    assert tess.get_face_at_coords(u, v) is face


def test_new_point_at_coords_round_trips_coords_at_point(tess):
  rng = random.Random(1)
  for _ in range(30):
    face = _random_stable_face(tess, rng)
    p = _random_point(face, rng)
    u, v = tess.coords_at_point(p)
    p2 = tess.new_point_at_coords(u, v)
    assert p2.grid is face
    assert p2.b == pytest.approx(p.b, abs=1e-6)
    assert p2.s == pytest.approx(p.s, abs=1e-6)


def test_coords_at_point_changes_after_recenter(tess8):
  face = tess8.faces[0]
  p = IsometricPoint(face, 0.4 * face.altitude, 0.3 * face.altitude)
  before = tess8.coords_at_point(p)
  other_face = tess8.faces[-1]
  try:
    tess8.recenter(other_face)
    after = tess8.coords_at_point(p)
    assert after != pytest.approx(before)
    # ...but the point itself, and the face it's on, are unaffected.
    assert p.grid is face
    assert tess8.get_face_at_coords(*after) is face
  finally:
    tess8.recenter(face)  # restore, so other tests sharing this fixture
                           # aren't affected by this test's recenter.


# ---------------------------------------------------------------------------
# geodesic_distance / shortest_path_by_segment
# ---------------------------------------------------------------------------

def test_geodesic_distance_same_face_matches_minkowski_distance_directly(tess):
  # Deliberately *not* compared against `p1.distance_from(p2)`
  # (`IsometricGrid.local_distance`'s flat local-frame formula): unlike
  # `GenericTessellation`, where that flat formula *is* the definition of
  # geodesic distance, here `geodesic_distance` uses the tessellation's
  # exact global embedding instead, which is not the same thing even for
  # two points on the same face -- a hyperbolic triangle's interior isn't
  # isometric to a flat one, only its 3 vertices are (by construction, to
  # the tiling's own edge length). This test instead cross-checks against
  # an independent, direct Minkowski-model computation.
  face = tess.faces[0]
  alt = face.altitude
  p1 = IsometricPoint(face, 0.2 * alt, 0.3 * alt)
  p2 = IsometricPoint(face, 0.5 * alt, 0.1 * alt)
  expected = _hyperbolic_distance(
      _minkowski_of_klein(*tess.coords_at_point(p1)),
      _minkowski_of_klein(*tess.coords_at_point(p2)))
  assert tess.geodesic_distance(p1, p2) == pytest.approx(expected)


def test_geodesic_distance_is_symmetric(tess):
  rng = random.Random(2)
  for _ in range(30):
    fa = _random_stable_face(tess, rng)
    fb = _random_stable_face(tess, rng)
    p1 = _random_point(fa, rng)
    p2 = _random_point(fb, rng)
    d12 = tess.geodesic_distance(p1, p2)
    d21 = tess.geodesic_distance(p2, p1)
    assert d12 == pytest.approx(d21, rel=1e-6)


def test_shortest_path_by_segment_same_face_is_a_single_segment(tess):
  face = tess.faces[0]
  alt = face.altitude
  p1 = IsometricPoint(face, 0.1 * alt, 0.1 * alt)
  p2 = IsometricPoint(face, 0.3 * alt, 0.2 * alt)
  assert tess.shortest_path_by_segment(p1, p2) == [(p1, p2)]


def test_shortest_path_by_segment_endpoints_match_request(tess):
  rng = random.Random(3)
  for _ in range(20):
    fa = _random_stable_face(tess, rng)
    fb = _random_stable_face(tess, rng)
    p1 = _random_point(fa, rng)
    p2 = _random_point(fb, rng)
    segs = tess.shortest_path_by_segment(p1, p2)
    assert segs[0][0] is p1
    assert segs[-1][1] is p2


def test_shortest_path_by_segment_boundary_crossings_agree_across_faces(tess):
  # Consecutive segments must "hand off" at the same physical (Klein)
  # location, each expressed in its own face's local coordinates.
  rng = random.Random(4)
  for _ in range(20):
    fa = _random_stable_face(tess, rng)
    fb = _random_stable_face(tess, rng)
    p1 = _random_point(fa, rng)
    p2 = _random_point(fb, rng)
    segs = tess.shortest_path_by_segment(p1, p2)
    for (_, exit_point), (entry_point, _) in zip(segs, segs[1:]):
      exit_uv = tess.coords_at_point(exit_point)
      entry_uv = tess.coords_at_point(entry_point)
      assert exit_uv[0] == pytest.approx(entry_uv[0], abs=1e-6)
      assert exit_uv[1] == pytest.approx(entry_uv[1], abs=1e-6)


def test_geodesic_distance_matches_sum_of_segment_hyperbolic_lengths(tess):
  # Cross-checks the O(1) direct distance formula against the segment-by-
  # segment walk, computing each segment's length independently via the
  # Minkowski-model distance formula (not `IsometricPoint.distance_from`,
  # which uses the flat local-frame formula -- correct within one face,
  # but a genuinely independent check here).
  rng = random.Random(5)
  for _ in range(15):
    fa = _random_stable_face(tess, rng)
    fb = _random_stable_face(tess, rng)
    p1 = _random_point(fa, rng)
    p2 = _random_point(fb, rng)
    direct = tess.geodesic_distance(p1, p2)
    segs = tess.shortest_path_by_segment(p1, p2)
    total = 0.0
    for a, b in segs:
      pa = _minkowski_of_klein(*tess.coords_at_point(a))
      pb = _minkowski_of_klein(*tess.coords_at_point(b))
      total += _hyperbolic_distance(pa, pb)
    assert total == pytest.approx(direct, rel=1e-5, abs=1e-6)


# ---------------------------------------------------------------------------
# flatten_region
# ---------------------------------------------------------------------------

def test_flatten_region_preserves_distance_to_center(tess):
  rng = random.Random(6)
  center = _random_point(_random_stable_face(tess, rng), rng)
  targets = [
      _random_point(_random_stable_face(tess, rng), rng) for _ in range(5)]
  positions = tess.flatten_region(center, targets)
  for target, (x, y) in zip(targets, positions):
    expected = tess.geodesic_distance(center, target)
    assert math.hypot(x, y) == pytest.approx(expected, rel=1e-6)


def test_flatten_region_of_center_itself_is_the_origin(tess):
  center = tess.faces[0].centroid_local_coords
  [(x, y)] = tess.flatten_region(center, [center])
  assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_flatten_region_different_vertices_of_a_face_get_different_angles(tess):
  face = tess.faces[0]
  center = face.centroid_local_coords
  vertex_b_point = IsometricPoint(face, face.altitude, 0.0)
  vertex_s_point = IsometricPoint(face, 0.0, face.altitude)
  (xb, yb), (xs, ys) = tess.flatten_region(
      center, [vertex_b_point, vertex_s_point])
  assert math.atan2(yb, xb) != pytest.approx(math.atan2(ys, xs), abs=1e-3)


def test_flatten_region_of_no_targets_returns_an_empty_list(tess):
  center = tess.faces[0].centroid_local_coords
  assert tess.flatten_region(center, []) == []


# ---------------------------------------------------------------------------
# geodesically_canonicalize_point
# ---------------------------------------------------------------------------

def test_geodesically_canonicalize_point_is_idempotent_when_in_bounds(tess):
  face = tess.faces[2]
  alt = face.altitude
  p = IsometricPoint(face, 0.3 * alt, 0.3 * alt)
  tess.geodesically_canonicalize_point(p)
  assert p.grid is face
  assert p.b == pytest.approx(0.3 * alt)
  assert p.s == pytest.approx(0.3 * alt)


def test_geodesically_canonicalize_point_lands_in_bounds():
  # A fresh, unshared tessellation (rather than a shared module-scoped
  # fixture): repeatedly canonicalizing out-of-bounds displacements grows
  # the mesh and can recenter the reference frame, so keeping this
  # self-contained avoids this test's pass/fail depending on unrelated
  # tests' recentering.
  tess = HyperbolicTessellation(order=7, rings=1, max_stable_hops=5)
  rng = random.Random(6)
  for _ in range(30):
    face = _random_stable_face(tess, rng)
    alt = face.altitude
    scale = rng.uniform(0.05, 0.8)
    angle = rng.uniform(0, 2 * math.pi)
    wb = 1.0 / 3.0 + scale * math.cos(angle)
    ws = 1.0 / 3.0 + scale * math.sin(angle)
    p = IsometricPoint(face, wb * alt, ws * alt)
    tess.geodesically_canonicalize_point(p)
    wb2, ws2, wd2 = p.barycentric
    assert -1e-6 <= wb2 <= 1 + 1e-6
    assert -1e-6 <= ws2 <= 1 + 1e-6
    assert -1e-6 <= wd2 <= 1 + 1e-6


# ---------------------------------------------------------------------------
# extend / extend_to_include
# ---------------------------------------------------------------------------

def test_extend_grows_the_mesh_and_preserves_existing_faces_and_vertices():
  tess = HyperbolicTessellation(order=7, rings=2)
  faces_before = list(tess.faces)
  vertices_before = list(tess.vertices)
  tess.extend(rings=1)
  assert len(tess.faces) > len(faces_before)
  assert len(tess.vertices) > len(vertices_before)
  assert all(f in tess.faces for f in faces_before)
  assert all(v in tess.vertices for v in vertices_before)


def test_extend_to_include_covers_a_distant_point():
  tess = HyperbolicTessellation(order=7, rings=1)
  far_point = tess.new_point_at_coords(0.97, 0.0)
  assert far_point.grid not in tess._stable_faces

  tess.extend_to_include(far_point)

  assert far_point.grid in tess._stable_faces
  # A subsequent lookup for the same point should need no further growth
  # or recentering.
  faces_before = list(tess.faces)
  ref_before = tess.reference_point
  assert tess.get_face_at_coords(*tess.coords_at_point(far_point)) is (
      far_point.grid)
  assert list(tess.faces) == faces_before
  assert tess.reference_point is ref_before


def test_get_face_at_coords_auto_extends_the_mesh_for_a_distant_target():
  tess = HyperbolicTessellation(order=7, rings=1)
  face_count_before = len(tess.faces)
  face = tess.get_face_at_coords(0.97, 0.0)
  assert face is not None
  assert len(tess.faces) > face_count_before


def test_growth_stays_path_limited():
  # Reaching one distant point should build only a thin corridor of faces,
  # not the whole ring-by-ring neighborhood a batch `extend` would.
  tess = HyperbolicTessellation(order=7, rings=1)
  tess.get_face_at_coords(0.97, 0.0)
  path_limited_face_count = len(tess.faces)

  full_rings_tess = HyperbolicTessellation(order=7, rings=1)
  full_rings_tess.extend(rings=6)  # enough rings to reach a comparable radius
  assert path_limited_face_count < len(full_rings_tess.faces)


# ---------------------------------------------------------------------------
# Reference frame (recenter, auto-recentering, the "too far" exception)
# ---------------------------------------------------------------------------

def test_recenter_on_face_uses_its_centroid():
  tess = HyperbolicTessellation(order=7, rings=1)
  face = tess.faces[3]
  tess.recenter(face)
  assert tess.reference_point.grid is face
  assert tess.coords_at_point(face.centroid_local_coords) == pytest.approx(
      (0.0, 0.0), abs=1e-9)


def test_recenter_on_isometric_point():
  tess = HyperbolicTessellation(order=7, rings=1)
  face = tess.faces[3]
  p = IsometricPoint(face, 0.2 * face.altitude, 0.6 * face.altitude)
  tess.recenter(p)
  assert tess.reference_point is p
  assert tess.coords_at_point(p) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_recenter_on_coordinate_tuple():
  tess = HyperbolicTessellation(order=7, rings=1)
  target = tess.new_point_at_coords(0.2, 0.1)
  tess.recenter((0.2, 0.1))
  assert tess.reference_point.grid is target.grid


def test_recenter_rejects_unregistered_face():
  tess = HyperbolicTessellation(order=7, rings=1)
  other = HyperbolicTessellation(order=7, rings=1)
  with pytest.raises(ValueError):
    tess.recenter(other.faces[0])


def test_recenter_rejects_unsupported_type():
  tess = HyperbolicTessellation(order=7, rings=1)
  with pytest.raises(TypeError):
    tess.recenter(42)


def test_ensure_in_range_auto_recenters_toward_a_distant_point():
  tess = HyperbolicTessellation(order=7, rings=1, max_stable_hops=3)
  original_reference = tess.reference_point
  far_point = tess.new_point_at_coords(0.95, 0.0)
  assert far_point.grid not in tess._stable_faces

  # `geodesic_distance` calls `_ensure_in_range` on both its arguments,
  # which recenters toward `far_point` since it's outside the stable
  # region (`original_reference`'s own face always is one, trivially).
  tess.geodesic_distance(original_reference, far_point)

  assert tess.reference_point is not original_reference
  assert far_point.grid in tess._stable_faces


def test_auto_recenter_raises_for_a_point_too_far_to_bridge():
  tess = HyperbolicTessellation(order=7, rings=1, max_stable_hops=2)
  far_point = tess.new_point_at_coords(0.999, 0.0)
  with pytest.raises(ValueError):
    tess.geodesic_distance(tess.reference_point, far_point)


# ---------------------------------------------------------------------------
# Construction depth / exact combinatorial deduplication
#
# These target `_closure_neighbor` and the O(log depth) binary-lifted
# `frame`/`ancestor_jumps` machinery that lets mesh *construction* itself
# reach far beyond one-hop-at-a-time chaining -- distinct from the
# query-time reference frame above, which only keeps precision good for
# whatever's already built.
# ---------------------------------------------------------------------------

def _naive_undeduplicated_tree_size(order: int, rings: int) -> int:
  """How many vertices a `rings`-deep BFS would create if *no* vertex were
  ever reused (every fan step but the first, known one, created fresh).

  An upper bound only, useful for confirming real reuse is happening.

  Args:
    order: Number of faces meeting at each vertex of the {3, order} tiling.
    rings: How many rings deep the (hypothetical, undeduplicated) BFS goes.

  Returns:
    The number of vertices such a BFS would create.
  """
  total = 1 + order  # seed vertex + its own full ring
  frontier = order
  for _ in range(rings - 1):
    frontier = frontier * (order - 1)
    total += frontier
  return total


def test_vertex_count_is_far_below_the_naive_undeduplicated_bound():
  order, rings = 7, 5
  tess = HyperbolicTessellation(order=order, rings=1, max_stable_hops=1)
  tess.extend(rings=rings - 1)
  assert len(tess.vertices) < _naive_undeduplicated_tree_size(order, rings)


@pytest.mark.parametrize("order,rings", [(7, 6), (8, 4), (10, 3)])
def test_deep_extend_keeps_every_interior_vertex_at_valence_order(order, rings):
  tess = HyperbolicTessellation(order=order, rings=1, max_stable_hops=1)
  tess.extend(rings=rings)
  frontier = set(tess._frontier)
  for v in tess.vertices:
    if v not in frontier:
      assert len(v.adjacent_faces()) == order


def test_no_duplicate_faces_and_adjacency_is_symmetric():
  # Wherever two independently-grown parts of the mesh reach the same
  # triangle (which happens constantly as the fan-completion frontier
  # grows), `_add_face_if_new`'s dedup must resolve to a single shared
  # `TessellationFace`, never two distinct objects for the same 3
  # vertices, and every edge crossing must be mutually consistent (each
  # side's `face_on_edge` points back at the other).
  tess = HyperbolicTessellation(order=7, rings=1, max_stable_hops=1)
  tess.extend(rings=5)
  triples = [
      frozenset((f.vertex_b, f.vertex_s, f.vertex_d)) for f in tess.faces]
  assert len(triples) == len(set(triples))
  for face in tess.faces:
    for direction in IsometricDirection:
      neighbor = face.face_on_edge(direction)
      if neighbor is None:
        continue
      back_direction = neighbor.direction_away_from_face(face)
      assert neighbor.face_on_edge(back_direction) is face


def test_closure_neighbor_matches_a_known_adjacent_face_exactly():
  # A direct, white-box check that the exact combinatorial rule -- not the
  # floating-point fallback -- is what resolves an already-built edge:
  # `_closure_neighbor(vertex, prev_neighbor, face)` looks up the *other*
  # face sharing edge `(vertex, prev_neighbor)` and returns its third
  # vertex -- find a face whose neighbor across such an edge is already
  # built, and confirm the exact match, with no geometry involved.
  tess = HyperbolicTessellation(order=7, rings=2)
  found_case = False
  for face in tess.faces:
    for vertex, prev_neighbor, prev_prev in (
        (face.vertex_b, face.vertex_s, face.vertex_d),
        (face.vertex_s, face.vertex_d, face.vertex_b),
        (face.vertex_d, face.vertex_b, face.vertex_s),
    ):
      neighbor_face = face.face_on_edge(face.direction_toward_vertex(prev_prev))
      if neighbor_face is None:
        continue
      expected_third = next(
          v for v in
          (neighbor_face.vertex_b, neighbor_face.vertex_s, neighbor_face.vertex_d)
          if v is not vertex and v is not prev_neighbor)
      assert _closure_neighbor(vertex, prev_neighbor, face) is expected_third
      found_case = True
  assert found_case  # sanity: the mesh actually exercised the check above


def test_closure_neighbor_returns_none_when_the_far_face_is_unbuilt():
  # A ring vertex's own fan starts with one pre-built neighboring face to
  # lean on (from the seed's own fan around v0), but nothing past it --
  # continuing far enough must eventually return `None` rather than
  # fabricate an answer. Tries every ring vertex still on the frontier,
  # since the constructor's own initial recenter may have already
  # advanced a couple of them partway.
  tess = HyperbolicTessellation(order=7, rings=1, max_stable_hops=1)
  found_unbuilt = False
  for vertex, (known_neighbor, discovering_face) in tess._frontier.items():
    prev_neighbor, prev_face = known_neighbor, discovering_face
    for _ in range(tess.order):
      step = _closure_neighbor(vertex, prev_neighbor, prev_face)
      if step is None:
        found_unbuilt = True
        break
      prev_face = tess._add_face_if_new(vertex, prev_neighbor, step)
      prev_neighbor = step
    if found_unbuilt:
      break
  assert found_unbuilt


def test_deep_panning_survives_far_past_the_original_construction_wall():
  # Steady panning (repeated recenter toward a point just past the
  # current reference frame) used to hit `EndOfMeshSurfaceException` or
  # `RuntimeError` after only a handful of steps, around hyperbolic
  # distance 4-5 from the seed, because mesh construction -- not just
  # querying -- accumulated error with BFS depth. Drives the same pattern
  # far past that point.
  tess = HyperbolicTessellation(order=7, rings=1, max_stable_hops=3)
  for _ in range(40):
    p = tess.new_point_at_coords(0.3, 0.0)
    tess.recenter(p)
  seed_minkowski = _minkowski_of_poincare(0j)
  distance_from_seed = _hyperbolic_distance(
      seed_minkowski, tess._reference_global_minkowski)
  assert distance_from_seed > 5.0
