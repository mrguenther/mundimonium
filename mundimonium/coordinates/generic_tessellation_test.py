import math

import numpy as np
import pytest

from mundimonium.coordinates import generic_tessellation
from mundimonium.coordinates.conftest import (
    build_icosahedron, build_stellated_icosahedron,
)
from mundimonium.coordinates.exceptions import EndOfMeshSurfaceException
from mundimonium.coordinates.generic_tessellation import (
    GenericTessellation, RelaxableFace, _canonical_local_frame,
    _LATTICE_ADJACENCY, _LATTICE_RESOLUTION, _LATTICE_TRIPLES,
)
from mundimonium.coordinates.isometric import IsometricDirection, IsometricPoint
from mundimonium.coordinates.tessellation import TessellationVertex


# `icosahedron` fixture comes from conftest.py.
#
# `generic_mesh` (also from conftest.py) is parametrized over every
# registered `GenericTessellationFixture` subclass, so tests using it
# automatically pick up any new mesh shape registered in the future with no
# changes needed here. Tests that assert something true of *any*
# GenericTessellation mesh should prefer it over the raw `icosahedron`
# fixture; tests that rely on icosahedron-specific facts (e.g. its exact
# antipodal symmetry) should keep using `icosahedron` directly.


# ---------------------------------------------------------------------------
# Abstract-ish stubs
# ---------------------------------------------------------------------------

def test_coords_related_methods_are_not_implemented(generic_mesh):
  tess, _, _ = generic_mesh.build()
  point = generic_mesh.any_face().centroid_local_coords
  with pytest.raises(NotImplementedError):
    tess.new_point_at_coords(0.0, 0.0)
  with pytest.raises(NotImplementedError):
    tess.get_face_at_coords(0.0, 0.0)
  with pytest.raises(NotImplementedError):
    tess.coords_at_point(point)


def test_flatten_region_requires_relaxable_face_type(generic_mesh):
  tess, _, _ = generic_mesh.build()
  point = generic_mesh.any_face().centroid_local_coords
  with pytest.raises(TypeError):
    tess.flatten_region(point, [point])


# ---------------------------------------------------------------------------
# Solver cache invalidation
# ---------------------------------------------------------------------------

def test_solvers_invalidated_when_a_new_face_is_registered():
  tess = GenericTessellation()
  v1 = TessellationVertex([0, 0, 0])
  v2 = TessellationVertex([1, 0, 0])
  v3 = TessellationVertex([0.5, 1, 0])
  tess.add_face([v1, v2, v3])
  tess.compute_distance_field(IsometricPoint.center(tess._faces[0]))
  assert tess._matrices_built

  v4 = TessellationVertex([1.5, 1, 0])
  tess.add_face([v2, v4, v3])
  assert not tess._matrices_built


def test_build_solvers_requires_at_least_one_face():
  tess = GenericTessellation()
  with pytest.raises(ValueError):
    tess._build_solvers()


# ---------------------------------------------------------------------------
# Local 2D <-> barycentric conversions
# ---------------------------------------------------------------------------

def test_local_2d_barycentric_round_trip(generic_mesh):
  tess, _, _ = generic_mesh.build()
  tess._build_solvers()
  rng = np.random.default_rng(0)
  for _ in range(50):
    w = rng.random(3)
    w /= w.sum()
    p2d = tess._barycentric_to_local_2d(tuple(w))
    w_back = tess._local_2d_to_barycentric(p2d)
    assert w_back == pytest.approx(tuple(w), abs=1e-9)


def test_canonical_local_frame_round_trip_matches_face_geometry(generic_mesh):
  face = generic_mesh.any_face()
  frame = _canonical_local_frame(face)
  assert frame[1, 0] == pytest.approx(face.side_length)
  assert frame[2, 1] == pytest.approx(face.altitude)

  rng = np.random.default_rng(1)
  for _ in range(50):
    w = rng.random(3)
    w /= w.sum()
    p2d = w[0] * frame[0] + w[1] * frame[1] + w[2] * frame[2]
    w_back = GenericTessellation._local_2d_to_barycentric_in_frame(p2d, frame)
    assert w_back == pytest.approx(tuple(w), abs=1e-9)


# ---------------------------------------------------------------------------
# geodesic_distance
# ---------------------------------------------------------------------------

def test_geodesic_distance_same_face_matches_local_isometric_distance(
    generic_mesh):
  tess, _, _ = generic_mesh.build()
  face = generic_mesh.any_face()
  alt = face.altitude
  p1 = IsometricPoint(face, 0.2 * alt, 0.3 * alt)
  p2 = IsometricPoint(face, 0.5 * alt, 0.1 * alt)
  assert tess.geodesic_distance(p1, p2) == pytest.approx(
      p1.distance_from(p2))


def test_geodesic_distance_adjacent_faces_matches_projection(generic_mesh):
  tess, _, _ = generic_mesh.build()
  face, neighbor = generic_mesh.two_adjacent_faces()
  p1 = face.centroid_local_coords
  p2 = neighbor.centroid_local_coords
  expected = p1.project_onto_adjacent_grid(neighbor).distance_from(p2)
  assert tess.geodesic_distance(p1, p2) == pytest.approx(expected)


def test_geodesic_distance_is_symmetric_and_zero_for_same_point(
    generic_mesh):
  tess, _, _ = generic_mesh.build()
  face_a, face_b = generic_mesh.two_far_apart_faces()
  p1 = face_a.centroid_local_coords
  p2 = face_b.centroid_local_coords
  d12 = tess.geodesic_distance(p1, p2)
  d21 = tess.geodesic_distance(p2, p1)
  assert d12 == pytest.approx(d21, rel=1e-6)
  assert d12 > 0
  assert tess.geodesic_distance(p1, p1) == pytest.approx(0.0, abs=1e-6)


def test_geodesic_distance_between_neighbors_is_the_right_order_of_magnitude(
    generic_mesh):
  # The Heat Method is only approximate -- and known to be less accurate for
  # short/near-field distances, especially on a coarse mesh -- so this only
  # checks it's in the right ballpark relative to the exact "project across
  # the shared edge" distance, not tightly accurate.
  tess, _, _ = generic_mesh.build()
  face, neighbor = generic_mesh.two_adjacent_faces()
  p1 = face.centroid_local_coords
  p2 = neighbor.centroid_local_coords
  exact = p1.project_onto_adjacent_grid(neighbor).distance_from(p2)
  heat_method_distance = tess.compute_distance_field(p1)
  tgt_f_idx = tess._face_index_map[neighbor]
  wb, ws, wd = p2.barycentric
  approx_dist = sum(
      float(w) * heat_method_distance[v_idx]
      for v_idx, w in zip(tess._face_indices[tgt_f_idx], [wb, ws, wd]))
  assert 0 < approx_dist < 3 * exact


# ---------------------------------------------------------------------------
# shortest_path
# ---------------------------------------------------------------------------

def test_shortest_path_same_face_is_trivial(generic_mesh):
  tess, _, _ = generic_mesh.build()
  face = generic_mesh.any_face()
  alt = face.altitude
  p1 = IsometricPoint(face, 0.1 * alt, 0.1 * alt)
  p2 = IsometricPoint(face, 0.4 * alt, 0.2 * alt)
  assert tess.shortest_path(p1, p2) == [p1, p2]


def test_shortest_path_endpoints_match_request(generic_mesh):
  tess, _, _ = generic_mesh.build()
  face_a, face_b = generic_mesh.two_far_apart_faces()
  p1 = face_a.centroid_local_coords
  p2 = face_b.centroid_local_coords
  path = tess.shortest_path(p1, p2)
  assert len(path) >= 2
  assert path[0] is p1
  assert path[-1].grid is p2.grid
  assert path[-1].b == pytest.approx(p2.b, abs=1e-6)
  assert path[-1].s == pytest.approx(p2.s, abs=1e-6)


def test_shortest_path_stays_within_valid_local_coordinates(generic_mesh):
  tess, _, _ = generic_mesh.build()
  face_a, face_b = generic_mesh.two_far_apart_faces()
  p1 = face_a.centroid_local_coords
  p2 = face_b.centroid_local_coords
  path = tess.shortest_path(p1, p2)
  for point in path:
    alt = point.grid.altitude
    assert -1e-6 <= point.b <= alt + 1e-6
    assert -1e-6 <= point.s <= alt + 1e-6
    assert -1e-6 <= point.d <= alt + 1e-6


# ---------------------------------------------------------------------------
# geodesically_canonicalize_point
# ---------------------------------------------------------------------------

def test_geodesically_canonicalize_point_one_hop_matches_projection(
    generic_mesh):
  tess, _, _ = generic_mesh.build()
  face = generic_mesh.any_face()
  alt = face.altitude
  new_grid = face.face_on_edge(IsometricDirection.S)

  b = alt * 0.3
  s = -alt * 0.05  # just past edge S, safely within bounds on b and d
  expected = IsometricPoint(face, b, s).project_onto_adjacent_grid(new_grid)

  p = IsometricPoint(face, b, s)
  tess.geodesically_canonicalize_point(p)
  assert p.grid is expected.grid
  assert p.b == pytest.approx(expected.b, abs=1e-6)
  assert p.s == pytest.approx(expected.s, abs=1e-6)


def test_geodesically_canonicalize_point_is_idempotent_when_in_bounds(
    generic_mesh):
  tess, _, _ = generic_mesh.build()
  face = generic_mesh.any_face()
  alt = face.altitude
  p = IsometricPoint(face, 0.3 * alt, 0.3 * alt)
  tess.geodesically_canonicalize_point(p)
  assert p.grid is face
  assert p.b == pytest.approx(0.3 * alt)
  assert p.s == pytest.approx(0.3 * alt)


def test_geodesically_canonicalize_point_lands_in_bounds_far_away(
    generic_mesh):
  tess, _, _ = generic_mesh.build()
  face = generic_mesh.any_face()
  alt = face.altitude
  p = IsometricPoint(face, -4.3 * alt, -2.1 * alt)
  tess.geodesically_canonicalize_point(p)
  assert -1e-6 <= p.b <= p.grid.altitude + 1e-6
  assert -1e-6 <= p.s <= p.grid.altitude + 1e-6
  assert -1e-6 <= p.d <= p.grid.altitude + 1e-6


def test_geodesically_canonicalize_point_handles_ray_through_a_vertex(
    generic_mesh):
  # Regression test: a ray whose direction is exactly symmetric (here,
  # b == s) can pass precisely through a shared mesh vertex, where every
  # edge touching that vertex reports the same crossing distance -- and,
  # after landing exactly on it, every edge reports distance 0, leaving no
  # well-defined next edge to cross. `_DEGENERATE_RAY_NUDGE_RADIANS` guards
  # against exactly this. This is a property of any equilateral face's own
  # local (b, s, d) frame (b == s always aims at the same vertex, regardless
  # of mesh topology), so it applies to any registered fixture.
  tess, _, _ = generic_mesh.build()
  face = generic_mesh.any_face()
  alt = face.altitude
  p = IsometricPoint(face, -3 * alt, -3 * alt)
  tess.geodesically_canonicalize_point(p)
  assert -1e-6 <= p.b <= p.grid.altitude + 1e-6
  assert -1e-6 <= p.s <= p.grid.altitude + 1e-6
  assert -1e-6 <= p.d <= p.grid.altitude + 1e-6


def test_geodesically_canonicalize_point_is_smooth_near_a_vertex_crossing(
    icosahedron):
  # The output should vary continuously as the query angle sweeps through a
  # direction that grazes a mesh vertex -- not jump to a wrong/invalid
  # result exactly at that angle (see the regression test above) or to a
  # wildly different one immediately next to it.
  #
  # Icosahedron-specific (not parametrized over `generic_mesh`): the angle
  # window tested is small enough to stay on one face only because this
  # mesh's vertices all have the same gentle (valence-5, 60-degree-defect)
  # curvature. A mesh with sharper vertices (e.g. a tetrahedron-cap apex)
  # can legitimately cross into a neighboring face within the same window.
  tess, _, faces = icosahedron
  face = faces[0]
  alt = face.altitude
  base_angle = math.atan2(-1, -1)  # the degenerate direction from that test
  scale = 3.0 * alt

  results = []
  for i in range(-5, 6):
    angle = base_angle + i * 1e-4
    p = IsometricPoint(
        face, scale * math.cos(angle), scale * math.sin(angle))
    tess.geodesically_canonicalize_point(p)
    results.append((p.grid, p.b, p.s))

  assert len({id(grid) for grid, _, _ in results}) == 1  # same face throughout
  for (_, b1, s1), (_, b2, s2) in zip(results, results[1:]):
    assert abs(b2 - b1) < 0.01 * alt
    assert abs(s2 - s1) < 0.01 * alt


def test_geodesically_canonicalize_point_raises_at_mesh_boundary():
  # Deliberately a single, disconnected triangle (not one of the registered
  # `GenericTessellationFixture` mesh shapes) -- this tests open-boundary
  # behavior specifically, not "does this work across weird mesh shapes".
  v1 = TessellationVertex([0, 0, 0])
  v2 = TessellationVertex([1, 0, 0])
  v3 = TessellationVertex([0.5, 1, 0])
  tess = GenericTessellation()
  face = tess.add_face([v1, v2, v3])
  p = IsometricPoint(face, -0.5, 0.3)
  with pytest.raises(EndOfMeshSurfaceException):
    tess.geodesically_canonicalize_point(p)


def test_geodesically_canonicalize_point_raises_if_max_steps_exceeded(
    generic_mesh):
  tess, _, _ = generic_mesh.build()
  face = generic_mesh.any_face()
  alt = face.altitude
  p = IsometricPoint(face, -4.3 * alt, -2.1 * alt)
  with pytest.raises(RuntimeError):
    tess.geodesically_canonicalize_point(p, max_steps=1)


# ---------------------------------------------------------------------------
# shortest_path_by_segment
# ---------------------------------------------------------------------------

def test_shortest_path_by_segment_same_face_is_a_single_segment(
    generic_mesh):
  tess, _, _ = generic_mesh.build()
  face = generic_mesh.any_face()
  alt = face.altitude
  p1 = IsometricPoint(face, 0.1 * alt, 0.1 * alt)
  p2 = IsometricPoint(face, 0.3 * alt, 0.2 * alt)
  assert tess.shortest_path_by_segment(p1, p2) == [(p1, p2)]


def test_shortest_path_by_segment_adjacent_faces_cross_directly(
    generic_mesh):
  # A single edge crossing should produce exactly two segments, meeting
  # exactly at the shared edge.
  tess, _, _ = generic_mesh.build()
  face, neighbor = generic_mesh.two_adjacent_faces()
  p1 = face.centroid_local_coords
  p2 = neighbor.centroid_local_coords
  segs = tess.shortest_path_by_segment(p1, p2)

  assert len(segs) == 2
  assert segs[0][0] is p1
  assert segs[0][1].grid is face
  assert segs[1][0].grid is neighbor
  assert segs[1][1] is p2
  expected = segs[0][1].project_onto_adjacent_grid(neighbor)
  assert segs[1][0].b == pytest.approx(expected.b)
  assert segs[1][0].s == pytest.approx(expected.s)


def test_shortest_path_by_segment_far_apart_endpoints_match_request(
    generic_mesh):
  tess, _, _ = generic_mesh.build()
  face_a, face_b = generic_mesh.two_far_apart_faces()
  p1 = face_a.centroid_local_coords
  p2 = face_b.centroid_local_coords
  segs = tess.shortest_path_by_segment(p1, p2)
  assert len(segs) >= 1
  assert segs[0][0] is p1
  assert segs[-1][1] is p2


def test_shortest_path_by_segment_each_segment_stays_within_one_grid(
    generic_mesh):
  tess, _, _ = generic_mesh.build()
  face_a, face_b = generic_mesh.two_far_apart_faces()
  p1 = face_a.centroid_local_coords
  p2 = face_b.centroid_local_coords
  segs = tess.shortest_path_by_segment(p1, p2)
  for a, b in segs:
    assert a.grid is b.grid


def test_shortest_path_by_segment_boundary_crossings_agree_across_grids(
    generic_mesh):
  tess, _, _ = generic_mesh.build()
  face_a, face_b = generic_mesh.two_far_apart_faces()
  p1 = face_a.centroid_local_coords
  p2 = face_b.centroid_local_coords
  segs = tess.shortest_path_by_segment(p1, p2)
  for (_, exit_point), (entry_point, _) in zip(segs, segs[1:]):
    projected = exit_point.project_onto_adjacent_grid(entry_point.grid)
    assert projected.b == pytest.approx(entry_point.b, abs=1e-6)
    assert projected.s == pytest.approx(entry_point.s, abs=1e-6)


def test_shortest_path_by_segment_endpoints_are_not_minuscule(generic_mesh):
  # The first and last segments are never minuscule (leading/trailing
  # near-zero segments -- e.g. an endpoint that started exactly on an
  # edge/vertex -- are dropped). A *middle* segment can legitimately still
  # be exactly zero-length, if the taut path happens to pass precisely
  # through a mesh vertex partway along an otherwise-straight stretch --
  # that's kept as-is (not filtered), since dropping it would splice its
  # two neighbors together directly, and they generally aren't themselves
  # adjacent faces.
  tess, _, _ = generic_mesh.build()
  face_a, face_b = generic_mesh.two_far_apart_faces()
  p1 = face_a.centroid_local_coords
  p2 = face_b.centroid_local_coords
  segs = tess.shortest_path_by_segment(p1, p2)
  a0, b0 = segs[0]
  assert a0.distance_from(b0) > 1e-6 * a0.grid.altitude
  a_last, b_last = segs[-1]
  assert a_last.distance_from(b_last) > 1e-6 * a_last.grid.altitude


def test_shortest_path_by_segment_handles_antipodal_centroids(icosahedron):
  # Icosahedron-specific regression test (not parametrized over
  # `generic_mesh`, since "antipodal centroids" relies on this mesh's exact
  # symmetry): exactly-antipodal face centroids put many equal-length
  # corridors in an exact A* tie. This must resolve to some single,
  # consistent, valid path rather than raising or looping.
  tess, _, faces = icosahedron
  p1 = faces[0].centroid_local_coords
  p2 = faces[13].centroid_local_coords
  segs = tess.shortest_path_by_segment(p1, p2)
  assert segs[0][0] is p1
  assert segs[-1][1] is p2
  for a, b in segs:
    assert a.grid is b.grid


# ---------------------------------------------------------------------------
# RelaxableFace
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def relaxable_icosahedron():
  return build_icosahedron(face_type=RelaxableFace)


def test_relaxation_points_and_dedup_keys_match_lattice_resolution_count(
    relaxable_icosahedron):
  _, _, faces = relaxable_icosahedron
  face = faces[0]
  n = _LATTICE_RESOLUTION
  expected_count = (n + 1) * (n + 2) // 2
  assert len(face.relaxation_points) == expected_count
  assert len(face.dedup_keys()) == expected_count


def test_relaxation_points_are_cached_not_recomputed(relaxable_icosahedron):
  _, _, faces = relaxable_icosahedron
  face = faces[1]
  assert face.relaxation_points is face.relaxation_points
  assert face.dedup_keys() is face.dedup_keys()


def test_dedup_keys_include_the_faces_own_three_vertices(relaxable_icosahedron):
  _, _, faces = relaxable_icosahedron
  face = faces[0]
  keys = face.dedup_keys()
  assert face.vertex_b in keys
  assert face.vertex_s in keys
  assert face.vertex_d in keys


def test_relaxation_point_corners_are_at_their_named_vertex(
    relaxable_icosahedron):
  _, _, faces = relaxable_icosahedron
  face = faces[0]
  points = face.relaxation_points
  keys = face.dedup_keys()
  alt = face.altitude
  for point, key in zip(points, keys):
    if key is face.vertex_b:
      assert (point.b, point.s) == pytest.approx((alt, 0.0))
    elif key is face.vertex_s:
      assert (point.b, point.s) == pytest.approx((0.0, alt))
    elif key is face.vertex_d:
      assert (point.b, point.s) == pytest.approx((0.0, 0.0))


def test_adjacent_faces_agree_on_every_shared_edge_point(
    relaxable_icosahedron):
  _, _, faces = relaxable_icosahedron
  face = faces[0]
  neighbor = face.face_on_edge(IsometricDirection.B)
  shared = set(face.dedup_keys()) & set(neighbor.dedup_keys())
  # The shared edge carries `_LATTICE_RESOLUTION + 1` points (its two
  # shared corners plus `_LATTICE_RESOLUTION - 1` edge-interior points).
  assert len(shared) == _LATTICE_RESOLUTION + 1


def test_non_adjacent_faces_share_at_most_one_vertex_and_no_edge_points(
    relaxable_icosahedron):
  _, _, faces = relaxable_icosahedron
  # faces[0] (vertices [0, 11, 5]) and faces[15] (vertices [4, 9, 5]) --
  # see `IcosahedronFixture.two_far_apart_faces` -- share exactly one
  # *vertex* (index 5) but no edge, so they should share exactly one
  # dedup key (that corner), never an edge-point key.
  face_a, face_b = faces[0], faces[15]
  shared = set(face_a.dedup_keys()) & set(face_b.dedup_keys())
  vertices_a = {face_a.vertex_b, face_a.vertex_s, face_a.vertex_d}
  vertices_b = {face_b.vertex_b, face_b.vertex_s, face_b.vertex_d}
  assert shared == vertices_a & vertices_b
  assert len(shared) == 1


# ---------------------------------------------------------------------------
# flatten_region
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def relaxable_stellated_icosahedron():
  return build_stellated_icosahedron(face_type=RelaxableFace)


# The one fully-interior lattice point (all three barycentric indices > 0)
# at `_LATTICE_RESOLUTION == 3` -- used as an anchor with a full set of 6
# immediate neighbors to test against, rather than a corner/edge point
# (fewer neighbors, less representative).
_INTERIOR_INDEX = _LATTICE_TRIPLES.index((1, 1, 1))


def test_flatten_region_anchor_lands_exactly_at_origin(relaxable_icosahedron):
  tess, _, faces = relaxable_icosahedron
  center = faces[0].relaxation_points[_INTERIOR_INDEX]
  (x, y), = tess.flatten_region(center, [center])
  assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_flatten_region_immediate_neighbors_land_near_their_true_distance(
    relaxable_icosahedron):
  tess, _, faces = relaxable_icosahedron
  face = faces[0]
  center = face.relaxation_points[_INTERIOR_INDEX]
  neighbor_indices = _LATTICE_ADJACENCY[_INTERIOR_INDEX]
  neighbors = [face.relaxation_points[i] for i in neighbor_indices]

  results = tess.flatten_region(center, neighbors)

  for neighbor, (x, y) in zip(neighbors, results):
    true_distance = tess.geodesic_distance(center, neighbor)
    flat_distance = math.hypot(x, y)
    # An immediate neighbor of the (pinned) anchor is only lightly relaxed
    # (low mobility, close to the anchor), so its flattened distance should
    # stay close to its true geodesic distance -- not exact, since it's
    # still pulled on by the rest of the relaxed lattice.
    assert flat_distance == pytest.approx(true_distance, rel=0.1)


def test_flatten_region_different_directions_get_different_angles(
    relaxable_icosahedron):
  tess, _, faces = relaxable_icosahedron
  face = faces[0]
  center = face.relaxation_points[_INTERIOR_INDEX]
  neighbor_indices = _LATTICE_ADJACENCY[_INTERIOR_INDEX]
  neighbors = [face.relaxation_points[i] for i in neighbor_indices[:3]]

  results = tess.flatten_region(center, neighbors)
  angles = [math.atan2(y, x) for x, y in results]

  # Every pairwise angle gap should be non-trivial -- direction isn't
  # collapsed by the relaxation.
  for i in range(len(angles)):
    for j in range(i + 1, len(angles)):
      gap = abs(angles[i] - angles[j])
      gap = min(gap, 2 * math.pi - gap)
      assert gap > 0.2


def test_flatten_region_same_center_is_stable(relaxable_icosahedron):
  tess, _, faces = relaxable_icosahedron
  face = faces[0]
  center = face.relaxation_points[_INTERIOR_INDEX]
  targets = list(face.relaxation_points)

  first = tess.flatten_region(center, targets)
  second = tess.flatten_region(center, targets)

  # Not bit-for-bit identical: the second call warm-starts from the
  # first's cached positions and runs a few more PBD passes on top of
  # them, and this mesh's constraint graph never fully settles to a
  # zero-movement fixed point (see `_SOR_FACTOR`'s comment) -- so a small
  # amount of further, ever-slowing settling between calls is expected.
  # "Stable" here means that further settling, not a jump to a
  # different layout.
  for (x1, y1), (x2, y2) in zip(first, second):
    assert (x1, y1) == pytest.approx((x2, y2), abs=1e-2)


def test_flatten_region_warm_start_reuses_cached_neighborhood():
  tess, _, faces = build_icosahedron(face_type=RelaxableFace)
  face = faces[0]
  center = face.relaxation_points[_INTERIOR_INDEX]

  tess.flatten_region(center, [center])
  assert len(tess._relaxation_neighborhoods) == 1
  cached_neighborhood = next(iter(tess._relaxation_neighborhoods.values()))

  tess.flatten_region(center, [center])
  assert len(tess._relaxation_neighborhoods) == 1
  assert next(iter(tess._relaxation_neighborhoods.values())) is (
      cached_neighborhood)


def test_flatten_region_warm_starts_when_anchor_moves_to_adjacent_point(
    relaxable_icosahedron):
  # The realistic case as a camera pans continuously: the anchor snaps to
  # a *different* (but nearby) lattice point almost every call, not the
  # exact same one repeatedly -- warm-starting needs to trigger then too,
  # not only when the anchor is bit-for-bit unchanged.
  tess, _, faces = relaxable_icosahedron
  face = faces[0]
  center_a = face.relaxation_points[_INTERIOR_INDEX]
  neighbor_index = _LATTICE_ADJACENCY[_INTERIOR_INDEX][0]
  center_b = face.relaxation_points[neighbor_index]

  tess.flatten_region(center_a, [center_a])
  anchor_b_key = face.dedup_keys()[neighbor_index]
  neighborhood_b = tess._relaxation_neighborhood(
      anchor_b_key, face, neighbor_index)
  already_cached = sum(
      1 for key in neighborhood_b.locations
      if key in tess._flatten_position_cache)
  assert already_cached > 0.9 * len(neighborhood_b.locations)

  (x, y), = tess.flatten_region(center_b, [center_b])
  assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_flatten_region_adjacent_faces_agree_at_shared_relaxed_points(
    relaxable_icosahedron):
  tess, _, faces = relaxable_icosahedron
  face = faces[0]
  neighbor = face.face_on_edge(IsometricDirection.B)
  center = face.relaxation_points[_INTERIOR_INDEX]

  shared_keys = set(face.dedup_keys()) & set(neighbor.dedup_keys())
  face_points = {
      key: point for key, point in zip(face.dedup_keys(), face.relaxation_points)
      if key in shared_keys
  }
  neighbor_points = {
      key: point
      for key, point in zip(neighbor.dedup_keys(), neighbor.relaxation_points)
      if key in shared_keys
  }

  # One call, so both faces' own copies of each shared point are read off
  # the exact same relaxed `positions` snapshot -- calling flatten_region
  # twice would let the second call's warm-started passes move the
  # (shared, cached-by-key) positions slightly between calls, which would
  # test warm-start settling (see the "same center is stable" test above)
  # rather than genuine cross-face agreement.
  keys = list(shared_keys)
  targets = [face_points[k] for k in keys] + [neighbor_points[k] for k in keys]
  results = tess.flatten_region(center, targets)
  from_face = dict(zip(keys, results[:len(keys)]))
  from_neighbor = dict(zip(keys, results[len(keys):]))

  for key in keys:
    assert from_face[key] == pytest.approx(from_neighbor[key], abs=1e-9)


def test_flatten_region_stellated_icosahedron_has_no_nan_across_valences(
    relaxable_stellated_icosahedron):
  # The motivating stress case for this whole phase: a neighborhood
  # spanning both a valence-5-to-10 original vertex and a valence-3 apex
  # vertex should relax to a finite, self-consistent layout.
  tess, _, faces = relaxable_stellated_icosahedron
  # `faces[0]` is `(apex, va, vb)` (see `build_stellated_icosahedron`'s
  # docstring): its own 3 corners already mix a valence-3 apex vertex with
  # two valence-10 original-icosahedron vertices, no need to reach further
  # out (which risks exceeding RELAXATION_RADIUS -- see the out-of-radius
  # test below).
  face = faces[0]
  center = face.relaxation_points[_INTERIOR_INDEX]
  targets = list(face.relaxation_points)
  neighbor = face.face_on_edge(IsometricDirection.B)
  if neighbor is not None:
    targets += list(neighbor.relaxation_points)

  results = tess.flatten_region(center, targets)

  for x, y in results:
    assert math.isfinite(x) and math.isfinite(y)


def test_flatten_region_raises_for_target_outside_relaxation_radius(
    monkeypatch):
  # `RELAXATION_RADIUS` (3.0) turns out to comfortably cover an entire
  # 20-face icosahedron from any single anchor (its faces are small
  # relative to the radius), so there's no naturally-far-enough point to
  # reach for on that fixture -- shrink the radius instead, on a fresh
  # tessellation, so the far corner of the very same face already falls
  # outside it.
  monkeypatch.setattr(generic_tessellation, "RELAXATION_RADIUS", 0.05)

  tess, _, faces = build_icosahedron(face_type=RelaxableFace)
  face = faces[0]
  center = face.relaxation_points[_INTERIOR_INDEX]
  far_point = IsometricPoint(face, face.altitude, 0.0)  # vertex_b's corner

  with pytest.raises(ValueError):
    tess.flatten_region(center, [far_point])


def test_flatten_region_cache_invalidated_after_new_face_added():
  tess, _, faces = build_icosahedron(face_type=RelaxableFace)
  center = faces[0].relaxation_points[_INTERIOR_INDEX]
  tess.flatten_region(center, [center])
  assert tess._relaxation_neighborhoods

  v_new = TessellationVertex([0, 0, 0])
  tess.add_face([faces[0].vertex_b, faces[0].vertex_s, v_new])

  assert not tess._relaxation_neighborhoods
  assert not tess._flatten_position_cache
