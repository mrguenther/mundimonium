import math

import numpy as np
import pytest

from mundimonium.coordinates import generic_tessellation
from mundimonium.coordinates.conftest import (
    build_icosahedron, build_stellated_icosahedron,
)
from mundimonium.coordinates.exceptions import EndOfMeshSurfaceException
from mundimonium.coordinates.generic_tessellation import (
    GenericTessellation, RelaxableFace, RelaxableVertex,
    _canonical_local_frame, _recentered_local_frame,
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
# point_to_3d_position
# ---------------------------------------------------------------------------

def test_point_to_3d_position_matches_barycentric_blend(generic_mesh):
  tess, _, _ = generic_mesh.build()
  face = generic_mesh.any_face()
  alt = face.altitude
  point = IsometricPoint(face, 0.3 * alt, 0.4 * alt)
  wb, ws, wd = point.barycentric
  expected = (
      wb * np.array(face.vertex_b.projection_coordinates)
      + ws * np.array(face.vertex_s.projection_coordinates)
      + wd * np.array(face.vertex_d.projection_coordinates))
  assert tuple(tess.point_to_3d_position(point)) == pytest.approx(
      tuple(expected))


def test_point_to_3d_position_at_a_vertex_is_that_vertex(generic_mesh):
  tess, _, _ = generic_mesh.build()
  face = generic_mesh.any_face()
  point = IsometricPoint(face, face.altitude, 0.0)  # vertex_b
  assert tuple(tess.point_to_3d_position(point)) == pytest.approx(
      face.vertex_b.projection_coordinates)


def test_point_to_3d_position_requires_euclidean_embedding():
  tess = GenericTessellation(euclidean=False)
  v1 = TessellationVertex([0.0, 0.0, 0.0])
  v2 = TessellationVertex([1.0, 0.0, 0.0])
  v3 = TessellationVertex([0.5, 1.0, 0.0])
  face = tess.add_face([v1, v2, v3])
  with pytest.raises(NotImplementedError):
    tess.point_to_3d_position(face.centroid_local_coords)


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


def _relaxable_icosahedron():
  return build_icosahedron(face_type=RelaxableFace, vertex_type=RelaxableVertex)


def _relaxable_stellated_icosahedron():
  return build_stellated_icosahedron(
      face_type=RelaxableFace, vertex_type=RelaxableVertex)


# ---------------------------------------------------------------------------
# RelaxableFace / RelaxableVertex
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def relaxable_icosahedron():
  return _relaxable_icosahedron()


@pytest.fixture(scope="module")
def relaxable_stellated_icosahedron():
  return _relaxable_stellated_icosahedron()


def test_discover_nearby_crosses_saddle_vertex():
  # Two triangles sharing only one vertex -- a "saddle" configuration
  # (`recalculate_adjacency_to` requires exactly 2 shared vertices for
  # edge adjacency, so these two are never edge-adjacent). `_discover_
  # nearby` must still cross it, since it reasons purely about vertex-
  # to-vertex edges, never face-to-face adjacency.
  tess = GenericTessellation(vertex_type=RelaxableVertex, face_type=RelaxableFace)
  center = RelaxableVertex([0.0, 0.0, 0.0])
  a = RelaxableVertex([1.0, 0.0, 0.0])
  b = RelaxableVertex([0.5, 1.0, 0.0])
  c = RelaxableVertex([-1.0, 0.0, 0.0])
  d = RelaxableVertex([-0.5, -1.0, 0.0])
  face_1 = tess.add_face([center, a, b])
  face_2 = tess.add_face([center, c, d])
  assert all(face_1.face_on_edge(d) is not face_2 for d in IsometricDirection)

  _, nearby_faces, _ = generic_tessellation._discover_nearby([center], 10)
  assert face_1 in nearby_faces
  assert face_2 in nearby_faces


def test_face_own_corners_are_pinned_at_canonical_positions(
    relaxable_icosahedron):
  _, _, faces = relaxable_icosahedron
  face = faces[1]
  positions = face.flattened_positions
  expected = _recentered_local_frame(face)
  assert positions[face.vertex_b] == pytest.approx(expected[0])
  assert positions[face.vertex_s] == pytest.approx(expected[1])
  assert positions[face.vertex_d] == pytest.approx(expected[2])


def test_flattened_positions_is_cached(relaxable_icosahedron):
  _, verts, faces = relaxable_icosahedron
  face = faces[2]
  assert face.flattened_positions is face.flattened_positions

  vertex = verts[2]
  assert vertex.flattened_positions is vertex.flattened_positions


def test_flattened_positions_invalidated_after_new_face_added():
  tess, _, faces = _relaxable_icosahedron()
  face = faces[0]
  cached = face.flattened_positions
  assert face._flattened_positions is cached

  v_new = RelaxableVertex([0.0, 0.0, 0.0])
  tess.add_face([face.vertex_b, face.vertex_s, v_new])

  assert face._flattened_positions is None
  assert face.vertex_b._flattened_positions is None
  assert face.vertex_s._flattened_positions is None


def test_flattened_positions_invalidation_is_selective(monkeypatch):
  # The default `RELAXATION_RADIUS` comfortably covers this entire
  # 20-face icosahedron from any single face, so nothing would be "far"
  # enough to survive invalidation at that radius -- shrink it first, on
  # a fresh tessellation, so a face reached only via a second hop is
  # genuinely out of range.
  monkeypatch.setattr(generic_tessellation, "RELAXATION_RADIUS", 1)

  tess, _, faces = _relaxable_icosahedron()
  face = faces[0]

  far_face = None
  for candidate in faces:
    if candidate is not face and candidate not in face.nearby_faces:
      far_face = candidate
      break
  assert far_face is not None, "expected some face outside the shrunk radius"

  far_face.flattened_positions  # force it to be cached
  cached = far_face._flattened_positions

  v_new = RelaxableVertex([0.0, 0.0, 0.0])
  tess.add_face([face.vertex_b, face.vertex_s, v_new])

  # `far_face` lies outside `RELAXATION_RADIUS` of the new face -- its
  # cache should be untouched.
  assert far_face._flattened_positions is cached


def test_relaxable_vertex_blend_matches_manual_equal_weighted_average(
    relaxable_icosahedron):
  # Every face here has the same (default) side length, so `RelaxableVertex`'s
  # circumradius-based weighting should reduce to a plain equal-weighted
  # average.
  _, verts, faces = relaxable_icosahedron
  vertex = verts[0]
  target_vertex = verts[1]  # present in every adjacent face's own dict,
                             # since the default radius covers this whole mesh
  adjacent = vertex.adjacent_faces()

  blended = vertex.flattened_positions[target_vertex]
  manual = generic_tessellation._blend_positions([
      (1.0, face.flattened_positions[target_vertex]
            - face.flattened_positions[vertex])
      for face in adjacent
  ])
  assert blended == pytest.approx(manual)


def test_relaxable_vertex_own_position_is_exactly_the_origin(
    relaxable_icosahedron):
  # Unlike an earlier, transform-blending design (which only landed
  # *close to* the origin here), blending raw positions recenters every
  # contribution exactly, so the blend is exactly `(0, 0)` too.
  _, verts, _faces = relaxable_icosahedron
  vertex = verts[0]
  assert vertex.flattened_positions[vertex] == pytest.approx(
      np.zeros(2), abs=1e-9)


# ---------------------------------------------------------------------------
# flatten_region
# ---------------------------------------------------------------------------

def test_flatten_region_requires_relaxable_vertex_type():
  # `face_type=RelaxableFace` alone isn't enough -- a plain
  # `TessellationVertex` has no `flattened_positions`.
  tess, _, faces = build_icosahedron(face_type=RelaxableFace)
  center = faces[0].centroid_local_coords
  with pytest.raises(AttributeError):
    tess.flatten_region(center, [center])


def test_flatten_region_centroid_lands_exactly_at_origin(relaxable_icosahedron):
  tess, _, faces = relaxable_icosahedron
  center = faces[0].centroid_local_coords
  (x, y), = tess.flatten_region(center, [center])
  assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_flatten_region_vertex_lands_exactly_at_origin(relaxable_icosahedron):
  # Unlike an earlier, transform-blending design (which only landed
  # *close to* the origin here), blending raw positions makes this exact
  # -- see `RelaxableVertex`'s own docstring.
  tess, _, faces = relaxable_icosahedron
  face = faces[0]
  vertex_point = IsometricPoint(face, face.altitude, 0.0)  # vertex_b
  (x, y), = tess.flatten_region(vertex_point, [vertex_point])
  assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_flatten_region_immediate_neighbors_land_near_their_true_distance(
    relaxable_icosahedron):
  tess, _, faces = relaxable_icosahedron
  face = faces[0]
  center = face.centroid_local_coords
  neighbors = [
      IsometricPoint(face, face.altitude, 0.0),
      IsometricPoint(face, 0.0, face.altitude),
      IsometricPoint(face, 0.0, 0.0),
  ]

  results = tess.flatten_region(center, neighbors)

  for neighbor, (x, y) in zip(neighbors, results):
    true_distance = tess.geodesic_distance(center, neighbor)
    flat_distance = math.hypot(x, y)
    assert flat_distance == pytest.approx(true_distance, rel=0.05)


def test_flatten_region_different_directions_get_different_angles(
    relaxable_icosahedron):
  tess, _, faces = relaxable_icosahedron
  face = faces[0]
  center = face.centroid_local_coords
  targets = [
      IsometricPoint(face, face.altitude, 0.0),
      IsometricPoint(face, 0.0, face.altitude),
      IsometricPoint(face, 0.0, 0.0),
  ]

  results = tess.flatten_region(center, targets)
  angles = [math.atan2(y, x) for x, y in results]

  for i in range(len(angles)):
    for j in range(i + 1, len(angles)):
      gap = abs(angles[i] - angles[j])
      gap = min(gap, 2 * math.pi - gap)
      assert gap > 0.5


def test_flatten_region_same_center_is_stable(relaxable_icosahedron):
  tess, _, faces = relaxable_icosahedron
  face = faces[0]
  center = face.centroid_local_coords
  targets = [
      IsometricPoint(face, face.altitude, 0.0),
      IsometricPoint(face, 0.0, face.altitude),
      IsometricPoint(face, 0.0, 0.0),
  ]

  first = tess.flatten_region(center, targets)
  second = tess.flatten_region(center, targets)

  # Bit-for-bit identical, unlike the earlier live-relaxation design:
  # everything involved is precomputed and cached, so repeating the same
  # call re-reads exactly the same cached positions every time.
  assert first == second


def test_flatten_region_degenerate_positions_match_direct_lookup(
    relaxable_icosahedron):
  # The plan's listed special cases (exactly at the centroid, at a
  # vertex, on a dividing edge) should all fall out of the same general
  # interpolation as a 1- or 2-anchor blend, with no separate code path.
  _, _, faces = relaxable_icosahedron
  face = faces[0]

  # Exactly at the centroid: only the face's own anchor should
  # contribute.
  centroid_blend = GenericTessellation._blended_positions_at(
      face, face.centroid_local_coords)
  for key, position in face.flattened_positions.items():
    assert centroid_blend[key] == pytest.approx(position)

  # Exactly at vertex_b: only vertex_b's own anchor should contribute.
  vertex_point = IsometricPoint(face, face.altitude, 0.0)
  vertex_blend = GenericTessellation._blended_positions_at(face, vertex_point)
  for key, position in face.vertex_b.flattened_positions.items():
    if key not in face.flattened_positions:
      continue
    assert vertex_blend[key] == pytest.approx(position)

  # On the b-s edge (the d-weight is exactly 0): only vertex_b and
  # vertex_s should contribute, not the centroid.
  edge_point = IsometricPoint(face, 0.5 * face.altitude, 0.5 * face.altitude)
  edge_blend = GenericTessellation._blended_positions_at(face, edge_point)
  manual = {}
  for key in face.flattened_positions:
    weighted = [
        (1.0, positions[key])
        for positions in (
            face.vertex_b.flattened_positions,
            face.vertex_s.flattened_positions)
        if key in positions
    ]
    if weighted:
      manual[key] = generic_tessellation._blend_positions(weighted)
  for key, position in manual.items():
    assert edge_blend[key] == pytest.approx(position)


def test_flatten_region_stellated_icosahedron_has_no_nan_across_valences(
    relaxable_stellated_icosahedron):
  # The motivating stress case for this whole phase: a neighborhood
  # spanning both a valence-5-to-10 original vertex and a valence-3 apex
  # vertex should relax to a finite, self-consistent layout.
  tess, _, faces = relaxable_stellated_icosahedron
  # `faces[0]` is `(apex, va, vb)` (see `build_stellated_icosahedron`'s
  # docstring): its own 3 corners already mix a valence-3 apex vertex
  # with two valence-10 original-icosahedron vertices.
  face = faces[0]
  center = face.centroid_local_coords
  targets = [
      IsometricPoint(face, face.altitude, 0.0),
      IsometricPoint(face, 0.0, face.altitude),
      IsometricPoint(face, 0.0, 0.0),
      center,
  ]

  results = tess.flatten_region(center, targets)

  for x, y in results:
    assert math.isfinite(x) and math.isfinite(y)


def test_flatten_region_raises_for_target_outside_every_anchors_range(
    monkeypatch):
  # The default `RELAXATION_RADIUS` turns out to comfortably cover an
  # entire 20-face icosahedron from any single anchor, so there's no
  # naturally-far-enough point to reach for on that fixture -- shrink the
  # radius instead, on a fresh tessellation, so a face reached only via a
  # second hop already falls outside it.
  monkeypatch.setattr(generic_tessellation, "RELAXATION_RADIUS", 1)

  tess, _, faces = _relaxable_icosahedron()
  face = faces[0]
  center = face.centroid_local_coords

  far_face = None
  for candidate in faces:
    if candidate is not face and candidate not in face.nearby_faces:
      far_face = candidate
      break
  assert far_face is not None, "expected some face outside the shrunk radius"
  far_target = far_face.centroid_local_coords

  with pytest.raises(ValueError):
    tess.flatten_region(center, [far_target])


def test_flatten_region_orientation_rotates_output(relaxable_icosahedron):
  tess, _, faces = relaxable_icosahedron
  face = faces[0]
  center = face.centroid_local_coords
  targets = [
      IsometricPoint(face, face.altitude, 0.0),
      IsometricPoint(face, 0.0, face.altitude),
  ]
  identity_results = tess.flatten_region(center, targets)

  angle = math.pi / 3
  rotation = np.array([
      [math.cos(angle), -math.sin(angle)],
      [math.sin(angle), math.cos(angle)],
  ])
  rotated_results = tess.flatten_region(center, targets, rotation)

  for (x, y), (rotated_x, rotated_y) in zip(identity_results, rotated_results):
    expected = rotation @ np.array([x, y])
    assert (rotated_x, rotated_y) == pytest.approx(tuple(expected))


# ---------------------------------------------------------------------------
# unflatten_point / unflatten_point_and_transport_orientation
# ---------------------------------------------------------------------------

def test_unflatten_point_round_trips_flatten_region(
    relaxable_stellated_icosahedron):
  # Uses the stellated (60-face) fixture rather than the plain
  # icosahedron: the latter's `RELAXATION_RADIUS` happens to comfortably
  # cover its entire 20-face mesh (see `test_flatten_region_raises_for_
  # target_outside_every_anchors_range`'s own comment), so unfolding it
  # flat from a single anchor is a full closed-surface unwrap that can
  # genuinely, harmlessly self-overlap far from center -- an unrelated,
  # pre-existing edge case of that specific small fixture, not something
  # this test is trying to exercise.
  tess, _, faces = relaxable_stellated_icosahedron
  face = faces[0]
  center = face.centroid_local_coords
  target = face.face_on_edge(IsometricDirection.S).centroid_local_coords

  (x, y), = tess.flatten_region(center, [target])
  resolved = tess.unflatten_point(center, x, y)

  assert resolved.grid is target.grid
  assert resolved.b == pytest.approx(target.b, abs=1e-6)
  assert resolved.s == pytest.approx(target.s, abs=1e-6)


def test_unflatten_point_and_transport_orientation_avoids_spurious_rotation(
    relaxable_stellated_icosahedron):
  # This is the regression case for the "panning to a new face snaps the
  # view to that face's own canonical orientation" bug: recentering on a
  # neighboring face and then, using the *transported* orientation,
  # looking back at the original center should land close to the exact
  # reverse of the original pan -- not rotated by whatever the two
  # faces' independent canonical frames happen to disagree by.
  tess, _, faces = relaxable_stellated_icosahedron
  face = faces[0]
  neighbor = face.face_on_edge(IsometricDirection.B)
  old_center = face.centroid_local_coords
  target = neighbor.centroid_local_coords

  (x, y), = tess.flatten_region(old_center, [target])
  new_center, new_orientation = tess.unflatten_point_and_transport_orientation(
      old_center, x, y, orientation=None)
  assert new_center.grid is neighbor

  (back_x, back_y), = tess.flatten_region(
      new_center, [old_center], new_orientation)
  assert (back_x, back_y) == pytest.approx((-x, -y), rel=0.2, abs=1e-6)
