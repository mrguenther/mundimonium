import math

import numpy as np
import pytest

from mundimonium.coordinates.exceptions import EndOfMeshSurfaceException
from mundimonium.coordinates.generic_tessellation import GenericTessellation
from mundimonium.coordinates.isometric import IsometricDirection, IsometricPoint
from mundimonium.coordinates.tessellation import TessellationVertex


# `icosahedron` fixture comes from conftest.py.


# ---------------------------------------------------------------------------
# Abstract-ish stubs
# ---------------------------------------------------------------------------

def test_coords_related_methods_are_not_implemented(icosahedron):
  tess, _, faces = icosahedron
  point = faces[0].centroid_local_coords
  with pytest.raises(NotImplementedError):
    tess.new_point_at_coords(0.0, 0.0)
  with pytest.raises(NotImplementedError):
    tess.get_face_at_coords(0.0, 0.0)
  with pytest.raises(NotImplementedError):
    tess.coords_at_point(point)


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

def test_local_2d_barycentric_round_trip(icosahedron):
  tess, _, _ = icosahedron
  tess._build_solvers()
  rng = np.random.default_rng(0)
  for _ in range(50):
    w = rng.random(3)
    w /= w.sum()
    p2d = tess._barycentric_to_local_2d(tuple(w))
    w_back = tess._local_2d_to_barycentric(p2d)
    assert w_back == pytest.approx(tuple(w), abs=1e-9)


def test_canonical_local_frame_round_trip_matches_face_geometry(icosahedron):
  _, _, faces = icosahedron
  face = faces[3]
  frame = GenericTessellation._canonical_local_frame(face)
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
    icosahedron):
  _, _, faces = icosahedron
  tess, _, _ = icosahedron
  face = faces[0]
  p1 = IsometricPoint(face, 0.2, 0.3)
  p2 = IsometricPoint(face, 0.5, 0.1)
  assert tess.geodesic_distance(p1, p2) == pytest.approx(
      p1.distance_from(p2))


def test_geodesic_distance_adjacent_faces_matches_projection(icosahedron):
  tess, _, faces = icosahedron
  face = faces[0]
  neighbor = face.face_on_edge(IsometricDirection.B)
  p1 = face.centroid_local_coords
  p2 = neighbor.centroid_local_coords
  expected = p1.project_onto_adjacent_grid(neighbor).distance_from(p2)
  assert tess.geodesic_distance(p1, p2) == pytest.approx(expected)


def test_geodesic_distance_is_symmetric_and_zero_for_same_point(icosahedron):
  tess, _, faces = icosahedron
  p1 = faces[0].centroid_local_coords
  p2 = faces[10].centroid_local_coords  # far away on the mesh
  d12 = tess.geodesic_distance(p1, p2)
  d21 = tess.geodesic_distance(p2, p1)
  assert d12 == pytest.approx(d21, rel=1e-6)
  assert d12 > 0
  assert tess.geodesic_distance(p1, p1) == pytest.approx(0.0, abs=1e-6)


def test_geodesic_distance_between_neighbors_is_the_right_order_of_magnitude(
    icosahedron):
  # The Heat Method is only approximate -- and known to be less accurate for
  # short/near-field distances, especially on a coarse (unsubdivided,
  # 20-face) mesh like this one -- so this only checks it's in the right
  # ballpark relative to the exact "project across the shared edge"
  # distance, not tightly accurate.
  tess, _, faces = icosahedron
  face = faces[0]
  neighbor = face.face_on_edge(IsometricDirection.B)
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

def test_shortest_path_same_face_is_trivial(icosahedron):
  tess, _, faces = icosahedron
  face = faces[0]
  p1 = IsometricPoint(face, 0.1, 0.1)
  p2 = IsometricPoint(face, 0.4, 0.2)
  assert tess.shortest_path(p1, p2) == [p1, p2]


def test_shortest_path_endpoints_match_request(icosahedron):
  tess, _, faces = icosahedron
  p1 = faces[0].centroid_local_coords
  p2 = faces[10].centroid_local_coords
  path = tess.shortest_path(p1, p2)
  assert len(path) >= 2
  assert path[0] is p1
  assert path[-1].grid is p2.grid
  assert path[-1].b == pytest.approx(p2.b, abs=1e-6)
  assert path[-1].s == pytest.approx(p2.s, abs=1e-6)


def test_shortest_path_stays_within_valid_local_coordinates(icosahedron):
  tess, _, faces = icosahedron
  p1 = faces[0].centroid_local_coords
  p2 = faces[15].centroid_local_coords
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
    icosahedron):
  tess, _, faces = icosahedron
  face = faces[0]
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
    icosahedron):
  _, _, faces = icosahedron
  tess, _, _ = icosahedron
  face = faces[2]
  alt = face.altitude
  p = IsometricPoint(face, 0.3 * alt, 0.3 * alt)
  tess.geodesically_canonicalize_point(p)
  assert p.grid is face
  assert p.b == pytest.approx(0.3 * alt)
  assert p.s == pytest.approx(0.3 * alt)


def test_geodesically_canonicalize_point_lands_in_bounds_far_away(
    icosahedron):
  tess, _, faces = icosahedron
  face = faces[5]
  alt = face.altitude
  p = IsometricPoint(face, -4.3 * alt, -2.1 * alt)
  tess.geodesically_canonicalize_point(p)
  assert -1e-6 <= p.b <= p.grid.altitude + 1e-6
  assert -1e-6 <= p.s <= p.grid.altitude + 1e-6
  assert -1e-6 <= p.d <= p.grid.altitude + 1e-6


def test_geodesically_canonicalize_point_handles_ray_through_a_vertex(
    icosahedron):
  # Regression test: a ray whose direction is exactly symmetric (here,
  # b == s) can pass precisely through a shared mesh vertex, where every
  # edge touching that vertex reports the same crossing distance -- and,
  # after landing exactly on it, every edge reports distance 0, leaving no
  # well-defined next edge to cross. `_DEGENERATE_RAY_NUDGE_RADIANS` guards
  # against exactly this.
  tess, _, faces = icosahedron
  face = faces[0]
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
  v1 = TessellationVertex([0, 0, 0])
  v2 = TessellationVertex([1, 0, 0])
  v3 = TessellationVertex([0.5, 1, 0])
  tess = GenericTessellation()
  face = tess.add_face([v1, v2, v3])
  p = IsometricPoint(face, -0.5, 0.3)
  with pytest.raises(EndOfMeshSurfaceException):
    tess.geodesically_canonicalize_point(p)


def test_geodesically_canonicalize_point_raises_if_max_steps_exceeded(
    icosahedron):
  tess, _, faces = icosahedron
  face = faces[0]
  alt = face.altitude
  p = IsometricPoint(face, -4.3 * alt, -2.1 * alt)
  with pytest.raises(RuntimeError):
    tess.geodesically_canonicalize_point(p, max_steps=1)
