import math
import random

import numpy as np
import pytest

from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation


@pytest.fixture(scope="module")
def sphere():
  return SphericalTessellation(radius=2.0, frequency=1, center=(1.0, 2.0, 3.0))


@pytest.fixture(scope="module")
def fine_sphere():
  return SphericalTessellation(radius=1.0, frequency=4)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

def test_base_icosahedron_has_20_faces_and_12_vertices():
  sphere = SphericalTessellation(radius=1.0, frequency=1)
  assert len(sphere._faces) == 20
  assert len(sphere._vertices) == 12


def test_frequency_scales_face_count_quadratically(fine_sphere):
  assert len(fine_sphere._faces) == 20 * 4 * 4


def test_all_vertices_lie_on_the_sphere_surface(fine_sphere):
  for v in fine_sphere._vertices:
    r = math.dist(v.projection_coordinates, (0.0, 0.0, 0.0))
    assert r == pytest.approx(1.0, abs=1e-9)


def test_vertices_are_offset_by_center(sphere):
  for v in sphere._vertices:
    r = math.dist(v.projection_coordinates, (1.0, 2.0, 3.0))
    assert r == pytest.approx(2.0, abs=1e-9)


def test_center_property_returns_the_actual_center(sphere):
  assert sphere.center == (1.0, 2.0, 3.0)


# ---------------------------------------------------------------------------
# _spherical_to_3d
# ---------------------------------------------------------------------------

def test_spherical_to_3d_pole_is_plus_z():
  sphere = SphericalTessellation(radius=1.0, frequency=1)
  v = sphere._spherical_to_3d(0.0, 0.0)
  assert v == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)


def test_spherical_to_3d_equator_directions():
  sphere = SphericalTessellation(radius=1.0, frequency=1)
  v_x = sphere._spherical_to_3d(math.pi / 2, 0.0)
  v_y = sphere._spherical_to_3d(math.pi / 2, math.pi / 2)
  assert v_x == pytest.approx([1.0, 0.0, 0.0], abs=1e-9)
  assert v_y == pytest.approx([0.0, 1.0, 0.0], abs=1e-9)


def test_spherical_to_3d_scales_with_radius():
  sphere = SphericalTessellation(radius=5.0, frequency=1)
  v = sphere._spherical_to_3d(math.pi / 2, 0.0)
  assert np.linalg.norm(v) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Face lookup / point construction and their inverse
# ---------------------------------------------------------------------------

def test_get_face_at_coords_returns_a_registered_face(fine_sphere):
  face = fine_sphere.get_face_at_coords(1.0, 2.0)
  assert face in fine_sphere._faces


def test_new_point_at_coords_has_valid_barycentric_weights(fine_sphere):
  point = fine_sphere.new_point_at_coords(1.0, 2.0)
  wb, ws, wd = point.barycentric
  assert wb + ws + wd == pytest.approx(1.0, abs=1e-9)
  assert wb >= -1e-9 and ws >= -1e-9 and wd >= -1e-9


def test_coords_at_point_is_the_inverse_of_new_point_at_coords(fine_sphere):
  rng = random.Random(42)
  for _ in range(200):
    colatitude = math.acos(1 - 2 * rng.random())
    longitude = rng.random() * 2 * math.pi
    point = fine_sphere.new_point_at_coords(colatitude, longitude)
    colatitude2, longitude2 = fine_sphere.coords_at_point(point)
    v1 = fine_sphere._spherical_to_3d(colatitude, longitude)
    v2 = fine_sphere._spherical_to_3d(colatitude2, longitude2)
    assert v1 == pytest.approx(v2, abs=1e-9)


def test_get_face_at_coords_agrees_with_brute_force_containment(fine_sphere):
  # Cross-checks the O(1) lookup against an exact great-circle half-space
  # test over every face, for a batch of random directions.
  def face_contains(face, direction):
    verts = [
        np.array(face.vertex_b.projection_coordinates) - fine_sphere.center,
        np.array(face.vertex_s.projection_coordinates) - fine_sphere.center,
        np.array(face.vertex_d.projection_coordinates) - fine_sphere.center,
    ]
    for i in range(3):
      j, k = (i + 1) % 3, (i + 2) % 3
      edge_normal = np.cross(verts[i], verts[j])
      if np.dot(edge_normal, direction) * np.dot(edge_normal, verts[k]) < -1e-9:
        return False
    return True

  rng = random.Random(7)
  for _ in range(30):
    colatitude = math.acos(1 - 2 * rng.random())
    longitude = rng.random() * 2 * math.pi
    direction = fine_sphere._spherical_to_3d(colatitude, longitude)
    fast_face = fine_sphere.get_face_at_coords(colatitude, longitude)
    assert face_contains(fast_face, direction)


# ---------------------------------------------------------------------------
# flatten_region
# ---------------------------------------------------------------------------

def test_flatten_region_preserves_distance_to_center(sphere):
  center = sphere.new_point_at_coords(1.0, 1.0)
  targets = [
      sphere.new_point_at_coords(0.5, 0.5),
      sphere.new_point_at_coords(2.0, 4.0),
      sphere.new_point_at_coords(1.2, 0.9),
  ]
  positions = sphere.flatten_region(center, targets)
  for target, (x, y) in zip(targets, positions):
    expected = sphere.geodesic_distance(center, target)
    assert math.hypot(x, y) == pytest.approx(expected, rel=1e-6)


def test_flatten_region_of_center_itself_is_the_origin(fine_sphere):
  center = fine_sphere.new_point_at_coords(1.0, 1.0)
  [(x, y)] = fine_sphere.flatten_region(center, [center])
  assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_flatten_region_from_the_pole_matches_colatitude_times_radius():
  sphere = SphericalTessellation(radius=3.0, frequency=3)
  center = sphere.new_point_at_coords(0.0, 0.0)  # north pole
  target = sphere.new_point_at_coords(0.4, 1.7)
  [(x, y)] = sphere.flatten_region(center, [target])
  assert math.hypot(x, y) == pytest.approx(0.4 * 3.0, rel=1e-6)


def test_flatten_region_targets_in_different_directions_get_different_angles(
    fine_sphere):
  center = fine_sphere.new_point_at_coords(1.0, 1.0)
  target_a = fine_sphere.new_point_at_coords(1.2, 1.0)
  target_b = fine_sphere.new_point_at_coords(1.0, 1.2)
  (xa, ya), (xb, yb) = fine_sphere.flatten_region(
      center, [target_a, target_b])
  assert math.atan2(ya, xa) != pytest.approx(math.atan2(yb, xb), abs=1e-3)


def test_flatten_region_raises_for_an_antipodal_target():
  sphere = SphericalTessellation(radius=1.0, frequency=2)
  center = sphere.new_point_at_coords(0.0, 0.0)
  target = sphere.new_point_at_coords(math.pi, 0.0)
  with pytest.raises(ValueError):
    sphere.flatten_region(center, [target])


def test_flatten_region_of_no_targets_returns_an_empty_list(fine_sphere):
  center = fine_sphere.new_point_at_coords(1.0, 1.0)
  assert fine_sphere.flatten_region(center, []) == []


# ---------------------------------------------------------------------------
# unflatten_point
# ---------------------------------------------------------------------------

def test_unflatten_point_round_trips_with_flatten_region(sphere):
  center = sphere.new_point_at_coords(1.0, 1.0)
  targets = [
      sphere.new_point_at_coords(0.5, 0.5),
      sphere.new_point_at_coords(2.0, 4.0),
      sphere.new_point_at_coords(1.2, 0.9),
  ]
  positions = sphere.flatten_region(center, targets)
  for target, (x, y) in zip(targets, positions):
    recovered = sphere.unflatten_point(center, x, y)
    assert sphere.point_to_3d_position(recovered) == pytest.approx(
        sphere.point_to_3d_position(target), abs=1e-6)


def test_unflatten_point_of_zero_offset_returns_center_unchanged(sphere):
  center = sphere.new_point_at_coords(1.0, 1.0)
  assert sphere.unflatten_point(center, 0.0, 0.0) is center


def test_unflatten_point_preserves_geodesic_distance(fine_sphere):
  center = fine_sphere.new_point_at_coords(1.0, 1.0)
  for x, y in [(0.3, 0.0), (0.0, -0.4), (0.2, 0.25)]:
    recovered = fine_sphere.unflatten_point(center, x, y)
    assert fine_sphere.geodesic_distance(center, recovered) == pytest.approx(
        math.hypot(x, y), rel=1e-6)


# ---------------------------------------------------------------------------
# geodesic_distance / shortest_path
# ---------------------------------------------------------------------------

def test_geodesic_distance_of_a_point_to_itself_is_zero(fine_sphere):
  p = fine_sphere.new_point_at_coords(1.0, 1.0)
  assert fine_sphere.geodesic_distance(p, p) == pytest.approx(0.0, abs=1e-9)


def test_geodesic_distance_between_antipodal_points_is_pi_r():
  sphere = SphericalTessellation(radius=3.0, frequency=2)
  p1 = sphere.new_point_at_coords(0.0, 0.0)  # north pole
  p2 = sphere.new_point_at_coords(math.pi, 0.0)  # south pole
  assert sphere.geodesic_distance(p1, p2) == pytest.approx(
      math.pi * 3.0, abs=1e-6)


def test_geodesic_distance_between_perpendicular_points_is_quarter_circle():
  sphere = SphericalTessellation(radius=1.0, frequency=3)
  p1 = sphere.new_point_at_coords(math.pi / 2, 0.0)
  p2 = sphere.new_point_at_coords(math.pi / 2, math.pi / 2)
  assert sphere.geodesic_distance(p1, p2) == pytest.approx(
      math.pi / 2, abs=1e-6)


def test_shortest_path_endpoints_and_radius(fine_sphere):
  p1 = fine_sphere.new_point_at_coords(0.5, 0.5)
  p2 = fine_sphere.new_point_at_coords(2.0, 4.0)
  path = fine_sphere.shortest_path(p1, p2, num_samples=10)
  assert len(path) == 10
  center = np.array(fine_sphere.center)
  for pos in path:
    assert np.linalg.norm(pos - center) == pytest.approx(
        fine_sphere.radius, abs=1e-9)


def test_shortest_path_length_matches_geodesic_distance(fine_sphere):
  p1 = fine_sphere.new_point_at_coords(0.5, 0.5)
  p2 = fine_sphere.new_point_at_coords(2.0, 4.0)
  path = fine_sphere.shortest_path(p1, p2, num_samples=200)
  total = sum(
      np.linalg.norm(np.array(path[i + 1]) - np.array(path[i]))
      for i in range(len(path) - 1))
  assert total == pytest.approx(
      fine_sphere.geodesic_distance(p1, p2), rel=1e-3)


# ---------------------------------------------------------------------------
# geodesically_canonicalize_point
# ---------------------------------------------------------------------------

def test_geodesically_canonicalize_point_preserves_direction(fine_sphere):
  point = fine_sphere.new_point_at_coords(1.0, 1.0)
  wb, ws, wd = point.barycentric
  alt = point.grid.altitude
  # Extrapolate far outside the current face.
  scale = 4.0
  out_point = point.grid.centroid_local_coords
  out_point.update(
      grid=point.grid,
      b=(1 / 3 + scale * (wb - 1 / 3)) * alt,
      s=(1 / 3 + scale * (ws - 1 / 3)) * alt,
  )

  direction_before = fine_sphere._point_to_3d_unit(out_point).copy()
  fine_sphere.geodesically_canonicalize_point(out_point)
  direction_after = fine_sphere._point_to_3d_unit(out_point)

  assert direction_after == pytest.approx(direction_before, abs=1e-9)
  alt2 = out_point.grid.altitude
  assert -1e-6 <= out_point.b <= alt2 + 1e-6
  assert -1e-6 <= out_point.s <= alt2 + 1e-6
  assert -1e-6 <= out_point.d <= alt2 + 1e-6


def test_geodesically_canonicalize_point_is_idempotent_when_in_bounds(
    fine_sphere):
  point = fine_sphere.new_point_at_coords(1.0, 1.0)
  b_before, s_before, grid_before = point.b, point.s, point.grid
  fine_sphere.geodesically_canonicalize_point(point)
  assert point.grid is grid_before
  assert point.b == pytest.approx(b_before)
  assert point.s == pytest.approx(s_before)


# ---------------------------------------------------------------------------
# shortest_path_by_segment
# ---------------------------------------------------------------------------

def test_shortest_path_by_segment_same_face_is_a_single_segment(fine_sphere):
  face = fine_sphere._faces[0]
  alt = face.altitude
  p1 = IsometricPoint(face, 0.1 * alt, 0.1 * alt)
  p2 = IsometricPoint(face, 0.3 * alt, 0.2 * alt)
  assert fine_sphere.shortest_path_by_segment(p1, p2) == [(p1, p2)]


def test_shortest_path_by_segment_identical_point_is_a_single_segment(
    fine_sphere):
  p1 = fine_sphere.new_point_at_coords(1.0, 1.0)
  p2 = fine_sphere.new_point_at_coords(1.0, 1.0)
  segs = fine_sphere.shortest_path_by_segment(p1, p2)
  assert len(segs) == 1


def test_shortest_path_by_segment_far_apart_endpoints_match_request(
    fine_sphere):
  p1 = fine_sphere.new_point_at_coords(0.5, 0.5)
  p2 = fine_sphere.new_point_at_coords(2.0, 4.0)
  segs = fine_sphere.shortest_path_by_segment(p1, p2)
  assert len(segs) > 1
  assert segs[0][0].grid is p1.grid
  assert segs[0][0].b == p1.b and segs[0][0].s == p1.s
  assert segs[-1][1].grid is p2.grid
  assert segs[-1][1].b == p2.b and segs[-1][1].s == p2.s


def test_shortest_path_by_segment_each_segment_stays_within_one_grid(
    fine_sphere):
  p1 = fine_sphere.new_point_at_coords(0.5, 0.5)
  p2 = fine_sphere.new_point_at_coords(2.0, 4.0)
  segs = fine_sphere.shortest_path_by_segment(p1, p2)
  for a, b in segs:
    assert a.grid is b.grid


def test_shortest_path_by_segment_boundary_crossings_are_physically_continuous(
    fine_sphere):
  p1 = fine_sphere.new_point_at_coords(0.5, 0.5)
  p2 = fine_sphere.new_point_at_coords(2.0, 4.0)
  segs = fine_sphere.shortest_path_by_segment(p1, p2)
  for (_, exit_point), (entry_point, _) in zip(segs, segs[1:]):
    v_exit = fine_sphere._point_to_3d_unit(exit_point)
    v_entry = fine_sphere._point_to_3d_unit(entry_point)
    assert v_exit == pytest.approx(v_entry, abs=1e-6)


def test_shortest_path_by_segment_has_no_minuscule_segments(fine_sphere):
  p1 = fine_sphere.new_point_at_coords(0.5, 0.5)
  p2 = fine_sphere.new_point_at_coords(2.0, 4.0)
  segs = fine_sphere.shortest_path_by_segment(p1, p2)
  for a, b in segs:
    assert a.distance_from(b) > 1e-6 * a.grid.altitude


def test_shortest_path_by_segment_handles_exact_antipodal_points():
  # Regression test: exactly-antipodal points make the great-circle SLERP
  # formula a 0/0 indeterminate form (every great circle through one point
  # also passes through its antipode), which must be resolved via the
  # deterministic reference-direction fallback rather than raising or
  # producing NaNs.
  sphere = SphericalTessellation(radius=1.0, frequency=3)
  p1 = sphere.new_point_at_coords(0.4, 0.0)
  p2 = sphere.new_point_at_coords(math.pi - 0.4, math.pi)
  segs = sphere.shortest_path_by_segment(p1, p2)
  assert segs[0][0].grid is p1.grid
  assert segs[-1][1].grid is p2.grid
  for a, b in segs:
    assert a.grid is b.grid
