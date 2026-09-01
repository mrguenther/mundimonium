import math
import random

import numpy as np
import pytest

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
