import math

import numpy as np
import pytest

from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshFace
from mundimonium.coordinates.nesting_iso_grid import SectorItem
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.rendering.flat_mesh_export import (
    center_point_from_camera, flatten_frontier_to_buffers,
    flatten_visible_items,
)
from mundimonium.rendering.lod_mesh_export import select_frontier


def _make_sphere(radius=1.0, frequency=1):
  return SphericalTessellation(
      radius=radius, frequency=frequency, face_type=LodMeshFace)


def test_center_point_from_camera_matches_the_cameras_direction():
  tess = _make_sphere(radius=2.0)
  center = center_point_from_camera(tess, [0.0, 0.0, 10.0])  # north pole

  colatitude, _longitude = tess.coords_at_point(center)
  assert colatitude == pytest.approx(0.0, abs=1e-6)


def test_center_point_from_camera_round_trips_an_arbitrary_direction():
  tess = _make_sphere(radius=3.0, frequency=2)
  camera_position = [5.0, -2.0, 1.0]
  center = center_point_from_camera(tess, camera_position)

  direction = np.array(camera_position) / np.linalg.norm(camera_position)
  expected_position = np.array(tess.center) + 3.0 * direction
  position = tess.point_to_3d_position(center)
  np.testing.assert_allclose(position, expected_position, atol=1e-6)


def test_flatten_frontier_buffer_lengths_and_counts_agree():
  tess = _make_sphere(frequency=1)
  camera_position = [0.0, 0.0, 10.0]
  center = center_point_from_camera(tess, camera_position)
  frontier = select_frontier(tess, camera_position)

  positions_bytes, indices_bytes, vertex_count, face_count = (
      flatten_frontier_to_buffers(tess, center, frontier))

  assert face_count == len(frontier)
  assert vertex_count == 3 * len(frontier)
  assert len(positions_bytes) == vertex_count * 3 * 4  # float32
  assert len(indices_bytes) == face_count * 3 * 4  # uint32


def test_flatten_frontier_positions_have_zero_z():
  tess = _make_sphere(frequency=1)
  camera_position = [0.0, 0.0, 10.0]
  center = center_point_from_camera(tess, camera_position)
  frontier = select_frontier(tess, camera_position)

  positions_bytes, _, _, _ = flatten_frontier_to_buffers(
      tess, center, frontier)
  positions = np.frombuffer(positions_bytes, dtype=np.float32).reshape(-1, 3)

  np.testing.assert_array_equal(positions[:, 2], 0.0)


def test_flatten_frontier_positions_preserve_distance_to_center():
  # Independent cross-check via `geodesic_distance` (not `flatten_region`
  # itself), the same style as `spherical_tessellation_test.py`'s
  # `test_flatten_region_preserves_distance_to_center`. Deliberately not
  # `Tessellation.distance`, which takes a faster, approximate shortcut
  # for nearby points that `flatten_region` doesn't match exactly (see
  # `Tessellation.flatten_region`'s docstring).
  tess = _make_sphere(frequency=1)
  camera_position = [0.0, 0.0, 10.0]
  center = center_point_from_camera(tess, camera_position)
  frontier = select_frontier(tess, camera_position)

  positions_bytes, _, _, _ = flatten_frontier_to_buffers(
      tess, center, frontier)
  positions = np.frombuffer(positions_bytes, dtype=np.float32).reshape(-1, 3)

  index = 0
  for sector, _address in frontier:
    altitude = sector.altitude
    local_corners = (
        IsometricPoint(sector, altitude, 0),
        IsometricPoint(sector, 0, altitude),
        IsometricPoint(sector, 0, 0),
    )
    for corner in local_corners:
      root_corner = sector.project_onto_root_grid(corner)
      expected = tess.geodesic_distance(center, root_corner)
      x, y, _z = positions[index]
      assert math.hypot(x, y) == pytest.approx(expected, rel=1e-3)
      index += 1


def test_flatten_visible_items_returns_flattened_positions():
  tess = _make_sphere(frequency=1)
  face = tess.faces[0]
  item_point = IsometricPoint.center(face)
  face.add_item(SectorItem(position=item_point, payload={'kind': 'city'}))
  camera_position = [0.0, 0.0, 10.0]
  center = center_point_from_camera(tess, camera_position)

  [(payload, x, y)] = flatten_visible_items(tess, center, camera_position)

  assert payload == {'kind': 'city'}
  [(expected_x, expected_y)] = tess.flatten_region(center, [item_point])
  assert (x, y) == pytest.approx((expected_x, expected_y), rel=1e-6)


def test_flatten_visible_items_returns_an_empty_list_with_no_items():
  tess = _make_sphere(frequency=1)
  camera_position = [0.0, 0.0, 10.0]
  center = center_point_from_camera(tess, camera_position)

  assert flatten_visible_items(tess, center, camera_position) == []
