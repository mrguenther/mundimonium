import math

import numpy as np
import pytest

from mundimonium.coordinates.generic_tessellation import (
    GenericTessellation, RelaxableFace, RelaxableVertex,
)
from mundimonium.coordinates.hyperbolic_tessellation import HyperbolicTessellation
from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshFace
from mundimonium.coordinates.nesting_iso_grid import SectorItem
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.coordinates.stellated_icosahedron import (
    build_stellated_icosahedron,
)
from mundimonium.coordinates.tessellation import TessellationVertex
from mundimonium.rendering.flat_mesh_export import (
    center_point_from_camera, flatten_frontier_to_buffers,
    flatten_visible_items, nearby_faces, poincare_frontier_to_buffers,
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
      flatten_frontier_to_buffers(
          tess, center, frontier, tess.tangent_basis_at(center)))

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
      tess, center, frontier, tess.tangent_basis_at(center))
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
      tess, center, frontier, tess.tangent_basis_at(center))
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

  basis = tess.tangent_basis_at(center)
  [(payload, x, y)] = flatten_visible_items(
      tess, center, camera_position, basis)

  assert payload == {'kind': 'city'}
  [(expected_x, expected_y)] = tess.flatten_region(
      center, [item_point], basis)
  assert (x, y) == pytest.approx((expected_x, expected_y), rel=1e-6)


def test_flatten_visible_items_returns_an_empty_list_with_no_items():
  tess = _make_sphere(frequency=1)
  camera_position = [0.0, 0.0, 10.0]
  center = center_point_from_camera(tess, camera_position)

  assert flatten_visible_items(
      tess, center, camera_position, tess.tangent_basis_at(center)) == []


# ---------------------------------------------------------------------------
# nearby_faces
# ---------------------------------------------------------------------------

def test_nearby_faces_for_spherical_excludes_most_of_the_mesh():
  tess = _make_sphere(radius=1.0, frequency=3)  # 180 faces total
  center = center_point_from_camera(tess, [0.0, 0.0, 1.3])

  nearby = nearby_faces(tess, center)

  assert center.grid in nearby
  assert 0 < len(nearby) < len(tess.faces)


def test_nearby_faces_for_spherical_costs_the_same_regardless_of_mesh_size():
  # A regression test for the actual point of using a bounded BFS instead
  # of scanning every face: the neighborhood size (and therefore the
  # work done) shouldn't grow just because the mesh has more faces overall.
  small = _make_sphere(radius=1.0, frequency=3)      # 180 faces
  large = _make_sphere(radius=1.0, frequency=10)     # 2000 faces

  small_center = center_point_from_camera(small, [0.0, 0.0, 1.3])
  large_center = center_point_from_camera(large, [0.0, 0.0, 1.3])

  small_count = len(nearby_faces(small, small_center))
  large_count = len(nearby_faces(large, large_center))

  # Not required to match exactly (face sizes differ slightly by
  # frequency), but should be the same order of magnitude, not scaling
  # with the roughly 11x difference in total face count.
  assert large_count < small_count * 2


def test_nearby_faces_for_generic_matches_the_precomputed_nearby_faces():
  tess, _vertices, _faces = build_stellated_icosahedron(
      face_type=RelaxableFace, vertex_type=RelaxableVertex)
  center = IsometricPoint.center(tess.faces[0])

  nearby = nearby_faces(tess, center)

  assert set(nearby) == set(tess.faces[0].nearby_faces)


def test_faces_within_hops_reaches_only_the_expected_ring():
  # Four faces in a fan around a shared central vertex `v1`, each face i
  # sharing an edge only with face i-1/i+1 -- a small, hand-built mesh
  # with a known, exact adjacency structure to check the hop-count cutoff
  # against directly.
  from mundimonium.rendering.flat_mesh_export import _faces_within_hops

  v1 = TessellationVertex([0.0, 0.0, 0.0])
  v2 = TessellationVertex([1.0, 0.0, 0.0])
  v3 = TessellationVertex([0.5, 1.0, 0.0])
  v4 = TessellationVertex([-0.5, 1.0, 0.0])
  v5 = TessellationVertex([-1.0, 0.0, 0.0])
  tess = GenericTessellation()
  face_a = tess.add_face([v1, v2, v3])
  face_b = tess.add_face([v1, v3, v4])
  face_c = tess.add_face([v1, v4, v5])

  assert _faces_within_hops(face_a, 0) == [face_a]
  assert set(_faces_within_hops(face_a, 1)) == {face_a, face_b}
  assert set(_faces_within_hops(face_a, 2)) == {face_a, face_b, face_c}


# ---------------------------------------------------------------------------
# poincare_frontier_to_buffers
# ---------------------------------------------------------------------------

def test_poincare_frontier_to_buffers_dedups_shared_vertices():
  tess = HyperbolicTessellation()
  faces = tess.faces_within_hops(tess.reference_point.grid, 2)

  positions_bytes, indices_bytes, vertex_count, face_count = (
      poincare_frontier_to_buffers(tess, faces))

  assert face_count == len(faces)
  assert len(indices_bytes) == face_count * 3 * 4  # uint32
  assert len(positions_bytes) == vertex_count * 3 * 4  # float32
  # Every face is a triangle sharing each of its edges with a neighbor, so
  # deduplicated vertex count must be well below one-per-triangle-corner.
  assert vertex_count < 3 * face_count


def test_poincare_frontier_to_buffers_positions_are_within_the_unit_disk():
  tess = HyperbolicTessellation()
  faces = tess.faces_within_hops(tess.reference_point.grid, 6)

  positions_bytes, _, _, _ = poincare_frontier_to_buffers(tess, faces)
  positions = np.frombuffer(positions_bytes, dtype=np.float32).reshape(-1, 3)

  np.testing.assert_array_equal(positions[:, 2], 0.0)
  radii = np.hypot(positions[:, 0], positions[:, 1])
  assert np.all(radii < 1.0)
