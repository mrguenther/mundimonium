import numpy as np

from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshFace
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.rendering.lod_mesh_export import (
    SectorAddress, lod_frontier_to_buffers, resolve_sector_address,
    select_frontier,
)

# Small relative to the sphere's overall size, so only a sector very close
# to the camera ever wants to subdivide -- keeps tests independent of the
# module's own default threshold/depth.
_THRESHOLD = 0.5
_MAX_DEPTH = 2


def _make_sphere(radius=1.0, frequency=1):
  return SphericalTessellation(
      radius=radius, frequency=frequency, face_type=LodMeshFace)


def _camera_near(tess, face, factor=1.001):
  """A camera position just outside the sphere, near `face`'s centroid."""
  center = np.array(tess.center, dtype=np.float64)
  centroid = tess.point_to_3d_position(IsometricPoint.center(face))
  return center + (centroid - center) * factor


def test_auto_subdivide_false_never_grows_the_tree():
  tess = _make_sphere()
  face = tess.faces[0]

  frontier = select_frontier(
      tess, _camera_near(tess, face), threshold=_THRESHOLD,
      max_depth=_MAX_DEPTH, auto_subdivide=False)

  assert face.children is None
  assert (face, SectorAddress(0)) in frontier


def test_auto_subdivide_true_subdivides_near_camera_and_stays_coarse_far_away():
  tess = _make_sphere()
  near_face = tess.faces[0]
  far_face = tess.faces[10]

  frontier = select_frontier(
      tess, _camera_near(tess, near_face), threshold=_THRESHOLD,
      max_depth=_MAX_DEPTH, auto_subdivide=True)

  assert near_face.children is not None
  assert far_face.children is None
  frontier_sectors = {sector for sector, _address in frontier}
  assert far_face in frontier_sectors
  assert near_face not in frontier_sectors  # replaced by its children


def test_auto_subdivide_true_is_idempotent_for_an_unchanged_camera():
  tess = _make_sphere()
  face = tess.faces[1]
  camera = _camera_near(tess, face)

  first = select_frontier(
      tess, camera, threshold=_THRESHOLD, max_depth=_MAX_DEPTH,
      auto_subdivide=True)
  second = select_frontier(
      tess, camera, threshold=_THRESHOLD, max_depth=_MAX_DEPTH,
      auto_subdivide=True)

  assert first == second


def test_zooming_back_out_uses_a_coarser_frontier_without_rebuilding():
  tess = _make_sphere()
  face = tess.faces[2]

  select_frontier(
      tess, _camera_near(tess, face), threshold=_THRESHOLD,
      max_depth=_MAX_DEPTH, auto_subdivide=True)
  children_after_zooming_in = face.children
  assert children_after_zooming_in is not None

  frontier_far = select_frontier(
      tess, _camera_near(tess, face, factor=20.0), threshold=_THRESHOLD,
      max_depth=_MAX_DEPTH, auto_subdivide=True)

  assert face.children is children_after_zooming_in  # nothing rebuilt
  assert (face, SectorAddress(2)) in frontier_far


def test_sector_address_resolves_top_level_and_nested_sectors():
  tess = _make_sphere()
  face = tess.faces[3]
  address = SectorAddress(3)
  assert resolve_sector_address(tess, address) is face

  face.subdivide(resolution=2)
  child = face.child_at(0, 0, False)
  child_address = address.child(0, 0, False)
  assert resolve_sector_address(tess, child_address) is child

  child.subdivide(resolution=2)
  grandchild = child.child_at(1, 0, False)
  grandchild_address = child_address.child(1, 0, False)
  assert resolve_sector_address(tess, grandchild_address) is grandchild


def test_sector_address_json_round_trips():
  address = SectorAddress(3, ((0, 0, False), (1, 0, True)))
  assert SectorAddress.from_json(address.to_json()) == address


def test_lod_frontier_buffer_lengths_and_counts_agree():
  tess = _make_sphere()
  face = tess.faces[5]

  frontier = select_frontier(
      tess, _camera_near(tess, face), threshold=_THRESHOLD,
      max_depth=_MAX_DEPTH, auto_subdivide=True)
  assert len(frontier) > 20  # `face` was replaced by more than one triangle

  positions_bytes, indices_bytes, vertex_count, face_count = (
      lod_frontier_to_buffers(tess, frontier))

  assert face_count == len(frontier)
  assert vertex_count == 3 * len(frontier)
  assert len(positions_bytes) == vertex_count * 3 * 4  # float32
  assert len(indices_bytes) == face_count * 3 * 4  # uint32


def test_lod_frontier_positions_lie_approximately_on_the_sphere():
  tess = _make_sphere(radius=3.0)
  face = tess.faces[7]

  frontier = select_frontier(
      tess, _camera_near(tess, face), threshold=_THRESHOLD,
      max_depth=_MAX_DEPTH, auto_subdivide=True)
  positions_bytes, _, _, _ = lod_frontier_to_buffers(tess, frontier)
  positions = np.frombuffer(positions_bytes, dtype=np.float32).reshape(-1, 3)

  radii = np.linalg.norm(positions - np.array(tess.center), axis=1)
  np.testing.assert_allclose(radii, 3.0, rtol=1e-4)
