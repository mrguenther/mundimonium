import numpy as np
import pytest

from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshFace
from mundimonium.coordinates.nesting_iso_grid import SectorItem
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.rendering.item_export import iter_visible_items


def _make_sphere(radius=1.0, frequency=1):
  return SphericalTessellation(
      radius=radius, frequency=frequency, face_type=LodMeshFace)


def _far_camera(tess, altitude_factor=1000.0):
  """A camera far above the sphere's surface (`altitude_factor * radius`,
  in an arbitrary direction) -- low `scale` regardless of direction."""
  return [
      tess.center[0], tess.center[1],
      tess.center[2] + tess.radius * (1.0 + altitude_factor)]


def _near_camera(tess, face, altitude_factor=0.4):
  """A camera close above `face`'s centroid (`altitude_factor * radius`
  above the surface) -- high `scale`. `0.4` matches `OrbitCameraController`
  `minDistance = 1.5`'s closest achievable altitude of `0.5 * radius`,
  comfortably past the `min_scale=2.0` a scale-gated demo item might use.
  """
  center = np.array(tess.center, dtype=np.float64)
  centroid = tess.point_to_3d_position(IsometricPoint.center(face))
  direction = (centroid - center) / np.linalg.norm(centroid - center)
  return center + direction * tess.radius * (1.0 + altitude_factor)


def test_default_scale_item_is_visible_regardless_of_camera_distance():
  tess = _make_sphere()
  face = tess.faces[0]
  face.add_item(SectorItem(
      position=IsometricPoint.center(face), payload={'kind': 'city'}))

  far_payloads = [
      payload for payload, *_ in iter_visible_items(tess, _far_camera(tess))]
  near_payloads = [
      payload for payload, *_
      in iter_visible_items(tess, _near_camera(tess, face))]

  assert {'kind': 'city'} in far_payloads
  assert {'kind': 'city'} in near_payloads


def test_scale_gated_item_only_visible_once_close_enough():
  tess = _make_sphere()
  face = tess.faces[1]
  face.add_item(SectorItem(
      position=IsometricPoint.center(face), payload={'kind': 'tavern'},
      min_scale=2.0))

  far_payloads = [
      payload for payload, *_ in iter_visible_items(tess, _far_camera(tess))]
  near_payloads = [
      payload for payload, *_
      in iter_visible_items(tess, _near_camera(tess, face))]

  assert {'kind': 'tavern'} not in far_payloads
  assert {'kind': 'tavern'} in near_payloads


def test_item_on_a_nested_sector_is_still_found():
  tess = _make_sphere()
  face = tess.faces[2]
  face.subdivide(resolution=2)
  child = face.child_at(0, 0, False)
  child.add_item(SectorItem(
      position=IsometricPoint.center(child), payload={'kind': 'village'}))

  payloads = [
      payload for payload, *_
      in iter_visible_items(tess, _near_camera(tess, face))]

  assert {'kind': 'village'} in payloads


def test_visible_item_positions_lie_approximately_on_the_sphere():
  tess = _make_sphere(radius=3.0)
  face = tess.faces[3]
  face.add_item(SectorItem(
      position=IsometricPoint.center(face), payload={'kind': 'city'}))

  [(_, x, y, z)] = list(
      iter_visible_items(tess, _near_camera(tess, face)))
  position = np.array([x, y, z]) - np.array(tess.center)

  assert np.linalg.norm(position) == pytest.approx(3.0, rel=1e-4)


def test_payload_identity_is_preserved():
  tess = _make_sphere()
  face = tess.faces[4]
  payload = object()
  face.add_item(SectorItem(position=IsometricPoint.center(face), payload=payload))

  [(returned_payload, *_)] = list(
      iter_visible_items(tess, _near_camera(tess, face)))

  assert returned_payload is payload
