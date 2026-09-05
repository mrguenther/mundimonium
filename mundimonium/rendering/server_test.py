import io

import numpy as np
import pytest

from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshFace
from mundimonium.coordinates.relaxable_lod_mesh import RelaxableLodMeshFace
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.rendering import flat_mesh_export
from mundimonium.rendering import server
from mundimonium.rendering.protocol import read_frame, write_frame
from mundimonium.rendering.server import dispatch, serve


def _camera_near_face_0(radius):
  """A camera position just outside `radius`'s sphere, near face 0."""
  probe = SphericalTessellation(
      radius=radius, frequency=1, face_type=LodMeshFace)
  centroid = probe.point_to_3d_position(IsometricPoint.center(probe.faces[0]))
  return [float(c) * 1.001 for c in centroid]


def test_get_mesh_returns_a_mesh_response_for_a_known_tessellation():
  header = {
      'type': 'get_mesh', 'id': '1', 'tessellation': 'spherical',
      'radius': 1.0, 'frequency': 1,
  }
  response_header, response_body = dispatch(header, b'')
  assert response_header['type'] == 'mesh'
  assert response_header['id'] == '1'
  assert response_header['vertex_count'] == 12
  assert response_header['face_count'] == 20
  assert len(response_body) == (
      response_header['positions_byte_length']
      + response_header['indices_byte_length'])


def test_get_mesh_returns_a_mesh_response_for_the_generic_demo():
  header = {'type': 'get_mesh', 'id': '1b', 'tessellation': 'generic'}
  response_header, response_body = dispatch(header, b'')
  assert response_header['type'] == 'mesh'
  assert response_header['id'] == '1b'
  assert response_header['vertex_count'] == 32
  assert response_header['face_count'] == 60
  assert len(response_body) == (
      response_header['positions_byte_length']
      + response_header['indices_byte_length'])

  # The demo tessellation's face type participates in the same LOD-tree
  # machinery the spherical case uses (see `relaxable_lod_mesh.py`), even
  # though this phase doesn't exercise LOD streaming for it.
  demo = server._get_generic_demo_tessellation()
  assert isinstance(demo.faces[0], RelaxableLodMeshFace)


def test_unknown_request_type_returns_an_error_not_a_crash():
  response_header, response_body = dispatch({'type': 'bogus', 'id': '2'}, b'')
  assert response_header['type'] == 'error'
  assert response_header['id'] == '2'
  assert response_body == b''


def test_bad_tessellation_kind_returns_an_error_not_a_crash():
  header = {'type': 'get_mesh', 'id': '3', 'tessellation': 'bogus'}
  response_header, response_body = dispatch(header, b'')
  assert response_header['type'] == 'error'
  assert response_header['id'] == '3'
  assert response_body == b''


def test_missing_type_returns_an_error_not_a_crash():
  response_header, _ = dispatch({'id': '4'}, b'')
  assert response_header['type'] == 'error'
  assert response_header['id'] == '4'


def test_serve_processes_one_request_then_stops_at_stream_close():
  request_buffer = io.BytesIO()
  write_frame(request_buffer, {
      'type': 'get_mesh', 'id': '1', 'tessellation': 'spherical',
      'radius': 1.0, 'frequency': 1,
  })
  request_buffer.seek(0)
  response_buffer = io.BytesIO()

  serve(request_buffer, response_buffer)

  response_buffer.seek(0)
  header, body = read_frame(response_buffer)
  assert header['type'] == 'mesh'
  assert len(body) == (
      header['positions_byte_length'] + header['indices_byte_length'])


def test_serve_processes_multiple_requests_in_sequence():
  request_buffer = io.BytesIO()
  for request_id in ('1', '2'):
    write_frame(request_buffer, {
        'type': 'get_mesh', 'id': request_id, 'tessellation': 'spherical',
        'radius': 1.0, 'frequency': 1,
    })
  request_buffer.seek(0)
  response_buffer = io.BytesIO()

  serve(request_buffer, response_buffer)

  response_buffer.seek(0)
  first_header, _ = read_frame(response_buffer)
  second_header, _ = read_frame(response_buffer)
  assert (first_header['id'], second_header['id']) == ('1', '2')


# ---------------------------------------------------------------------------
# get_lod_mesh / subdivide_sector
#
# Each test below uses its own `radius` value: `server.py`'s LOD tessellation
# cache is keyed by `(radius, frequency)` and persists for the process's
# lifetime (by design -- see `_get_lod_tessellation`), so sharing a radius
# across tests would leak subdivisions from one test into another.
# ---------------------------------------------------------------------------

def test_get_lod_mesh_returns_a_valid_response_with_sector_addresses():
  header = {
      'type': 'get_lod_mesh', 'id': '5', 'tessellation': 'spherical',
      'radius': 10.0, 'frequency': 1, 'camera_position': [0.0, 0.0, 1000.0],
  }
  response_header, response_body = dispatch(header, b'')
  assert response_header['type'] == 'lod_mesh'
  assert response_header['id'] == '5'
  assert response_header['face_count'] == len(response_header['sectors'])
  assert len(response_body) == (
      response_header['positions_byte_length']
      + response_header['indices_byte_length'])


def test_subdivide_sector_grows_exactly_that_sector():
  radius = 20.0
  header = {
      'type': 'get_lod_mesh', 'id': '6', 'tessellation': 'spherical',
      'radius': radius, 'frequency': 1,
      'camera_position': _camera_near_face_0(radius),
  }
  before_header, _ = dispatch(header, b'')
  face_count_before = before_header['face_count']
  target_sector = before_header['sectors'][0]

  ack_header, ack_body = dispatch({
      'type': 'subdivide_sector', 'id': '7', 'tessellation': 'spherical',
      'radius': radius, 'frequency': 1, 'sector': target_sector,
  }, b'')
  assert ack_header['type'] == 'subdivide_sector_ack'
  assert ack_header['id'] == '7'
  assert ack_body == b''

  after_header, _ = dispatch(header, b'')
  assert after_header['face_count'] > face_count_before


def test_subdivide_sector_on_a_bad_address_returns_an_error_not_a_crash():
  header = {
      'type': 'subdivide_sector', 'id': '8', 'tessellation': 'spherical',
      'radius': 30.0, 'frequency': 1, 'sector': {'face': 999, 'path': []},
  }
  response_header, response_body = dispatch(header, b'')
  assert response_header['type'] == 'error'
  assert response_header['id'] == '8'
  assert response_body == b''


# ---------------------------------------------------------------------------
# get_items
# ---------------------------------------------------------------------------

def test_get_items_returns_seeded_demo_items_but_not_the_scale_gated_one_far_away():
  header = {
      'type': 'get_items', 'id': '9', 'tessellation': 'spherical',
      'radius': 40.0, 'frequency': 1, 'camera_position': [0.0, 0.0, 4000.0],
  }
  response_header, response_body = dispatch(header, b'')
  assert response_header['type'] == 'items'
  assert response_header['id'] == '9'
  assert response_body == b''

  labels = {item['label'] for item in response_header['items']}
  assert {'Anchorhold', 'Millbrook', 'Stonegate'} <= labels
  assert 'Hiddenreach' not in labels
  for item in response_header['items']:
    assert item['kind'] == 'city'
    assert {'x', 'y', 'z'} <= item.keys()


def test_get_items_returns_seeded_generic_demo_items():
  header = {
      'type': 'get_items', 'id': '9b', 'tessellation': 'generic',
      'camera_position': [10.0, 0.0, 0.0],
  }
  response_header, response_body = dispatch(header, b'')
  assert response_header['type'] == 'items'
  assert response_header['id'] == '9b'
  assert response_body == b''

  # `GenericTessellation` has no generic notion of camera altitude yet
  # (see `item_export._scale_from_camera`'s docstring), so every seeded
  # item -- including the one that would be scale-gated on the spherical
  # demo -- is visible regardless of `camera_position`.
  labels = {item['label'] for item in response_header['items']}
  assert labels == {'Anchorhold', 'Millbrook', 'Stonegate', 'Hiddenreach'}
  for item in response_header['items']:
    assert item['kind'] == 'city'
    assert {'x', 'y', 'z'} <= item.keys()


def test_get_items_reveals_the_scale_gated_item_once_close():
  radius = 50.0
  header = {
      'type': 'get_items', 'id': '10', 'tessellation': 'spherical',
      'radius': radius, 'frequency': 1,
      'camera_position': _camera_near_face_0(radius),
  }
  response_header, _ = dispatch(header, b'')
  labels = {item['label'] for item in response_header['items']}
  assert 'Hiddenreach' in labels


# ---------------------------------------------------------------------------
# get_flat_mesh / get_flat_items
# ---------------------------------------------------------------------------

def test_get_flat_mesh_returns_a_valid_response_with_zero_z():
  header = {
      'type': 'get_flat_mesh', 'id': '11', 'tessellation': 'spherical',
      'radius': 60.0, 'frequency': 1, 'camera_position': [0.0, 0.0, 6000.0],
  }
  response_header, response_body = dispatch(header, b'')
  assert response_header['type'] == 'flat_mesh'
  assert response_header['id'] == '11'
  assert response_header['face_count'] == len(response_header['sectors'])
  assert len(response_body) == (
      response_header['positions_byte_length']
      + response_header['indices_byte_length'])

  positions = np.frombuffer(
      response_body[:response_header['positions_byte_length']],
      dtype=np.float32).reshape(-1, 3)
  assert np.all(positions[:, 2] == 0.0)
  assert {'face', 'b', 's'} <= response_header['center'].keys()
  assert {'e_x', 'e_y'} <= response_header['basis'].keys()


def test_get_flat_mesh_with_pan_offset_recenters_via_unflatten_point():
  radius = 60.0
  entry_header = {
      'type': 'get_flat_mesh', 'id': '11b', 'tessellation': 'spherical',
      'radius': radius, 'frequency': 1, 'camera_position': [0.0, 0.0, 6000.0],
  }
  entry_response, _ = dispatch(entry_header, b'')
  first_center = entry_response['center']
  first_basis = entry_response['basis']

  pan_offset = [5.0, -3.0]
  panned_header = {
      'type': 'get_flat_mesh', 'id': '11c', 'tessellation': 'spherical',
      'radius': radius, 'frequency': 1, 'camera_position': [0.0, 0.0, 6000.0],
      'center': first_center, 'pan_offset': pan_offset, 'basis': first_basis,
  }
  panned_response, _ = dispatch(panned_header, b'')

  # Independently compute the expected new center/basis the same way the
  # server should have, and confirm the response's own resolved values
  # match them.
  tessellation = server._get_lod_tessellation(radius, 1)
  reference_point = flat_mesh_export.center_from_json(
      tessellation, first_center)
  reference_basis = flat_mesh_export.basis_from_json(first_basis)
  expected_center, expected_basis = (
      tessellation.unflatten_point_and_transport_basis(
          reference_point, *pan_offset, reference_basis))
  actual_center = flat_mesh_export.center_from_json(
      tessellation, panned_response['center'])
  actual_basis = flat_mesh_export.basis_from_json(panned_response['basis'])
  assert tessellation.point_to_3d_position(
      actual_center) == pytest.approx(
          tessellation.point_to_3d_position(expected_center))
  assert actual_basis[0] == pytest.approx(expected_basis[0])
  assert actual_basis[1] == pytest.approx(expected_basis[1])


def test_get_flat_items_returns_seeded_demo_items_with_zero_z():
  header = {
      'type': 'get_flat_items', 'id': '12', 'tessellation': 'spherical',
      'radius': 70.0, 'frequency': 1, 'camera_position': [0.0, 0.0, 7000.0],
  }
  response_header, response_body = dispatch(header, b'')
  assert response_header['type'] == 'items'
  assert response_header['id'] == '12'
  assert response_body == b''

  labels = {item['label'] for item in response_header['items']}
  assert {'Anchorhold', 'Millbrook', 'Stonegate'} <= labels
  for item in response_header['items']:
    assert item['z'] == 0.0
  assert {'face', 'b', 's'} <= response_header['center'].keys()
  assert {'e_x', 'e_y'} <= response_header['basis'].keys()


def test_get_flat_items_with_pan_offset_uses_the_recentered_point():
  radius = 70.0
  camera_position = [0.0, 0.0, 7000.0]
  entry_header = {
      'type': 'get_flat_items', 'id': '12b', 'tessellation': 'spherical',
      'radius': radius, 'frequency': 1, 'camera_position': camera_position,
  }
  entry_response, _ = dispatch(entry_header, b'')

  pan_offset = [2.0, 1.5]
  panned_header = {
      'type': 'get_flat_items', 'id': '12c', 'tessellation': 'spherical',
      'radius': radius, 'frequency': 1, 'camera_position': camera_position,
      'center': entry_response['center'], 'pan_offset': pan_offset,
      'basis': entry_response['basis'],
  }
  panned_response, _ = dispatch(panned_header, b'')

  tessellation = server._get_lod_tessellation(radius, 1)
  reference_point = flat_mesh_export.center_from_json(
      tessellation, entry_response['center'])
  reference_basis = flat_mesh_export.basis_from_json(entry_response['basis'])
  expected_center, expected_basis = (
      tessellation.unflatten_point_and_transport_basis(
          reference_point, *pan_offset, reference_basis))
  actual_center = flat_mesh_export.center_from_json(
      tessellation, panned_response['center'])
  actual_basis = flat_mesh_export.basis_from_json(panned_response['basis'])
  assert tessellation.point_to_3d_position(
      actual_center) == pytest.approx(
          tessellation.point_to_3d_position(expected_center))
  assert actual_basis[0] == pytest.approx(expected_basis[0])
  assert actual_basis[1] == pytest.approx(expected_basis[1])


def _generic_demo_center():
  """The generic demo tessellation, plus a `center_to_json`-encoded point
  at face 0's own centroid, for tests that need a starting `center`
  request field (generic flat mode has no `camera_position` fallback --
  see `_resolve_generic_flat_center`).
  """
  tessellation = server._get_generic_demo_tessellation()
  center_point = IsometricPoint.center(tessellation.faces[0])
  return tessellation, flat_mesh_export.center_to_json(
      tessellation, center_point)


def test_get_flat_mesh_returns_a_valid_response_for_generic_with_zero_z():
  _tessellation, center = _generic_demo_center()
  header = {
      'type': 'get_flat_mesh', 'id': '11d', 'tessellation': 'generic',
      'center': center,
  }
  response_header, response_body = dispatch(header, b'')
  assert response_header['type'] == 'flat_mesh'
  assert response_header['id'] == '11d'
  assert response_header['face_count'] == len(response_header['sectors'])
  assert len(response_body) == (
      response_header['positions_byte_length']
      + response_header['indices_byte_length'])

  positions = np.frombuffer(
      response_body[:response_header['positions_byte_length']],
      dtype=np.float32).reshape(-1, 3)
  assert np.all(positions[:, 2] == 0.0)
  assert {'face', 'b', 's'} <= response_header['center'].keys()
  assert 'basis' not in response_header


def test_get_flat_mesh_with_pan_offset_recenters_via_unflatten_point_for_generic():
  tessellation, center = _generic_demo_center()
  entry_header = {
      'type': 'get_flat_mesh', 'id': '11e', 'tessellation': 'generic',
      'center': center,
  }
  entry_response, _ = dispatch(entry_header, b'')
  first_center = entry_response['center']

  pan_offset = [0.3, -0.2]
  panned_header = {
      'type': 'get_flat_mesh', 'id': '11f', 'tessellation': 'generic',
      'center': first_center, 'pan_offset': pan_offset,
  }
  panned_response, _ = dispatch(panned_header, b'')

  # Independently compute the expected new center the same way the server
  # should have, and confirm the response's own resolved value matches.
  reference_point = flat_mesh_export.center_from_json(
      tessellation, first_center)
  expected_center = tessellation.unflatten_point(reference_point, *pan_offset)
  actual_center = flat_mesh_export.center_from_json(
      tessellation, panned_response['center'])
  assert actual_center.grid is expected_center.grid
  assert actual_center.b == pytest.approx(expected_center.b)
  assert actual_center.s == pytest.approx(expected_center.s)


def test_get_flat_mesh_with_pan_offset_transports_orientation_for_generic():
  tessellation, center = _generic_demo_center()
  entry_header = {
      'type': 'get_flat_mesh', 'id': '11g', 'tessellation': 'generic',
      'center': center,
  }
  entry_response, _ = dispatch(entry_header, b'')
  assert 'orientation' not in entry_response  # nothing to transport yet

  pan_offset = [0.3, -0.2]
  panned_header = {
      'type': 'get_flat_mesh', 'id': '11h', 'tessellation': 'generic',
      'center': entry_response['center'], 'pan_offset': pan_offset,
  }
  panned_response, _ = dispatch(panned_header, b'')

  reference_point = flat_mesh_export.center_from_json(
      tessellation, entry_response['center'])
  _expected_center, expected_orientation = (
      tessellation.unflatten_point_and_transport_orientation(
          reference_point, *pan_offset, orientation=None))
  actual_orientation = flat_mesh_export.orientation_from_json(
      panned_response['orientation'])
  assert actual_orientation == pytest.approx(expected_orientation)

  # A third call, carrying the second response's own orientation forward,
  # should continue from it rather than silently resetting to identity.
  second_pan_offset = [-0.1, 0.15]
  second_panned_header = {
      'type': 'get_flat_mesh', 'id': '11i', 'tessellation': 'generic',
      'center': panned_response['center'], 'pan_offset': second_pan_offset,
      'orientation': panned_response['orientation'],
  }
  second_panned_response, _ = dispatch(second_panned_header, b'')

  second_reference_point = flat_mesh_export.center_from_json(
      tessellation, panned_response['center'])
  _expected_second_center, expected_second_orientation = (
      tessellation.unflatten_point_and_transport_orientation(
          second_reference_point, *second_pan_offset,
          orientation=expected_orientation))
  actual_second_orientation = flat_mesh_export.orientation_from_json(
      second_panned_response['orientation'])
  assert actual_second_orientation == pytest.approx(
      expected_second_orientation)


def test_get_flat_items_returns_seeded_generic_demo_items_with_zero_z():
  _tessellation, center = _generic_demo_center()
  header = {
      'type': 'get_flat_items', 'id': '12d', 'tessellation': 'generic',
      'center': center,
  }
  response_header, response_body = dispatch(header, b'')
  assert response_header['type'] == 'items'
  assert response_header['id'] == '12d'
  assert response_body == b''

  labels = {item['label'] for item in response_header['items']}
  assert 'Anchorhold' in labels  # seeded at face 0, this request's own center
  for item in response_header['items']:
    assert item['z'] == 0.0
  assert {'face', 'b', 's'} <= response_header['center'].keys()
  assert 'basis' not in response_header


def test_get_flat_items_with_pan_offset_uses_the_recentered_point_for_generic():
  tessellation, center = _generic_demo_center()
  entry_header = {
      'type': 'get_flat_items', 'id': '12e', 'tessellation': 'generic',
      'center': center,
  }
  entry_response, _ = dispatch(entry_header, b'')

  pan_offset = [0.2, 0.1]
  panned_header = {
      'type': 'get_flat_items', 'id': '12f', 'tessellation': 'generic',
      'center': entry_response['center'], 'pan_offset': pan_offset,
  }
  panned_response, _ = dispatch(panned_header, b'')

  reference_point = flat_mesh_export.center_from_json(
      tessellation, entry_response['center'])
  expected_center = tessellation.unflatten_point(reference_point, *pan_offset)
  actual_center = flat_mesh_export.center_from_json(
      tessellation, panned_response['center'])
  assert actual_center.grid is expected_center.grid
  assert actual_center.b == pytest.approx(expected_center.b)
  assert actual_center.s == pytest.approx(expected_center.s)
