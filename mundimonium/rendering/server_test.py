import io

from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshFace
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
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
