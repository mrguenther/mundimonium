import io

from mundimonium.rendering.protocol import read_frame, write_frame
from mundimonium.rendering.server import dispatch, serve


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
