from __future__ import annotations

from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.rendering.mesh_export import tessellation_to_buffers
from mundimonium.rendering.protocol import read_frame, write_frame

from typing import BinaryIO, Callable
import sys
import traceback


def _handle_get_mesh(header: dict) -> tuple[dict, bytes]:
  """Handles a `get_mesh` request.

  Args:
    header: The request header. Must include `tessellation` (currently
      only `'spherical'` is supported) plus that tessellation type's own
      construction parameters (`radius`, `frequency` for `'spherical'`).

  Returns:
    A `(response_header, response_body)` pair -- a `mesh` response with
    the exported geometry.

  Raises:
    ValueError: If `header["tessellation"]` isn't a supported kind.
  """
  tessellation_kind = header.get('tessellation')
  if tessellation_kind == 'spherical':
    tessellation = SphericalTessellation(
        radius=header.get('radius', 1.0),
        frequency=header.get('frequency', 1))
  else:
    raise ValueError(f"Unknown tessellation kind: {tessellation_kind!r}")

  positions, indices, vertex_count, face_count = tessellation_to_buffers(
      tessellation)
  response_header = {
      'type': 'mesh',
      'id': header.get('id'),
      'vertex_count': vertex_count,
      'face_count': face_count,
      'positions_byte_length': len(positions),
      'indices_byte_length': len(indices),
  }
  return response_header, positions + indices


_HANDLERS: dict[str, Callable[[dict], tuple[dict, bytes]]] = {
    'get_mesh': _handle_get_mesh,
}


def dispatch(header: dict, body: bytes) -> tuple[dict, bytes]:
  """Handles one request.

  Never raises -- failures become `error` responses instead, so a single
  bad request can't take down the persistent subprocess.

  Args:
    header: The request header. Must include `type`.
    body: The request body (unused by any handler so far, but part of
      the signature so handlers needing one can be added later).

  Returns:
    A `(response_header, response_body)` pair.
  """
  request_type = header.get('type')
  handler = _HANDLERS.get(request_type)
  if handler is None:
    return (
        {'type': 'error', 'id': header.get('id'),
         'message': f"Unknown request type: {request_type!r}"},
        b'')
  try:
    return handler(header)
  except Exception as error:  # A request boundary: report, don't crash.
    traceback.print_exc(file=sys.stderr)
    return (
        {'type': 'error', 'id': header.get('id'), 'message': str(error)},
        b'')


def serve(input_stream: BinaryIO, output_stream: BinaryIO) -> None:
  """Runs the request/response loop until `input_stream` closes.

  Args:
    input_stream: Binary stream to read framed requests from (e.g.
      `sys.stdin.buffer`).
    output_stream: Binary stream to write framed responses to (e.g.
      `sys.stdout.buffer`).
  """
  while True:
    try:
      header, body = read_frame(input_stream)
    except EOFError:
      return
    response_header, response_body = dispatch(header, body)
    write_frame(output_stream, response_header, response_body)
