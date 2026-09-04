from __future__ import annotations

from mundimonium.coordinates.lod_mesh import LodMeshFace
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.rendering import lod_mesh_export
from mundimonium.rendering.mesh_export import tessellation_to_buffers
from mundimonium.rendering.protocol import read_frame, write_frame

from typing import BinaryIO, Callable
import sys
import traceback


# Reused across `get_lod_mesh`/`subdivide_sector` requests within a
# subprocess's lifetime, keyed by `(radius, frequency)`, so that a sector's
# subdivisions persist between requests instead of being rebuilt from
# scratch every time the camera moves.
_lod_tessellations: dict[tuple[float, int], SphericalTessellation] = {}


def _get_lod_tessellation(radius: float, frequency: int) -> SphericalTessellation:
  """Returns the persistent LOD tessellation for `(radius, frequency)`."""
  key = (radius, frequency)
  tessellation = _lod_tessellations.get(key)
  if tessellation is None:
    tessellation = SphericalTessellation(
        radius=radius, frequency=frequency, face_type=LodMeshFace)
    _lod_tessellations[key] = tessellation
  return tessellation


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


def _handle_get_lod_mesh(header: dict) -> tuple[dict, bytes]:
  """Handles a `get_lod_mesh` request.

  Args:
    header: The request header. Must include `tessellation` (currently
      only `'spherical'` is supported), that tessellation type's own
      construction parameters (`radius`, `frequency`), and
      `camera_position`. May include `auto_subdivide` (default `False`).

  Returns:
    A `(response_header, response_body)` pair -- an `lod_mesh` response
    with the currently-selected level of detail, plus each rendered
    sector's own address (for a later `subdivide_sector` request).

  Raises:
    ValueError: If `header["tessellation"]` isn't a supported kind.
  """
  tessellation_kind = header.get('tessellation')
  if tessellation_kind != 'spherical':
    raise ValueError(f"Unknown tessellation kind: {tessellation_kind!r}")

  tessellation = _get_lod_tessellation(
      header.get('radius', 1.0), header.get('frequency', 1))
  frontier = lod_mesh_export.select_frontier(
      tessellation, header['camera_position'],
      auto_subdivide=header.get('auto_subdivide', False))
  positions, indices, vertex_count, face_count = (
      lod_mesh_export.lod_frontier_to_buffers(tessellation, frontier))

  response_header = {
      'type': 'lod_mesh',
      'id': header.get('id'),
      'vertex_count': vertex_count,
      'face_count': face_count,
      'positions_byte_length': len(positions),
      'indices_byte_length': len(indices),
      'sectors': [address.to_json() for _sector, address in frontier],
  }
  return response_header, positions + indices


def _handle_subdivide_sector(header: dict) -> tuple[dict, bytes]:
  """Handles a `subdivide_sector` request.

  Args:
    header: The request header. Must include `tessellation` (currently
      only `'spherical'` is supported), that tessellation type's own
      construction parameters (`radius`, `frequency`), and `sector` (a
      `lod_mesh_export.SectorAddress.to_json()` address, typically taken
      from a prior `get_lod_mesh` response's `sectors` list).

  Returns:
    A `(response_header, response_body)` pair -- a `subdivide_sector_ack`
    response with an empty body. Doesn't return geometry itself; the
    caller re-requests `get_lod_mesh` to pick up the newly available
    detail.

  Raises:
    ValueError: If `header["tessellation"]` isn't a supported kind.
    IndexError: If `header["sector"]`'s face index is out of range.
    ValueError: If `header["sector"]`'s path names a sector whose parent
      chain isn't fully built yet.
  """
  tessellation_kind = header.get('tessellation')
  if tessellation_kind != 'spherical':
    raise ValueError(f"Unknown tessellation kind: {tessellation_kind!r}")

  tessellation = _get_lod_tessellation(
      header.get('radius', 1.0), header.get('frequency', 1))
  address = lod_mesh_export.SectorAddress.from_json(header['sector'])
  sector = lod_mesh_export.resolve_sector_address(tessellation, address)
  sector.subdivide(resolution=lod_mesh_export.SUBDIVISION_RESOLUTION)

  return {'type': 'subdivide_sector_ack', 'id': header.get('id')}, b''


_HANDLERS: dict[str, Callable[[dict], tuple[dict, bytes]]] = {
    'get_mesh': _handle_get_mesh,
    'get_lod_mesh': _handle_get_lod_mesh,
    'subdivide_sector': _handle_subdivide_sector,
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
