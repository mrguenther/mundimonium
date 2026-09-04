from __future__ import annotations

from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshFace
from mundimonium.coordinates.nesting_iso_grid import SectorItem
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.rendering import flat_mesh_export
from mundimonium.rendering import item_export
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

# Hardcoded (face index, label, min_scale, max_scale) demo markers, seeded
# onto every fresh LOD tessellation purely to prove the item pipeline
# end-to-end -- not real world-generation content, which doesn't exist yet
# (see README.md).
#
# The last one is deliberately scale-gated to exercise that path; see
# `item_export.iter_visible_items`'s docstring for what `scale` means. The
# others use `SectorItem`'s always-visible defaults.
_DEMO_ITEMS = [
    (0, 'Anchorhold', 0.0, float('inf')),
    (5, 'Millbrook', 0.0, float('inf')),
    (10, 'Stonegate', 0.0, float('inf')),
    (15, 'Hiddenreach', 2.0, float('inf')),
]


def _get_lod_tessellation(radius: float, frequency: int) -> SphericalTessellation:
  """Returns the persistent LOD tessellation for `(radius, frequency)`,
  seeded with `_DEMO_ITEMS` the first time it's constructed."""
  key = (radius, frequency)
  tessellation = _lod_tessellations.get(key)
  if tessellation is None:
    tessellation = SphericalTessellation(
        radius=radius, frequency=frequency, face_type=LodMeshFace)
    for face_index, label, min_scale, max_scale in _DEMO_ITEMS:
      face = tessellation.faces[face_index]
      face.add_item(SectorItem(
          position=IsometricPoint.center(face),
          payload={'kind': 'city', 'label': label},
          min_scale=min_scale, max_scale=max_scale))
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


def _handle_get_flat_mesh(header: dict) -> tuple[dict, bytes]:
  """Handles a `get_flat_mesh` request.

  Args:
    header: The request header. Must include `tessellation` (currently
      only `'spherical'` is supported), that tessellation type's own
      construction parameters (`radius`, `frequency`), and
      `camera_position`. May include `auto_subdivide` (default `False`).

  Returns:
    A `(response_header, response_body)` pair -- a `flat_mesh` response,
    the same shape as `get_lod_mesh`'s `lod_mesh` response, but with
    positions flattened (via `Tessellation.flatten_region`) around the
    mesh point the camera is currently over, rather than in true 3D.

  Raises:
    ValueError: If `header["tessellation"]` isn't a supported kind.
  """
  tessellation_kind = header.get('tessellation')
  if tessellation_kind != 'spherical':
    raise ValueError(f"Unknown tessellation kind: {tessellation_kind!r}")

  tessellation = _get_lod_tessellation(
      header.get('radius', 1.0), header.get('frequency', 1))
  camera_position = header['camera_position']
  center = flat_mesh_export.center_point_from_camera(
      tessellation, camera_position)
  frontier = lod_mesh_export.select_frontier(
      tessellation, camera_position,
      auto_subdivide=header.get('auto_subdivide', False))
  positions, indices, vertex_count, face_count = (
      flat_mesh_export.flatten_frontier_to_buffers(
          tessellation, center, frontier))

  response_header = {
      'type': 'flat_mesh',
      'id': header.get('id'),
      'vertex_count': vertex_count,
      'face_count': face_count,
      'positions_byte_length': len(positions),
      'indices_byte_length': len(indices),
      'sectors': [address.to_json() for _sector, address in frontier],
  }
  return response_header, positions + indices


def _handle_get_items(header: dict) -> tuple[dict, bytes]:
  """Handles a `get_items` request.

  Args:
    header: The request header. Must include `tessellation` (currently
      only `'spherical'` is supported), that tessellation type's own
      construction parameters (`radius`, `frequency`), and
      `camera_position`.

  Returns:
    A `(response_header, response_body)` pair -- an `items` response
    listing every currently-visible item, each a copy of its payload dict
    plus `x`/`y`/`z`. Always an empty body; everything is in the header.

  Raises:
    ValueError: If `header["tessellation"]` isn't a supported kind.
  """
  tessellation_kind = header.get('tessellation')
  if tessellation_kind != 'spherical':
    raise ValueError(f"Unknown tessellation kind: {tessellation_kind!r}")

  tessellation = _get_lod_tessellation(
      header.get('radius', 1.0), header.get('frequency', 1))
  items = [
      {**payload, 'x': x, 'y': y, 'z': z}
      for payload, x, y, z
      in item_export.iter_visible_items(tessellation, header['camera_position'])
  ]

  return {'type': 'items', 'id': header.get('id'), 'items': items}, b''


def _handle_get_flat_items(header: dict) -> tuple[dict, bytes]:
  """Handles a `get_flat_items` request.

  Args:
    header: The request header. Must include `tessellation` (currently
      only `'spherical'` is supported), that tessellation type's own
      construction parameters (`radius`, `frequency`), and
      `camera_position`.

  Returns:
    A `(response_header, response_body)` pair -- an `items` response, the
    same shape as `get_items`'s, but with `x`/`y` flattened (via
    `Tessellation.flatten_region`) around the mesh point the camera is
    currently over, and `z` always `0.0`.

  Raises:
    ValueError: If `header["tessellation"]` isn't a supported kind.
  """
  tessellation_kind = header.get('tessellation')
  if tessellation_kind != 'spherical':
    raise ValueError(f"Unknown tessellation kind: {tessellation_kind!r}")

  tessellation = _get_lod_tessellation(
      header.get('radius', 1.0), header.get('frequency', 1))
  camera_position = header['camera_position']
  center = flat_mesh_export.center_point_from_camera(
      tessellation, camera_position)
  items = [
      {**payload, 'x': x, 'y': y, 'z': 0.0}
      for payload, x, y
      in flat_mesh_export.flatten_visible_items(
          tessellation, center, camera_position)
  ]

  return {'type': 'items', 'id': header.get('id'), 'items': items}, b''


_HANDLERS: dict[str, Callable[[dict], tuple[dict, bytes]]] = {
    'get_mesh': _handle_get_mesh,
    'get_lod_mesh': _handle_get_lod_mesh,
    'subdivide_sector': _handle_subdivide_sector,
    'get_flat_mesh': _handle_get_flat_mesh,
    'get_items': _handle_get_items,
    'get_flat_items': _handle_get_flat_items,
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
