from __future__ import annotations

from mundimonium.coordinates.generic_tessellation import GenericTessellation
from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshFace
from mundimonium.coordinates.nesting_iso_grid import SectorItem
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.coordinates.tessellation import Tessellation
from mundimonium.rendering import flat_mesh_export
from mundimonium.rendering import generic_demo
from mundimonium.rendering import item_export
from mundimonium.rendering import lod_mesh_export
from mundimonium.rendering.mesh_export import tessellation_to_buffers
from mundimonium.rendering.protocol import read_frame, write_frame

from typing import BinaryIO, Callable
import sys
import traceback

import numpy as np


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


# The `GenericTessellation` demo content is a single fixed shape (unlike
# the spherical case's tunable `radius`/`frequency`), so this needs no
# cache key -- just a lazily-built singleton, persisting across requests
# within a subprocess's lifetime the same way `_lod_tessellations` does.
_generic_demo_tessellation: Tessellation | None = None


def _get_generic_demo_tessellation() -> Tessellation:
  """Returns the persistent demo `GenericTessellation`, building
  (and `generic_demo`-seeding) it the first time it's requested."""
  global _generic_demo_tessellation
  if _generic_demo_tessellation is None:
    _generic_demo_tessellation = generic_demo.build_demo_tessellation()
  return _generic_demo_tessellation


def _handle_get_mesh(header: dict) -> tuple[dict, bytes]:
  """Handles a `get_mesh` request.

  Args:
    header: The request header. Must include `tessellation` (`'spherical'`
      or `'generic'`) plus, for `'spherical'`, that tessellation type's
      own construction parameters (`radius`, `frequency`). `'generic'`
      always returns the same fixed demo shape (see `generic_demo.py`),
      no construction parameters needed.

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
  elif tessellation_kind == 'generic':
    tessellation = _get_generic_demo_tessellation()
  else:
    raise ValueError(f"Unknown tessellation kind: {tessellation_kind!r}")

  positions, indices, vertex_count, face_count, adjacency, vertex_faces = (
      tessellation_to_buffers(tessellation))
  response_header = {
      'type': 'mesh',
      'id': header.get('id'),
      'vertex_count': vertex_count,
      'face_count': face_count,
      'positions_byte_length': len(positions),
      'indices_byte_length': len(indices),
      'adjacency': adjacency,
      'vertex_faces': vertex_faces,
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


def _resolve_flat_center_and_basis(
    tessellation: SphericalTessellation, header: dict,
) -> tuple[IsometricPoint, tuple[np.ndarray, np.ndarray]]:
  """The mesh point and tangent basis a `get_flat_mesh`/`get_flat_items`
  request should center and orient its projection on.

  If `header` carries `center`/`pan_offset`/`basis` (every request after
  flat mode's first, once a reference frame is already known), resolves
  the new center and basis exactly via `SphericalTessellation.
  unflatten_point_and_transport_basis` -- parallel-transporting `basis`
  keeps the projection's orientation evolving continuously as the center
  moves, with no net rotation relative to the path panned, unlike
  independently recomputing a basis at each new center. Otherwise (flat
  mode's initial entry, when only a real 3D `camera_position` exists yet)
  falls back to `flat_mesh_export.center_point_from_camera` plus a fresh
  `tangent_basis_at` -- the starting frame later calls will transport.
  """
  if 'center' in header:
    reference_point = flat_mesh_export.center_from_json(
        tessellation, header['center'])
    pan_x, pan_y = header['pan_offset']
    basis = flat_mesh_export.basis_from_json(header['basis'])
    return tessellation.unflatten_point_and_transport_basis(
        reference_point, pan_x, pan_y, basis)
  center = flat_mesh_export.center_point_from_camera(
      tessellation, header['camera_position'])
  return center, tessellation.tangent_basis_at(center)


def _resolve_generic_flat_center_and_orientation(
    tessellation: GenericTessellation, header: dict,
) -> tuple[IsometricPoint, np.ndarray | None]:
  """The mesh point and screen-space orientation a generic `get_flat_
  mesh`/`get_flat_items` request should center and orient its projection
  on.

  Always starts from `header['center']` -- the surface-following camera
  that drives `GenericTessellation`'s flat mode always knows its own
  exact face-local position directly, unlike `SphericalTessellation`,
  which needs a real 3D `camera_position` resolved only on its first
  request (see `_resolve_flat_center_and_basis`). If `header` also
  carries `pan_offset` (every request after flat mode's first, once that
  center is already the *previous* response's own resolved value),
  advances both center and orientation via `GenericTessellation.
  unflatten_point_and_transport_orientation` -- keeping the projection
  updating in real time as the view pans (the same real-time re-centering
  `SphericalTessellation`'s own flat mode already has), while keeping its
  screen-space orientation continuous rather than snapping to whichever
  new anchor face's own canonical frame happens to be (see that method's
  own docstring for why this is an approximation, not exact transport).
  `header['orientation']`, present whenever `pan_offset` is, is the
  previous response's own returned orientation to continue from.
  """
  center = flat_mesh_export.center_from_json(tessellation, header['center'])
  if 'pan_offset' in header:
    pan_x, pan_y = header['pan_offset']
    orientation = (
        flat_mesh_export.orientation_from_json(header['orientation'])
        if 'orientation' in header else None)
    return tessellation.unflatten_point_and_transport_orientation(
        center, pan_x, pan_y, orientation)
  return center, None


def _handle_get_flat_mesh(header: dict) -> tuple[dict, bytes]:
  """Handles a `get_flat_mesh` request.

  Args:
    header: The request header. Must include `tessellation` (`'spherical'`
      or `'generic'`).

      For `'spherical'`: that tessellation type's own construction
      parameters (`radius`, `frequency`), and either `camera_position`
      (flat mode's initial entry) or `center` + `pan_offset` + `basis`
      (every subsequent call, once a reference frame is already known --
      see `_resolve_flat_center_and_basis`). May include `auto_subdivide`
      (default `False`).

      For `'generic'`: `center` (a `{face, b, s}` referring to the fixed
      demo tessellation, `flat_mesh_export.center_to_json`-encoded), plus
      `pan_offset` (and, once available, `orientation`) on every call
      after flat mode's first (see `_resolve_generic_flat_center_and_
      orientation`). No `camera_position`/`basis`: the surface-following
      camera that drives `GenericTessellation`'s flat mode always knows
      its own exact face-local position directly, with no "map a 3D
      position back to a mesh point" resolution step needed (unlike
      `SphericalTessellation`), and no tangent basis of its own (see
      `GenericTessellation.flatten_region`'s own docstring) -- it carries
      an `orientation` (a 2x2 rotation) forward instead, for the same
      screen-space-continuity purpose.

  Returns:
    A `(response_header, response_body)` pair -- a `flat_mesh` response,
    the same shape as `get_lod_mesh`'s `lod_mesh` response, but with
    positions flattened (via `Tessellation.flatten_region`) around the
    resolved center rather than in true 3D. Also includes `center`
    (`flat_mesh_export.center_to_json`-encoded) for the caller to pass
    into its own next request; (`'spherical'` only) `basis` (`basis_to_
    json`-encoded) and `center_position` (the resolved center's own true
    3D position, `[x, y, z]`) -- for the caller to resume the 3D orbit
    camera at, if it exits flat mode, wherever the view has actually
    panned to by then; and (`'generic'` only, once resolved via a
    `pan_offset`) `orientation` (`orientation_to_json`-encoded).

  Raises:
    ValueError: If `header["tessellation"]` isn't a supported kind.
  """
  tessellation_kind = header.get('tessellation')
  center_position = None
  orientation_out = None
  if tessellation_kind == 'spherical':
    tessellation = _get_lod_tessellation(
        header.get('radius', 1.0), header.get('frequency', 1))
    center, basis = _resolve_flat_center_and_basis(tessellation, header)
    # `select_frontier`'s own distance-to-camera check still wants a real
    # 3D position -- reuse `center`'s (exact) 3D position rather than the
    # original (possibly long-stale, once panning has moved on) 3D
    # `camera_position`, since `center` is always the freshest estimate
    # of where the camera actually is now. Also returned to the caller
    # (see `center_position` below) so it can resume the 3D orbit camera
    # at wherever flat mode's own view has actually panned to by the time
    # it exits, rather than the stale pre-entry position.
    center_position = tessellation.point_to_3d_position(center)
    nearby = flat_mesh_export.nearby_faces(tessellation, center)
    face_indices = [tessellation.faces.index(face) for face in nearby]
    frontier = lod_mesh_export.select_frontier(
        tessellation, center_position,
        auto_subdivide=header.get('auto_subdivide', False),
        face_indices=face_indices)
  elif tessellation_kind == 'generic':
    tessellation = _get_generic_demo_tessellation()
    center, orientation_out = _resolve_generic_flat_center_and_orientation(
        tessellation, header)
    basis = None
    # No `select_frontier` call needed: the fixed demo mesh has no LOD
    # tree in active use, so each nearby top-level face is its own,
    # un-subdivided sector.
    frontier = [
        (face, lod_mesh_export.SectorAddress(tessellation.faces.index(face)))
        for face in flat_mesh_export.nearby_faces(tessellation, center)
    ]
  else:
    raise ValueError(f"Unknown tessellation kind: {tessellation_kind!r}")

  positions, indices, vertex_count, face_count = (
      flat_mesh_export.flatten_frontier_to_buffers(
          tessellation, center, frontier, basis, orientation_out))

  response_header = {
      'type': 'flat_mesh',
      'id': header.get('id'),
      'vertex_count': vertex_count,
      'face_count': face_count,
      'positions_byte_length': len(positions),
      'indices_byte_length': len(indices),
      'sectors': [address.to_json() for _sector, address in frontier],
      'center': flat_mesh_export.center_to_json(tessellation, center),
  }
  if basis is not None:
    response_header['basis'] = flat_mesh_export.basis_to_json(basis)
  if center_position is not None:
    response_header['center_position'] = [float(c) for c in center_position]
  if orientation_out is not None:
    response_header['orientation'] = (
        flat_mesh_export.orientation_to_json(orientation_out))
  return response_header, positions + indices


def _handle_get_items(header: dict) -> tuple[dict, bytes]:
  """Handles a `get_items` request.

  Args:
    header: The request header. Must include `tessellation` (`'spherical'`
      or `'generic'`), that tessellation type's own construction
      parameters (`radius`, `frequency` for `'spherical'`; none for
      `'generic'`), and `camera_position`.

  Returns:
    A `(response_header, response_body)` pair -- an `items` response
    listing every currently-visible item, each a copy of its payload dict
    plus `x`/`y`/`z`. Always an empty body; everything is in the header.

  Raises:
    ValueError: If `header["tessellation"]` isn't a supported kind.
  """
  tessellation_kind = header.get('tessellation')
  if tessellation_kind == 'spherical':
    tessellation = _get_lod_tessellation(
        header.get('radius', 1.0), header.get('frequency', 1))
  elif tessellation_kind == 'generic':
    tessellation = _get_generic_demo_tessellation()
  else:
    raise ValueError(f"Unknown tessellation kind: {tessellation_kind!r}")

  items = [
      {**payload, 'x': x, 'y': y, 'z': z}
      for payload, x, y, z
      in item_export.iter_visible_items(tessellation, header['camera_position'])
  ]

  return {'type': 'items', 'id': header.get('id'), 'items': items}, b''


def _handle_get_flat_items(header: dict) -> tuple[dict, bytes]:
  """Handles a `get_flat_items` request.

  Args:
    header: The request header. Must include `tessellation` (`'spherical'`
      or `'generic'`) -- see `_handle_get_flat_mesh` for each kind's exact
      request shape (`'spherical'` also requires `camera_position` here,
      unlike `get_flat_mesh`, since `flatten_visible_items` still uses it
      for item-visibility `scale` even once `center`/`pan_offset`/`basis`
      are also present and doing the actual positioning -- not yet
      updated to track flat-mode zoom instead, so this scale grows
      increasingly stale the further the view pans from where flat mode
      was originally entered; `'generic'` has no equivalent scale concept
      yet, so every item is always visible there).

  Returns:
    A `(response_header, response_body)` pair -- an `items` response, the
    same shape as `get_items`'s, but with `x`/`y` flattened (via
    `Tessellation.flatten_region`) around the resolved center, and `z`
    always `0.0`. Also includes `center` (and, `'spherical'` only,
    `basis`; `'generic'` only, once resolved via a `pan_offset`,
    `orientation`), like `get_flat_mesh`'s response.

  Raises:
    ValueError: If `header["tessellation"]` isn't a supported kind.
  """
  tessellation_kind = header.get('tessellation')
  orientation_out = None
  if tessellation_kind == 'spherical':
    tessellation = _get_lod_tessellation(
        header.get('radius', 1.0), header.get('frequency', 1))
    center, basis = _resolve_flat_center_and_basis(tessellation, header)
  elif tessellation_kind == 'generic':
    tessellation = _get_generic_demo_tessellation()
    center, orientation_out = _resolve_generic_flat_center_and_orientation(
        tessellation, header)
    basis = None
  else:
    raise ValueError(f"Unknown tessellation kind: {tessellation_kind!r}")

  items = [
      {**payload, 'x': x, 'y': y, 'z': 0.0}
      for payload, x, y
      in flat_mesh_export.flatten_visible_items(
          tessellation, center, header.get('camera_position'), basis,
          orientation_out)
  ]

  response_header = {
      'type': 'items', 'id': header.get('id'), 'items': items,
      'center': flat_mesh_export.center_to_json(tessellation, center),
  }
  if basis is not None:
    response_header['basis'] = flat_mesh_export.basis_to_json(basis)
  if orientation_out is not None:
    response_header['orientation'] = (
        flat_mesh_export.orientation_to_json(orientation_out))
  return response_header, b''


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
