from __future__ import annotations

from mundimonium.coordinates.isometric import IsometricDirection, IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshSector
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.coordinates.tessellation import Tessellation, TessellationFace
from mundimonium.rendering import item_export
from mundimonium.rendering.lod_mesh_export import SectorAddress

from collections.abc import Sequence
from typing import Any

import math
import numpy as np

# How many face-to-face edge crossings from a flat-mode `center`'s own
# face a face may be and still be rendered -- see `nearby_faces`'s own
# docstring for why this matters. A tunable placeholder, expected to
# need empirical adjustment (4 hops measured at ~29 faces on a
# frequency-3 geodesic sphere); kept in sync only by convention (not
# shared code) with `index.js`'s own `FLAT_MODE_THRESHOLD`, so the
# rendered patch is sized to roughly match what's actually visible right
# at the 3D/flat-mode switch.
_FLAT_MODE_RENDER_RADIUS_HOPS = 4


def center_point_from_camera(
    tessellation: SphericalTessellation,
    camera_position: Sequence[float],
) -> IsometricPoint:
  """The mesh point directly "below" a camera at `camera_position`.

  Args:
    tessellation: The tessellation to locate a point on.
    camera_position: The camera's `(x, y, z)` world position.

  Returns:
    The point on `tessellation`'s surface in the direction of
    `camera_position` from `tessellation.center`.
  """
  direction = (
      np.array(camera_position, dtype=np.float64)
      - np.array(tessellation.center, dtype=np.float64))
  direction /= np.linalg.norm(direction)
  colatitude = math.acos(np.clip(direction[2], -1.0, 1.0))
  longitude = math.atan2(direction[1], direction[0]) % (2.0 * math.pi)
  return tessellation.new_point_at_coords(colatitude, longitude)


def center_to_json(tessellation: Tessellation, point: IsometricPoint) -> dict:
  """A JSON-safe encoding of `point`, for round-tripping a flat-mode
  center through a response and back into a later request's own
  `center_from_json` call.

  `point.grid` must be one of `tessellation`'s own top-level faces (true
  of anything `new_point_at_coords`/`unflatten_point` return for
  `SphericalTessellation`, or a `GenericTessellation` surface-camera's
  own always-exact face reference -- neither ever resolves onto a nested
  LOD sub-sector) -- this doesn't handle a nested `SectorAddress`-style
  path, unlike `lod_mesh_export.py`'s own address scheme, since it
  doesn't need to. Works for any `Tessellation` (only calls `tessellation
  .faces.index(...)` and constructs a plain `IsometricPoint`), not just
  `SphericalTessellation`.

  Args:
    tessellation: The tessellation `point` belongs to.
    point: The center to encode.

  Returns:
    A `{'face': int, 'b': float, 's': float}` dict.
  """
  return {
      'face': tessellation.faces.index(point.grid),
      'b': point.b,
      's': point.s,
  }


def center_from_json(tessellation: Tessellation, data: dict) -> IsometricPoint:
  """Inverse of `center_to_json`.

  Args:
    tessellation: The tessellation `data['face']` indexes into.
    data: A dict as produced by `center_to_json`.

  Returns:
    The decoded center point.
  """
  face = tessellation.faces[data['face']]
  return IsometricPoint(face, data['b'], data['s'])


def nearby_faces(
    tessellation: Tessellation, center: IsometricPoint,
) -> list[TessellationFace]:
  """The top-level faces close enough to `center` to render in flat mode.

  Rendering every top-level face regardless of distance (as `lod_mesh_
  export.select_frontier`'s 3D-orbit callers do, where a far face just
  becomes one coarse, barely-visible triangle) is wrong for flat mode
  specifically: `flatten_region` maps the far side of the mesh to wildly
  different, badly distorted positions that visually stretch across and
  obscure the actually-relevant nearby region -- and for
  `GenericTessellation`, a face too far from `center` isn't just
  distorted, it's outside `flatten_region`'s own precomputed range
  entirely (`RelaxableFace.flattened_positions`) and raises `ValueError`.

  For `SphericalTessellation`: every face within `_FLAT_MODE_RENDER_
  RADIUS_HOPS` face-to-face edge crossings of `center.grid`, found by a
  breadth-first flood fill over `face_on_edge` (see `_faces_within_hops`)
  -- cost proportional to the faces actually included, never to the
  mesh's total size, unlike a naive scan checking every face's distance.

  For any other tessellation (`GenericTessellation`): exactly
  `center.grid`'s own precomputed `nearby_faces` -- already the correct,
  cheap-to-look-up "nearby" set described above, with no separate
  distance computation needed.
  """
  if isinstance(tessellation, SphericalTessellation):
    return _faces_within_hops(center.grid, _FLAT_MODE_RENDER_RADIUS_HOPS)
  return list(center.grid.nearby_faces)


def _faces_within_hops(
    start_face: TessellationFace, max_hops: int,
) -> list[TessellationFace]:
  """Every face reachable from `start_face` within `max_hops` face-to-face
  edge crossings (`face_on_edge`) -- a breadth-first flood fill over the
  face-adjacency graph, not a scan of the whole mesh, so its cost is
  proportional to the (small) neighborhood returned, regardless of how
  large the mesh as a whole is.
  """
  visited = {start_face}
  frontier = [start_face]
  for _ in range(max_hops):
    next_frontier = []
    for face in frontier:
      for direction in IsometricDirection:
        neighbor = face.face_on_edge(direction)
        if neighbor is not None and neighbor not in visited:
          visited.add(neighbor)
          next_frontier.append(neighbor)
    frontier = next_frontier
  return list(visited)


def basis_to_json(basis: tuple[np.ndarray, np.ndarray]) -> dict:
  """A JSON-safe encoding of a tangent basis (see `SphericalTessellation
  .tangent_basis_at`/`unflatten_point_and_transport_basis`), for
  round-tripping through a response and back into a later request's own
  `basis_from_json` call.

  Args:
    basis: An `(e_x, e_y)` tangent basis.

  Returns:
    A `{'e_x': [x, y, z], 'e_y': [x, y, z]}` dict.
  """
  e_x, e_y = basis
  return {'e_x': [float(c) for c in e_x], 'e_y': [float(c) for c in e_y]}


def basis_from_json(data: dict) -> tuple[np.ndarray, np.ndarray]:
  """Inverse of `basis_to_json`.

  Args:
    data: A dict as produced by `basis_to_json`.

  Returns:
    The decoded `(e_x, e_y)` tangent basis.
  """
  return (
      np.array(data['e_x'], dtype=np.float64),
      np.array(data['e_y'], dtype=np.float64),
  )


def _flatten_region(
    tessellation: Tessellation,
    center: IsometricPoint,
    targets: list[IsometricPoint],
    basis: tuple[np.ndarray, np.ndarray] | None,
) -> list[tuple[float, float]]:
  """Calls `tessellation.flatten_region`, passing `basis` through only for
  `SphericalTessellation` -- `GenericTessellation.flatten_region` takes
  no `basis` parameter at all (it has no tangent-basis concept; see its
  own docstring), so passing one would raise `TypeError`.
  """
  if isinstance(tessellation, SphericalTessellation):
    return tessellation.flatten_region(center, targets, basis)
  return tessellation.flatten_region(center, targets)


def flatten_frontier_to_buffers(
    tessellation: Tessellation,
    center: IsometricPoint,
    frontier: list[tuple[LodMeshSector, SectorAddress]],
    basis: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[bytes, bytes, int, int]:
  """Exports a `select_frontier` result as flat (`z = 0`) renderer-ready
  buffers, centered at `center`.

  Same shape as `lod_mesh_export.lod_frontier_to_buffers`, but each
  corner's position comes from `tessellation.flatten_region` instead of
  its true 3D position -- still emitted as 3-component `float32` triples
  (`(x, y, 0.0)`), so the wire format and `MeshLoader.buildGeometry` need
  no changes: a flat mesh is just a 3D mesh lying in the `z = 0` plane.

  Args:
    tessellation: The `frontier`'s owning tessellation.
    center: The point `flatten_region` centers the projection on.
    frontier: A `select_frontier` result.
    basis: The tangent basis to project onto -- see `flatten_region`'s
      own `basis` argument. Only meaningful for `SphericalTessellation`;
      ignored (must be omitted or `None`) for any other tessellation.

  Returns:
    A `(positions_bytes, indices_bytes, vertex_count, face_count)` tuple,
    matching `lod_frontier_to_buffers`'s buffer layout.
  """
  targets: list[IsometricPoint] = []
  indices: list[tuple[int, int, int]] = []
  for sector, _address in frontier:
    altitude = sector.altitude
    local_corners = (
        IsometricPoint(sector, altitude, 0),  # vertex B
        IsometricPoint(sector, 0, altitude),  # vertex S
        IsometricPoint(sector, 0, 0),         # vertex D
    )
    base_index = len(targets)
    # `flatten_region` needs each target resolved on a top-level face --
    # a frontier sector below the top level (subdivided by the LOD tree)
    # has no `vertex_b/s/d` of its own to compute a position from.
    targets.extend(
        sector.project_onto_root_grid(corner) for corner in local_corners)
    indices.append((base_index, base_index + 1, base_index + 2))

  flat_positions = _flatten_region(tessellation, center, targets, basis)
  positions_array = np.array(
      [(x, y, 0.0) for x, y in flat_positions], dtype=np.float32)
  indices_array = np.array(indices, dtype=np.uint32)
  return (
      positions_array.tobytes(), indices_array.tobytes(),
      len(targets), len(frontier))


def flatten_visible_items(
    tessellation: Tessellation,
    center: IsometricPoint,
    camera_position: Sequence[float] | None,
    basis: tuple[np.ndarray, np.ndarray] | None = None,
) -> list[tuple[Any, float, float]]:
  """Like `item_export.iter_visible_items`, but positions are flattened
  (via `flatten_region`) around `center` instead of in true 3D.

  Args:
    tessellation: The tessellation to look up items on.
    center: The point `flatten_region` centers the projection on.
    camera_position: The camera's `(x, y, z)` world position -- used only
      to derive the same visibility scale `item_export.iter_visible_items`
      would use, not for positioning. May be omitted (`None`) for a
      tessellation with no such scale concept (see `item_export.
      _scale_from_camera`), which never actually reads it.
    basis: The tangent basis to project onto -- see `flatten_region`'s
      own `basis` argument. Only meaningful for `SphericalTessellation`;
      ignored (must be omitted or `None`) for any other tessellation.

  Returns:
    A `(payload, x, y)` tuple per currently-visible item.
  """
  nearby = set(nearby_faces(tessellation, center))
  payloads = []
  points = []
  for payload, point in item_export.iter_visible_item_points(
      tessellation, camera_position):
    if point.grid not in nearby:
      continue  # see `nearby_faces`'s own docstring for why
    payloads.append(payload)
    points.append(point)
  if not points:
    return []

  flat_positions = _flatten_region(tessellation, center, points, basis)
  return [
      (payload, x, y)
      for payload, (x, y) in zip(payloads, flat_positions)
  ]
