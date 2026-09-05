from __future__ import annotations

from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshSector
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.rendering import item_export
from mundimonium.rendering.lod_mesh_export import SectorAddress

from collections.abc import Sequence
from typing import Any

import math
import numpy as np


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


def center_to_json(
    tessellation: SphericalTessellation, point: IsometricPoint) -> dict:
  """A JSON-safe encoding of `point`, for round-tripping a flat-mode
  center through a response and back into a later request's own
  `center_from_json` call.

  `point.grid` must be one of `tessellation`'s own top-level faces (true
  of anything `new_point_at_coords`/`unflatten_point` return -- neither
  ever resolves onto a nested LOD sub-sector) -- this doesn't handle a
  nested `SectorAddress`-style path, unlike `lod_mesh_export.py`'s own
  address scheme, since it doesn't need to.

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


def center_from_json(
    tessellation: SphericalTessellation, data: dict) -> IsometricPoint:
  """Inverse of `center_to_json`.

  Args:
    tessellation: The tessellation `data['face']` indexes into.
    data: A dict as produced by `center_to_json`.

  Returns:
    The decoded center point.
  """
  face = tessellation.faces[data['face']]
  return IsometricPoint(face, data['b'], data['s'])


def flatten_frontier_to_buffers(
    tessellation: SphericalTessellation,
    center: IsometricPoint,
    frontier: list[tuple[LodMeshSector, SectorAddress]],
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

  flat_positions = tessellation.flatten_region(center, targets)
  positions_array = np.array(
      [(x, y, 0.0) for x, y in flat_positions], dtype=np.float32)
  indices_array = np.array(indices, dtype=np.uint32)
  return (
      positions_array.tobytes(), indices_array.tobytes(),
      len(targets), len(frontier))


def flatten_visible_items(
    tessellation: SphericalTessellation,
    center: IsometricPoint,
    camera_position: Sequence[float],
) -> list[tuple[Any, float, float]]:
  """Like `item_export.iter_visible_items`, but positions are flattened
  (via `flatten_region`) around `center` instead of in true 3D.

  Args:
    tessellation: The tessellation to look up items on.
    center: The point `flatten_region` centers the projection on.
    camera_position: The camera's `(x, y, z)` world position -- used only
      to derive the same visibility scale `item_export.iter_visible_items`
      would use, not for positioning.

  Returns:
    A `(payload, x, y)` tuple per currently-visible item.
  """
  payloads = []
  points = []
  for payload, point in item_export.iter_visible_item_points(
      tessellation, camera_position):
    payloads.append(payload)
    points.append(point)
  if not points:
    return []

  flat_positions = tessellation.flatten_region(center, points)
  return [
      (payload, x, y)
      for payload, (x, y) in zip(payloads, flat_positions)
  ]
