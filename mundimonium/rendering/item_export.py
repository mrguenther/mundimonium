from __future__ import annotations

from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshSector
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation

from collections.abc import Generator, Sequence
from typing import Any

import math
import numpy as np


def iter_visible_items(
    tessellation: SphericalTessellation,
    camera_position: Sequence[float],
    scale: float | None = None,
) -> Generator[tuple[Any, float, float, float]]:
  """Yields every item currently visible at `scale`, with its 3D position.

  Walks every top-level face and recurses through however much of the LOD
  tree currently exists. Unlike `lod_mesh_export.select_frontier`, this
  isn't gated by distance or depth -- an item is found wherever it was
  attached (`NestingIsoGrid.add_item`), regardless of how coarse or fine
  the mesh LOD state happens to be there.

  Args:
    tessellation: A `SphericalTessellation` built with `face_type=
      LodMeshFace`.
    camera_position: The camera's `(x, y, z)` world position.
    scale: The zoom scale to filter items by (an item is yielded if
      `item.min_scale <= scale < item.max_scale`). If not given, derived
      from `camera_position`: `tessellation.radius` divided by the
      camera's altitude above the sphere's surface (distance from
      `tessellation.center`, minus `tessellation.radius`), so `scale`
      grows as the camera approaches the surface -- e.g. a camera at 1.5x
      the radius from the center (`OrbitCameraController`'s own closest
      allowed distance) gives `scale == 2.0`. This is a starting
      convention, not a fixed meaning -- nothing else yet depends on its
      exact shape besides whatever items are calibrated against it.

  Yields:
    `(payload, x, y, z)` for each visible item, in tree traversal order.
  """
  for payload, point in iter_visible_item_points(
      tessellation, camera_position, scale):
    x, y, z = tessellation.point_to_3d_position(point)
    yield (payload, float(x), float(y), float(z))


def iter_visible_item_points(
    tessellation: SphericalTessellation,
    camera_position: Sequence[float],
    scale: float | None = None,
) -> Generator[tuple[Any, IsometricPoint]]:
  """Yields every item currently visible at `scale`, with its raw mesh
  point rather than a position in any particular output coordinate
  system.

  The lower-level building block behind `iter_visible_items`, which
  further projects each point to a 3D world position -- useful for
  callers that need a different projection instead, e.g.
  `flat_mesh_export.py`, which flattens item positions via
  `Tessellation.flatten_region`. See `iter_visible_items` for the full
  scale/traversal semantics.

  Yields:
    `(payload, point)` for each visible item, `point` already projected
    onto its top-level face (`NestingIsoGrid.project_onto_root_grid`), in
    tree traversal order.
  """
  if scale is None:
    scale = _scale_from_camera(tessellation, camera_position)

  for face in tessellation.faces:
    yield from _iter_item_points_in_subtree(tessellation, face, scale)


def _scale_from_camera(
    tessellation: SphericalTessellation, camera_position: Sequence[float],
) -> float:
  """The visibility `scale` implied by a camera at `camera_position` --
  see `iter_visible_items`'s docstring for the formula and its meaning.
  """
  center = np.array(tessellation.center, dtype=np.float64)
  distance = float(np.linalg.norm(np.array(camera_position) - center))
  altitude = distance - tessellation.radius
  return tessellation.radius / altitude if altitude > 1e-12 else math.inf


def _iter_item_points_in_subtree(
    tessellation: SphericalTessellation,
    sector: LodMeshSector,
    scale: float,
) -> Generator[tuple[Any, IsometricPoint]]:
  """`iter_visible_item_points`'s recursion over one sector's subtree."""
  for item in sector.items:
    if item.min_scale <= scale < item.max_scale:
      yield (item.payload, sector.project_onto_root_grid(item.position))

  if sector.children is not None:
    for child in sector.children:
      yield from _iter_item_points_in_subtree(tessellation, child, scale)
