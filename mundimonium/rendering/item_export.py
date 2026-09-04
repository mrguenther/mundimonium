from __future__ import annotations

from mundimonium.coordinates.lod_mesh import LodMeshSector
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.rendering.lod_mesh_export import sector_world_position

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
  if scale is None:
    center = np.array(tessellation.center, dtype=np.float64)
    distance = float(np.linalg.norm(np.array(camera_position) - center))
    altitude = distance - tessellation.radius
    scale = tessellation.radius / altitude if altitude > 1e-12 else math.inf

  for face in tessellation.faces:
    yield from _iter_items_in_subtree(tessellation, face, scale)


def _iter_items_in_subtree(
    tessellation: SphericalTessellation,
    sector: LodMeshSector,
    scale: float,
) -> Generator[tuple[Any, float, float, float]]:
  """`iter_visible_items`'s recursion over one sector's own subtree."""
  for item in sector.items:
    if item.min_scale <= scale < item.max_scale:
      x, y, z = sector_world_position(tessellation, sector, item.position)
      yield (item.payload, float(x), float(y), float(z))

  if sector.children is not None:
    for child in sector.children:
      yield from _iter_items_in_subtree(tessellation, child, scale)
