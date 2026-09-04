from __future__ import annotations

from mundimonium.coordinates.isometric import IsometricPoint
from mundimonium.coordinates.lod_mesh import LodMeshSector
from mundimonium.coordinates.spherical_tessellation import SphericalTessellation
from mundimonium.coordinates.tessellation import Tessellation

from collections.abc import Generator, Sequence
from typing import Any

import math
import numpy as np

# A stand-in "camera is very close" scale for tessellation types with no
# generic notion of camera altitude yet (see `_scale_from_camera`) -- large
# enough to satisfy any demo item's `min_scale` gate, but deliberately
# finite so it doesn't fail a `scale < max_scale` check against the common
# `max_scale = inf` default.
_ALWAYS_VISIBLE_SCALE = 1e6


def iter_visible_items(
    tessellation: Tessellation,
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
    tessellation: A `Tessellation` built with a LOD-tree-aware `face_type`
      (`LodMeshFace`, or `RelaxableLodMeshFace` for a `GenericTessellation`).
    camera_position: The camera's `(x, y, z)` world position.
    scale: The zoom scale to filter items by (an item is yielded if
      `item.min_scale <= scale < item.max_scale`). If not given, derived
      from `camera_position` via `_scale_from_camera` -- see its own
      docstring for the formula and its meaning, and for which
      tessellation types it actually applies to.

  Yields:
    `(payload, x, y, z)` for each visible item, in tree traversal order.
  """
  for payload, point in iter_visible_item_points(
      tessellation, camera_position, scale):
    x, y, z = tessellation.point_to_3d_position(point)
    yield (payload, float(x), float(y), float(z))


def iter_visible_item_points(
    tessellation: Tessellation,
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
    tessellation: Tessellation, camera_position: Sequence[float],
) -> float:
  """The visibility `scale` implied by a camera at `camera_position`.

  For a `SphericalTessellation`: `tessellation.radius` divided by the
  camera's altitude above the sphere's surface (distance from
  `tessellation.center`, minus `tessellation.radius`), so `scale` grows
  as the camera approaches the surface -- e.g. a camera at 1.5x the
  radius from the center (`OrbitCameraController`'s own closest allowed
  distance) gives `scale == 2.0`. This is a starting convention, not a
  fixed meaning -- nothing else yet depends on its exact shape besides
  whatever items are calibrated against it.

  For any other tessellation type (e.g. `GenericTessellation`, which has
  no `.center`/`.radius` to measure an altitude against): every item is
  treated as always visible, via `_ALWAYS_VISIBLE_SCALE` -- there's no
  generic notion of "camera altitude" yet for an arbitrary mesh shape
  (see the plan's Phase 9, which introduces one via the surface-following
  camera's own known hover distance). Deliberately a large *finite*
  number, not `math.inf`: an item's visibility test is `min_scale <=
  scale < max_scale`, and the common (default) `max_scale` is itself
  `inf` -- `inf < inf` is `False`, so an actual `inf` scale would exclude
  every normally-always-visible item, the opposite of the intent here.
  """
  if not isinstance(tessellation, SphericalTessellation):
    return _ALWAYS_VISIBLE_SCALE
  center = np.array(tessellation.center, dtype=np.float64)
  distance = float(np.linalg.norm(np.array(camera_position) - center))
  altitude = distance - tessellation.radius
  return tessellation.radius / altitude if altitude > 1e-12 else math.inf


def _iter_item_points_in_subtree(
    tessellation: Tessellation,
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
