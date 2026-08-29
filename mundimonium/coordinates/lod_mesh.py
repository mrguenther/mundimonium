from __future__ import annotations

from mundimonium.coordinates.tessellation import (
    Tessellation, TessellationFace, TessellationVertex
)
from mundimonium.coordinates.isometric import (
    IsometricPoint
)
from mundimonium.coordinates.nesting_iso_grid import (
    NestingIsoGrid
)
from mundimonium.utils import classproperty

from numbers import Number
from typing import override


class LodMeshSector(NestingIsoGrid):
  """A triangular region of a mesh with nested regions for LOD rendering."""
  @classmethod
  @override
  def nearby_grid_distance(cls, p1, p2):
    grid_1 = p1.grid
    grid_2 = p2.grid
    if not isinstance(grid_1, cls) or not isinstance(grid_2, cls):
      return None

    root_grid_1 = grid_1.root
    root_grid_2 = grid_2.root

    if root_grid_1 is root_grid_2:
      return NestingIsoGrid.nearby_grid_distance(p1, p2)

    if root_grid_1.is_adjacent_to_face(root_grid_2):
      return TessellationFace.nearby_grid_distance(
          p1.project_onto_root_grid(),
          p2.project_onto_root_grid(),
      )

    return None

  @classmethod
  @override
  def geodesic_distance(cls, p1: IsometricPoint, p2: IsometricPoint) -> Number:
    return p1.grid.root.tessellation.geodesic_distance(
        p1.project_onto_root_grid(),
        p2.project_onto_root_grid(),
    )


class LodMeshFace(LodMeshSector, TessellationFace):
  @classproperty
  @override
  def child_type(cls) -> type[NestingIsoGrid]:
    # A top-level `LodMeshFace` is also an `LodMeshSector`, but a nested
    # `LodMeshSector` is *not* an `LodMeshFace`. Therefore, an `LodMeshFace`
    # must contain child sectors of type `LodMeshSector` rather than the default
    # child-sector type of `cls` (which would evaluate to `LodMeshFace`).
    return LodMeshSector

