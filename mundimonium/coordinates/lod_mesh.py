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
    """Distance between two points whose root grids share at least one edge."""
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
    """Geodesic distance along the mesh containing `p1` and `p2`."""
    return p1.grid.root.tessellation.geodesic_distance(
        p1.project_onto_root_grid(),
        p2.project_onto_root_grid(),
    )

  @override
  def to_mesh_coordinates(self, point: IsometricPoint) -> tuple[Number, ...]:
    """Converts a point on `self` to a tessellation-defined coordinate system.

    The length of the returned coordinate vector is tessellation-defined, as is
    the meaning of each individual coordinate in the vector.

    For example, on a spherical world, this would return a 2-vector of spherical
    coordinates `(colatitude, longitude)`, with the radial-distance coordinate
    implicitly equal to the world's radius.
    """
    return self.root.tessellation.coords_at_point(
        self.project_onto_root_grid(point))

  @classmethod
  @override
  def canonicalize_point(cls, point: IsometricPoint) -> IsometricPoint:
    """Moves `point` to a new grid if located outside its current grid's bounds.

    Mutates and returns `point`, not a copy.
    """
    b: float = float(point.b)
    s: float = float(point.s)
    d: float = float(point.d)
    grid: NestingIsoGrid = point.grid

    while grid.parent is not None and (b < 0 or s < 0 or d < 0):
      b, s, d = grid.local_to_parent(b, s, d)
      grid = grid.parent

    delta_depth = point.grid.depth - grid.depth

    if b < 0 or s < 0 or d < 0:
      assert isinstance(grid, LodMeshFace)
      point.update(grid=grid, b=b, s=s)
      grid = grid.canonicalize_point(point).grid
      b = float(point.b)
      s = float(point.s)
      d = float(point.d)
      if b < 0 or s < 0 or d < 0:
        raise ValueError(
            "Failed to find a top-level face containing the point.")

    for i in range(delta_depth):
      if grid.children is None:
        grid.subdivide()
      grid = grid.child_containing(b, s, d)
      b, s, d = grid.parent_to_local(b, s, d)

    return point.update(grid=grid, b=b, s=s)


class LodMeshFace(LodMeshSector, TessellationFace):
  """A top-level mesh face.

  Both a `TessellationFace` and the root `LodMeshSector` of its own LOD tree."""

  @classproperty
  @override
  def child_type(cls) -> type[NestingIsoGrid]:
    """Non-top-level sectors are plain `LodMeshSector`s, not `LodMeshFace`s.

    A top-level `LodMeshFace` is also an `LodMeshSector`, but a nested
    `LodMeshSector` is *not* an `LodMeshFace`. Therefore, an `LodMeshFace` must
    contain child sectors of type `LodMeshSector` instead of the default
    child-sector type `cls` (i.e. same as the parent-sector type).
    """
    return LodMeshSector

  @override
  def to_mesh_coordinates(self, point: IsometricPoint) -> tuple[Number, ...]:
    """Converts a point on `self` to a tessellation-defined coordinate system.

    The length of the returned coordinate vector is tessellation-defined, as is
    the meaning of each individual coordinate in the vector.

    For example, on a spherical world, this would return a 2-vector of spherical
    coordinates `(colatitude, longitude)`, with the radial-distance coordinate
    implicitly equal to the world's radius.
    """
    return self.tessellation.coords_at_point(point)

  @classmethod
  @override
  def canonicalize_point(cls, point: IsometricPoint) -> IsometricPoint:
    """Moves `point` to a new grid if located outside its current grid's bounds.

    Mutates and returns `point`, not a copy.
    """
    return TessellationFace.canonicalize_point(point)

