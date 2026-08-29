from __future__ import annotations

from mundimonium.layers.coordinates.isometric import (
    IsometricGrid, IsometricPoint,
)

from collections.abc import Generator
from dataclasses import dataclass
from numbers import Number
from typing import Any, override
import math


_SQRT3 = math.sqrt(3.0)


def isometric_to_cartesian(b: float, s: float, d: float) -> tuple[float, float]:
  """Maps isometric (b, s, d) to 2D Cartesian (x, y).

  B (distance from base) maps to y.  The base of the triangle sits at
  y = 0, the apex (vertex B) at y = altitude::

      x = (d - s) / sqrt(3)
      y = b
  """
  return ((d - s) / _SQRT3, b)


# ---------------------------------------------------------------------------
# Data classes for LOD rendering
# ---------------------------------------------------------------------------

@dataclass
class SectorItem:
  """A positioned object with a LOD visibility range.

  Attributes:
      position:  Location within the owning sector's local grid.
      payload:   The renderable object (drawing primitives, etc.).
      min_scale: Minimum zoom scale at which to render (inclusive).
      max_scale: Maximum zoom scale at which to render (exclusive).
  """
  position: IsometricPoint
  payload: Any
  min_scale: float = 0.0
  max_scale: float = float('inf')


@dataclass
class RenderItem:
  """Output of 2D rendering: a Cartesian position plus the payload."""
  x: float
  y: float
  payload: Any


# ---------------------------------------------------------------------------
# Sector – a single sub-triangle in a NestingIsoGrid
# ---------------------------------------------------------------------------

class IsoGridSector:
  """A sub-triangle within a NestingIsoGrid.

  Each sector knows its parent grid, its position within that grid
  (row/column indices and orientation), and can hold both renderable
  items and an optional child NestingIsoGrid for further subdivision.
  """

  def __init__(
      self,
      parent_grid: NestingIsoGrid,
      i_b: int,
      i_s: int,
      inverted: bool,
  ):
    self._parent_grid = parent_grid
    self._root_grid = parent_grid.root_grid
    self._i_b = i_b
    self._i_s = i_s
    self._inverted = inverted
    self._items: list[SectorItem] = []
    self._child_grid: NestingIsoGrid | None = None

  # ---- Properties ----

  @property
  def parent_grid(self) -> NestingIsoGrid:
    return self._parent_grid

  @property
  def root_grid(self) -> NestingIsoGrid:
    return self._root_grid

  @property
  def i_b(self) -> int:
    return self._i_b

  @property
  def i_s(self) -> int:
    return self._i_s

  @property
  def i_d(self) -> int:
    N = self._parent_grid.resolution
    if self._inverted:
      return N - 2 - self._i_b - self._i_s
    else:
      return N - 1 - self._i_b - self._i_s

  @property
  def indices(self) -> tuple[int, int, int]:
    """Integer (i_b, i_s, i_d) indices within the parent grid."""
    return (self._i_b, self._i_s, self.i_d)

  @property
  def inverted(self) -> bool:
    """True for downward-pointing sub-triangles."""
    return self._inverted

  @property
  def items(self) -> list[SectorItem]:
    return self._items

  @property
  def child_grid(self) -> NestingIsoGrid | None:
    return self._child_grid

  @child_grid.setter
  def child_grid(self, grid: NestingIsoGrid | None) -> None:
    self._child_grid = grid

  # ---- Mutators ----

  def add_item(self, item: SectorItem) -> None:
    self._items.append(item)

  def subdivide(self, resolution: int) -> NestingIsoGrid:
    """Create a child NestingIsoGrid for further LOD subdivision.

    The child grid's altitude is ``parent_altitude / parent_resolution``,
    i.e. it fills this sector's sub-triangle exactly.
    """
    sub_altitude = self._parent_grid.altitude / self._parent_grid.resolution
    child = NestingIsoGrid(
        resolution=resolution,
        altitude=sub_altitude,
        parent_sector=self,
    )
    self._child_grid = child
    return child

  # ---- Display ----

  def __repr__(self) -> str:
    kind = "\u25BD" if self._inverted else "\u25B3"
    return (
        f"IsoGridSector({kind} i_b={self._i_b}, i_s={self._i_s}, "
        f"i_d={self.i_d}, items={len(self._items)}, "
        f"child={'yes' if self._child_grid else 'no'})")


# ---------------------------------------------------------------------------
# Sector table – flat-indexed lookup of N² sectors
# ---------------------------------------------------------------------------

class IsoGridSectorTable(IsometricGrid):
  """Flat-indexed spatial lookup table of IsoGridSectors.

  A resolution-N equilateral triangle is subdivided into N**2 sub-
  triangles (sectors).  Sectors are stored in a flat list of length N**2
  with O(1) lookup by index or by IsometricPoint.

  Within each row *i_b*, upward and downward sub-triangles are
  interleaved for cache locality::

      flat_index = i_b * (2*N - i_b) + 2*i_s + (1 if inverted else 0)
  """

  def __init__(self):
    super().__init__()
    N = self.resolution
    self._sectors: list[IsoGridSector] = []
    for i_b in range(N):
      for i_s in range(N - i_b):
        # Upward triangle
        self._sectors.append(
            IsoGridSector(self, i_b, i_s, inverted=False))
        # Downward triangle (exists when there is a next column)
        if i_s < N - 1 - i_b:
          self._sectors.append(
              IsoGridSector(self, i_b, i_s, inverted=True))
    assert len(self._sectors) == N * N

  # ---- Flat indexing ----

  def _flat_index(self, i_b: int, i_s: int, inverted: bool) -> int:
    """O(1) flat index into ``self._sectors``."""
    N = self.resolution
    if i_b < 0 or i_s < 0 or i_b >= N or i_s >= N - i_b:
      raise IndexError(
          f"Sector indices (i_b={i_b}, i_s={i_s}) out of range "
          f"for resolution {N}.")
    if inverted and i_s >= N - 1 - i_b:
      raise IndexError(
          f"No inverted sector at (i_b={i_b}, i_s={i_s}) for "
          f"resolution {N}.")
    return i_b * (2 * N - i_b) + 2 * i_s + (1 if inverted else 0)

  def sector_at(self, i_b: int, i_s: int, inverted: bool) -> IsoGridSector:
    """Look up a sector by its grid indices and orientation."""
    return self._sectors[self._flat_index(i_b, i_s, inverted)]

  def sector_containing(self, point: IsometricPoint) -> IsoGridSector:
    """Find the sector that contains a given IsometricPoint.

    Determines whether the point falls in an upward or downward
    sub-triangle by checking the sum of the floored scaled coordinates.
    """
    if point.grid is not self:
      raise KeyError("The provided IsometricPoint is not on this grid.")

    N = self.resolution
    h = self.altitude
    scale = N / h

    b_s = point.b * scale
    s_s = point.s * scale
    d_s = point.d * scale

    i_b = max(0, min(int(b_s), N - 1))
    i_s = max(0, min(int(s_s), N - 1 - i_b))
    i_d = max(0, min(int(d_s), N - 1 - i_b))

    floor_sum = i_b + i_s + i_d

    if floor_sum == N - 1:
      inverted = False
    elif floor_sum == N - 2:
      inverted = True
    else:
      # Floating-point edge case – fall back to upward triangle
      inverted = False
      i_s = min(i_s, N - 1 - i_b)

    return self._sectors[self._flat_index(i_b, i_s, inverted)]

  # ---- Subscript interface (preserved from original API) ----

  def __getitem__(self, near_point: IsometricPoint) -> IsoGridSector:
    return self.sector_containing(near_point)

  def __setitem__(
      self, near_point: IsometricPoint, sector: IsoGridSector,
  ) -> None:
    target = self.sector_containing(near_point)
    idx = self._flat_index(target.i_b, target.i_s, target.inverted)
    self._sectors[idx] = sector

  def get_or_insert(
      self, near_point: IsometricPoint, sector: IsoGridSector,
  ) -> IsoGridSector:
    """Return the existing sector at *near_point*.

    With pre-populated flat indexing, sectors always exist; the
    *sector* argument is ignored (retained for API compatibility).
    """
    return self.sector_containing(near_point)

  def get_or_emplace(
      self,
      near_point: IsometricPoint,
      sector_type: type[IsoGridSector] | None = None,
      *args,
      **kwargs,
  ) -> IsoGridSector:
    """Return the existing sector at *near_point*.

    With pre-populated flat indexing, sectors always exist; the
    factory arguments are ignored (retained for API compatibility).
    """
    return self.sector_containing(near_point)

  @property
  def resolution(self):
    raise NotImplementedError()


# ---------------------------------------------------------------------------
# NestingIsoGrid – the recursive LOD tree node
# ---------------------------------------------------------------------------

class NestingIsoGrid(IsoGridSectorTable):
  """Hierarchical equilateral-triangle LOD grid.

  Each NestingIsoGrid subdivides an equilateral triangle into N**2
  sub-triangles (sectors).  Each sector can hold renderable items with
  scale-dependent visibility, and can optionally contain a child
  NestingIsoGrid for further subdivision — forming a tree of arbitrary
  depth for level-of-detail rendering.

  Parameters:
      resolution:    Number of subdivisions per edge (N).  Produces N**2
                     sectors.  Pass None for a leaf grid with no sectors.
      altitude:      Height of this triangle in world units.
      parent_sector: The IsoGridSector that contains this grid (if this
                     grid is a child of another NestingIsoGrid).
  """

  BASE_TO_ALTITUDE = _SQRT3 / 2
  APOTHEM_TO_ALTITUDE = 3

  def __init__(
      self,
      resolution: int | None,
      altitude: float = 1.0,
      parent_sector: IsoGridSector | None = None,
  ):
    self._resolution = resolution
    self._altitude = float(altitude)
    self._side_length = 2.0 * self._altitude / _SQRT3
    self._apothem = self._altitude / 3.0
    self._parent_sector = parent_sector
    self._root_grid = (
        parent_sector.root_grid if parent_sector is not None else self
    )

    if resolution is not None and resolution >= 1:
      IsoGridSectorTable.__init__(self)

  # ---- IsometricGrid abstract property implementations ----

  @property
  @override
  def resolution(self) -> int | None:
    return self._resolution

  @property
  @override
  def altitude(self) -> float:
    return self._altitude

  @property
  @override
  def side_length(self) -> float:
    return self._side_length

  @property
  @override
  def apothem(self) -> float:
    return self._apothem

  @property
  def parent_sector(self) -> IsoGridSector | None:
    return self._parent_sector

  @property
  def root_grid(self) -> NestingIsoGrid:
    return self._root_grid

  # ---- IsometricGrid abstract method implementations ----

  @classmethod
  @override
  def nearby_grid_distance(
      cls, p1: IsometricPoint, p2: IsometricPoint,
  ) -> Number | None:
    grid_1 = p1.grid
    grid_2 = p2.grid
    if not isinstance(grid_1, cls) or not isinstance(grid_2, cls):
      return None

    root_grid = grid_1.root_grid
    if root_grid is grid_2.root_grid:
      return root_grid.local_distance(
          p1.project_onto_root_grid(),
          p2.project_onto_root_grid(),
      )

    return None

  @classmethod
  @override
  def geodesic_distance(
      cls, p1: IsometricPoint, p2: IsometricPoint,
  ) -> Number | None:
    return None

  # ---- Coordinate transforms ----

  def parent_to_local(
      self,
      b_parent: float, s_parent: float, d_parent: float,
      sector: IsoGridSector,
  ) -> tuple[float, float, float]:
    """Convert parent-grid isometric coords to sector-local coords.

    For an **upward** sector at (i_b, i_s, i_d):
        b_local = b_parent - i_b * sector_altitude
        s_local = s_parent - i_s * sector_altitude
        d_local = d_parent - i_d * sector_altitude

    For a **downward** (inverted) sector:
        b_local = (i_b + 1) * sector_altitude - b_parent
        s_local = (i_s + 1) * sector_altitude - s_parent
        d_local = (i_d + 1) * sector_altitude - d_parent
    """
    sector_altitude = self._altitude / self._resolution
    i_b, i_s, i_d = sector.indices
    if sector.inverted:
      return (
          (i_b + 1) * sector_altitude - b_parent,
          (i_s + 1) * sector_altitude - s_parent,
          (i_d + 1) * sector_altitude - d_parent,
      )
    else:
      return (
          b_parent - i_b * sector_altitude,
          s_parent - i_s * sector_altitude,
          d_parent - i_d * sector_altitude,
      )

  def local_to_parent(
      self,
      b_local: float, s_local: float, d_local: float,
      sector: IsoGridSector,
  ) -> tuple[float, float, float]:
    """Convert sector-local isometric coords to parent-grid coords.

    The formulas are the same as parent_to_local (the transform is its
    own inverse), but applied in the opposite direction.
    """
    sector_altitude = self._altitude / self._resolution
    i_b, i_s, i_d = sector.indices
    if sector.inverted:
      return (
          (i_b + 1) * sector_altitude - b_local,
          (i_s + 1) * sector_altitude - s_local,
          (i_d + 1) * sector_altitude - d_local,
      )
    else:
      return (
          b_local + i_b * sector_altitude,
          s_local + i_s * sector_altitude,
          d_local + i_d * sector_altitude,
      )

  def project_onto_root_grid(self, point) -> IsometricPoint:
    root_b, root_s, root_d = self.to_root_isometric(point.b, point.s, point.d)
    return IsometricPoint(self.root_grid, root_b, root_s)

  # def project_onto_ancestor_grid(
  #     self, point: IsometricPoint, ancestor: NestingIsoGrid) -> IsometricPoint:

  def to_root_isometric(
      self, b: float, s: float, d: float) -> tuple[float, float, float]:
    """Convert local isometric (b, s, d) to root-level isometric (b, s, d).

    Walks up the tree through parent sectors, composing coordinate transforms
    until we reach the root grid.
    """
    if self._parent_sector is None:
      return (b, s, d)

    parent_grid = self._parent_sector.parent_grid
    b_p, s_p, d_p = parent_grid.local_to_parent(
        b, s, d, self._parent_sector)
    return parent_grid.to_root_isometric(b_p, s_p, d_p)

  def to_root_cartesian(
      self, b: float, s: float, d: float,
  ) -> tuple[float, float]:
    """Convert local isometric (b, s, d) to root-level Cartesian (x, y).

    Walks up the tree through parent sectors, composing coordinate transforms
    until we reach the root grid.
    """
    return isometric_to_cartesian(*self.to_root_isometric(b, s, d))

  # ---- 2D rendering ----

  def render_2d(
      self,
      scale: float,
      viewport: tuple[float, float, float, float] | None = None,
      _offset_x: float = 0.0,
      _offset_y: float = 0.0,
      _sign: float = 1.0,
  ) -> Generator[RenderItem]:
    """Traverse the LOD tree and yield visible items as RenderItems.

    Args:
        scale:    Current zoom scale.  Items whose ``[min_scale,
                  max_scale)`` range includes this value are yielded.
        viewport: Optional ``(x_min, y_min, x_max, y_max)`` culling
                  rectangle in Cartesian coordinates.  Sectors whose
                  bounding box doesn't intersect are skipped entirely
                  (along with their whole subtree).
        _offset_x, _offset_y:
                  (internal) Accumulated Cartesian offset from the
                  recursive traversal.
        _sign:    (internal) ``+1.0`` or ``-1.0``, tracking accumulated
                  orientation flips from nested inversions.

    Yields:
        ``RenderItem(x, y, payload)`` for each visible SectorItem.
    """
    if self._resolution is None or not hasattr(self, '_sectors'):
      return

    h_sub = self._altitude / self._resolution
    half_base = h_sub / _SQRT3

    for sector in self._sectors:
      # Sector Cartesian offset in this grid's local frame
      i_b, i_s, i_d = sector.indices
      sec_x = (i_d - i_s) * h_sub / _SQRT3
      sec_y = (i_b + 1) * h_sub if sector.inverted else i_b * h_sub

      # Absolute offset of this sector's reference point
      abs_x = _offset_x + _sign * sec_x
      abs_y = _offset_y + _sign * sec_y

      # Sign for mapping local Cartesian within this sector
      item_sign = -_sign if sector.inverted else _sign

      # Viewport culling: AABB of the sector's triangle
      if viewport is not None:
        vp_xmin, vp_ymin, vp_xmax, vp_ymax = viewport
        # Sector bounding box
        y_lo = abs_y
        y_hi = abs_y + item_sign * h_sub
        bb_xmin = abs_x - half_base
        bb_xmax = abs_x + half_base
        bb_ymin = min(y_lo, y_hi)
        bb_ymax = max(y_lo, y_hi)
        if bb_xmax < vp_xmin or bb_xmin > vp_xmax or \
           bb_ymax < vp_ymin or bb_ymin > vp_ymax:
          continue

      # Yield visible items at the current scale.
      # Item positions are IsometricPoints in this grid's coordinate
      # system, so we convert them to Cartesian using the *grid-level*
      # transform (_offset + _sign), not the per-sector offset.
      for item in sector.items:
        if item.min_scale <= scale < item.max_scale:
          lx, ly = isometric_to_cartesian(
              item.position.b, item.position.s, item.position.d)
          yield RenderItem(
              _offset_x + _sign * lx,
              _offset_y + _sign * ly,
              item.payload,
          )

      # Recurse into child grid
      if sector.child_grid is not None:
        yield from sector.child_grid.render_2d(
            scale, viewport, abs_x, abs_y, item_sign)

  # ---- Convenience ----

  @property
  def default_sector_type(self):
    return IsoGridSector


# ---------------------------------------------------------------------------
# Main – demo / smoke test
# ---------------------------------------------------------------------------

def main():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  print(f"Root grid: resolution={grid.resolution}, altitude={grid.altitude}, "
        f"side_length={grid.side_length:.4f}, apothem={grid.apothem:.4f}")
  print(f"Sectors: {len(grid._sectors)} (expected {grid.resolution**2})")
  print()

  for i, sector in enumerate(grid._sectors):
    print(f"  [{i:>2d}] {sector}")

  print()
  p1 = IsometricPoint(grid, 0.05, 0.05)
  p2 = IsometricPoint(grid, 0.40, 0.40)
  s1 = grid.sector_containing(p1)
  s2 = grid.sector_containing(p2)
  print(f"p1 = {p1} -> {s1}")
  print(f"p2 = {p2} -> {s2}")
  print()

  # Place items
  s1.add_item(SectorItem(position=p1, payload="village"))
  s2.add_item(SectorItem(position=p2, payload="mountain", min_scale=0.0))

  # Subdivide a sector and place an item in the child
  child = s1.subdivide(resolution=3)
  child_point = IsometricPoint(child, 0.03, 0.03)
  child_sector = child.sector_containing(child_point)
  child_sector.add_item(SectorItem(
      position=child_point, payload="tavern", min_scale=2.0))

  print(f"Child grid: resolution={child.resolution}, "
        f"altitude={child.altitude:.4f}")
  print(f"Child sector for {child_point}: {child_sector}")
  print()

  # Render
  print("Render at scale=1.0:")
  for item in grid.render_2d(scale=1.0):
    print(f"  ({item.x:.4f}, {item.y:.4f}): {item.payload}")

  print()
  print("Render at scale=3.0 (tavern becomes visible):")
  for item in grid.render_2d(scale=3.0):
    print(f"  ({item.x:.4f}, {item.y:.4f}): {item.payload}")

  # Coordinate round-trip
  print()
  b_l, s_l, d_l = 0.03, 0.03, child.altitude - 0.06
  cx, cy = child.to_root_cartesian(b_l, s_l, d_l)
  rx, ry = isometric_to_cartesian(p1.b, p1.s, p1.d)
  print(f"Child local ({b_l:.2f}, {s_l:.2f}, {d_l:.2f}) "
        f"-> root Cartesian ({cx:.4f}, {cy:.4f})")

  print("         p1 -> p2:         ", p1.distance_from(p2))
  print("         p2 -> p1:         ", p2.distance_from(p1))
  print("         p1 -> child_point:", p1.distance_from(child_point))
  print("child_point -> p1:         ", child_point.distance_from(p1))
  print("         p2 -> child_point:", p2.distance_from(child_point))
  print("child_point -> p2:         ", child_point.distance_from(p2))


if __name__ == "__main__":
  main()
