from __future__ import annotations

from mundimonium.coordinates.isometric import (
    IsometricGrid, IsometricPoint,
)
from mundimonium.utils import classproperty

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
      position:  Location within the owning grid's local frame.
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
# NestingIsoGrid -- a recursive equilateral-triangle LOD node
# ---------------------------------------------------------------------------

class NestingIsoGrid(IsometricGrid):
  """Hierarchical equilateral-triangle LOD grid.

  Each NestingIsoGrid represents a single equilateral triangle that can:

  - Hold renderable items (with scale-dependent visibility).
  - Be subdivided into ``resolution**2`` child triangles, each of which
    is itself a NestingIsoGrid -- forming an arbitrarily deep LOD tree.
  - Convert local isometric coordinates to root-level Cartesian or
    isometric coordinates in **O(1)** time, regardless of nesting depth.

  Children are stored in a flat list with O(1) lookup by index or by
  :class:`IsometricPoint`.  Within each row *i_b*, upward and downward
  sub-triangles are interleaved for cache locality::

      flat_index = i_b * (2*N - i_b) + 2*i_s + (1 if inverted else 0)

  Parameters:
      resolution: Number of subdivisions per edge (N).  Produces N^2
                  children.  ``None`` for a leaf with no children.
      altitude:   Height of this triangle in world units.
  """

  BASE_TO_ALTITUDE = _SQRT3 / 2
  APOTHEM_TO_ALTITUDE = 3

  def __init__(
      self,
      *,
      resolution: int | None = None,
      altitude: float = 1.0,
      _parent: NestingIsoGrid | None = None,
      _i_b: int = 0,
      _i_s: int = 0,
      _inverted: bool = False,
      **kwargs):
    super().__init__(**kwargs)
    self._resolution = resolution
    self._altitude = float(altitude)
    self._side_length = 2.0 * self._altitude / _SQRT3
    self._apothem = self._altitude / 3.0

    # Position within parent (set internally during subdivision)
    self._parent = _parent
    self._i_b = _i_b
    self._i_s = _i_s
    self._inverted = _inverted

    # Content
    self._items: list[SectorItem] = []

    # Children (None = not subdivided)
    self._children: list[NestingIsoGrid] | None = None

    # Cache O(1) root transform and bounding box
    self._cache_root_transform()
    self._cache_bounding_box()

    # Pre-create children if resolution is given
    if resolution is not None and resolution >= 1:
      self._init_children()

  @classproperty
  def child_type(cls) -> type[NestingIsoGrid]:
    # By default, a grid's children will be of the same type as the parent grid.
    return cls

  # ==================================================================
  # IsometricGrid abstract property implementations
  # ==================================================================

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

  # ==================================================================
  # IsometricGrid abstract method implementations
  # ==================================================================

  @classmethod
  @override
  def nearby_grid_distance(
      cls, p1: IsometricPoint, p2: IsometricPoint,
  ) -> Number | None:
    grid_1 = p1.grid
    grid_2 = p2.grid
    if not isinstance(grid_1, cls) or not isinstance(grid_2, cls):
      return None

    root_grid = grid_1.root
    if root_grid is grid_2.root:
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

  # ==================================================================
  # Tree-position properties
  # ==================================================================

  @property
  def parent(self) -> NestingIsoGrid | None:
    """The parent grid, or ``None`` if this is the root."""
    return self._parent

  @property
  def root(self) -> NestingIsoGrid:
    """The root grid of this LOD tree."""
    return self._root

  @property
  def i_b(self) -> int:
    """Row index within the parent grid (``0`` for the root)."""
    return self._i_b

  @property
  def i_s(self) -> int:
    """Column index within the parent grid (``0`` for the root)."""
    return self._i_s

  @property
  def i_d(self) -> int:
    """Derived third index: ``N - 1 - i_b - i_s`` (upward) or
    ``N - 2 - i_b - i_s`` (inverted).  ``0`` for the root."""
    if self._parent is None:
      return 0
    N = self._parent._resolution
    return (N - 2 - self._i_b - self._i_s
            if self._inverted
            else N - 1 - self._i_b - self._i_s)

  @property
  def indices(self) -> tuple[int, int, int]:
    """Integer ``(i_b, i_s, i_d)`` position within the parent grid."""
    return (self._i_b, self._i_s, self.i_d)

  @property
  def inverted(self) -> bool:
    """``True`` if this is a downward-pointing triangle."""
    return self._inverted

  @property
  def items(self) -> list[SectorItem]:
    """Renderable items placed in this triangle."""
    return self._items

  @property
  def children(self) -> list[NestingIsoGrid] | None:
    """Child triangles, or ``None`` if not subdivided."""
    return self._children

  # ==================================================================
  # Mutators
  # ==================================================================

  def add_item(self, item: SectorItem) -> None:
    """Add a renderable item to this triangle."""
    self._items.append(item)

  def subdivide(self, resolution: int) -> None:
    """Subdivide this triangle into ``resolution**2`` children.

    Each child is a :class:`NestingIsoGrid` with
    ``altitude = self.altitude / resolution``.  Children start as
    leaves (no items, no further subdivision).
    """
    if self._children is not None:
      raise ValueError("This grid has already been subdivided.")
    self._resolution = resolution
    self._init_children()

  # ==================================================================
  # Child lookup
  # ==================================================================

  def _flat_index(self, i_b: int, i_s: int, inverted: bool) -> int:
    """O(1) flat index into ``self._children``."""
    N = self._resolution
    if i_b < 0 or i_s < 0 or i_b >= N or i_s >= N - i_b:
      raise IndexError(
          f"Child indices (i_b={i_b}, i_s={i_s}) out of range "
          f"for resolution {N}.")
    if inverted and i_s >= N - 1 - i_b:
      raise IndexError(
          f"No inverted child at (i_b={i_b}, i_s={i_s}) for "
          f"resolution {N}.")
    return i_b * (2 * N - i_b) + 2 * i_s + (1 if inverted else 0)

  def child_at(self, i_b: int, i_s: int, inverted: bool) -> NestingIsoGrid:
    """Look up a child by its grid indices and orientation."""
    if self._children is None:
      raise ValueError("This grid has not been subdivided.")
    return self._children[self._flat_index(i_b, i_s, inverted)]

  def child_containing(self, point: IsometricPoint) -> NestingIsoGrid:
    """Find the child that contains a given :class:`IsometricPoint`.

    The point must be on this grid (``point.grid is self``).
    """
    if point.grid is not self:
      raise KeyError("The provided IsometricPoint is not on this grid.")
    if self._children is None:
      raise ValueError("This grid has not been subdivided.")

    N = self._resolution
    scale = N / self._altitude

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
      # Floating-point edge case -- fall back to upward triangle
      inverted = False
      i_s = min(i_s, N - 1 - i_b)

    return self._children[self._flat_index(i_b, i_s, inverted)]

  def __getitem__(self, point: IsometricPoint) -> NestingIsoGrid:
    """Subscript access: ``grid[point]`` returns the child containing *point*."""
    return self.child_containing(point)

  # ==================================================================
  # Coordinate transforms
  # ==================================================================

  def local_to_parent(
      self,
      b_local: float, s_local: float, d_local: float,
  ) -> tuple[float, float, float]:
    """Convert this triangle's local isometric coords to parent coords.

    For an **upward** child at ``(i_b, i_s, i_d)``::

        b_parent = i_b * h + b_local
        s_parent = i_s * h + s_local
        d_parent = i_d * h + d_local

    For a **downward** (inverted) child (self-inverse)::

        b_parent = (i_b + 1) * h - b_local
        s_parent = (i_s + 1) * h - s_local
        d_parent = (i_d + 1) * h - d_local

    where ``h = self.altitude`` (this child's altitude, which equals
    the parent's per-sector height ``parent.altitude / parent.resolution``).
    """
    if self._parent is None:
      raise ValueError("Root grid has no parent to transform into.")
    h = self._altitude
    i_b, i_s, i_d = self._i_b, self._i_s, self.i_d
    if self._inverted:
      return (
          (i_b + 1) * h - b_local,
          (i_s + 1) * h - s_local,
          (i_d + 1) * h - d_local,
      )
    else:
      return (
          i_b * h + b_local,
          i_s * h + s_local,
          i_d * h + d_local,
      )

  def parent_to_local(
      self,
      b_parent: float, s_parent: float, d_parent: float,
  ) -> tuple[float, float, float]:
    """Convert parent grid coords to this triangle's local coords.

    For upward children, subtracts the sector origin.
    For inverted children, applies the same formula as
    :meth:`local_to_parent` (the transform is its own inverse).
    """
    if self._parent is None:
      raise ValueError("Root grid has no parent to transform from.")
    h = self._altitude
    i_b, i_s, i_d = self._i_b, self._i_s, self.i_d
    if self._inverted:
      return (
          (i_b + 1) * h - b_parent,
          (i_s + 1) * h - s_parent,
          (i_d + 1) * h - d_parent,
      )
    else:
      return (
          b_parent - i_b * h,
          s_parent - i_s * h,
          d_parent - i_d * h,
      )

  def to_root_isometric(
      self, b: float, s: float, d: float,
  ) -> tuple[float, float, float]:
    """Convert local isometric coords to root-level isometric coords in O(1).

    Uses the cached affine transform::

        (b_root, s_root, d_root) = (off_b, off_s, off_d) + sign * (b, s, d)

    The cache is built at construction time, so this is O(1) regardless
    of how deeply nested this grid is.
    """
    return (
        self._root_offset_b + self._root_sign * b,
        self._root_offset_s + self._root_sign * s,
        self._root_offset_d + self._root_sign * d,
    )

  def to_root_cartesian(
      self, b: float, s: float, d: float,
  ) -> tuple[float, float]:
    """Convert local isometric (b, s, d) to root Cartesian (x, y) in O(1).

    Equivalent to ``isometric_to_cartesian(*self.to_root_isometric(b, s, d))``.
    """
    return isometric_to_cartesian(*self.to_root_isometric(b, s, d))

  @override
  def project_onto_root_grid(self, point: IsometricPoint) -> IsometricPoint:
    """Project a local point onto the root grid's coordinate system."""
    root_b, root_s, root_d = self.to_root_isometric(
        point.b, point.s, point.d)
    return IsometricPoint(self._root, root_b, root_s)

  # ==================================================================
  # 2D rendering
  # ==================================================================

  def render_2d(
      self,
      scale: float,
      viewport: tuple[float, float, float, float] | None = None,
  ) -> Generator[RenderItem]:
    """Traverse the LOD tree and yield visible items as RenderItems.

    Args:
        scale:    Current zoom scale.  Items whose ``[min_scale,
                  max_scale)`` range includes this value are yielded.
        viewport: Optional ``(x_min, y_min, x_max, y_max)`` culling
                  rectangle in root Cartesian coordinates.  Triangles
                  whose bounding box doesn't intersect are skipped
                  (along with their entire subtree).

    Yields:
        ``RenderItem(x, y, payload)`` for each visible SectorItem.
    """
    # Viewport culling: skip this entire subtree if out of view
    if viewport is not None and not self._intersects_viewport(viewport):
      return

    # Yield visible items at this node
    for item in self._items:
      if item.min_scale <= scale < item.max_scale:
        x, y = self.to_root_cartesian(
            item.position.b, item.position.s, item.position.d)
        yield RenderItem(x, y, item.payload)

    # Recurse into children
    if self._children is not None:
      for child in self._children:
        yield from child.render_2d(scale, viewport)

  # ==================================================================
  # Private helpers
  # ==================================================================

  def _init_children(self) -> None:
    """Create N^2 child triangles."""
    N = self._resolution
    h_sub = self._altitude / N
    self._children = []
    for i_b in range(N):
      for i_s in range(N - i_b):
        # Upward child
        self._children.append(self.child_type(
            altitude=h_sub,
            _parent=self, _i_b=i_b, _i_s=i_s, _inverted=False,
        ))
        # Downward child (exists when there is a next column)
        if i_s < N - 1 - i_b:
          self._children.append(self.child_type(
              altitude=h_sub,
              _parent=self, _i_b=i_b, _i_s=i_s, _inverted=True,
          ))
    assert len(self._children) == N * N

  def _cache_root_transform(self) -> None:
    """Compute the cached affine transform to root isometric coordinates.

    For any local point (b, s, d)::

        (b_root, s_root, d_root) = (off_b, off_s, off_d) + sign * (b, s, d)

    The transform composes from the parent's cached transform and this
    child's ``local_to_parent`` offset:

    - Upward child:   ``local_offset = (i_b*h, i_s*h, i_d*h)``,  ``local_sign = +1``
    - Inverted child:  ``local_offset = ((i_b+1)*h, (i_s+1)*h, (i_d+1)*h)``, ``local_sign = -1``
    - Composition: ``root_offset = parent_offset + parent_sign * local_offset``,
      ``root_sign = parent_sign * local_sign``
    """
    if self._parent is None:
      self._root_offset_b = 0.0
      self._root_offset_s = 0.0
      self._root_offset_d = 0.0
      self._root_sign = 1.0
      self._root = self
    else:
      h = self._altitude
      i_b, i_s, i_d = self._i_b, self._i_s, self.i_d

      if self._inverted:
        local_off_b = (i_b + 1) * h
        local_off_s = (i_s + 1) * h
        local_off_d = (i_d + 1) * h
        local_sign = -1.0
      else:
        local_off_b = i_b * h
        local_off_s = i_s * h
        local_off_d = i_d * h
        local_sign = 1.0

      parent_sign = self._parent._root_sign
      self._root_offset_b = (self._parent._root_offset_b
                              + parent_sign * local_off_b)
      self._root_offset_s = (self._parent._root_offset_s
                              + parent_sign * local_off_s)
      self._root_offset_d = (self._parent._root_offset_d
                              + parent_sign * local_off_d)
      self._root_sign = parent_sign * local_sign
      self._root = self._parent._root

  def _cache_bounding_box(self) -> None:
    """Compute the AABB in root Cartesian coordinates."""
    h = self._altitude
    vb = self.to_root_cartesian(h, 0.0, 0.0)
    vs = self.to_root_cartesian(0.0, h, 0.0)
    vd = self.to_root_cartesian(0.0, 0.0, h)
    self._bb_xmin = min(vb[0], vs[0], vd[0])
    self._bb_xmax = max(vb[0], vs[0], vd[0])
    self._bb_ymin = min(vb[1], vs[1], vd[1])
    self._bb_ymax = max(vb[1], vs[1], vd[1])

  def _intersects_viewport(
      self, viewport: tuple[float, float, float, float],
  ) -> bool:
    """Check if this triangle's AABB intersects the viewport."""
    vp_xmin, vp_ymin, vp_xmax, vp_ymax = viewport
    return not (
        self._bb_xmax < vp_xmin or self._bb_xmin > vp_xmax or
        self._bb_ymax < vp_ymin or self._bb_ymin > vp_ymax
    )

  # ==================================================================
  # Display
  # ==================================================================

  def __repr__(self) -> str:
    kind = "v" if self._inverted else "^"
    children_str = (f"{len(self._children)}"
                    if self._children else "none")
    return (
        f"NestingIsoGrid({kind} altitude={self._altitude:.4g}, "
        f"res={self._resolution}, "
        f"items={len(self._items)}, children={children_str})")


# ---------------------------------------------------------------------------
# Main -- demo / smoke test
# ---------------------------------------------------------------------------

def main():
  grid = NestingIsoGrid(resolution=4, altitude=1.0)
  print(f"Root grid: resolution={grid.resolution}, altitude={grid.altitude}, "
        f"side_length={grid.side_length:.4f}, apothem={grid.apothem:.4f}")
  print(f"Children: {len(grid._children)} (expected {grid.resolution**2})")
  print()

  for i, child in enumerate(grid._children):
    kind = "v" if child.inverted else "^"
    print(f"  [{i:>2d}] {kind} i_b={child.i_b}, i_s={child.i_s}, "
          f"i_d={child.i_d}")

  print()
  p1 = IsometricPoint(grid, 0.05, 0.05)
  p2 = IsometricPoint(grid, 0.40, 0.40)
  c1 = grid.child_containing(p1)
  c2 = grid.child_containing(p2)
  print(f"p1 = {p1} -> {c1}")
  print(f"p2 = {p2} -> {c2}")
  print()

  # Place items on children (positions in the child's local frame)
  p1_local = IsometricPoint(c1, 0.05, 0.05)
  c1.add_item(SectorItem(position=p1_local, payload="village"))
  p2_local = IsometricPoint(c2, 0.10, 0.10)
  c2.add_item(SectorItem(position=p2_local, payload="mountain"))

  # Subdivide a child and place an item in a grandchild
  c1.subdivide(resolution=3)
  grandchild = c1.child_at(0, 0, False)
  gc_point = IsometricPoint(grandchild, 0.03, 0.03)
  grandchild.add_item(SectorItem(
      position=gc_point, payload="tavern", min_scale=2.0))

  print(f"Subdivided child: {c1}")
  print(f"Grandchild: {grandchild}")
  print()

  # Render at different scales
  print("Render at scale=1.0:")
  for item in grid.render_2d(scale=1.0):
    print(f"  ({item.x:.4f}, {item.y:.4f}): {item.payload}")

  print()
  print("Render at scale=3.0 (tavern becomes visible):")
  for item in grid.render_2d(scale=3.0):
    print(f"  ({item.x:.4f}, {item.y:.4f}): {item.payload}")

  # Verify O(1) to_root_cartesian matches to_root_isometric
  print()
  b_l, s_l = 0.03, 0.03
  d_l = grandchild.altitude - b_l - s_l
  cx, cy = grandchild.to_root_cartesian(b_l, s_l, d_l)
  rb, rs, rd = grandchild.to_root_isometric(b_l, s_l, d_l)
  ex, ey = isometric_to_cartesian(rb, rs, rd)
  print(f"Grandchild to_root_cartesian: ({cx:.6f}, {cy:.6f})")
  print(f"Grandchild to_root_isometric: ({ex:.6f}, {ey:.6f})")
  assert abs(cx - ex) < 1e-12 and abs(cy - ey) < 1e-12, "Mismatch!"
  print("O(1) cached transform matches recursive transform: OK")

  print()
  print("      p1 -> p2:      ", p1.distance_from(p2))
  print("      p2 -> p1:      ", p2.distance_from(p1))
  print("      p1 -> gc_point:", p1.distance_from(gc_point))
  print("gc_point -> p1:      ", gc_point.distance_from(p1))
  print("      p2 -> gc_point:", p2.distance_from(gc_point))
  print("gc_point -> p2:      ", gc_point.distance_from(p2))


if __name__ == "__main__":
  main()
