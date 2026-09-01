from __future__ import annotations

from mundimonium.coordinates.exceptions import EndOfMeshSurfaceException
from mundimonium.coordinates.isometric import (
    IsometricGrid, IsometricPoint,
)
from mundimonium.utils import classproperty

from collections.abc import Generator
from dataclasses import dataclass
from numbers import Number
from typing import Any, override

import functools
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

  Children are stored in a flat list with O(1) lookup by index or by
  :class:`IsometricPoint`. Within each row *i_b*, upright and inverted children
  are interleaved for cache locality:
  ```
  flat_index = i_b * (2*N - i_b) + 2*i_s + (1 if inverted else 0)
  ```

  Parameters:
    resolution: Number of subdivisions per edge. Produces resolution^2 children.
                `None` for a leaf with no children.
    **kwargs:   Forwarded up the method resolution order.
  """

  def __init__(
      self,
      *,
      resolution: int | None = None,
      default_resolution: int = 2,
      _parent: NestingIsoGrid | None = None,
      _i_b: int = 0,
      _i_s: int = 0,
      _inverted: bool = False,
      **kwargs):
    """Constructs a grid node.

    `_parent`/`_i_b`/`_i_s`/`_inverted` are set internally by the parent during
    subdivision and shouldn't normally be passed under other circumstances.

    Args:
      resolution:         Number of subdivisions per edge. Produces resolution^2
                          children immediately if given. `None` for a leaf with
                          no children.
      default_resolution: The default subdivision resolution for this node's LOD
                          tree if subdivided after construction. (Default `2`.)
      _parent:            This node's parent node, or `None` for a root.
      _i_b:               This node's row index within `_parent`. (Unused for a
                          root.)
      _i_s:               This node's column index within `_parent`. (Unused for
                          a root.)
      _inverted:          Whether this node is a downward-pointing triangle
                          within `_parent`. (Unused for a root.)
      **kwargs:           Forwarded up the method resolution order.
    """
    super().__init__(**kwargs)
    self._resolution = resolution
    self._default_resolution = default_resolution

    # Position within parent (set internally during subdivision)
    self._parent = _parent
    self._i_b = _i_b
    self._i_s = _i_s
    self._inverted = _inverted

    # Relationship to LOD tree
    self._root: NestingIsoGrid
    self._depth: int
    if _parent is None:
      self._root = self
      self._depth = 0
    else:
      self._root = self._parent.root
      self._depth = self._parent.depth + 1

    # Content
    self._items: list[SectorItem] = []

    # Children (None = not subdivided)
    self._children: list[NestingIsoGrid] | None = None

    # Cache root transform
    self._root_offset_b: float
    self._root_offset_s: float
    self._root_offset_d: float
    self._root_sign: float
    self._cache_root_transform()

    # Cache bounding box
    self._bb_xmin: float
    self._bb_xmax: float
    self._bb_ymin: float
    self._bb_ymax: float
    self._cache_bounding_box()

    # Pre-create children if resolution is given
    if resolution is not None and resolution >= 1:
      self._init_children()

  @classproperty
  def child_type(cls) -> type[NestingIsoGrid]:
    """The type used for this grid's children when subdivided.

    By default, a grid's children will be of the same type as the parent grid.
    """
    return cls

  # ==================================================================
  # IsometricGrid abstract property implementations
  # ==================================================================

  @property
  def resolution(self) -> int | None:
    """Number of subdivisions per edge, or `None` if this grid is a leaf."""
    return self._resolution

  # ==================================================================
  # IsometricGrid abstract method implementations
  # ==================================================================

  @classmethod
  @override
  def nearby_grid_distance(
      cls, p1: IsometricPoint, p2: IsometricPoint,
  ) -> Number | None:
    """Distance between two points on adjacent/nearby grids.

    The notion of adjacency or proximity depends on the implementation and may
    or may not apply. If it doesn't apply, this function should return None.
    """
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
    """Not applicable at this level of abstraction.

    A bare `NestingIsoGrid` has no notion of meshes along which to calculate
    geodesics. (See `LodMeshSector` for the mesh-aware subclass.)
    """
    return None

  # ==================================================================
  # Tree-position properties
  # ==================================================================

  @property
  def parent(self) -> NestingIsoGrid | None:
    """The parent grid, or `None` if this is the root."""
    return self._parent

  @property
  def root(self) -> NestingIsoGrid:
    """The root grid of this LOD tree. (`self` if this is the root grid.)"""
    return self._root

  @property
  def depth(self) -> int:
    """The depth of this node in the LOD tree. (`0` for the root.)"""
    return self._depth

  @property
  def i_b(self) -> int:
    """Row index within the parent grid (`0` for the root)."""
    return self._i_b

  @property
  def i_s(self) -> int:
    """Column index within the parent grid (`0` for the root)."""
    return self._i_s

  @property
  def i_d(self) -> int:
    """Derived third index: `N - 1 - i_b - i_s` (upward) or
    `N - 2 - i_b - i_s` (inverted).  `0` for the root."""
    if self._parent is None:
      return 0
    N = self._parent._resolution
    return (N - 2 - self._i_b - self._i_s
            if self._inverted
            else N - 1 - self._i_b - self._i_s)

  @property
  def indices(self) -> tuple[int, int, int]:
    """Integer `(i_b, i_s, i_d)` position within the parent grid."""
    return (self._i_b, self._i_s, self.i_d)

  @property
  def inverted(self) -> bool:
    """`True` if this is a downward-pointing triangle."""
    return self._inverted

  @property
  def items(self) -> list[SectorItem]:
    """Renderable items placed in this triangle."""
    return self._items

  @property
  def children(self) -> list[NestingIsoGrid] | None:
    """Child triangles, or `None` if not subdivided."""
    return self._children

  # ==================================================================
  # Mutators
  # ==================================================================

  def add_item(self, item: SectorItem) -> None:
    """Add a renderable item to this triangle."""
    self._items.append(item)

  def subdivide(self, resolution: int | None = None) -> None:
    """Subdivide this triangle into `resolution**2` children.

    Each child is a `NestingIsoGrid` with
    `altitude = self.altitude / resolution`.
    Children start as leaves with no further subdivision.
    """
    if resolution is None:
      resolution = self._default_resolution

    if self._children is not None:
      raise ValueError("This grid has already been subdivided.")
    self._resolution = resolution
    self._init_children()

  # ==================================================================
  # Child lookup
  # ==================================================================

  def _flat_index(self, i_b: int, i_s: int, inverted: bool) -> int:
    """O(1) flat index into `self._children`."""
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

  @functools.singledispatchmethod
  def child_containing(
      self,
      b: float | IsometricPoint,
      s: float | None = None,
      d: float | None = None) -> NestingIsoGrid:
    """Find the child containing the given `IsometricPoint` or `(b,s,d)` coords.

    The point must be on this grid (`point.grid is self`).
    """
    raise TypeError(
        "Unexpected argument type(s) for 'NestingIsoGrid.child_containing'.")

  @child_containing.register(IsometricPoint)
  def _child_containing_point(self, point: IsometricPoint) -> NestingIsoGrid:
    if point.grid is not self:
      raise KeyError("The provided 'IsometricPoint' is not on this grid.")

    return self._child_containing_coords(point.b, point.s, point.d)

  @child_containing.register(float)
  def _child_containing_coords(
      self, b: float, s: float, d: float) -> NestingIsoGrid:
    if self._children is None:
      raise ValueError("This grid has not been subdivided.")

    N = self._resolution
    scale = N / self._altitude

    b_s = b * scale
    s_s = s * scale
    d_s = d * scale

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
    """Subscript access: `grid[point]` returns the child containing *point*."""
    return self.child_containing(point)

  # ==================================================================
  # Coordinate transforms
  # ==================================================================

  def local_to_parent(
      self,
      b_local: float, s_local: float, d_local: float,
  ) -> tuple[float, float, float]:
    """Convert this triangle's local isometric coords to parent coords.

    For an upright child at `(i_b, i_s, i_d)`:
    ```
    b_parent = i_b * h + b_local
    s_parent = i_s * h + s_local
    d_parent = i_d * h + d_local
    ```
    For an inverted child:
    ```
    b_parent = (i_b + 1) * h - b_local
    s_parent = (i_s + 1) * h - s_local
    d_parent = (i_d + 1) * h - d_local
    ```
    where `h = self.altitude` (this child's altitude, which equals the parent's
    per-sector height `parent.altitude / parent.resolution`).
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

    For inverted children, applies the same formula as :meth:`local_to_parent`.
    (The transformation is its own inverse).
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

    Uses the cached affine transformation:
    ```
    (b_root, s_root, d_root) = (off_b, off_s, off_d) + sign * (b, s, d)
    ```
    """
    return (
        self._root_offset_b + self._root_sign * b,
        self._root_offset_s + self._root_sign * s,
        self._root_offset_d + self._root_sign * d,
    )

  def to_root_cartesian(
      self, b: float, s: float, d: float,
  ) -> tuple[float, float]:
    """Converts local isometric (b, s, d) to root Cartesian (x, y).

    Equivalent to `isometric_to_cartesian(*self.to_root_isometric(b, s, d))`.
    """
    return isometric_to_cartesian(*self.to_root_isometric(b, s, d))

  @override
  def project_onto_root_grid(
      self, point: IsometricPoint, in_place: bool = False) -> IsometricPoint:
    """Project a local point onto the root grid's coordinate system."""
    root_b, root_s, root_d = self.to_root_isometric(
        point.b, point.s, point.d)
    if in_place:
      return point.update(grid=self._root, b=root_b, s=root_s)
    else:
      return IsometricPoint(self._root, root_b, root_s)

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
      raise EndOfMeshSurfaceException(
          "Cannot canonicalize an 'IsometricPoint' located outside the bounds "
          "of a root 'NestingIsoGrid' that isn't part of a mesh. "
          "(See 'LodMeshFace' for the mesh-aware implementation.)")

    for i in range(delta_depth):
      if grid.children is None:
        grid.subdivide()
      grid = grid.child_containing(b, s, d)
      b, s, d = grid.parent_to_local(b, s, d)

    return point.update(grid=grid, b=b, s=s)

  # ==================================================================
  # 2D rendering
  # ==================================================================

  def render_2d(
      self,
      scale: float,
      viewport: tuple[float, float, float, float] | None = None,
  ) -> Generator[RenderItem]:
    """Traverses the LOD tree and yield visible items as `RenderItem`s.

    Args:
      scale:    Current zoom scale. Items whose `[min_scale, max_scale)` range
                includes this value are yielded.
      viewport: Optional `(x_min, y_min, x_max, y_max)` culling rectangle in
                root Cartesian coordinates. Triangles whose bounding boxes don't
                intersect are skipped (along with their entire subtrees).

    Yields:
      `RenderItem(x, y, payload)` for each visible `SectorItem`.
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

    For any local point (b, s, d):
    ```
    (b_root, s_root, d_root) = (off_b, off_s, off_d) + sign * (b, s, d)
    ```
    """
    if self._parent is None:
      self._root_offset_b = 0.0
      self._root_offset_s = 0.0
      self._root_offset_d = 0.0
      self._root_sign = 1.0
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
    """Debug summary: orientation, altitude, resolution, item and child counts.
    """
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
  """Demo/smoke test exercising subdivision, lookup, transforms, and rendering.
  """
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
