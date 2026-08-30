from __future__ import annotations

from mundimonium.coordinates.exceptions import NotAdjacentException
from mundimonium.coordinates.hash_by_index import HashByIndex
from mundimonium.utils.helper_functions import argc

from collections.abc import Iterable
from enum import Enum
from numbers import Number
from typing import Self, override

import abc
import math


_SQRT_3 = math.sqrt(3.0)

BASE_TO_ALTITUDE = 0.5 * _SQRT_3
BASE_TO_APOTHEM = 0.5 / _SQRT_3
ALTITUDE_TO_BASE = 2.0 / _SQRT_3
ALTITUDE_TO_APOTHEM = 1.0 / 3.0
APOTHEM_TO_BASE = 2.0 * _SQRT_3
APOTHEM_TO_ALTITUDE = 3.0


def isometric_distance(
    component_axis_1: Number, component_axis_2: Number) -> Number:
  """
  Finds distance on the isometric grid given components in two isometric axes.

  By the law of cosines, this is sqrt(a**2 + b**2 - 2*a*b*cos(angle C)). The
  angle in question is always 60 degrees, making 2*cos(angle C) equal to 1.
  This leaves sqrt(a**2 + b**2 - a*b).
  """
  return math.sqrt(
      component_axis_1**2 + component_axis_2**2 -
      component_axis_1 * component_axis_2
  )


class IsometricDirection(Enum):
  """
  One of the three edge/vertex directions (B, S, D) on an equilateral triangle.
  """

  B = 0
  S = 1
  D = 2

  def rotated_cw_by_index(self, index: int) -> IsometricDirection:
    """Returns the direction `index` 120-deg steps clockwise of `self`."""
    return IsometricDirection(
        (self.value + index) % len(IsometricDirection))

  def rotated_ccw_by_index(self, index: int) -> IsometricDirection:
    """Returns the direction `index` 120-deg steps counterclockwise of `self`.
    """
    return IsometricDirection(
        (self.value - index) % len(IsometricDirection))

  def __repr__(self) -> str:
    """Lowercase single-letter form ('b', 's', 'd')."""
    return chr(ord(self.name) | 0x20)

  def __str__(self) -> str:
    """Lowercase single-letter form ('b', 's', 'd')."""
    return repr(self)


class IsometricGrid(abc.ABC):
  """Abstract base for a triangular grid that `IsometricPoint`s live on."""

  def __init__(self,
               side_length: Number | None = None,
               altitude: Number | None = None,
               apothem: Number | None = None,
               **kwargs):
    if sum(1 for x in (side_length, altitude, apothem) if x is not None) != 1:
      raise ValueError(
          "Exactly one of ('side_length', 'altitude', 'apothem') must be "
          "provided."
      )
    elif side_length is not None:
      altitude = side_length * BASE_TO_ALTITUDE
      apothem = side_length * BASE_TO_APOTHEM
    elif altitude is not None:
      side_length = altitude * ALTITUDE_TO_BASE
      apothem = altitude * ALTITUDE_TO_APOTHEM
    else:  # apothem is not None
      side_length = APOTHEM_TO_BASE
      altitude = APOTHEM_TO_ALTITUDE

    self._side_length = side_length
    self._altitude = altitude
    self._apothem = apothem

  @property
  def side_length(self) -> Number:
    """The length of one edge of the triangle."""
    return self._side_length

  @property
  def altitude(self) -> Number:
    """The distance from one vertex to the closest point on the opposite edge.

    Since the triangle is equilateral, the closest point is also the midpoint.
    """
    return self._altitude

  @property
  def apothem(self) -> Number:
    """The distance from the triangle's center to the midpoint of an edge."""
    return self._apothem

  @classmethod
  def distance(cls, p1: IsometricPoint, p2: IsometricPoint) -> Number | None:
    """Distance between two points that may or may not share a grid.

    If the two points share a grid, this will forward to `local_distance`.
    If the two points don't share a grid but are on adjacent/nearby grids (as
    defined by the implementation), this will forward to `nearby_grid_distance`.
    If the two points don't share a grid and aren't adjacent/nearby but are on
    transitively connected grids, this will forward to `geodesic_distance`.
    If the two points are entirely disjoint, this will return `None`.
    """
    if p1.grid is p2.grid:
      return p1.grid.local_distance(p1, p2)

    common_grid_type = cls.common_grid_type(p1, p2)

    distance = common_grid_type.nearby_grid_distance(p1, p2)
    if distance is not None:
      return distance

    return common_grid_type.geodesic_distance(p1, p2)

  @classmethod
  def local_distance(
      cls, p1: IsometricPoint, p2: IsometricPoint) -> Number | None:
    """Distance between two points on the same grid.

    If the two points aren't on the same grid, returns `None`.
    """
    if p1.grid is not p2.grid:
      return None
    b_component = p2.b - p1.b
    s_component = p2.s - p1.s
    return isometric_distance(
        b_component - 0.5 * s_component,
        s_component - 0.5 * b_component,
    )

  @classmethod
  @abc.abstractmethod
  def nearby_grid_distance(
      cls, p1: IsometricPoint, p2: IsometricPoint) -> Number | None:
    """Distance between two points on adjacent/nearby grids.

    The notion of adjacency or proximity depends on the implementation and may
    or may not apply. If it doesn't apply, this function should return None.
    """
    raise NotImplementedError()

  @classmethod
  @abc.abstractmethod
  def geodesic_distance(
      cls, p1: IsometricPoint, p2: IsometricPoint) -> Number | None:
    """Distance between two points along a geodesic through N connected grids.

    If the notion of connected grids doesn't apply to a given implementation,
    this function should return None.
    """
    raise NotImplementedError()

  def project_onto_root_grid(self, point: IsometricPoint):
    """Project `point` onto the root grid if this grid is a `NestedIsoGrid`.

    Otherwise, simply return `point` since this grid doesn't have a parent and
    is thus a root grid by default.
    """
    return point

  def to_world_coordinates(self, point: IsometricPoint) -> tuple[Number, ...]:
    """Converts a point on `self` to a tessellation-defined coordinate system.

    The length of the returned coordinate vector is tessellation-defined, as is
    the meaning of each individual coordinate in the vector.

    For example, on a spherical world, this would return a 2-vector of spherical
    coordinates `(colatitude, longitude)`, with the radial-distance coordinate
    implicitly equal to the world's radius.
    """
    raise NotImplementedError()

  @classmethod
  def common_grid_type(
      cls,
      *points: Iterable[IsometricPoint],
  ) -> type[IsometricGrid]:
    """
    Returns the most specific `IsometricGrid` subclass common to all `points`.

    Raises `TypeError` if the points share no common `IsometricGrid` subclass.
    """
    if not points:
      raise ValueError("No points provided.")

    classes = tuple(point.grid_type for point in points)
    cls_0 = classes[0]

    if all(c is cls_0 for c in classes[1:]):
      return cls_0

    common_bases = set.intersection(*(set(c.mro()) for c in classes[1:]))

    for base_class in cls_0.mro():
      if base_class in common_bases:
        return base_class

    raise TypeError(
        "Error: No common grid type between IsometricPoints. (At worst, any N "
        "IsometricGrids should all share the 'IsometricGrid' abstract type.)"
    )

  @classmethod
  def common_grid(cls, *points: Iterable[IsometricPoint]) -> IsometricGrid | None:
    """Returns a common grid containing all `points` if applicable, else `None`.
    """
    if not points:
      return None

    common_grid = points[0].grid

    if all(point.grid is common_grid for point in points[1:]):
      return common_grid

    return None


class IsometricPoint(HashByIndex):
  """
  A point on an `IsometricGrid`, addressed by `(b, s, d)` isometric coordinates.

  Each `(b, s, d)` coordinate represents the distance from a particular edge of
  the local `IsometricGrid`, which is an equilateral triangle:
  - Coordinate `b` represents distance from edge b (which is opposite vertex B).
  - Coordinate `s` represents distance from edge s (which is opposite vertex S).
  - Coordinate `d` represents distance from edge d (which is opposite vertex D).

  Only two of the three `(b, s, d)` coordinates are linearly independent; one of
  them is always implied by the other two. (Specifically, we store `(b, s)` and
  derive `d`.)
  """

  def __init__(self, grid: IsometricGrid, b: Number, s: Number):
    """Constructs a point at `(b, s, d)` on `grid`, where `d` is derived."""
    self._grid: IsometricGrid = grid
    self._b: Number = b
    self._s: Number = s

  @classmethod
  def center(cls, grid: IsometricGrid) -> Self:
    """Returns the centroid of `grid`."""
    return cls(grid, grid.apothem, grid.apothem)

  @classmethod
  def at_coordinates(
      cls,
      grid: IsometricGrid,
      b: Number | None = None,
      s: Number | None = None,
      d: Number | None = None,
  ) -> Self:
    """Constructs a point on `grid` from exactly two of `(b, s, d)`."""
    new_point = cls(grid, 0, 0)
    new_point.move_to(b, s, d)
    return new_point

  @classmethod
  def from_barycentric(
      cls,
      grid: IsometricGrid,
      wb: Number,
      ws: Number,
      wd: Number | None = None,
  ) -> Self:
    """Creates an IsometricPoint from barycentric coordinates `(wb, ws, wd)`."""
    alt = grid.altitude
    return cls(grid, wb * alt, ws * alt)

  @property
  def barycentric(self) -> tuple[Number, Number, Number]:
    """Returns normalized `(wb, ws, wd)` barycentric coordinates."""
    alt = self.grid.altitude
    return (self.b / alt, self.s / alt, self.d / alt)

  def project_onto_adjacent_grid(self, adjacent_grid) -> Self:
    """
    Initialize as a projection of point `other` onto `grid`. In order to be well
    defined, this requires that `other.grid` and `grid` share an edge.
    """
    if adjacent_grid is self.grid:
      return IsometricPoint(self.grid, self.b, self.s)
    altitude_mean = (adjacent_grid.altitude + self.grid.altitude) / 2
    old_border_edge = self.grid.direction_away_from_face(adjacent_grid)
    new_border_edge = adjacent_grid.direction_away_from_face(self.grid)
    local_complement_of_new_b = old_border_edge.rotated_ccw_by_index(
        new_border_edge.value)
    local_complement_of_new_s = \
        local_complement_of_new_b.rotated_cw_by_index(1)
    projected_point = IsometricPoint(
        adjacent_grid,
        altitude_mean - self[local_complement_of_new_b],
        altitude_mean - self[local_complement_of_new_s])
    if new_border_edge == IsometricDirection.B:
      projected_point._b -= altitude_mean
    elif new_border_edge == IsometricDirection.S:
      projected_point._s -= altitude_mean
    return projected_point

  def project_onto_root_grid(self) -> Self:
    """Project `self` onto the root grid if `self.grid` is a `NestedIsoGrid`.

    Otherwise, simply return `self` since the local grid doesn't have a parent
    and is thus a root grid by default.
    """
    return self.grid.project_onto_root_grid(self)

  def to_world_coordinates(self) -> tuple[Number, ...]:
    """Converts this point to the world's (mesh-defined) coordinate system."""
    return self.grid.to_world_coordinates(self)

  def distance_from(self, other: IsometricPoint) -> Number:
    """Distance from this point to `other` anywhere on the same mesh/world."""
    return self.grid.distance(self, other)

  def __repr__(self) -> str:
    """Debug form including this point's hash and grid."""
    return f"<id {hash(self)}: {str(self)} in grid {repr(self.grid)}>"

  def __str__(self) -> str:
    """The point as its `(b, s, d)` coordinates."""
    return f"({self.b}, {self.s}, {self.d})"

  def __getitem__(self, key: IsometricDirection) -> Number:
    """Returns the `b`, `s`, or `d` coordinate named by `key`."""
    if key == IsometricDirection.B:
      return self.b
    elif key == IsometricDirection.S:
      return self.s
    elif key == IsometricDirection.D:
      return self.d
    else:
      raise ValueError(f"{key} is not a valid IsometricDirection.")

  def __setitem__(self, key: IsometricDirection, item: Number) -> None:
    """Sets the `b`, `s`, or `d` coordinate named by `key`."""
    if key == IsometricDirection.B:
      self.b = item
    elif key == IsometricDirection.S:
      self.s = item
    elif key == IsometricDirection.D:
      self.d = item
    else:
      raise ValueError(f"{key} is not a valid IsometricDirection.")

  def __add__(self, vector: IsometricVector) -> IsometricPoint:
    """Returns a new `IsometricPoint` translated from `self` by `vector`."""
    assert type(vector) is IsometricVector, "Invalid __add__() operand."
    return IsometricPoint(
        self.grid, self.b + vector.delta_b,
        self.s + vector.delta_s)

  def _sub_point(self, other: IsometricPoint) -> IsometricVector:
    """The `IsometricVector` pointing from `other` to `self`."""
    return IsometricVector.with_net_b_s(self.b - other.b, self.s - other.s)

  def _sub_vector(self, vector: IsometricVector) -> IsometricPoint:
    """This point translated backward by an `IsometricVector`."""
    return IsometricPoint(
        self.grid, self.b - vector.delta_b,
        self.s - vector.delta_s)

  def __sub__(self, other):
    """Dispatches to `_sub_point` or `_sub_vector` depending on operand type."""
    if isinstance(other, IsometricPoint):
      return self._sub_point(other)
    elif isinstance(other, IsometricVector):
      return self._sub_vector(other)
    else:
      raise ValueError("Invalid __sub__() operand.")

  def move_to(
      self,
      b: Number | None = None,
      s: Number | None = None,
      d: Number | None = None,
  ) -> None:
    """Repositions this point in place, given exactly two of `(b, s, d)`."""
    assert argc(b, s, d) == 2, \
        "move_to() must be provided exactly two of (b, s, d)."
    if b is None:
      self.b = self.grid.side_length - s - d
      self.s = s
    elif s is None:
      self.b = b
      self.s = self.grid.side_length - b - d
    else:  # d is None
      self.b = b
      self.s = s

  # def move_by(
  #     self,
  #     b: Optional[Number] = None,
  #     s: Optional[Number] = None,
  #     d: Optional[Number] = None
  # ) -> None:
  #   assert argc(b, s, d) == 2, \
  #       "move_by() must be provided exactly two of (b,s,d)."
  #   if b is None:
  #     self.b -= (s + d)
  #     self.s += s
  #   elif s is None:
  #     self.b += b
  #     self.s -= (b + d)
  #   else:  # d is None
  #     self.b += b
  #     self.s += s

  @property
  def grid(self) -> IsometricGrid:
    """The `IsometricGrid` this point lives on."""
    return self._grid

  @property
  def grid_type(self) -> type[IsometricGrid]:
    """The concrete type of `self.grid`."""
    return type(self._grid)

  @property
  def b(self) -> Number:
    """Distance from the edge opposite vertex B."""
    return self._b

  @property
  def s(self) -> Number:
    """Distance from the edge opposite vertex S."""
    return self._s

  @property
  def d(self) -> Number:
    """Distance from the edge opposite vertex D, derived from `b` and `s`."""
    return self.grid.altitude - self._b - self._s

  @b.setter
  def b(self, b: Number) -> None:
    """Sets `b`, adjusting `s` to hold `d` fixed."""
    self._s -= 0.5 * (b - self._b)
    self._b = b

  @s.setter
  def s(self, s: Number) -> None:
    """Sets `s`, adjusting `b` to hold `d` fixed."""
    self._b -= 0.5 * (s - self._s)
    self._s = s

  @d.setter
  def d(self, d: Number) -> None:
    """Sets `d`, adjusting `b` and `s` equally to compensate."""
    delta = 0.5 * (d - self.d)
    self._b -= delta
    self._s -= delta


class IsometricVector:
  """Displacement on an isometric grid, stored as `(b_component, s_component)`.

  `d_component` is implicitly treated as zero since there are only two linearly
  independent coordinates.
  """

  def __init__(self, b_component: Number, s_component: Number):
    """Constructs a vector directly from its B and S components."""
    self._b_component = b_component
    self._s_component = s_component
    self._cached_length = None
    self._length_dirty = True

  @classmethod
  def with_net_b_s(cls, delta_b: Number, delta_s: Number):
    """Constructs a vector from its net changes along the B and S axes."""
    return IsometricVector(delta_b - 0.5 * delta_s, delta_s - 0.5 * delta_b)

  @classmethod
  def with_net_b_d(cls, delta_b: Number, delta_d: Number):
    """Constructs a vector from its net changes along the B and D axes."""
    return IsometricVector(
        delta_b - 0.5 * delta_d, -0.5 * (delta_b + delta_d))

  @classmethod
  def with_net_s_d(cls, delta_s: Number, delta_d: Number):
    """Constructs a vector from its net changes along the S and D axes."""
    return IsometricVector(
        -0.5 * (delta_s + delta_d), delta_s - 0.5 * delta_d)

  @classmethod
  def between_points(
      cls, start: IsometricPoint, end: IsometricPoint) -> IsometricVector:
    """Returns the vector pointing from `start` to `end`."""
    return end._sub_point(start)

  def unit_vector(self) -> IsometricVector:
    """Returns this vector scaled to unit length."""
    return IsometricVector(
        self.b_component / self.length, self.s_component / self.length)

  @property
  def b_component(self) -> Number:
    """B component of the vector. This is NOT the net change in B."""
    return self._b_component

  @property
  def s_component(self) -> Number:
    """S component of the vector. This is NOT the net change in S."""
    return self._s_component

  @property
  def d_component(self) -> Number:
    """
    D component of the vector. This is NOT the net change in D.

    Note that this component of the vector is always zero since only two
    linearly independent vectors are necessary to represent a 2D vector.
    More would overconstrain the system. The projection along D, however, is
    not constrained to zero.
    """
    return 0

  @b_component.setter
  def b_component(self, b_component: Number) -> None:
    """Sets `b_component`, adjusting `s_component` to hold `d_component` at 0.
    """
    self._length_dirty = True
    self._s_component -= 0.5 * (b_component - self._b_component)
    self._b_component = b_component

  @s_component.setter
  def s_component(self, s_component: Number) -> None:
    """Sets `s_component`, adjusting `b_component` to hold `d_component` at 0.
    """
    self._length_dirty = True
    self._b_component -= 0.5 * (s_component - self._s_component)
    self._s_component = s_component

  @d_component.setter
  def d_component(self, d_component: Number) -> None:
    """Always invalid: `d_component` is constrained to 0 and cannot be assigned.
    """
    raise TypeError("d_component is constrained to 0, but delta_d can be set.")

  @property
  def delta_b(self) -> Number:
    """The vector's net change along the B axis."""
    return self._b_component - 0.5 * self._s_component

  @property
  def delta_s(self) -> Number:
    """The vector's net change along the S axis."""
    return self._s_component - 0.5 * self._b_component

  @property
  def delta_d(self) -> Number:
    """The vector's net change along the D axis."""
    return -0.5 * (self._b_component + self._s_component)

  @delta_b.setter
  def delta_b(self, delta_b):
    """Sets the vector's net change along the B axis."""
    self.b_component += (delta_b - self.delta_b)

  @delta_b.setter
  def delta_s(self, delta_s):
    """Sets the vector's net change along the S axis."""
    self.s_component += (delta_s - self.delta_s)

  @delta_d.setter
  def delta_d(self, delta_d):
    """Sets the vector's net change along the D axis."""
    self._length_dirty = True
    delta_delta_b_s = 0.5 * (delta_d - self.delta_d)
    self._b_component -= delta_delta_b_s
    self._s_component -= delta_delta_b_s

  @property
  def length(self) -> Number:
    """This vector's length, computed lazily and cached until it's invalidated.
    """
    if self._length_dirty:
      self._cached_length = \
          isometric_distance(self._b_component, self._s_component)
      self._length_dirty = False
    return self._cached_length

  @length.setter
  def length(self, length) -> None:
    """Rescales this vector in place to the given length."""
    scale_factor = length / self.length
    self.b_component *= scale_factor
    self.s_component *= scale_factor
    # NOTE: This may cause rare issues with floating-point errors, but the
    # alternative of setting `self._length_dirth = True` would run the same
    # risks, albeit in different situations, should the call site assume the
    # newly set length exactly matched the provided value. This way is also
    # faster in many cases.
    self._cached_length = length
    self._length_dirty = False

  def __repr__(self) -> str:
    """The vector as its `(delta_b, delta_s, delta_d)` net changes."""
    return f"<{self.delta_b}, {self.delta_s}, {self.delta_d}>"

  def __str__(self) -> str:
    """Same as `__repr__`."""
    return repr(self)

  def __getitem__(self, key: IsometricDirection) -> Number:
    """Returns the net change along the axis named by `key`."""
    if key == IsometricDirection.B:
      return self.delta_b
    elif key == IsometricDirection.S:
      return self.delta_s
    elif key == IsometricDirection.D:
      return self.delta_d
    else:
      raise ValueError(f"{key} is not a valid IsometricDirection.")

  def __setitem__(self, key: IsometricDirection, item: Number) -> None:
    """Sets the net change along the axis named by `key`."""
    if key == IsometricDirection.B:
      self.delta_b = item
    elif key == IsometricDirection.S:
      self.delta_s = item
    elif key == IsometricDirection.D:
      self.delta_d = item
    else:
      raise ValueError(f"{key} is not a valid IsometricDirection.")

  def __add__(self, other: IsometricVector) -> IsometricVector:
    """Vector addition."""
    return IsometricVector(
        self._b_component + other.b_component,
        self._s_component + other.s_component)

  def __sub__(self, other: IsometricVector) -> IsometricVector:
    """Vector subtraction."""
    return IsometricVector(
        self._b_component - other.b_component,
        self._s_component - other.s_component)

  def __mul__(self, scalar: Number) -> IsometricVector:
    """Scales this vector by `scalar`."""
    return IsometricVector(
        self._b_component * scalar, self._s_component * scalar)

  def __truediv__(self, scalar: Number) -> IsometricVector:
    """Scales this vector by `1/scalar`."""
    return IsometricVector(
        self._b_component / scalar, self._s_component / scalar)
