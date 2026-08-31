from __future__ import annotations

from mundimonium.coordinates.exceptions import (
    NotAdjacentException, EndOfMeshSurfaceException
)
from mundimonium.coordinates.hash_by_index import HashByIndex
from mundimonium.coordinates.isometric import (
    IsometricDirection, IsometricGrid, IsometricPoint, IsometricVector,
    isometric_distance
)

import abc
import functools
import itertools
import math
from numbers import Number
from typing import get_type_hints, override
import numpy as np


_EQUILATERAL_TRIANGLE_AREA_TO_BASE_SQUARED = 4.0 / math.sqrt(3.0)


class _TessellationImplMetadata:
  """Metadata about a subclass of `Tessellation`.

  Instantiated for a given `Tessellation` subclass when that subclass is defined
  (via an `__init_subclass__` hook).
  """

  def __init__(self, impl: type[Tessellation]):
    """Initializes the metadata object pertaining to a given subclass `impl`.

    Records metadata about `impl`, and validates that `impl.__init__` accepts
    (and hopefully forwards) the keyword arguments of `Tessellation.__init__`.
    """
    base = Tessellation
    base_init_params = get_type_hints(base.__init__, include_extras=True)
    impl_init_params = get_type_hints(impl.__init__, include_extras=True)
    for param, type_hint in base_init_params.items():
      if param in impl_init_params:
        import pprint
        raise AttributeError(
            "A 'Tessellation' implementation must not shadow the base class's "
            "'__init__' keyword arguments:\n" + pprint.pformat(base_init_params)
        )
    self._init_params = impl_init_params


class Tessellation(abc.ABC):
  """
  Abstract base class for equilateral triangular meshes.
  """

  _impls: dict[type[Tessellation], _TessellationImplMetadata] = {}

  def __init_subclass__(cls, **kwargs):
    """Registers each `Tessellation` subclass and validates its `__init__`."""
    assert cls not in Tessellation._impls
    Tessellation._impls[cls] = _TessellationImplMetadata(cls)

  def __init__(self,
               *,
               vertex_type: type[TessellationVertex] | None = None,
               face_type: type[TessellationFace] | None = None,
               face_side_length: Number = 1):
    """Constructs an empty tessellation with no vertices or faces."""
    if vertex_type is None:
      vertex_type = TessellationVertex

    if face_type is None:
      face_type = TessellationFace

    self._vertex_type: type[TessellationVertex] = vertex_type
    self._face_type: type[TessellationFace] = face_type
    self._face_side_length: Number = face_side_length
    self._vertex_graph: dict[TessellationVertex, list[TessellationVertex]] = {}
    self._vertices: list[TessellationVertex] = []
    self._faces: list[TessellationFace] = []
    self._vertex_index_map: dict[TessellationVertex, int] = {}
    self._face_index_map: dict[TessellationFace, int] = {}

  @property
  def vertex_type(self) -> type[TessellationVertex]:
    """The `TessellationVertex` subclass used for this tessellation's vertices.
    """
    return self._vertex_type

  @property
  def face_type(self) -> type[TessellationFace]:
    """The `TessellationFace` subclass used for this tessellation's faces."""
    return self._face_type

  @abc.abstractmethod
  def coords_at_point(self, point: IsometricPoint) -> tuple[Number, ...]:
    """
    Returns the coordinates of `point` in this tessellation's coordinate system.

    The coordinate system of the returned tuple (including the number of
    coordinates in the tuple) is implementation-defined.

    Inverse of `new_point_at_coords`.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def new_point_at_coords(
      self, *coords: tuple[Number, ...]) -> IsometricPoint | None:
    """Returns a new `IsometricPoint` at the specified coordinates.

    The coordinate system used by `coords` (including the number of coordinates
    in the tuple) is implementation-defined.

    Inverse of `new_point_at_coords`.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_face_at_coords(
      self, *coords: tuple[Number, ...]) -> TessellationFace | None:
    """Returns the face containing the specified coordinates.

    The coordinate system used by `coords` (including the number of coordinates
    in the tuple) is implementation-defined.
    """
    raise NotImplementedError()

  def on_vertex_added(self, vertex: TessellationVertex) -> None:
    """Hook for subclasses to update internal state when mesh topology changes.

    Implementation is optional. If implemented, the implementation should call
    `super().on_vertex_added(vertex)` to safely handle potential MRO traversal.
    """
    pass

  def on_face_added(self, face: TessellationFace) -> None:
    """Hook for subclasses to update internal state when mesh topology changes.

    Implementation is optional. If implemented, the implementation should call
    `super().on_face_added(face)` to safely handle potential MRO traversal.
    """
    pass

  def add_vertex(
      self,
      new_vertex: TessellationVertex,
      adjacent_vertices: list[TessellationVertex] | None = None) -> int:
    """
    Registers `new_vertex` (linking `adjacent_vertices`) and returns its index.
    """
    if adjacent_vertices is None:
      adjacent_vertices = list()
    self._vertex_graph[new_vertex] = list(adjacent_vertices)
    for vertex in adjacent_vertices:
      if vertex in self._vertex_graph:
        self._vertex_graph[vertex].append(new_vertex)

    if new_vertex not in self._vertex_index_map:
      idx = len(self._vertices)
      self._vertex_index_map[new_vertex] = idx
      self._vertices.append(new_vertex)
      new_vertex.tessellation = self
      self.on_vertex_added(new_vertex)
    return self._vertex_index_map[new_vertex]

  def register_face(self, face: TessellationFace) -> int:
    """Registers an existing `TessellationFace` in this `Tessellation`."""
    if not isinstance(face, self.face_type):
      raise TypeError("'face' is not an instance of type 'self.face_type'")

    if face not in self._face_index_map:
      idx = len(self._faces)
      self._faces.append(face)
      self._face_index_map[face] = idx
      face.tessellation = self
      for v in face._adjacent_vertices:
        if v not in self._vertex_index_map:
          self.add_vertex(v)
      self.on_face_added(face)
    return self._face_index_map[face]

  def add_face(
      self,
      bounding_vertices: list[TessellationVertex] | TessellationFace,
      *,
      side_length: Number | None = None,
      area: Number | None = None,
      linear_distortion_factor: Number | None = None,
      area_distortion_factor: Number | None = None,
  ) -> TessellationFace:
    """Creates and registers a new face bounded by three vertices.

    `side_length`, `area`, `linear_distortion_factor`, and
    `area_distortion_factor` are optional and mutually exclusive. If none are
    passed, the tessellation's default `_face_side_length` is used. It is an
    error to pass multiple of these arguments.

    Args:
      bounding_vertices:        A list of the three bounding vertices.
      side_length:              An explicit side length for the face. Overrides
                                the tessellation's default `_face_side_length`.
      area:                     An explicit area for the face. Overrides the
                                tessellation's default `_face_side_length`.
      linear_distortion_factor: A 1D scale multiplier for the face relative to
                                the tessellation's default `_face_side_length`.
      area_distortion_factor:   A 2D scale multiplier for the face relative to
                                the area implied by the tessellation's default
                                `_face_side_length`.

    Returns:
      The newly created and registered face.
    """
    if len(bounding_vertices) != 3:
      raise ValueError("A face must be bounded by exactly three vertices.")

    if sum(1 for arg in (side_length, area, linear_distortion_factor,
                         area_distortion_factor)
           if arg is not None) > 1:
      raise ValueError(
          "Arguments 'side_length', 'area', 'linear_distortion_factor', and "
          "'area_distortion_factor' are mutually exclusive.")
    elif area is not None:
      side_length = math.sqrt(area * _EQUILATERAL_TRIANGLE_AREA_TO_BASE_SQUARED)
    elif linear_distortion_factor is not None:
      side_length = self._face_side_length * linear_distortion_factor
    elif area_distortion_factor is not None:
      side_length = self._face_side_length * math.sqrt(area_distortion_factor)
    elif side_length is None:
      side_length = self._face_side_length

    new_face = self.face_type(
        side_length=side_length,
        vertex_b=bounding_vertices[0],
        vertex_s=bounding_vertices[1],
        vertex_d=bounding_vertices[2],
    )
    self.register_face(new_face)
    return new_face

  def distance(self, p1: IsometricPoint, p2: IsometricPoint) -> Number:
    """Computes geodesic distance between two arbitrary points on the mesh.

    If the points are close to one another (on the same face or nearby faces),
    uses faster local calculations instead of general geodesic calculations.
    This may result in slight numerical discontinuities at face boundaries.
    """
    return self.face_type.distance(p1, p2)

  @abc.abstractmethod
  def geodesic_distance(
      self, p1: IsometricPoint, p2: IsometricPoint) -> Number | None:
    """Distance between two points along a geodesic through connected faces."""
    raise NotImplementedError()

  @abc.abstractmethod
  def shortest_path(
      self,
      p1: IsometricPoint,
      p2: IsometricPoint
  ) -> list[IsometricPoint]:
    """
    Traces the geodesic path from p1 to p2 across faces.

    Subclasses should override this method with specific path tracers.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def geodesically_canonicalize_point(
      self, point: IsometricPoint) -> IsometricPoint:
    """Moves `point` to a new grid if located outside its current grid's bounds.

    Mutates and returns `point`, not a copy.
    """
    raise NotImplementedError()


class TessellationVertex(HashByIndex):
  """A vertex of a `Tessellation`, positioned in 3D projection space."""

  def __init__(self, projection_coordinates: list[Number]):
    """Constructs a vertex at `projection_coordinates` with no adjacent faces.
    """
    self._projection_coordinates = projection_coordinates
    self._adjacent_faces = list()
    self._tessellation: Tessellation | None = None

  @property
  def tessellation(self) -> Tessellation | None:
    """The `Tessellation` this vertex belongs to, or `None` if unregistered."""
    return self._tessellation

  @tessellation.setter
  def tessellation(self, val: Tessellation) -> None:
    """Sets the `Tessellation` this vertex belongs to."""
    self._tessellation = val

  def add_adjacent_face(self, face: TessellationFace) -> None:
    """Registers `face` as adjacent to this vertex. Refreshes local adjacencies.
    """
    if face in self._adjacent_faces:
      return

    self._adjacent_faces.append(face)

    for other_face in self._adjacent_faces:
      if other_face is not face:
        other_face.recalculate_adjacency_to(face)

  def is_adjacent_to_face(self, face: TessellationFace) -> bool:
    """Whether this vertex is adjacent to `face`."""
    return face in self._adjacent_faces

  def is_adjacent_to_vertex(self, vertex: TessellationVertex) -> bool:
    """Whether `vertex` shares an edge with this vertex on some common face."""
    for face in self._adjacent_faces:
      if face.is_adjacent_to_vertex(vertex):
        return vertex is not self
    return False

  @functools.singledispatchmethod
  def is_adjacent_to(self, other):
    """Whether `other` (a face or vertex) is adjacent to this face."""
    # Dispatches to `is_adjacent_to_face` or `is_adjacent_to_vertex`. These
    # delegations are registered outside of the class body, when both
    # `TessellationVertex` and `TessellationFace` are fully defined.
    raise TypeError(f"Unexpected 'other' type '{type(other).__name__}'")

  @property
  def x(self) -> Number:
    """X projection coordinate."""
    return self._projection_coordinates[0]

  @property
  def y(self) -> Number:
    """Y projection coordinate."""
    return self._projection_coordinates[1]

  @property
  def z(self) -> Number:
    """Z projection coordinate."""
    return self._projection_coordinates[2]

  @property
  def projection_coordinates(self):
    """The (x, y, z) projection coordinates as a tuple."""
    return tuple(self._projection_coordinates)

  @x.setter
  def x(self, new_x: Number) -> None:
    """Sets the X coordinate and refreshes adjacent faces' centroids."""
    self._projection_coordinates[0] = new_x
    for face in self._adjacent_faces:
      face.recalculate_centroid_projection_coords()

  @y.setter
  def y(self, new_y: Number) -> None:
    """Sets the Y coordinate and refreshes adjacent faces' centroids."""
    self._projection_coordinates[1] = new_y
    for face in self._adjacent_faces:
      face.recalculate_centroid_projection_coords()

  @z.setter
  def z(self, new_z: Number) -> None:
    """Sets the Z coordinate and refreshes adjacent faces' centroids."""
    self._projection_coordinates[2] = new_z
    for face in self._adjacent_faces:
      face.recalculate_centroid_projection_coords()

  def adjacent_faces(self) -> list[TessellationFace]:
    """The faces that touch this vertex."""
    return self._adjacent_faces


class TessellationFace(IsometricGrid):
  def __init__(
      self,
      *,
      vertex_b: TessellationVertex,
      vertex_s: TessellationVertex,
      vertex_d: TessellationVertex,
      **kwargs):
    """Constructs a face bounded by three given vertices."""
    super().__init__(**kwargs)
    self._adjacent_faces = [None] * len(IsometricDirection)
    self._adjacent_vertices = [vertex_b, vertex_s, vertex_d]
    self._tessellation: Tessellation | None = None

    for vertex in self._adjacent_vertices:
      vertex.add_adjacent_face(self)

    self._centroid_projection_coords = None
    self.recalculate_centroid_projection_coords()

  @classmethod
  @override
  def nearby_grid_distance(
      cls, p1: IsometricPoint, p2: IsometricPoint) -> Number | None:
    """Distance between two points on adjacent faces, projecting `p1` across
    the shared edge; `None` if the faces aren't adjacent."""
    grid_1 = p1.grid
    grid_2 = p2.grid
    if not isinstance(grid_1, cls) or not isinstance(grid_2, cls):
      return None

    if grid_1.is_adjacent_to_face(grid_2):
      return p1.project_onto_adjacent_grid(grid_2).distance_from(p2)

    return None

  @classmethod
  @override
  def geodesic_distance(cls, p1: IsometricPoint, p2: IsometricPoint) -> Number:
    """Delegates to the shared tessellation's geodesic distance solver."""
    return p1.grid.tessellation.geodesic_distance(p1, p2)

  @override
  def to_mesh_coordinates(self, point: IsometricPoint) -> tuple[Number, ...]:
    """Converts `point` to the owning tessellation's coordinate system."""
    return self._tessellation.coords_at_point(point)

  @classmethod
  @override
  def canonicalize_point(cls, point: IsometricPoint) -> IsometricPoint:
    """Moves `point` to a new grid if located outside its current grid's bounds.

    Mutates and returns `point`, not a copy.
    """
    grid: TessellationFace = point.grid
    if not isinstance(grid, TessellationFace):
      raise TypeError(
          "The provided 'IsometricPoint' is not on a grid of type "
          "'TessellationFace'.")
    altitude: Number = grid.altitude
    project_to_face_on_edge: IsometricDirection | None = None
    if point.b < 0:
      if point.s <= altitude and point.d <= altitude:
        project_to_face_on_edge = IsometricDirection.B
    elif point.s < 0:
      if point.b <= altitude and point.d <= altitude:
        project_to_face_on_edge = IsometricDirection.B
    elif point.d < 0:
      if point.b <= altitude and point.s <= altitude:
        project_to_face_on_edge = IsometricDirection.B
    else:
      return point

    if project_to_face_on_edge is None:
      return grid._tessellation.geodesically_canonicalize_point(point)

    new_grid = grid.face_on_edge(project_to_face_on_edge)
    if new_grid is None:
      raise EndOfMeshSurfaceException(
          "Cannot canonicalize point located outside of mesh-surface boundary.")
    point.project_onto_adjacent_grid(new_grid, in_place=True)

    # If the grids have different altitudes, the projection isn't exact and
    # might land slightly outside of `new_grid`. To account for this, repeat the
    # canonicalization process if the altitudes differ.
    # (If the point landed inside of `new_grid` anyway, which should happen most
    # of the time assuming scale distortion is small, the repeat will quickly
    # return the unmodified point.)
    if new_grid.altitude != altitude:
      new_grid.canonicalize_point(point)
    return point


  @property
  def tessellation(self) -> Tessellation | None:
    """The `Tessellation` this face belongs to, or `None` if unregistered."""
    return self._tessellation

  @tessellation.setter
  def tessellation(self, val: Tessellation) -> None:
    """Sets the `Tessellation` this face belongs to."""
    self._tessellation = val

  @property
  def vertex_b(self) -> TessellationVertex:
    """The vertex in the B direction."""
    return self._adjacent_vertices[0]

  @property
  def vertex_s(self) -> TessellationVertex:
    """The vertex in the S direction."""
    return self._adjacent_vertices[1]

  @property
  def vertex_d(self) -> TessellationVertex:
    """The vertex in the D direction."""
    return self._adjacent_vertices[2]

  def is_adjacent_to_face(self, face: TessellationFace) -> bool:
    """Whether `face` shares an edge with this face."""
    return face in self._adjacent_faces

  def is_adjacent_to_vertex(self, vertex: TessellationVertex) -> bool:
    """Whether `vertex` bounds this face."""
    return vertex in self._adjacent_vertices

  @functools.singledispatchmethod
  def is_adjacent_to(self, other):
    """Whether `other` (a face or vertex) is adjacent to this face."""
    # Dispatches to `is_adjacent_to_face` or `is_adjacent_to_vertex`. These
    # delegations are registered outside of the class body, when both
    # `TessellationVertex` and `TessellationFace` are fully defined.
    raise TypeError(f"Unexpected 'other' type '{type(other).__name__}'")

  def vertex_at(self, opposite_edge: IsometricDirection) -> TessellationVertex:
    """The vertex opposite the given edge direction."""
    return self._adjacent_vertices[opposite_edge.value]

  def face_on_edge(
      self, intervening_edge: IsometricDirection) -> TessellationFace:
    """The neighboring face across the given edge direction."""
    return self._adjacent_faces[intervening_edge.value]

  def direction_toward_vertex(
      self, adjacent_vertex: TessellationVertex) -> IsometricDirection:
    """The `IsometricDirection` pointing toward `adjacent_vertex` on this face.

    Raises `NotAdjacentException` if `adjacent_vertex` isn't one of this face's
    bounding vertices.
    """
    try:
      return IsometricDirection(
          self._adjacent_vertices.index(adjacent_vertex))
    except ValueError:
      raise NotAdjacentException(
          "The provided vertex is not adjacent to this face."
      ) from None

  def direction_away_from_face(
      self, adjacent_face: TessellationFace) -> IsometricDirection:
    """The `IsometricDirection` pointing away from `adjacent_face` on this face.

    Raises `NotAdjacentException` if the faces aren't adjacent.
    """
    try:
      return IsometricDirection(self._adjacent_faces.index(adjacent_face))
    except ValueError:
      raise NotAdjacentException(
          "The provided faces are not adjacent."
      ) from None

  def recalculate_centroid_projection_coords(self) -> None:
    """Refreshes cached 3D-projection centroid from current vertex positions."""
    self._centroid_projection_coords = tuple(
        sum([getattr(v, axis) for v in self._adjacent_vertices]) /
            len(self._adjacent_vertices)
        for axis in "xyz"
    )

  def recalculate_adjacency_to(
      self,
      other_face: TessellationFace,
      call_bilaterally: bool = True) -> None:
    """Refreshes which edge direction (if any) borders `other_face`.

    When `call_bilaterally` is set, also refreshes `other_face`'s adjacency back
    to this face.
    """
    shared_vertices = [
        v.is_adjacent_to_face(other_face) \
        for v in self._adjacent_vertices]

    adjacent = (sum(shared_vertices) == 2)

    for direction in IsometricDirection:
      if adjacent and not shared_vertices[direction.value]:
        # The `not` is because the face associated with `direction` is
        # opposite the vertex associated with `direction`.
        self._adjacent_faces[direction.value] = other_face
      elif self._adjacent_faces[direction.value] is other_face:
        self._adjacent_faces[direction.value] = None

    if (adjacent and call_bilaterally):
      other_face.recalculate_adjacency_to(self, call_bilaterally=False)

  @property
  def centroid_projection_coords(self) -> tuple[Number, Number, Number]:
    """This face's centroid in 3D-projection coordinates."""
    return self._centroid_projection_coords

  @property
  def centroid_mesh_coords(self) -> tuple[Number, ...]:
    """This face's centroid in a mesh-defined coordinate system."""
    return self.to_mesh_coordinates(self.centroid_local_coords)

  @property
  def centroid_local_coords(self) -> IsometricPoint:
    """This face's centroid as an `IsometricPoint` on itself."""
    return IsometricPoint.center(self)


TessellationFace.is_adjacent_to.register(TessellationFace)(
    TessellationFace.is_adjacent_to_face
)
TessellationFace.is_adjacent_to.register(TessellationVertex)(
    TessellationFace.is_adjacent_to_vertex
)
TessellationVertex.is_adjacent_to.register(TessellationFace)(
    TessellationVertex.is_adjacent_to_face
)
TessellationVertex.is_adjacent_to.register(TessellationVertex)(
    TessellationVertex.is_adjacent_to_vertex
)


if __name__ == '__main__':
  apothem = math.sqrt(3)/6

  vert_b = TessellationVertex([0, 2*apothem, 0])
  vert_s = TessellationVertex([1/2, -apothem, 0])
  vert_d = TessellationVertex([-1/2, -apothem, 0])

  grid = TessellationFace(vert_b, vert_s, vert_d)

  vert_b_prime = TessellationVertex([
      0, -(math.sqrt(3)*3/2*apothem + 1), -3/2*apothem])
  vert_s_prime = vert_d
  vert_d_prime = vert_s

  grid_prime = TessellationFace(vert_b_prime, vert_s_prime, vert_d_prime)

  assert(apothem == grid.apothem)
  assert(apothem == grid_prime.apothem)

  pt_a = IsometricPoint.center(grid)
  pt_b = IsometricPoint(grid, apothem + 0.2, apothem + 0.2)
  pt_c = IsometricPoint(grid, apothem + 0.2, apothem - 0.2)

  adjacency_string = lambda node: " ".join(
      [str(node.is_adjacent_to(other)).ljust(5) for other in [
          grid, vert_b, vert_s, vert_d,
          grid_prime, vert_b_prime, vert_s_prime, vert_d_prime]])

  print()
  print(r"   (1)")
  print(r"  /   \   grid : <b,  s,  d > = <(1), (2), (3)>")
  print(r"(3)---(2)")
  print(r"  \   /   grid': <b', s', d'> = <(4), (3), (2)>")
  print(r"   (4)")
  print()
  print("Coordinates (x,y,z) of vertices:")
  print("(1):", vert_b.projection_coordinates)
  print("(2):", vert_s.projection_coordinates)
  print("(3):", vert_d.projection_coordinates)
  print("(4):", vert_b_prime.projection_coordinates)
  print()
  print("Adjacency matrix:")
  print("      | grid   b     s     d    grid'  b'    s'    d'  ")
  print("------+------------------------------------------------")
  print("grid  |", adjacency_string(grid))
  print("b     |", adjacency_string(vert_b))
  print("s     |", adjacency_string(vert_s))
  print("d     |", adjacency_string(vert_d))
  print("grid' |", adjacency_string(grid_prime))
  print("b'    |", adjacency_string(vert_b_prime))
  print("s'    |", adjacency_string(vert_s_prime))
  print("d'    |", adjacency_string(vert_d_prime))
  print()
  print("Centroid (x,y,z) of grid: ", grid.centroid_projection_coords)
  print("Centroid (x,y,z) of grid':", grid_prime.centroid_projection_coords)
  print()
  print("Local directions facing toward adjacent vertices:")
  print(f"grid  -> b:  {grid.direction_toward_vertex(vert_b)}")
  print(f"grid  -> s:  {grid.direction_toward_vertex(vert_s)}")
  print(f"grid  -> d:  {grid.direction_toward_vertex(vert_d)}")
  print(f"grid' -> b': {grid_prime.direction_toward_vertex(vert_b_prime)}")
  print(f"grid' -> s': {grid_prime.direction_toward_vertex(vert_s_prime)}")
  print(f"grid' -> d': {grid_prime.direction_toward_vertex(vert_d_prime)}")
  print()
  print("Local directions facing toward adjacent faces:")
  print(f"grid -> grid': -{grid.direction_away_from_face(grid_prime)}")
  print(f"grid' -> grid: -{grid_prime.direction_away_from_face(grid)}")
  print()
  print("Points (b,s,d) within grid:")
  print("a:", pt_a)
  print("b:", pt_b)
  print("c:", pt_c)
  print()
  print("Vectors <db,ds,dd> within grid:")
  assert((pt_a - pt_b).length == pt_a.distance_from(pt_b))
  assert((pt_b - pt_a).length == pt_b.distance_from(pt_a))
  assert((pt_a - pt_c).length == pt_a.distance_from(pt_c))
  assert((pt_c - pt_a).length == pt_c.distance_from(pt_a))
  assert((pt_b - pt_c).length == pt_b.distance_from(pt_c))
  assert((pt_c - pt_b).length == pt_c.distance_from(pt_b))
  print(f"(a - b): |{pt_a - pt_b}| = {(pt_a - pt_b).length}")
  print(f"(b - a): |{pt_b - pt_a}| = {(pt_b - pt_a).length}")
  print(f"(a - c): |{pt_a - pt_c}| = {(pt_a - pt_c).length}")
  print(f"(c - a): |{pt_c - pt_a}| = {(pt_c - pt_a).length}")
  print(f"(b - c): |{pt_b - pt_c}| = {(pt_b - pt_c).length}")
  print(f"(c - b): |{pt_c - pt_b}| = {(pt_c - pt_b).length}")
  print()
  print("Distances from points within grid to centroid (b,s,d) of grid':")
  print("from a:", pt_a.distance_from(IsometricPoint.center(grid_prime)))
  print("from b:", pt_b.distance_from(IsometricPoint.center(grid_prime)))
  print("from c:", pt_c.distance_from(IsometricPoint.center(grid_prime)))
  print()

  print(pt_a)
  print(pt_b)
  print(pt_a - pt_b)
  print()

  print(pt_a)
  print(pt_a.project_onto_adjacent_grid(grid_prime))
  print()

  v1 = IsometricVector(-2*apothem, -2*apothem)
  print(v1.b_component, v1.s_component, v1.d_component)
  print(v1.delta_b, v1.delta_s, v1.delta_d)
  print(IsometricPoint.center(grid) + v1)
