from __future__ import annotations

from mundimonium.coordinates.exceptions import NotAdjacentException
from mundimonium.coordinates.hash_by_index import HashByIndex
from mundimonium.coordinates.isometric import (
    IsometricDirection, IsometricGrid, IsometricPoint, IsometricVector,
    isometric_distance
)

import abc
import itertools
import math
from numbers import Number
from typing import get_type_hints, override
import numpy as np


class _TessellationImplMetadata:
  def __init__(self, impl: type[Tessellation]):
    base = Tessellation
    base_init_params = get_type_hints(base.__init__, include_extras=True)
    impl_init_params = get_type_hints(impl.__init__, include_extras=True)
    self._init_params = impl_init_params
    for param, type_hint in base_init_params.items():
      if param not in impl_init_params:
        import pprint
        raise AttributeError(
            "A 'Tessellation' implementation must accept at least these "
            "'__init__' keyword arguments:\n" + pprint.pformat(base_init_params)
        )
      # The user shouldn't actually pass the base Tessellation args, though.
      del self._init_params[param]
    self._init_params = impl_init_params


class Tessellation(abc.ABC):
  """
  Abstract base class for equilateral triangular meshes.
  """

  _impls: dict[type[Tessellation], _TessellationImplMetadata] = {}

  def __init_subclass__(cls, **kwargs):
    assert cls not in Tessellation._impls
    Tessellation._impls[cls] = _TessellationImplMetadata(cls)

  def __init__(self,
               *,
               vertex_type: type[TessellationVertex] | None = None,
               face_type: type[TessellationFace] | None = None):
    if vertex_type is None:
      vertex_type = TessellationVertex

    if face_type is None:
      face_type = TessellationFace

    self._vertex_type: type[TessellationVertex] = vertex_type
    self._face_type: type[TessellationFace] = face_type
    self._vertex_graph: dict[TessellationVertex, list[TessellationVertex]] = {}
    self._vertices: list[TessellationVertex] = []
    self._faces: list[TessellationFace] = []
    self._vertex_index_map: dict[TessellationVertex, int] = {}
    self._face_index_map: dict[TessellationFace, int] = {}

  @property
  def vertex_type(self) -> type[TessellationVertex]:
    return self._vertex_type

  @property
  def face_type(self) -> type[TessellationFace]:
    return self._face_type

  @abc.abstractmethod
  def new_point_at_coords(
      self, *coords: tuple[Number, ...]) -> IsometricPoint | None:
    """Returns a new `IsometricPoint` at the specified coordinates.

    The coordinate system (including `len(coords)`) is implementation-defined.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def get_face_at_coords(
      self, *coords: tuple[Number, ...]) -> TessellationFace | None:
    """Returns the face containing the specified coordinates.

    The coordinate system (including `len(coords)`) is implementation-defined.
    """
    raise NotImplementedError()

  @abc.abstractmethod
  def coords_at_point(self, point: IsometricPoint) -> tuple[Number, ...]:
    raise NotImplementedError()

  def _generate_tessellation(self) -> None:
    raise NotImplementedError()

  def invalidate_solvers(self) -> None:
    """Hook for subclasses to invalidate solver caches when topology changes."""
    pass

  def add_vertex(
      self,
      new_vertex: TessellationVertex,
      adjacent_vertices: list[TessellationVertex] | None = None) -> int:
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
      self.invalidate_solvers()
    return self._vertex_index_map[new_vertex]

  def register_face(self, face: TessellationFace) -> int:
    """Registers an existing TessellationFace in this tessellation."""
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
      self.invalidate_solvers()
    return self._face_index_map[face]

  def add_face(
      self,
      bounding_vertices: list[TessellationVertex] | TessellationFace
  ) -> TessellationFace:
    if isinstance(bounding_vertices, TessellationFace):
      self.register_face(bounding_vertices)
      return bounding_vertices

    assert len(bounding_vertices) == 3, (
        "A face must be bounded by exactly three vertices.")
    new_face = self.face_type(
        vertex_b=bounding_vertices[0],
        vertex_s=bounding_vertices[1],
        vertex_d=bounding_vertices[2],
    )
    self.register_face(new_face)
    return new_face

  def distance(self, p1: IsometricPoint, p2: IsometricPoint) -> Number:
    """
    Computes geodesic distance between two arbitrary points on the mesh.
    Subclasses should override this method with specific solvers.
    """
    return self.face_type.distance(p1, p2)

  @abc.abstractmethod
  def geodesic_distance(
      self, p1: IsometricPoint, p2: IsometricPoint) -> Number | None:
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


class TessellationVertex(HashByIndex):
  def __init__(self, projection_coordinates: list[Number]):
    self._projection_coordinates = projection_coordinates
    self._adjacent_faces = list()
    self._tessellation: Tessellation | None = None

  @property
  def tessellation(self) -> Tessellation | None:
    return self._tessellation

  @tessellation.setter
  def tessellation(self, val: Tessellation) -> None:
    self._tessellation = val

  def add_adjacent_face(self, face: TessellationFace) -> None:
    if face in self._adjacent_faces:
      return

    self._adjacent_faces.append(face)

    for other_face in self._adjacent_faces:
      if other_face is not face:
        other_face.recalculate_adjacency_to(face)

  def is_adjacent_to_face(self, face: TessellationFace) -> bool:
    return face in self._adjacent_faces

  def is_adjacent_to_vertex(self, vertex: TessellationVertex) -> bool:
    for face in self._adjacent_faces:
      if face.is_adjacent_to_vertex(vertex):
        return vertex is not self
    return False

  def _is_adjacent_to_selector(self, arg_type: type):
    return {
        TessellationFace: self.is_adjacent_to_face,
        TessellationVertex: self.is_adjacent_to_vertex,
    }[arg_type]

  def is_adjacent_to(self, other):
    return self._is_adjacent_to_selector(type(other))(other)

  @property
  def tessellation_type(self):
    raise NotImplementedError()

  @property
  def face_type(self):
    raise NotImplementedError()

  @property
  def x(self) -> Number:
    return self._projection_coordinates[0]

  @property
  def y(self) -> Number:
    return self._projection_coordinates[1]

  @property
  def z(self) -> Number:
    return self._projection_coordinates[2]

  @property
  def projection_coordinates(self):
    return tuple(self._projection_coordinates)

  @x.setter
  def x(self, new_x: Number) -> None:
    self._projection_coordinates[0] = new_x
    for face in self._adjacent_faces:
      face.recalculate_centroid()

  @y.setter
  def y(self, new_y: Number) -> None:
    self._projection_coordinates[1] = new_y
    for face in self._adjacent_faces:
      face.recalculate_centroid()

  @z.setter
  def z(self, new_z: Number) -> None:
    self._projection_coordinates[2] = new_z
    for face in self._adjacent_faces:
      face.recalculate_centroid()

  def adjacent_faces(self) -> list[TessellationFace]:
    return self._adjacent_faces


class TessellationFace(HashByIndex, IsometricGrid):
  BASE_TO_ALTITUDE = math.sqrt(3) / 2
  APOTHEM_TO_ALTITUDE = 3
  BASE_TO_APOTHEM = BASE_TO_ALTITUDE / APOTHEM_TO_ALTITUDE

  def __init__(
      self,
      *,
      vertex_b: TessellationVertex,
      vertex_s: TessellationVertex,
      vertex_d: TessellationVertex,
      **kwargs):
    super().__init__(**kwargs)
    self._adjacent_faces = [None] * len(IsometricDirection)
    self._adjacent_vertices = [vertex_b, vertex_s, vertex_d]
    self._tessellation: Tessellation | None = None

    for vertex in self._adjacent_vertices:
      vertex.add_adjacent_face(self)

    self._side_length = 1  # TODO: Set this to a global scale value times
    #                              a local scale-distortion modifier.
    self._apothem = self._side_length * TessellationFace.BASE_TO_APOTHEM
    self._altitude = self._side_length * TessellationFace.BASE_TO_ALTITUDE

    self._centroid_external = None
    self.recalculate_centroid()

  @classmethod
  @override
  def nearby_grid_distance(
      cls, p1: IsometricPoint, p2: IsometricPoint) -> Number | None:
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
    return p1.grid.tessellation.geodesic_distance(p1, p2)

  @override
  def to_world_coordinates(self, point: IsometricPoint) -> tuple[Number, ...]:
    return self.tessellation.coords_at_point(point)

  @property
  def tessellation(self) -> Tessellation | None:
    return self._tessellation

  @tessellation.setter
  def tessellation(self, val: Tessellation) -> None:
    self._tessellation = val

  @property
  def vertex_b(self) -> TessellationVertex:
    return self._adjacent_vertices[0]

  @property
  def vertex_s(self) -> TessellationVertex:
    return self._adjacent_vertices[1]

  @property
  def vertex_d(self) -> TessellationVertex:
    return self._adjacent_vertices[2]

  def is_adjacent_to_face(self, face: TessellationFace) -> bool:
    return face in self._adjacent_faces

  def is_adjacent_to_vertex(self, vertex: TessellationVertex) -> bool:
    return vertex in self._adjacent_vertices

  def _is_adjacent_to_selector(self, arg_type: type):
    return {
        TessellationFace: self.is_adjacent_to_face,
        TessellationVertex: self.is_adjacent_to_vertex,
    }[arg_type]

  def is_adjacent_to(self, other):
    return self._is_adjacent_to_selector(type(other))(other)

  def vertex_at(self, opposite_edge: IsometricDirection) -> TessellationVertex:
    return self._adjacent_vertices[opposite_edge.value]

  def face_on_edge(
      self, intervening_edge: IsometricDirection) -> TessellationFace:
    return self._adjacent_faces[intervening_edge.value]

  def direction_toward_vertex(
      self, adjacent_vertex: TessellationVertex) -> IsometricDirection:
    try:
      return IsometricDirection(
          self._adjacent_vertices.index(adjacent_vertex))
    except ValueError:
      raise NotAdjacentException(
          "The provided vertex is not adjacent to this face."
      ) from None

  def direction_away_from_face(
      self, adjacent_face: TessellationFace) -> IsometricDirection:
    try:
      return IsometricDirection(self._adjacent_faces.index(adjacent_face))
    except ValueError:
      raise NotAdjacentException(
          "The provided faces are not adjacent."
      ) from None

  def recalculate_centroid(self) -> None:
    self._centroid_external = tuple(
        sum([getattr(v, axis) for v in self._adjacent_vertices]) /
            len(self._adjacent_vertices)
        for axis in "xyz"
    )

  def recalculate_adjacency_to(
      self,
      other_face: TessellationFace,
      call_bilaterally: bool = True) -> None:
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
  def side_length(self):
    return self._side_length

  @property
  def apothem(self):
    return self._apothem

  @property
  def altitude(self):
    return self._altitude

  @property
  def centroid_external(self):
    return self._centroid_external

  @property
  def centroid_internal(self):
    return IsometricPoint.center(self)

  @property
  def tessellation_type(self):
    raise NotImplementedError()

  @property
  def vertex_type(self):
    raise NotImplementedError()


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
  print("Centroid (x,y,z) of grid: ", grid.centroid_external)
  print("Centroid (x,y,z) of grid':", grid_prime.centroid_external)
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
