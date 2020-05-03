from mundimonium.layers.coordinates.exceptions import NotAdjacentException
from mundimonium.layers.coordinates.hash_by_index import HashByIndex
from mundimonium.layers.coordinates.isometric import \
		IsometricDirection, IsometricGrid, IsometricPoint, IsometricVector

import itertools
import math
from numbers import Number
from typing import Optional, List


class Tessellation:
	def __init__(self):
		self._vertex_graph = dict()

	@property
	def vertex_type(self) -> type:
		raise NotImplementedError()

	@property
	def face_type(self) -> type:
		raise NotImplementedError()

	def _generate_tessellation(self) -> None:
		raise NotImplementedError()

	def add_vertex(
			self,
			new_vertex: "TessellationVertex",
			adjacent_vertices: List["TessellationVertex"] = list()) -> None:
		self._vertex_graph[new_vertex] = adjacent_vertices
		for vertex in adjacent_vertices:
			self._vertex_graph[vertex].append(new_vertex)

	def add_face(self, bounding_vertices: List["TessellationFace"]) -> None:
		assert len(bounding_vertices) == 3, (
				"A face must be bounded by exactly three vertices.")
		new_face = self.face_type(*bounding_vertices)


class TessellationVertex(HashByIndex):
	def __init__(self, projection_coordinates: List[Number]):
		self._projection_coordinates = projection_coordinates
		self._adjacent_faces = list()

	def add_adjacent_face(self, face: "TessellationFace") -> None:
		if face in self._adjacent_faces:
			return

		self._adjacent_faces.append(face)

		for other_face in self._adjacent_faces:
			if other_face is not face:
				other_face.recalculate_adjacency_to(face)

	def is_adjacent_to_face(self, face: "TessellationFace") -> bool:
		return face in self._adjacent_faces

	def is_adjacent_to_vertex(self, vertex: "TessellationVertex") -> bool:
		for face in self._adjacent_faces:
			if face.is_adjacent_to_vertex(vertex):
				return vertex is not self
		return False

	def _is_adjacent_to_selector(self, arg_type: type):
		return {TessellationFace: self.is_adjacent_to_face,
				TessellationVertex: self.is_adjacent_to_vertex}[arg_type]

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

	def adjacent_faces(self) -> List["TessellationFace"]:
		return self._adjacent_faces


class TessellationFace(HashByIndex, IsometricGrid):
	BASE_TO_ALTITUDE = math.sqrt(3) / 2
	APOTHEM_TO_ALTITUDE = 3
	BASE_TO_APOTHEM = BASE_TO_ALTITUDE / APOTHEM_TO_ALTITUDE

	def __init__(
			self,
			vertex_b: "TessellationVertex",
			vertex_s: "TessellationVertex",
			vertex_d: "TessellationVertex"):
		self._adjacent_faces = [None] * len(IsometricDirection)
		self._adjacent_vertices = [vertex_b, vertex_s, vertex_d]

		for vertex in self._adjacent_vertices:
			vertex.add_adjacent_face(self)

		self._side_length = 1  # TODO: Set this to a global scale value times
		#                              a local scale-distortion modifier.
		self._apothem = self._side_length * TessellationFace.BASE_TO_APOTHEM
		self._altitude = self._side_length * TessellationFace.BASE_TO_ALTITUDE

		self._centroid = None
		self.recalculate_centroid()

	def is_adjacent_to_face(self, face: "TessellationFace") -> bool:
		return face in self._adjacent_faces

	def is_adjacent_to_vertex(self, vertex: "TessellationVertex") -> bool:
		return vertex in self._adjacent_vertices

	def _is_adjacent_to_selector(self, arg_type: type):
		return {TessellationFace: self.is_adjacent_to_face,
				TessellationVertex: self.is_adjacent_to_vertex}[arg_type]

	def is_adjacent_to(self, other):
		return self._is_adjacent_to_selector(type(other))(other)

	def vertex_at(self, opposite_edge: "IsometricDirection"
			) -> "TessellationVertex":
		return self._adjacent_vertices[opposite_edge.value]

	def face_on_edge(self, intervening_edge: "IsometricDirection"
			) -> "TessellationFace":
		return self._adjacent_faces[intervening_edge.value]

	def direction_toward_vertex(self, adjacent_vertex: "TessellationVertex"
			) -> "IsometricDirection":
		try:
			return IsometricDirection(
					self._adjacent_vertices.index(adjacent_vertex))
		except ValueError:
			raise NotAdjacentException(
					"The provided vertex is not adjacent to this face."
					) from None

	def direction_opposite_face(self, adjacent_face: "TessellationFace"
			) -> "IsometricDirection":
		try:
			return IsometricDirection(self._adjacent_faces.index(adjacent_face))
		except ValueError:
			raise NotAdjacentException(
					"The provided faces are not adjacent.") from None

	def recalculate_centroid(self) -> None:
		self._centroid = tuple(
			sum([getattr(v, axis) for v in self._adjacent_vertices]) /
			len(self._adjacent_vertices) for axis in "xyz")

	def recalculate_adjacency_to(
			self, other_face: "TessellationFace", call_bilaterally: bool = True
			) -> None:
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
	def centroid(self):
		return self._centroid

	@property
	def tessellation_type(self):
		raise NotImplementedError()

	@property
	def vertex_type(self):
		raise NotImplementedError()


if __name__ == '__main__':
	apothem = math.sqrt(3)/6

	vert_b = TessellationVertex([0, 2*apothem, 0])
	vert_s = TessellationVertex([-1/2, -apothem, 0])
	vert_d = TessellationVertex([1/2, -apothem, 0])

	grid = TessellationFace(vert_b, vert_s, vert_d)

	vert_b_prime = TessellationVertex([
			0, -(math.sqrt(3)*3/2*apothem + 1), -3/2*apothem])
	vert_s_prime = vert_d
	vert_d_prime = vert_s

	grid_prime = TessellationFace(vert_b_prime, vert_s_prime, vert_d_prime)

	a = IsometricPoint.center(grid)
	b = IsometricPoint(grid, apothem + 0.2, apothem + 0.2)
	c = IsometricPoint(grid, apothem + 0.2, apothem - 0.2)

	adjacency_string = lambda node: " ".join(
			[str(node.is_adjacent_to(other)).ljust(5) for other in [
					grid, vert_b, vert_s, vert_d,
					grid_prime, vert_b_prime, vert_s_prime, vert_d_prime]])

	print()
	print(r"   (1)")
	print(r"  /   \   grid : <b,  s,  d > = <(1), (2), (3)>")
	print(r"(2)---(3)")
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
	print("Centroid of grid: ", grid.centroid)
	print("Centroid of grid':", grid_prime.centroid)
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
	print(f"grid -> grid': -{grid.direction_opposite_face(grid_prime)}")
	print(f"grid' -> grid: -{grid_prime.direction_opposite_face(grid)}")
	print()
	print("Points (b,s,d) within grid:")
	print("a:", a)
	print("b:", b)
	print("c:", c)
	print()
	print("Vectors <Δb,Δs,Δd> within grid:")
	assert((a - b).length == a.distance_from(b))
	assert((b - a).length == b.distance_from(a))
	assert((a - c).length == a.distance_from(c))
	assert((c - a).length == c.distance_from(a))
	assert((b - c).length == b.distance_from(c))
	assert((c - b).length == c.distance_from(b))
	print(f"(a - b): |{a - b}| = {(a - b).length}")
	print(f"(b - a): |{b - a}| = {(b - a).length}")
	print(f"(a - c): |{a - c}| = {(a - c).length}")
	print(f"(c - a): |{c - a}| = {(c - a).length}")
	print(f"(b - c): |{b - c}| = {(b - c).length}")
	print(f"(c - b): |{c - b}| = {(c - b).length}")
