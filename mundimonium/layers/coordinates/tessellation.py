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
			adjacent_vertices: List["TessellationVertex"] = list()
	) -> None:
		self._vertex_graph[new_vertex] = adjacent_vertices
		for vertex in adjacent_vertices:
			self._vertex_graph[vertex].append(new_vertex)

	def add_face(self, bounding_vertices: List["TessellationFace"]):
		assert len(bounding_vertices) == 3, (
				"A face must be bounded by exactly three vertices.")
		new_face = self.face_type(*bounding_vertices)


class TessellationVertex(HashByIndex):
	def __init__(self, projection_coordinates: List[Number]):
		self._projection_coordinates = projection_coordinates
		self._adjacent_faces = list()

	def add_adjacent_face(self, face):
		if face not in self._adjacent_faces:
			self._adjacent_faces.append(face)

	def is_adjacent_to(self, face):
		return face in self._adjacent_faces

	def is_adjacent_to(self, vertex):
		for face in self._adjacent_faces:
			if face.is_adjacent_to(vertex):
				return vertex is not self
		return False

	@property
	def tessellation_type(self):
		raise NotImplementedError()

	@property
	def face_type(self):
		raise NotImplementedError()

	@property
	def x(self):
		return self._projection_coordinates[0]

	@property
	def y(self):
		return self._projection_coordinates[1]

	@property
	def z(self):
		return self._projection_coordinates[2]

	@x.setter
	def x(self, new_x):
		self._projection_coordinates[0] = new_x
		for face in self._adjacent_faces:
			face.recalculate_centroid()

	@y.setter
	def y(self, new_y):
		self._projection_coordinates[1] = new_y
		for face in self._adjacent_faces:
			face.recalculate_centroid()

	@z.setter
	def z(self, new_z):
		self._projection_coordinates[2] = new_z
		for face in self._adjacent_faces:
			face.recalculate_centroid()

	def adjacent_faces(self):
		return self._adjacent_faces


class TessellationFace(HashByIndex, IsometricGrid):
	BASE_TO_ALTITUDE = math.sqrt(3) / 2
	APOTHEM_TO_ALTITUDE = 3
	BASE_TO_APOTHEM = BASE_TO_ALTITUDE / APOTHEM_TO_ALTITUDE

	def __init__(self, vertex_b, vertex_s, vertex_d):
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

	def vertex_at(self, opposite_edge):
		return self._adjacent_vertices[opposite_edge.value]

	def face_on_edge(self, intervening_edge):
		return self._adjacent_faces[intervening_edge.value]

	def direction_toward_vertex(self, adjacent_vertex):
		try:
			return IsometricDirection(
					self._adjacent_vertices.index(adjacent_vertex))
		except ValueError:
			raise NotAdjacentException(
					"The provided vertex is not adjacent to this face.")

	def direction_from_face(self, adjacent_face):
		try:
			return IsometricDirection(self._adjacent_faces.index(adjacent_face))
		except ValueError:
			raise NotAdjacentException("The provided faces are not adjacent.")

	def recalculate_centroid(self):
		self._centroid = tuple(
			sum([getattr(vert, axis) for vert in self._adjacent_vertices]) /
			len(self._adjacent_vertices) for axis in "xyz")

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
	print("Centroid:", grid.centroid)
	a = IsometricPoint.center(grid)
	b = IsometricPoint(grid, apothem + 0.2, apothem + 0.2)
	c = IsometricPoint(grid, apothem + 0.2, apothem - 0.2)
	print("a:", a)
	print("b:", b)
	print("c:", b)
	print(f"(a - b): |{a - b}| = {(a - b).length} = {a.distance_from(b)}")
	print(f"(b - a): |{b - a}| = {(b - a).length} = {b.distance_from(a)}")
	print(f"(a - c): |{a - c}| = {(a - c).length} = {a.distance_from(c)}")
	print(f"(c - a): |{c - a}| = {(c - a).length} = {c.distance_from(a)}")
	print(f"(c - b): |{c - b}| = {(c - b).length} = {c.distance_from(b)}")
	print(f"(b - c): |{b - c}| = {(b - c).length} = {b.distance_from(c)}")
