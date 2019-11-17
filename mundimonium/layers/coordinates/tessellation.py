from mundimonium.layers.coordinates.exceptions import NotAdjacentException
from mundimonium.layers.coordinates.hash_by_index import HashByIndex
from mundimonium.layers.coordinates.isometric \
		import IsometricGrid, IsometricDirection

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
		# TODO: remove all remaining uses of self._adjacent_vertices
		self._adjacent_faces = list()

	def add_adjacent_faces(self):
		num_neighbors = len(self._adjacent_vertices)
		for i in range(num_neighbors):
			neighbor_i = self._adjacent_vertices[i]
			for j in range(i + 1, num_neighbors):
				neighbor_j = self._adjacent_vertices[j]
				if neighbor_i.is_adjacent_to(neighbor_j):
					new_face = self.face_type(self, neighbor_i, neighbor_j)  # TODO: make sure I order these so the normal vector isn't inverted.
					self._add_adjacent_face(new_face)
					neighbor_i._add_adjacent_face(new_face)
					neighbor_j._add_adjacent_face(new_face)

	def _add_adjacent_vertex(self, vertex):
		if vertex not in self._adjacent_vertices:
			self._adjacent_vertices.append(vertex)

	def _add_adjacent_face(self, face):
		if face not in self._adjacent_faces:
			self._adjacent_faces.append(face)

	def is_adjacent_to(self, vertex):
		return vertex in self._adjacent_vertices

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

	@y.setter
	def y(self, new_y):
		self._projection_coordinates[1] = new_y

	@z.setter
	def z(self, new_z):
		self._projection_coordinates[2] = new_z

	def adjacent_vertices(self):
		return self._adjacent_vertices

	def adjacent_faces(self):
		return self._adjacent_faces


class TessellationFace(HashByIndex, IsometricGrid):
	BASE_TO_ALTITUDE = math.sqrt(3) / 2
	APOTHEM_TO_ALTITUDE = 3
	BASE_TO_APOTHEM = BASE_TO_ALTITUDE / APOTHEM_TO_ALTITUDE

	def __init__(self, vertex_b, vertex_s, vertex_d):
		self._adjacent_faces = [None] * len(IsometricDirection)
		self._adjacent_vertices = [vertex_b, vertex_s, vertex_d]
		self._coordinates = [
			self.calculate_external_x(),
			self.calculate_external_y(),
			self.calculate_external_z()]
		self._side_length = None  # TODO
		self._apothem = self._side_length * TessellationFace.BASE_TO_APOTHEM
		self._altitude = self._side_length * TessellationFace.BASE_TO_ALTITUDE

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
	def tessellation_type(self):
		raise NotImplementedError()

	@property
	def vertex_type(self):
		raise NotImplementedError()
