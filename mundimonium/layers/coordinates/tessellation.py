from exceptions import NotAdjacentException
from isometric import IsometricGrid

import math
from typing import Optional


class Tessellation:
	def __init__(self):
		self._vertex_graph = self._generate_vertex_graph()
		self._face_graph = self._generate_face_graph()

	@property
	def vertex_type(self):
		raise NotImplementedError()

	@property
	def face_type(self):
		raise NotImplementedError()

	def _generate_vertex_graph(self):
		raise NotImplementedError()

	def _generate_face_graph(self):
		raise NotImplementedError()


class TessellationVertex:
	def __init__(self, projection_coordinates, adjacent_vertices,
				suppress_faces_between=list()):
		# TODO: handle suppression of unwanted faces when intentionally creating holes to be extended later with more polygons.
		self._projection_coordinates = projection_coordinates
		self._adjacent_vertices = adjacent_vertices
		self._adjacent_faces = list()
		self._add_reciprocal_edges_to_neighbors()
		self._create_adjacent_faces()

	def _add_reciprocal_edges_to_neighbors(self):
		for v in self._adjacent_vertices:
			v._add_adjacent_vertex(self)

	def _create_adjacent_faces(self):
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


class TessellationFace(IsometricGrid):
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
