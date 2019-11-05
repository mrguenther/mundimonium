from enum import Enum
from types import MethodType


class NotAdjacentException(ValueError):
	"""
	Raised when an operation expects adjacent graph nodes but encounters
	non-adjacent nodes.
	"""
	pass


class Point:
	"""TODO"""
	pass


class IsometricPoint(Point):
	"""TODO"""
	def __init__(self, grid, b=None, s=None, d=None):
		"""TODO"""
		self._grid = grid
		if b is None and s is None and d is None:
			self._b = self.grid.apothem
			self._s = self.grid.apothem
		else:
			self.move_to(b, s, d)

	def __init__(self, grid, other):
		"""
		Initialize as a projection of point `other` onto `grid`. In order to be
		well defined, this requires that `other.grid` and `grid` share an edge.
		"""
		self._grid = grid
		if grid is other.grid:
			self._b = other.b
			self._s = other.s
			return
		apothem_sum = self.grid.apothem + other.grid.apothem
		old_border_edge = other.grid.direction_from_face(self.grid)
		new_border_edge = self.grid.direction_from_face(other.grid)
		other_b_complement = IsometricDirection(
				new_border_edge.value - old_border_edge.value)
		other_s_complement = other_b_complement.rotated_cw_by_index(1)
		self._b = apothem_sum - other[other_b_complement]
		self._s = apothem_sum - other[other_s_complement]
		self[new_border_edge] -= apothem_sum

	def __repr__(self):
		"""TODO"""
		return f"({self.b},{self.s},{self.d}) in {repr(self.grid)}"

	def __getitem__(self, key):
		"""TODO"""
		if key == IsometricDirection.B:
			return self.b
		elif key == IsometricDirection.S:
			return self.s
		elif key == IsometricDirection.D:
			return self.d
		else:
			raise ValueError(f"{key} is not a valid IsometricDirection.")

	def __setitem__(self, key, item):
		"""TODO"""
		if key == IsometricDirection.B:
			self.b = item
		elif key == IsometricDirection.S:
			self.s = item
		elif key == IsometricDirection.D:
			self.d = item
		else:
			raise ValueError(f"{key} is not a valid IsometricDirection.")

	def __add__(self, other):
		"""TODO"""
		assert type(other) is IsometricVector, "Invalid __add__() operand."
		return IsometricPoint(
				self.grid,
				self.b + other.delta_b,
				self.s + other.delta_s)

	def __sub__(self, other):
		"""TODO"""
		assert type(other) is IsometricPoint, "Invalid __sub__() operand."
		return IsometricVector(self.b - other.b, self.s - other.s)

	def move_to(self, b=None, s=None, d=None):
		"""TODO"""
		assert int(b is None) + int(s is None) + int(d is None) == 1, (
				"move_to() must be provided exactly two of (b,s,d).")
		if b is None:
			self.b = self.grid.side_length - s - d
			self.s = s
		elif s is None:
			self.b = b
			self.s = self.grid.side_length - b - d
		else:  # d is None
			self.b = b
			self.s = s

	def move_by(self, b=None, s=None, d=None):
		"""TODO"""
		assert int(b is None) + int(s is None) + int(d is None) == 1, (
				"move_by() must be provided exactly two of (b,s,d).")
		if b is None:
			self.b -= (s + d)
			self.s += s
		elif s is None:
			self.b += b
			self.s -= (b + d)
		else:  # d is None
			self.b += b
			self.s += s

	def project_onto_adjacent_grid(self, adjacent_grid):
		"""TODO"""
		return IsometricPoint(adjacent_grid, self)

	@property
	def grid(self):
		"""TODO"""
		return self._grid

	@property
	def b(self):
		"""TODO"""
		return self._b

	@property
	def s(self):
		"""TODO"""
		return self._s

	@property
	def d(self):
		"""TODO"""
		return self.grid.side_length - self._b - self._s

	@b.setter
	def b(self, b):
		"""TODO"""
		self._s -= 0.5 * (b - self._b)
		self._b = b

	@s.setter
	def s(self, s):
		"""TODO"""
		self._b -= 0.5 * (s - self._s)
		self._s = s

	@d.setter
	def d(self, d):
		"""TODO"""
		delta = 0.5 * (d - self.d)
		self._b -= delta
		self._s -= delta


class IsometricVector:
	"""TODO"""
	def __init__(self, p1, p2):
		"""TODO"""
		self._delta_b = p2.b - p1.b
		self._delta_s = p2.s - p1.s

	def __init__(delta_b=None, delta_s=None, delta_d=None):
		"""TODO"""
		assert (int(delta_b is None) + int(delta_s is None) +
				int(delta_d is None)) == 1, (
				"IsometricVector() must be provided exactly two arguments.")
		if delta_b is None:
			self._delta_b = -(delta_s + delta_d)
			self._delta_s = delta_s
		elif s is None:
			self._delta_b = delta_b
			self._delta_s = -(delta_b + delta_d)
		else:  # d is None
			self._delta_b = delta_b
			self._delta_s = delta_s

	@property
	def delta_b(self):
		"""TODO"""
		return self._delta_b

	@property
	def delta_s(self):
		"""TODO"""
		return self._delta_s

	@property
	def delta_d(self):
		"""TODO"""
		return -(self._delta_b + self._delta_s)


class IsometricDirection(Enum):
	"""TODO"""
	B = 0
	S = 1
	D = 2

	def rotated_cw_by_index(self, index):
		"""TODO"""
		return IsometricDirection(
				(self.value + index) % len(IsometricDirection))

	def rotated_ccw_by_index(self, index):
		"""TODO"""
		return IsometricDirection(
				(self.value - index) % len(IsometricDirection))


class Tessellation:
	"""TODO"""
	def __init__(self):
		"""TODO"""
		self._vertex_graph = self._generate_vertex_graph()
		self._face_graph = self._generate_face_graph()

	@property
	def vertex_type(self):
		"""TODO"""
		raise NotImplementedError()

	@property
	def face_type(self):
		"""TODO"""
		raise NotImplementedError()

	def _generate_vertex_graph(self):
		"""TODO"""
		raise NotImplementedError()

	def _generate_face_graph(self):
		"""TODO"""
		raise NotImplementedError()


class TessellationVertex:
	"""TODO"""
	def __init__(self, projection_coordinates, adjacent_vertices,
				suppress_faces_between=list()):
		# TODO: handle suppression of unwanted faces when intentionally creating holes to be extended later with more polygons.
		self._projection_coordinates = projection_coordinates
		self._adjacent_vertices = adjacent_vertices
		self._adjacent_faces = list()
		self._add_reciprocal_edges_to_neighbors()
		self._create_adjacent_faces()

	def _add_reciprocal_edges_to_neighbors(self):
		"""TODO"""
		for v in self._adjacent_vertices:
			v._add_adjacent_vertex(self)

	def _create_adjacent_faces(self):
		"""TODO"""
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
		"""TODO"""
		if vertex not in self._adjacent_vertices:
			self._adjacent_vertices.append(vertex)

	def _add_adjacent_face(self, face):
		"""TODO"""
		if face not in self._adjacent_faces:
			self._adjacent_faces.append(face)

	def is_adjacent_to(self, vertex):
		"""TODO"""
		return vertex in self._adjacent_vertices

	@property
	def tessellation_type(self):
		"""TODO"""
		raise NotImplementedError()

	@property
	def face_type(self):
		"""TODO"""
		raise NotImplementedError()

	@property
	def x(self):
		"""TODO"""
		return self._projection_coordinates[0]

	@property
	def y(self):
		"""TODO"""
		return self._projection_coordinates[1]

	@property
	def z(self):
		"""TODO"""
		return self._projection_coordinates[2]

	@x.setter
	def x(self, new_x):
		"""TODO"""
		self._projection_coordinates[0] = new_x

	@y.setter
	def y(self, new_y):
		"""TODO"""
		self._projection_coordinates[1] = new_y

	@z.setter
	def z(self, new_z):
		"""TODO"""
		self._projection_coordinates[2] = new_z

	def adjacent_vertices(self):
		"""TODO"""
		return self._adjacent_vertices

	def adjacent_faces(self):
		"""TODO"""
		return self._adjacent_faces


class TessellationFace:
	"""TODO"""
	BASE_TO_APOTHEM = sqrt(3) / 6

	def __init__(self, vertex_b, vertex_s, vertex_d):
		"""TODO"""
		self._adjacent_faces = [None] * len(IsometricDirection)
		self._adjacent_vertices = [vertex_b, vertex_s, vertex_d]
		self._coordinates = [
			self.calculate_external_x(),
			self.calculate_external_y(),
			self.calculate_external_z()]
		self._side_length = None  # TODO
		self._apothem = self._side_length * TessellationFace.BASE_TO_APOTHEM

	def vertex_at(self, opposite_edge):
		"""TODO"""
		return self._adjacent_vertices[opposite_edge.value]

	def face_on_edge(self, intervening_edge):
		"""TODO"""
		return self._adjacent_faces[intervening_edge.value]

	def direction_toward_vertex(self, adjacent_vertex):
		"""TODO"""
		try:
			return IsometricDirection(
					self._adjacent_vertices.index(adjacent_vertex))
		except ValueError:
			raise NotAdjacentException(
					"The provided vertex is not adjacent to this face.")

	def direction_from_face(self, adjacent_face):
		"""TODO"""
		try:
			return IsometricDirection(self._adjacent_faces.index(adjacent_face))
		except ValueError:
			raise NotAdjacentException("The provided faces are not adjacent.")

	@property
	def side_length(self):
		"""TODO"""
		return self._side_length

	@property
	def apothem(self):
		"""TODO"""
		return self._apothem

	@property
	def tessellation_type(self):
		"""TODO"""
		raise NotImplementedError()

	@property
	def vertex_type(self):
		"""TODO"""
		raise NotImplementedError()

