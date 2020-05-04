from mundimonium.layers.coordinates.exceptions import NotAdjacentException
from mundimonium.layers.coordinates.hash_by_index import HashByIndex
from mundimonium.utils.helper_functions import argc

from enum import Enum
from math import sqrt
from numbers import Number
from typing import Optional


def isometric_distance(delta_axis_1: Number, delta_axis_2: Number) -> Number:
	"""
	Finds a distance on the isometric grid given deltas in two isometric axes.

	By the law of cosines, this is sqrt(a**2 + b**2 - 2*a*b*cos(angle C)). The
	angle in question is always 60 degrees, making 2*cos(angle C) equal to 1.
	This leaves sqrt(a**2 + b**2 - a*b).
	"""
	return sqrt(delta_axis_1**2 + delta_axis_2**2 - delta_axis_1 * delta_axis_2)


class IsometricDirection(Enum):
	B = 0
	S = 1
	D = 2

	def rotated_cw_by_index(self, index: int) -> "IsometricDirection":
		return IsometricDirection(
				(self.value + index) % len(IsometricDirection))

	def rotated_ccw_by_index(self, index: int) -> "IsometricDirection":
		return IsometricDirection(
				(self.value - index) % len(IsometricDirection))

	def __repr__(self) -> str:
		return chr(ord(self.name) | 0x20)  # 'b', 's', 'd'

	def __str__(self) -> str:
		return repr(self)


class IsometricGrid:
	@property
	def side_length(self) -> Number:
		raise NotImplementedError()

	@property
	def apothem(self) -> Number:
		raise NotImplementedError()

	@property
	def altitude(self) -> Number:
		raise NotImplementedError()


class IsometricPoint(HashByIndex):
	def __init__(self, grid: IsometricGrid, b: Number, s: Number):
		self._grid = grid
		self._b = b
		self._s = s

	@classmethod
	def center(cls, grid: IsometricGrid) -> "IsometricPoint":
		return cls(grid, grid.apothem, grid.apothem)

	@classmethod
	def at_coordinates(
			cls,
			grid: IsometricGrid,
			b: Optional[Number] = None,
			s: Optional[Number] = None,
			d: Optional[Number] = None
	) -> "IsometricPoint":
		new_point = cls(grid, 0, 0)
		new_point.move_to(b, s, d)
		return new_point

	def project_onto_adjacent_grid(self, adjacent_grid) -> "IsometricPoint":
		"""
		Initialize as a projection of point `other` onto `grid`. In order to be
		well defined, this requires that `other.grid` and `grid` share an edge.
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
		# print()
		# print(self.b / self.grid.apothem, projected_point.b / self.grid.apothem)
		# print(self.s / self.grid.apothem, projected_point.s / self.grid.apothem)
		# print(self.d / self.grid.apothem, projected_point.d / self.grid.apothem)
		# print()
		return projected_point

	def distance_from(self, other: "IsometricPoint") -> Number:
		if self.grid is other.grid:
			return isometric_distance(other.b - self.b, other.s - self.s)
		elif self.grid.is_adjacent_to_face(other.grid):
			return self.project_onto_adjacent_grid(other.grid).distance_from(
					other)
		else:
			return distance_from_remote_point(other)

	def distance_from_remote_point(self, other: "IsometricPoint") -> Number:
		return NotImplementedError(
				"The general case of this function is not yet implemented.")

	def __repr__(self) -> str:
		return f"<id {hash(self)}: {str(self)} in grid {repr(self.grid)}>"

	def __str__(self) -> str:
		return f"({self.b},{self.s},{self.d})"

	def __getitem__(self, key: IsometricDirection) -> Number:
		if key == IsometricDirection.B:
			return self.b
		elif key == IsometricDirection.S:
			return self.s
		elif key == IsometricDirection.D:
			return self.d
		else:
			raise ValueError(f"{key} is not a valid IsometricDirection.")

	def __setitem__(self, key: IsometricDirection, item: Number) -> None:
		if key == IsometricDirection.B:
			self.b = item
		elif key == IsometricDirection.S:
			self.s = item
		elif key == IsometricDirection.D:
			self.d = item
		else:
			raise ValueError(f"{key} is not a valid IsometricDirection.")

	def __add__(self, other: "IsometricVector") -> "IsometricPoint":
		assert type(other) is IsometricVector, "Invalid __add__() operand."
		return IsometricPoint(
				self.grid, self.b + other.delta_b, self.s + other.delta_s)

	def _sub_point(self, other: "IsometricPoint") -> "IsometricVector":
		return IsometricVector(self.b - other.b, self.s - other.s)

	def _sub_vector(self, other: "IsometricVector") -> "IsometricPoint":
		return IsometricPoint(
				self.grid, self.b - other.delta_b, self.s - other.delta_s)

	def __sub__(self, other):
		if isinstance(other, IsometricPoint):
			return self._sub_point(other)
		elif isinstance(other, IsometricVector):
			return self._sub_vector(other)
		else:
			raise ValueError("Invalid __sub__() operand.")

	def move_to(
			self,
			b: Optional[Number] = None,
			s: Optional[Number] = None,
			d: Optional[Number] = None
	) -> None:
		assert argc(b, s, d) == 2, \
				"move_to() must be provided exactly two of (b,s,d)."
		if b is None:
			self.b = self.grid.side_length - s - d
			self.s = s
		elif s is None:
			self.b = b
			self.s = self.grid.side_length - b - d
		else:  # d is None
			self.b = b
			self.s = s

	def move_by(
			self,
			b: Optional[Number] = None,
			s: Optional[Number] = None,
			d: Optional[Number] = None
	) -> None:
		assert argc(b, s, d) == 2, \
				"move_by() must be provided exactly two of (b,s,d)."
		if b is None:
			self.b -= (s + d)
			self.s += s
		elif s is None:
			self.b += b
			self.s -= (b + d)
		else:  # d is None
			self.b += b
			self.s += s

	@property
	def grid(self) -> IsometricGrid:
		return self._grid

	@property
	def b(self) -> Number:
		return self._b

	@property
	def s(self) -> Number:
		return self._s

	@property
	def d(self) -> Number:
		return self.grid.altitude - self._b - self._s

	@b.setter
	def b(self, b: Number) -> None:
		self._s -= 0.5 * (b - self._b)
		self._b = b

	@s.setter
	def s(self, s: Number) -> None:
		self._b -= 0.5 * (s - self._s)
		self._s = s

	@d.setter
	def d(self, d: Number) -> None:
		delta = 0.5 * (d - self.d)
		self._b -= delta
		self._s -= delta


class IsometricVector:
	def __init__(self, delta_b: Number, delta_s: Number):
		self._delta_b = delta_b
		self._delta_s = delta_s

	@classmethod
	def between_points(cls, start: IsometricPoint, end: IsometricPoint
	) -> "IsometricVector":
		return cls(end.b - start.b, end.s - start.s)

	@classmethod
	def from_components(
			cls,
			delta_b: Optional[Number] = None,
			delta_s: Optional[Number] = None,
			delta_d: Optional[Number] = None
	) -> "IsometricVector":
		assert argc(delta_b, delta_s, delta_d) == 2, \
				"from_components() must be provided exactly two arguments."
		return IsometricVector(
				delta_b if delta_b is not None else -0.5 * (delta_s + delta_d),
				delta_s if delta_s is not None else -0.5 * (delta_b + delta_d))

	def unit_vector(self) -> "IsometricVector":
		length = self.length
		return IsometricVector(self.delta_b / length, self.delta_s / length)

	@property
	def delta_b(self) -> Number:
		return self._delta_b

	@property
	def delta_s(self) -> Number:
		return self._delta_s

	@property
	def delta_d(self) -> Number:
		return -(self._delta_b + self._delta_s)

	@delta_b.setter
	def delta_b(self, delta_b: Number) -> None:
		self._delta_s -= 0.5 * (delta_b - self._delta_b)
		self._delta_b = delta_b

	@delta_s.setter
	def delta_s(self, delta_s: Number) -> None:
		self._delta_b -= 0.5 * (delta_s - self._delta_s)
		self._delta_s = delta_s

	@delta_d.setter
	def delta_d(self, delta_d: Number) -> None:
		delta_delta = 0.5 * (delta_d - self.delta_d)
		self._delta_b -= delta_delta
		self._delta_s -= delta_delta

	@property
	def length(self) -> Number:
		return isometric_distance(self.delta_b, self.delta_s)

	@length.setter
	def length(self, length) -> None:
		scale_factor = length / self.length
		self.delta_b *= scale_factor
		self.delta_s *= scale_factor

	def __repr__(self) -> str:
		return f"<{self.delta_b}, {self.delta_s}, {self.delta_d}>"

	def __str__(self) -> str:
		return repr(self)

	def __getitem__(self, key: IsometricDirection) -> Number:
		if key == IsometricDirection.B:
			return self.delta_b
		elif key == IsometricDirection.S:
			return self.delta_s
		elif key == IsometricDirection.D:
			return self.delta_d
		else:
			raise ValueError(f"{key} is not a valid IsometricDirection.")

	def __setitem__(self, key: IsometricDirection, item: Number) -> None:
		if key == IsometricDirection.B:
			self.delta_b = item
		elif key == IsometricDirection.S:
			self.delta_s = item
		elif key == IsometricDirection.D:
			self.delta_d = item
		else:
			raise ValueError(f"{key} is not a valid IsometricDirection.")

	def __add__(self, other: "IsometricVector") -> "IsometricVector":
		return IsometricVector(
				self.delta_b + other.delta_b,
				self.delta_s + other.delta_s)

	def __sub__(self, other: "IsometricVector") -> "IsometricVector":
		return IsometricVector(
				self.delta_b - other.delta_b,
				self.delta_s - other.delta_s)

	def __mul__(self, scalar: Number) -> "IsometricVector":
		return IsometricVector(self.delta_b * scalar, self.delta_s * scalar)

	def __div__(self, scalar: Number) -> "IsometricVector":
		return IsometricVector(self.delta_b / scalar, self.delta_s / scalar)
