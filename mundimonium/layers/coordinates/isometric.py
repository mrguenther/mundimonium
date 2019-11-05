from exceptions import NotAdjacentException
from type_names import Number
from mundimonium.utils.helper_funcs import argc

from enum import Enum
from typing import Optional


class IsometricDirection(Enum):
	B = 0
	S = 1
	D = 2

	def rotated_cw_by_index(self, index: int):
		return IsometricDirection(
				(self.value + index) % len(IsometricDirection))

	def rotated_ccw_by_index(self, index: int):
		return IsometricDirection(
				(self.value - index) % len(IsometricDirection))


class IsometricGrid:
	@property
	def side_length(self):
		raise NotImplementedError()

	@property
	def apothem(self):
		raise NotImplementedError()

	@property
	def altitude(self):
		raise NotImplementedError()


class IsometricPoint:
	def __init__(
			self,
			grid: IsometricGrid,
			b: Optional[Number] = None,
			s: Optional[Number] = None,
			d: Optional[Number] = None):
		self._grid = grid
		if b is None and s is None and d is None:
			self._b = self.grid.apothem
			self._s = self.grid.apothem
		else:
			self.move_to(b, s, d)

	def __init__(self, grid: IsometricGrid, other: "IsometricPoint"):
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
		return f"({self.b},{self.s},{self.d}) in {repr(self.grid)}"

	def __getitem__(self, key: IsometricDirection):
		if key == IsometricDirection.B:
			return self.b
		elif key == IsometricDirection.S:
			return self.s
		elif key == IsometricDirection.D:
			return self.d
		else:
			raise ValueError(f"{key} is not a valid IsometricDirection.")

	def __setitem__(self, key: IsometricDirection, item: Number):
		if key == IsometricDirection.B:
			self.b = item
		elif key == IsometricDirection.S:
			self.s = item
		elif key == IsometricDirection.D:
			self.d = item
		else:
			raise ValueError(f"{key} is not a valid IsometricDirection.")

	def __add__(self, other: "IsometricVector"):
		assert type(other) is IsometricVector, "Invalid __add__() operand."
		return IsometricPoint(
				self.grid,
				self.b + other.delta_b,
				self.s + other.delta_s)

	def __sub__(self, other: "IsometricVector"):
		assert type(other) is IsometricPoint, "Invalid __sub__() operand."
		return IsometricVector(self.b - other.b, self.s - other.s)

	def move_to(
			self,
			b: Optional[Number] = None,
			s: Optional[Number] = None,
			d: Optional[Number] = None):
		assert argc(b, s, d) == 2, (
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

	def move_by(
			self,
			b: Optional[Number] = None,
			s: Optional[Number] = None,
			d: Optional[Number] = None):
		assert argc(b, s, d) == 2, (
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
		return IsometricPoint(adjacent_grid, self)

	@property
	def grid(self):
		return self._grid

	@property
	def b(self):
		return self._b

	@property
	def s(self):
		return self._s

	@property
	def d(self):
		return self.grid.altitude - self._b - self._s

	@b.setter
	def b(self, b):
		self._s -= 0.5 * (b - self._b)
		self._b = b

	@s.setter
	def s(self, s):
		self._b -= 0.5 * (s - self._s)
		self._s = s

	@d.setter
	def d(self, d):
		delta = 0.5 * (d - self.d)
		self._b -= delta
		self._s -= delta


class IsometricVector:
	"""TODO"""
	def __init__(self, start: IsometricPoint, end: IsometricPoint):
		"""TODO"""
		self._delta_b = end.b - start.b
		self._delta_s = end.s - start.s

	def __init__(self, delta_b: Number, delta_s: Number):
		"""TODO"""
		print("Fast constructor!")
		self._delta_b = delta_b
		self._delta_s = delta_s

	def __init__(self,
			delta_b: Optional[Number] = None,
			delta_s: Optional[Number] = None,
			delta_d: Optional[Number] = None):
		"""TODO"""
		print("Slow constructor!")
		assert argc(delta_b, delta_s, delta_d) == 2, (
				"IsometricVector() must be provided exactly two arguments.")
		if delta_b is None:
			self._delta_b = -(delta_s + delta_d)
			self._delta_s = delta_s
		elif delta_s is None:
			self._delta_b = delta_b
			self._delta_s = -(delta_b + delta_d)
		else:  # delta_d is None
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

	def __add__(self, other: "IsometricVector"):
		return IsometricVector(
				self.delta_b + other.delta_b,
				self.delta_s + other.delta_s)

	def __sub__(self, other: "IsometricVector"):
		return IsometricVector(
				self.delta_b - other.delta_b,
				self.delta_s - other.delta_s)

	def __mul__(self, scalar: Number):
		return IsometricVector(self.delta_b * scalar, self.delta_s * scalar)

	def __div__(self, scalar: Number):
		return IsometricVector(self.delta_b / scalar, self.delta_s / scalar)
