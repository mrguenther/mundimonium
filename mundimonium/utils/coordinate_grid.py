from typing import Tuple, Union

from mundimonium.utils.distance_unit import DistanceUnit
from numbers import Number
import numpy as np
from enum import Enum, auto


class CoordinateSpace(Enum):
	"""
	Enum for specifying the coordinate space of a grid.
	"""

	CARTESIAN = auto()
	SPHERICAL = auto()


class CoordinateGrid:
	"""
	TODO
	"""

	defaultDistanceUnit = DistanceUnit.MILE
	coordinateSpace = None

	def __init__(self):
		self.distanceUnit = self.defaultDistanceUnit

	def distance(self, pt1, pt2):
		"""
		Calculates the distance between two points in the grid, given as coordinate-pair tuples.
		"""
		raise NotImplementedError("Requires a coordinate system.")


class CartesianPoint:
	"""
	A point in a 3d CartesianGrid

	Attributes:
		coords (Tuple[Number, Number, Number]): (x, y, z)
	"""

	def __init__(self, location: Tuple[Number, Number, Number]):
		"""
		Create the point at a location

		Arguments:
			location (Tuple[Number, Number, Number]): Point creation location (x, y, z)
		"""

		if (not isinstance(location, tuple) or
				not all(isinstance(i, Number) for i in location) or
				not len(location) == 3):
			raise TypeError("location must be Tuple[Number, Number, Number]")

		self.coords = location

	def __hash__(self):
		"""
		Hash function - allows CartesianPoints objects to be used as hashables
		"""

		return(hash(self.coords))

	def __eq__(self, other) -> bool:
		"""
		Equivalence function - allows comparison and usage of CartesianPoint objects as hashables
		Note that tuples are compared against CartesianPoint.coords for equivalency
		"""

		if type(self) == type(other):
			return(self.__hash__() == other.__hash__())
		if type(other) == tuple:
			return(self.coords == other)
		return(False)

	def __repr__(self) -> str:
		"""
		Human-readable print function
		"""

		return("CartesianPoint(%(coords)s)" %
			{'coords': str(self.coords)})

	def distanceTo(self, point: Union['CartesianPoint', tuple]) -> float:
		"""
		Determine the distance to a point

		Arguments:
			point {CartesianPoint, tuple}

		Returns:
			distance {float}
		"""

		if type(point) is tuple:
			point = CartesianPoint(point)
		dist = float(np.sum(np.power(np.subtract(self.coords, point.coords), 2)) ** .5)
		return (dist)


class CartesianGrid(CoordinateGrid):
	"""
	A Cartesian grid for mapping areas on a local scale.
	"""

	def __init__(self):
		self.coordinateSpace = CoordinateSpace.CARTESIAN

	def distance(self, pt1, pt2):
		"""
		Calculates the distance between two points in the grid, given as coordinate-pair tuples.
		"""

		pass


class SphericalGrid(CoordinateGrid):
	"""
	A spherical polar grid (with constant r) for mapping areas on a global scale.

	Parameters theta and phi are internally stored as radians to simplify calculations.
	"""

	def __init__(self, sphereRadius):
		self.coordinateSpace = CoordinateSpace.SPHERICAL
		self.sphereRadius = sphereRadius  # Radius in self.distanceUnit

	def distance(self, pt1, pt2):
		"""
		Calculates the distance between two points in the grid, given as coordinate-pair tuples.
		"""

		pass
