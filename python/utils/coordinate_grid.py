import numpy as np
from enum import Enum
from distance_unit import DistanceUnit


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
		self.distanceUnit = defaultDistanceUnit

	def distance(pt1, pt2):
		"""
		Calculates the distance between two points in the grid, given as coordinate-pair tuples.
		"""
		raise NotImplementedError("Requires a coordinate system.")


class CartesianGrid(CoordinateGrid):
	"""
	A Cartesian grid for mapping areas on a local scale.
	"""

	def __init__(self):
		self.coordinateSpace = CoordinateSpace.CARTESIAN

	def distance(pt1, pt2):
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
		self.sphereRadius = sphereRadius # Radius in self.distanceUnit

	def distance(pt1, pt2):
		"""
		Calculates the distance between two points in the grid, given as coordinate-pair tuples.
		"""
		pass