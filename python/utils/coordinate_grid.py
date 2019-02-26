from typing import Tuple

import numpy as np
from enum import Enum, auto
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
		coords {tuple[float, float, float]}
	"""

	def __init__(self, location: Tuple[float, float, float]):
		"""
		Create the point at a location

		Arguments:
			location {Tuple[float, float, float]} 
				-- Point creation location
		"""

		self.coords = location



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
		self.sphereRadius = sphereRadius # Radius in self.distanceUnit

	def distance(self, pt1, pt2):
		"""
		Calculates the distance between two points in the grid, given as coordinate-pair tuples.
		"""
		pass