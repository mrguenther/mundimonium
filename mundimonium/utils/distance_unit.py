from enum import Enum


class DistanceUnit(Enum):
	"""
	Enum for various distance measurements.

	Contains helper functions for converting between units.
	"""

	# Fine measurements
	CENTIMETER = 100
	INCH = 254

	# Local measurements
	FOOT = 3048
	METER = 10000

	# Geographical measurements
	KILOMETER = 10000000
	MILE = 16093440

	def __str__(self):
		return self.name

	def convertTo(self, otherUnit, distance=1):
		return distance * (self.value / otherUnit.value)

	def convertFrom(self, otherUnit, distance=1):
		return distance * (otherUnit.value / self.value)
