#!/usr/local/bin/python36

# TODO: Delete this file, as it is now extraneous
# It remains for now as it may still have useful bits to pull into other files
# But your days are numbered, city_object.py

from typing import Tuple, List, Optional

from utils.coordinate_grid import CartesianPoint
from road_graph import RoadGraph
import numpy


class CityObject(CartesianPoint):
	"""
	Base class for objects in the City layer, such as Buildings
	Contain functions for road-management, location evaluation, etc

	Attributes:
		location {CartesianPoint} 
			-- The object's location
		args {dict}
			-- The arguments with which to call locationFactor.determineValue
		locationFactors {list [locationFactor {SelectionProperty}]}
			-- The factors to be evaluated when checking a location's suitability
				locationFactor is a generic for a subclass of SelectionProperty
		locationWeights {numpy.array}
			-- The array of weights associated with each locationFactor
	"""
	
	locationFactors = None
	locationWeights = None
	
	def __init__(self, location: Optional[CartesianPoint] = None, args: Optional[dict] = None):
		"""[summary]
		
		Arguments:
			location {CartesianPoint} -- [description]
		"""

		if self.locationFactors is None:
			self.locationFactors = {}
		if self.locationWeights is None:
			self.locationWeights = numpy.array([])
		if args is None:
			args = {}
		self.args = args

		self.location = location

	def getValues(self, potentialLocations: List[CartesianPoint]) -> numpy.array:
		"""
		Calculate an array of values for each potential location in an array thereof
		
		Arguments:
			potentialLocations {List[CartesianPoint]} 
				-- A list of locations at which to check each location factor
					This should preferably be chosen somewhat heuristically
						Or at a minimum narrowed down significantly from all points
					We may also want to save results to the local point, for future reference
		
		Returns:
			returnValues {numpy.array} 
				-- An array of the resulting values, normalized as a total probability distribution
					TODO: Consider returning potentialLocations as well, so we can more easily reference the chosen spot
		"""

		# TODO: Save these results for any given factor/arg set, so we don't recalculate all the time

		locationValues = numpy.array([])

		for potentialLocation in potentialLocations:
			factorValues = numpy.array([])
			for i in range(self.locationFactors):
				factorValue = self.locationFactors[i].determineValue(potentialLocation) * self.locationWeights[i]
				factorValues.append(factorValue)
			locationValues.append(numpy.sum(factorValues))
		
		total = locationValues.dot(self.locationWeights)
		returnValues = numpy.array([])
		for value in locationValues:
			returnValues.append(value/total)

		return(returnValues)



class SelectionProperty:
	"""
	Base class for properties that influence the selection of locations.
	Most objects in the city will have a list of properties that must be considered when determining placement.
	"""
	

	def determineValue(
		self, 
		location: CartesianPoint, 
		objectSelecting: Optional[CityObject] = None,
		args: Optional[dict] = None) -> float:
		"""
		Abstract Method: Determine the value of a location for the property.
		
		Arguments:
			location {CartesianPoint} 
				-- The location at which to check
			objectSelecting {CityObject} 
				-- The object determining the value at the location
			args {dict}
				-- Dictionary of all args that might be passed to children of SelectionProperty

		Returns:
			valueMultiplier {float} 
				-- The effect of the result on the point's value
		"""

		raise NotImplementedError


class DistanceToRoadSelection(SelectionProperty):
	"""
	Class for determining the influence of the distance to the nearest road for a location.
	This should be weighted significantly more for certain buildings, such as residences.
	"""

	def determineValue(
		self, 
		location: CartesianPoint,
		args: dict) -> float:
		"""
		Determine the value.
		
		Arguments:
			location {CartesianPoint} 
				-- The location at which to check, and from where to determine distance
			args {dict}
				-- Generic arg container. Must contain:
				args ['RoadGraph'] {RoadGraph} 
					-- The RoadGraph to check for nearest road within
		
		Returns:
			valueMultiplier {float} 
				-- The effect of the result on the point's value\
		"""

		raise NotImplementedError("This is not an abstract class, it's just not written yet.")


