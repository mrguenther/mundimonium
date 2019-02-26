from typing import Tuple, List, Optional

from utils.coordinate_grid import CartesianPoint

class SelectionProperty:
	"""
	Base class for property checkers that influence the selection of locations.
	Most objects will have a list of property checkers that must be considered when determining placement.
	
	Children of this class should implement determineValue (and, optionally, additional helper functions)
		This is the function that's called by ExtensibleObjects while determining locations etc.

	TODO: Decide if these should just be functions instead
		As classes, they support inheritance, which might be nice
	"""
	
	def determineValue(
		self, 
		caller: 'ExtensibleObject',
		location: CartesianPoint) -> float:
		"""
		Abstract Method: Determine the value of a location for the property for an object.
		
		Arguments:
			caller {ExtensibleObject}
				-- The object you wish to check the value of the property for
			location {CartesianPoint}
				-- The potential location at which to check the value of the property

		Returns:
			value {float} 
				-- The value of the location
		"""

		raise NotImplementedError


class DistanceToRoadSelection(SelectionProperty):
	"""
	Class for determining the influence of the distance to the nearest road for a location.
		This should be weighted significantly more for certain buildings, such as residences.
	"""

	def determineValue(
		self, 
		caller: 'ExtensibleObject',
		location: CartesianPoint) -> float:
		"""
		Determine the value of a location based on the distance to a road.
		
		Arguments:
			caller {ExtensibleObject}
				-- The object you wish to check the value of the property for
			location {CartesianPoint}
				-- The potential location at which to check the value of the property

		Returns:
			value {float} 
				-- The value of the location
		"""

		caller.getAssociatedNetwork() # The road network against which to check
		caller.properties['associatedNetwork'] # The alternative

		raise NotImplementedError("This is not an abstract class, it's just not written yet.")


class SelectionPropertyManager:
	"""
	Class for associating strings (loaded by ExtensibleObjects from extension files)
		with their respective SelectionProperty classes, allowing them to be
		arbitrarily called without full arbitrary function execution

	Attributes:
		propertyGetters {dict[str: function]}
			-- The dictionary of getter functions that can be called by SelectionProperty objects
	"""

	selectionProperties = {"DistanceToRoadSelection": DistanceToRoadSelection}