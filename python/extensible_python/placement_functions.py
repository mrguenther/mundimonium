from typing import Tuple, List, Optional

from ..component_functions import ComponentFunction
from .utils.coordinate_grid import CartesianPoint
from .utils.helper_funcs import descendants
from . import trigger_functions
from . import pathfinding

class SelectionProperty(ComponentFunction):
	"""
	Base class for property checkers that influence the selection of locations.
	Most objects will have a list of property checkers that must be considered when determining placement.
	
	Children of this class should implement execute (and, optionally, additional helper functions)
		This is the function that's called by MapComponents while determining locations etc.

	TODO: Decide if these should just be functions instead
		As classes, they support inheritance, which might be nice
	TODO: Decide how best to strip the caller argument from execute, and pass all relevant info through args instead
		This will allow for memoisation, as we can then store results by args
	"""
	
	def execute(
			self, 
			caller: 'MapComponent',
			args: dict) -> float:
		"""
		Abstract Method: Determine the value of a location for the property for an object.
		
		Arguments:
			caller (MapComponent): The object you wish to check the value of the property for
			location (CartesianPoint): The potential location at which to check the value of the property
			args (dict): Static values passed from the layer file for this selection property

		Returns:
			value (float): The value of the location
		"""

		raise NotImplementedError

class GradeSelection(SelectionProperty):
	"""
	Class for determining the influence of the slope for a location
		This becomes exponentially more negative the steeper the slope
	"""

	def execute(
			self, 
			caller: 'MapComponent',
			args: dict) -> float:
		"""[summary]
		
		Arguments:
			caller (MapComponent): The object you wish to check the value of the property for
			location (CartesianPoint): The potential location at which to check the value of the property
			args (dict): Static values passed from the layer file for this selection property. Relevant for this class:
				gradeExp (float): Exponent to which the grade value is raised. Default: 1.5
				gradeMult (Number): Multiplier for the grade value, before subtraction: Default: 50

		Returns:
			value (float): The value of the location
				With default arguments, this varies between 1 (flat ground) and -3.5 (20% grade)
		"""

		gradeMult = 50
		if 'gradeMult' in args: gradeMult = args['gradeMult']
		gradeExp = 1.5
		if 'gradeExp' in args: gradeExp = args['gradeExp']

		(x,y) = args['location'].coords[:2]
		associatedTerrain = caller.properties['associatedTerrain']
		grade = associatedTerrain.terrainObject.getGrade(x,y)
		gradeVal = gradeMult * grade**gradeExp
		
		value = 1.0 - gradeVal

		return (value)



class TraversableDistanceToRoadSelection(SelectionProperty):
	"""
	Class for determining the influence of the traversable distance to the nearest road for a location.
		This should be weighted significantly more for certain buildings, such as residences.
		It should take into account the terrain and pathfind along slopes, based on the associatedNetwork's maxGrade property

	TODO: (Consider) Pathfinding over terrain may well need to be imported from ../triggers/terrain
	"""

	def execute(
			self, 
			caller: 'MapComponent',
			args: dict) -> float:
		"""
		Determine the value of a location based on the distance to a road.
		
		Arguments:
			caller (MapComponent): The object you wish to check the value of the property for
			location (CartesianPoint): The potential location at which to check the value of the property
			args (dict): Static values passed from the layer file for this selection property.

		Returns:
			value (float): The value of the location
		"""

		caller.properties['associatedNetwork'] # The road network against which to check

		#raise NotImplementedError("This is not an abstract class, it's just not written yet.")
		return(1) # Dummy return value, for testing purposes


class RoadBranchPointSelection(SelectionProperty):
	"""
	Class for determining the value of a point for creating a point of a new road branch.
		This is the "meat and bones" class for road creation
		It should try to put the point as far as possible from the origin (exponentially higher values closer to maxEdgeLength)
		Though it MUST remain within maxEdgeLength of the creating node, via traversable distance (no more than maxGrade)
		It should, however, be closer in situations where such is necesarry for preserving pathing information
		TODO: Decide how, exactly, we'll do this:
			A possible (expensive) option: Pathfind from creator to the nearest node on network proper
			Only allow points on that path
			Require that a straight path to the creator never exceeds maxGrade
			NOTE: If we do it this way, we'll *definitely* need to implement caching
				Otherwise it's going to be doing the same computations dozens of times
	
	TODO: (Consider) Perhaps inherit from TraversableDistanceToRoadSelection
		In this case, make sure the above is using helper functions that this can steal
	"""

	# Temp function
	def execute(
			self, 
			caller: 'MapComponent',
			args: dict) -> float:
		"""
		Temp function that simply provides value based on distance to the nearest point in the disconnected network
		"""

		maxEdgeLength = caller.properties['associatedNetwork'].properties['maxEdgeLength']
		oldLocation = caller.location
		caller.location = args['location']
		distanceToOtherComponent = trigger_functions.DistanceToOtherConnectedComponent().execute(caller, args)
		distanceToLocation = args['location'].distanceTo(caller.creator.location) # Distance to location from previous point
		caller.location = oldLocation
		if distanceToLocation < maxEdgeLength and distanceToLocation > maxEdgeLength - 1:
			return (1.0/distanceToOtherComponent)
		return (0)


class CreatorLocationSelection(SelectionProperty):
	"""
	Class for deciding location based off the creating object's location.
		This is a binary property (returning 1 if the location is equal, or 0 otherwise)
		Therefore, it should *almost always* be the only property used in placement
	Note that caller.creator MUST be a MapComponent with a location attribute
	"""

	def execute(
			self, 
			caller: 'MapComponent',
			args: dict) -> float:
		"""
		
		"""

		distanceThreshold = 0.5
		if 'distanceThreshold' in args: distanceThreshold = args['distanceThreshold']
		if all(abs(coord-creatorCoord) < distanceThreshold for (coord, creatorCoord) in zip(args['location'].coords, caller.creator.location.coords)):
			return(1.0)
		return(0.0)