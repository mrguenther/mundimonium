from typing import Tuple, List, Optional

from utils.coordinate_grid import CartesianPoint
from road_graph import RoadGraph
from selection_properties import *
from property_functions import *
import numpy


class ExtensibleObject:
	"""
	Base class for objects that are generated
	These objects load most of their properties from extensible JSON files
	Contains functions for getting various properties, though most of these will be in subclasses
		as well as a mapping of the function names to the functions themselves, for calling by extension files

	Attributes:
		properties {dict[str: Any]}
			-- Dictionary of properties loaded by extension files
		triggers {dict[Trigger: function]}

		locationFactors {list [locationFactor {SelectionProperty}]}
			-- The factors to be evaluated when checking a location's suitability
				locationFactor is a generic for a subclass of SelectionProperty
		locationWeights {numpy.array}
			-- The array of weights associated with each locationFactor
	"""
	
	def __init__(
		self,
		jsonObject: type,
		creator: Optional[ExtensibleObject] = None):
		"""[summary]
		
		Arguments:
			jsonObject {[type]}
				-- The JSON Object from which to load the ExtensibleObject
			creator {ExtensibleObject}
				-- The ExtensibleObject creating this new ExtensibleObject
					TriggerManager, while usually being the object actually instantiating a new ExtensibleObject,
					should set creator to the ExtensibleObject that activated the OnCalled CreationTrigger,
					or the parent layer if it was an Ordered trigger
		"""

		self.properties = {
			'creator': creator}

		self.triggers = {}

		self.loadObject(jsonObject)

	def loadObject(self, jsonObject: type):
		"""
		Load an ExtensibleObject from a given JSON Object
		
		Arguments:
			jsonObject {[type]} -- [description]
		"""

		# TODO
		raise NotImplementedError("This is not an abstract class, it's just not written yet.")

		self.initializeObject()

	def initializeObject(self):
		"""
		After loading the ExtensibleObject from JSON, initialize it
			This is separate from __init__ as it uses information derived in loadObject
		"""

		# TODO
		raise NotImplementedError("This is not an abstract class, it's just not written yet.")
		# Call property functions
		# Call placeObject (add object to parent layer)
		# Add object to helper objects (e.g. bin manager)
		# Add object to renderer if relevant
		# Call any AfterCreation triggers

	def placeObject(self):
		"""
		Abstract Method: Place the object on its parent layer object
			Each subclass should overload this with their own placement method
			Some ExtensibleObjects in the future may, additionally, use a different placeObject
			method than their default.
		"""
		
		# Subclasses will usually call getValues here.
		raise NotImplementedError

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

		"""locationValues = numpy.array([])

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

		return(returnValues)"""

class NetworkObject(ExtensibleObject): pass

class TerrainObject(ExtensibleObject): pass

class LayerObject(ExtensibleObject):
	"""[summary]

	Attributes:
		objectManager {ObjectManager}
			-- 
	"""

	pass


class PointObject(ExtensibleObject):
	"""
	Class for objects that exist at a specific point
	For example, buildings
	
	Attributes:
		properties {dict[str: Any]}
			-- Dictionary of properties loaded by extension files
		location {CartesianPoint}
			-- Location of the PointObject
	"""

	def __init__(self, location: CartesianPoint):
		"""[summary]
		
		Arguments:
			location {CartesianPoint} -- [description]
		"""

		self.properties = {
			'location': location}

	def place(self):
		"""
		Uses loaded object's SelectionFactors to determine a location and place the object there.
			Usually called

		Some objects may use a different 
		"""

