from typing import Tuple, List, Optional

from utils.coordinate_grid import CartesianPoint
from road_graph import RoadGraph
from selection_properties import *
from property_functions import *
from object_manager import *
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
		objectName: str,
		jsonObject: dict,
		parentLayer: Optional[LayerObject] = None,
		creator: Optional[ExtensibleObject] = None):
		"""[summary]
		
		Arguments:
			objectName {str}
				-- The name of the object (e.g. Residence, RoadNetwork)
			jsonObject {dict}
				-- The JSON Object from which to load the ExtensibleObject
			parentLayer {Optional[LayerObject]}
				-- The parent layer of the object
					This should only be None in the case of the highest-tier layer
			creator {Optional[ExtensibleObject]}
				-- The ExtensibleObject creating this new ExtensibleObject
					TriggerManager, while usually being the object actually instantiating a new ExtensibleObject,
					should set creator to the ExtensibleObject that activated the OnCalled CreationTrigger,
					or the parent layer if it was an Ordered trigger
		"""

		self.properties = {
			'creator': creator}
		self.objectName = objectName
		self.selectionFactors = None
		self.creator = creator
		self.parentLayer = parentLayer
		self.afterCreationTriggers = []
		self.initializeObject(jsonObject)

	def loadObject(self, jsonObject: dict):
		"""
		Load an ExtensibleObject from a given JSON Object
		
		Arguments:
			jsonObject {[type]} -- [description]
		"""

		for prop, data in jsonObject['Properties'].items():
			if data['Origin'] == 'Static':
				self.properties[prop] = data['Value']
			elif data['Origin'] == 'PropertyFunction':
				funcName = data['PropertyFunction']
				args = data['Args']
				result = PropertyFunctionManager.propertyFunctions[funcName](args)
				self.properties[prop] = result
			else:
				raise KeyError('Invalid Origin value %(originValue)s for property %(propertyName)s' % 
					{'originValue': str(data['Origin']),
					'propertyName': str(prop)})
		if 'SelectionFactors' in jsonObject:
			self.selectionFactors = jsonObject['SelectionFactors']
		if 'AfterCreation' in jsonObject:
			for triggerName in jsonObject['AfterCreation']:
				trigger = Trigger(self.parentLayer, triggerName=triggerName)
				self.afterCreationTriggers.append(trigger)

	def initializeObject(self, jsonObject: dict):
		"""
		Load the object from the JSON file and initialize it
		"""

		self.loadObject(jsonObject)
		self.placeObject()
		# TODO
		# Call placeObject (add object to parent layer)
		# Add object to helper objects (e.g. bin manager)
		# Add object to renderer if relevant
		# Call any AfterCreation triggers
		self.subclassInit()
		self.callAfterCreationTriggers()

	def subclassInit(self):
		"""
		A function that's called just before AfterCreation triggers
			Subclasses should overload this if they need their own initialization
			While super().__init__ is also an option, triggers will then be called at that time
		"""
		pass

	def callAfterCreationTriggers(self):
		"""[summary]
		"""

		for trigger in self.afterCreationTriggers:
			self.parentLayer.objectManager.triggerManager.resolveTrigger(trigger, self)

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

	def subclassInit(self):
		"""[summary]
		
		Arguments:
			layerFilePath {str} -- [description]
		"""

		self.objectManager = ObjectManager(self, self.properties['layerFilePath'])
		self.objectManager.triggerManager.resolveOrdered(self)

	def loadObject(self, caller: ExtensibleObject, objectData: Tuple[str, dict]):
		"""[summary]
		
		Arguments:
			caller {ExtensibleObject} -- [description]
			objectData {tuple[dict, str]}
				(objectName, objectInfo)
		"""

		(objectName, objectInfo) = objectData

		# This could be done in a more DRY manner with a dict or similar
		# But then if we decide to have different __init__ args or something for a type of object later
		# everything becomes a massive pain
		# So for now I'm going with this
		if objectInfo['Type'] == 'PointObject':
			newObject = PointObject(objectName, objectInfo, self, caller)
		elif objectInfo['Type'] == 'NetworkObject':
			newObject = NetworkObject(objectName, objectInfo, self, caller)
		elif objectInfo['Type'] == 'TerrainObject':
			newObject = TerrainObject(objectName, objectInfo, self, caller)
		elif objectInfo['Type'] == 'LayerObject':
			newObject = LayerObject(objectName, objectInfo, self, caller)
		else:
			raise KeyError('Invalid object Type %(type)s for object %(objectName)s' % 
						{'type': str(objectInfo['Type']),
						'objectName': str(objectName)})
		self.objectManager.objects.append(newObject)


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

	def placeObject(self):
		"""
		Uses loaded object's SelectionFactors to determine a location and place the object there.
			Usually called

		Some objects may use a different 
		"""

