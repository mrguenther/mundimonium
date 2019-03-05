from typing import Tuple, List, Optional, Dict

from .utils.coordinate_grid import CartesianPoint
from .extendable_python.placements.placement_functions import SelectionPropertyManager
from .extendable_python.properties.property_functions import PropertyFunctionManager
from .extendable_python.triggers.trigger_functions import TriggerThresholdManager
from .layer_classes import LayerManager, Trigger
import numpy
import networkx
import warnings

warnings.simplefilter('always', UserWarning)

# Force warnings.warn() to omit the source code line in the message
formatwarning_orig = warnings.formatwarning
warnings.formatwarning = lambda message, category, filename, lineno, line=None: \
	formatwarning_orig(message, category, filename, lineno, line='')
# TODO (long-term): Catch warnings and do something more user-palatable with them

class MapComponent:
	"""
	Base class for objects that are generated
	These objects load most of their properties from extensible JSON files
	Contains functions for getting various properties, though most of these will be in subclasses
		as well as a mapping of the function names to the functions themselves, for calling by extension files

	Attributes:
		properties (dict[str: Any]): Dictionary of properties loaded by extension files
		objectName (str): The name of the object (e.g. Residence, RoadNetwork)
		jsonObject (dict): The JSON Object from which to load the MapComponent
		parentLayer (Optional[MapLayer]): The parent layer of the object
				This should only be None in the case of the highest-tier layer
		creator (Optional[MapComponent]): The MapComponent creating this new MapComponent
				TriggerManager, while usually being the object actually instantiating a new MapComponent,
				should set creator to the MapComponent that activated the OnCalled CreationTrigger,
				or the parent layer if it was an Ordered trigger
		afterCreationTriggers (list): Triggers to be called on object creation
	"""
	
	def __init__(
			self,
			objectName: str,
			jsonObject: dict,
			parentLayer: Optional['MapLayer'] = None,
			creator: Optional['MapComponent'] = None):

		self.properties = {
			'creator': creator}
		self.objectName = objectName
		self.jsonObject = jsonObject
		self.selectionFactors = None
		self.creator = creator
		self.parentLayer = parentLayer
		if parentLayer is not None:
			self.parentLayer.addObject(self)
		self.afterCreationTriggers = []
		self.initializeObject(jsonObject)

	def loadObject(self, jsonObject: dict):
		"""
		Load an MapComponent from a given JSON Object
		
		Arguments:
			jsonObject {[type]} -- [description]
		"""

		for prop, data in jsonObject['Properties'].items():
			if data['Origin'] == 'Static':
				self.properties[prop] = data['Value']
			elif data['Origin'] == 'PropertyFunction':
				funcName = data['PropertyFunction']
				args = data['Args']
				try:
					result = PropertyFunctionManager.propertyFunctions[funcName].determineValue(None, self, args)
					self.properties[prop] = result
				except (Exception) as e:
					warnings.warn('''%(eType)s raised while getting property %(prop)s: 
					%(eStr)s
					Using default value of %(value)s''' %
						{'eType': type(e),
						'prop': prop,
						'eStr': str(e),
						'value': str(data['Value'])})
					self.properties[prop] = data['Value']
			else:
				raise KeyError('Invalid Origin value %(originValue)s for property %(propertyName)s' % 
					{'originValue': str(data['Origin']),
					'propertyName': str(prop)})
		if 'SelectionFactors' in jsonObject:
			self.selectionFactors = jsonObject['SelectionFactors']
		if 'Triggers' in jsonObject:
			self.loadTriggers(jsonObject['Triggers'])

	def loadTriggers(self, jsonTriggers: dict):
		"""
		Load triggers from a subset of the jsonObject
			For now, this just involves populating self.afterCreationTriggers
			There may later be additional types of triggers to be called
		
		Arguments:
			jsonTriggers (dict): Dict of triggers from layer.json files
		"""

		# TODO: Flesh this out for more trigger varieties

		# TODO: Probably move this whole chunk into the Trigger or TriggerManager class
		# Give them a loadFromJSON method or something, then just do a nice short call here
		for triggerName, trigger in jsonTriggers.items():
			repeatCount = 0
			if 'RepeatCount' in trigger: repeatCount = trigger['RepeatCount']
			stopThreshold = None
			if 'StopThreshold' in trigger: stopThreshold = trigger['StopThreshold']
			continueFunction = None
			if 'ContinueFunction' in trigger: 
				continueFunction = TriggerThresholdManager.triggerThresholds[trigger['ContinueFunction']].testThreshold
			continueArgs = None
			if 'ContinueArgs' in trigger: continueArgs = trigger['ContinueArgs']
			if trigger['TriggerType'] == 'AfterCreation':
				triggerObject = Trigger(
					self.parentLayer, 
					repeatCount=repeatCount, 
					stopThreshold=stopThreshold, 
					continueFunction=continueFunction, 
					continueArgs=continueArgs,
					triggerName=triggerName)
				self.afterCreationTriggers.append(triggerObject)


	def initializeObject(self, jsonObject: dict):
		"""
		Load the object from the JSON file and initialize it
		"""

		self.loadObject(jsonObject)
		self.placeObject()
		# TODO
		# Add object to helper objects (e.g. bin manager)
		# Add object to renderer if relevant
		# Add any non-AfterCreation triggers
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
		"""
		Calls any AfterCreation Triggers for the object
		"""

		for trigger in self.afterCreationTriggers:
			self.parentLayer.layerManager.triggerManager.resolveTrigger(trigger, self)

	def placeObject(self):
		"""
		Determine potential locations for the object and place it, if applicable
			Each subclass that cares should overload this with their own placement method
			Note that subclasses that *do* have a location should place themselves on the
			location lookup table in this function in the future
		"""
		pass

class MapNetwork(MapComponent):
	"""[summary]
	
	Attributes:
		graph (networkx.Graph): The network this object is built around
	"""

	def subclassInit(self):
		"""[summary]
		"""

		self.graph = networkx.Graph()

	def addPoint(self, point: 'MapPoint'):
		"""[summary]
		
		Arguments:
			point {MapPoint} -- [description]
		"""

		self.graph.add_node(point)


class MapTerrain(MapComponent):
	"""[summary]
	
	Attributes:
		heightmap (Dict[Tuple[int, int]: float]): Heightmap for the terrain
	"""

	def subclassInit(self):
		"""
		Do terrain-specific init stuff
		For now, that's limited to creating an empty heightmap dict, to be filled by a triggered function later
		While the heightmap could be generated here instead, using a triggered function lets us more flexibly
		change algorithms and parameters passed to those algorithms
		"""

		self.heightmap = {}


class MapLayer(MapComponent):
	"""[summary]

	Attributes:
		layerManager (LayerManager): 
	"""

	def subclassInit(self):
		"""[summary]
		
		Arguments:
			layerFilePath (str): The filepath of the layer JSON file from which to load... everything in the layer
		"""

		self.layerManager = LayerManager(self, self.properties['layerFilePath'])
		self.layerManager.triggerManager.resolveOrdered(self)

	def addObject(self, mapObject: MapComponent):
		"""
		Add a MapComponent to the layer's LayerManager
		
		Arguments:
			objectName {[type]} -- [description]
		"""

		self.layerManager.objects.append(mapObject)
		if mapObject.objectName not in self.layerManager.objectsByName:
			self.layerManager.objectsByName[mapObject.objectName] = []
		self.layerManager.objectsByName[mapObject.objectName].append(mapObject)


	def loadOtherObject(self, caller: MapComponent, objectData: Tuple[str, dict]):
		"""[summary]
		
		Arguments:
			caller {MapComponent} -- [description]
			objectData {tuple[dict, str]}
				(objectName, objectInfo)
		"""

		(objectName, objectInfo) = objectData

		# This could be done in a more DRY manner with a dict or similar
		# But then if we decide to have different __init__ args or something for a type of object later
		# everything becomes a massive pain
		# So for now I'm going with this
		if objectInfo['Type'] == 'MapPoint':
			MapPoint(objectName, objectInfo, self, caller)
		elif objectInfo['Type'] == 'MapNetwork':
			MapNetwork(objectName, objectInfo, self, caller)
		elif objectInfo['Type'] == 'MapTerrain':
			MapTerrain(objectName, objectInfo, self, caller)
		elif objectInfo['Type'] == 'MapLayer':
			MapLayer(objectName, objectInfo, self, caller)
		else:
			raise KeyError('Invalid object Type %(type)s for object %(objectName)s' % 
						{'type': str(objectInfo['Type']),
						'objectName': str(objectName)})


class MapPoint(MapComponent):
	"""
	Class for objects that exist at a specific point
	For example, buildings
	
	Attributes:
		location (CartesianPoint): Location of the MapPoint
	"""

	def subclassInit(self):
		"""[summary]
		"""

		self.placeObject()


	def findPotentialLocations(self) -> List[CartesianPoint]:
		"""
		Returns a list of potential locations
			Ideally, this will use some sort of heuristics to narrow down the options from 'everywhere'

		Returns:
			potentialLocations (List[CartesianPoint]): List of potential locations
		"""

		# TODO: Make this actually do something
		# This is just a temp function to get a bunch of potential points
		potentialLocations = []
		for x in range(100):
			for y in range(100):
				potentialLocations.append(CartesianPoint((x,y,0)))
		return(potentialLocations)


	def placeObject(self):
		"""
		Uses loaded object's SelectionFactors to determine a location and place the object there.
		"""

		potentialLocations = self.findPotentialLocations()
		locationValues = self.getValues(potentialLocations)
		values = numpy.array(list(locationValues.values()))
		values /= values.sum()
		self.location = numpy.random.choice(list(locationValues.keys()),1,p=values)[0]


	def getValue(self, potentialLocation: CartesianPoint) -> float:
		"""[summary]
		
		Arguments:
			potentialLocation {CartesianPoint} -- [description]
		
		Returns:
			float -- [description]
		"""

		valueTot = 0.0

		for selectionFactor, data in self.selectionFactors.items():
			if data['Origin'] == 'SelectionProperty':
				factorWeight = 1.0
				if 'FactorWeight' in data: factorWeight = data['FactorWeight']
				funcName = data['SelectionProperty']
				try:
					result = SelectionPropertyManager.selectionProperties[funcName].determineValue(None, self, potentialLocation)
					value = result * factorWeight
				except (Exception) as e:
					warnings.warn('''%(eType)s raised while getting value: 
					%(eStr)s
					Defaulting to value of 0''' %
						{'eType': type(e),
						'eStr': str(e)})
					value = 0
			else:
				raise KeyError('Invalid Origin value %(originValue)s for selectionFactor %(selectionFactor)s' % 
					{'originValue': str(data['Origin']),
					'selectionFactor': str(selectionFactor)})

			valueTot += value

		return(valueTot)

	def getValues(self, potentialLocations: List[CartesianPoint]) -> dict:
		"""
		Calculate an array of values for each potential location in an array thereof
		
		Arguments:
			potentialLocations (List[CartesianPoint]): A list of locations at which to check each location factor
					This should preferably be chosen somewhat heuristically
						Or at a minimum narrowed down significantly from all points
					We may also want to save results to the local point, for future reference
		
		Returns:
			values (dict[CartesianPoint: float]): A dict of the resulting values (keyed by location), NOT yet normalized
					TODO: Consider returning potentialLocations as well, so we can more easily reference the chosen spot
		"""

		# TODO: Save these results for any given factor/arg set, so we don't recalculate all the time
		# Also change this to a dot product eventually
		# But right now that's far from the least efficient part here

		values = {}

		for potentialLocation in potentialLocations:
			values[potentialLocation] = self.getValue(potentialLocation)

		return(values)

