from typing import Tuple, List, Optional, Dict
from .utils.helper_funcs import descendants
from .terrain.terrain_functions import CityTerrain

class TriggerThreshold:
	"""
	Base class for functions that will be used to determine if a trigger has been called enough times
		TODO
	"""
	
	def testThreshold(
			self, 
			layer: 'MapLayer',
			args: dict) -> float:
		"""
		[summary]
		"""

		raise NotImplementedError


class TriggerThresholdManager:
	"""
	[summary]
	"""

	triggerThresholds = {descendant.__name__:descendant for descendant in descendants(TriggerThreshold)}


class TriggeredFunction:
	"""
	Base class for functions that triggers can call to change something
	"""
	
	def executeFunction(
			self,
			caller: 'MapComponent',
			args: dict) -> float:
		"""
		[summary]
		"""

		raise NotImplementedError


class AssociatePointWithNetwork(TriggeredFunction):
	"""
	Class for associating a calling MapPoint with a given MapNetwork
	"""

	def executeFunction(
			self,
			caller: 'MapPoint',
			args: Dict[str, any]):
		"""
		Find a MapNetwork and add caller as a node, then return the MapNetwork

		Arguments:
			caller (MapPoint): The object you wish to add as a node
			args (dict): Dict of args passed along from the JSON object
					Required Values:
					'NetworkByProperty' (str): The name of the property in which the network to associate with is stored
							Usually this is associatedNetwork
		"""

		networkToAssociate = caller.properties[args['NetworkByProperty']]
		networkToAssociate.addPoint(caller)


class GenerateSimplexHeightmap(TriggeredFunction):
	"""[summary]
	
	Arguments:
		TriggeredFunction {[type]} -- [description]
	"""
	# TODO: Consider how to move this to another file most likely

	def executeFunction(
			self,
			caller: 'MapPoint',
			args: Dict[str, any]):
		"""[summary]
		
		Arguments:
			caller {[type]} -- [description]
			args {Dict[str, any]} -- [description]
		
		Returns:
			dict -- [description]
		"""


		size = caller.properties['size']
		seed = caller.properties['seed']

		cityTerrain = CityTerrain(size, seed, args)

		caller.heightmap = cityTerrain.heightmap


class TriggeredFunctionManager:
	"""
	Class for allowing excecution of functions by string reference

	Attributes:
		triggeredFunctions (Dict[str: class]): Dict of names: TriggeredFunction subclass
	"""

	triggeredFunctions = {descendant.__name__:descendant for descendant in descendants(TriggeredFunction)}