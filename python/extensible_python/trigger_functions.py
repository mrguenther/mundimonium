from typing import Tuple, List, Optional, Dict
from ..component_functions import ComponentFunction
from .utils.helper_funcs import descendants
from .utils import component_exceptions
from .terrain.terrain_functions import SimplexTerrain
import networkx
from . import pathfinding

class TriggerThreshold(ComponentFunction):
	"""
	Base class for functions that will be used to determine if a trigger has been called enough times
		TODO
	"""
	
	def execute(
			self, 
			layer: 'MapLayer',
			args: dict) -> float:
		"""
		[summary]
		"""

		raise NotImplementedError


class AssociatedNetworkIsConnected(TriggerThreshold):
	"""
	Class for determining if the associatedNetwork is Complete - that is, every point is connected to the network by at least one edge
		This is a boolean function, and should return either 0 (there are still unconnected nodes) or 1.0 (all nodes are connected by edges)
	"""
	
	def execute(
			self,
			caller: 'MapComponent',
			args: dict) -> float:
		"""[summary]
		
		Arguments:
			caller {[type]} -- [description]
			args {dict} -- [description]
		
		Returns:
			float -- [description]
		"""

		net = caller.properties['associatedNetwork']
		if networkx.is_connected(net.graph):
			return (1.0)
		return (0.0)

class NearestPointInOtherConnectedComponent(TriggerThreshold):
	"""
	Class for determining the nearest node in a part of associatedNetwork not in the same connected component
		TODO: This should be measured by way of traversable distance, based on the RoadNetwork's maxGrade
		TODO: Consider how to make this *not* terribly inefficient
	"""
	
	def execute(
			self,
			caller: 'MapPoint',
			args: dict) -> float:
		"""[summary]
		
		Arguments:
			caller {[type]} -- [description]
			args {dict} -- [description]
		
		Returns:
			float -- [description]
		"""
		
		net = caller.properties['associatedNetwork']
		components = networkx.connected_components(net.graph)
		for component in components:
			component = list(component)
			if caller not in component:
				minDist = pathfinding.distanceBetweenPoints(caller.location, component[0].location)
				minDistNode = component[0]
				for node in component[1:]:
					dist = pathfinding.distanceBetweenPoints(caller.location, node.location)
					if dist < minDist: 
						minDist = dist
						minDistNode = node
				return (minDistNode)
		raise component_exceptions.GraphAlreadyConnectedError


class DistanceToOtherConnectedComponent(NearestPointInOtherConnectedComponent):
	"""
	Class for determining the distance from the caller to a part of associatedNetwork not in the same connected component
		TODO: This should be measured by way of traversable distance, based on the RoadNetwork's maxGrade
		TODO: Consider how to make this *not* terribly inefficient
	"""
	
	def execute(
			self,
			caller: 'MapPoint',
			args: dict) -> float:
		"""[summary]
		
		Arguments:
			caller {[type]} -- [description]
			args {dict} -- [description]
		
		Returns:
			float -- [description]
		"""
		
		nearestNode = super().execute(caller, args)
		return(pathfinding.distanceBetweenPoints(caller.location, nearestNode.location))


class TriggeredFunction(ComponentFunction):
	"""
	Base class for functions that triggers can call to change something
	"""
	
	def execute(
			self,
			caller: 'MapComponent',
			args: dict) -> float:
		"""
		[summary]
		"""

		raise NotImplementedError

class AddEdgeBetweenCreatorAndSelfToNetwork(TriggeredFunction):
	"""
	Class for adding an edge to a point's associatedNetwork, with 'distance' equal to the pathfinding distance along the edge
	"""

	def execute(
			self,
			caller: 'MapPoint',
			args: dict):
		"""
		...

		Arguments:
			caller (MapPoint): The object you wish to add as a node
			args (dict): Dict of args passed along from the JSON object
		"""

		network = caller.properties['associatedNetwork']
		distance = pathfinding.distanceBetweenPoints(caller.location, caller.creator.location)
		network.graph.add_edge(caller, caller.creator, distance = distance)

class AddEdgeFromSelfToExtantNetwork(TriggeredFunction):
	"""[summary]
	"""

	def execute(
			self,
			caller: 'MapPoint',
			args: dict):
		"""
		...

		Arguments:
			caller (MapPoint): The object you wish to add as a node
			args (dict): Dict of args passed along from the JSON object
		"""

		nearestNode = NearestPointInOtherConnectedComponent().execute(caller, args)
		network = caller.properties['associatedNetwork']
		distance = pathfinding.distanceBetweenPoints(caller.location, nearestNode.location)
		network.graph.add_edge(caller, nearestNode, distance = distance)


class AssociatePointWithNetwork(TriggeredFunction):
	"""
	Class for associating a calling MapPoint with a given MapNetwork
	"""

	def execute(
			self,
			caller: 'MapPoint',
			args: dict):
		"""
		Find a MapNetwork and add caller as a node, then return the MapNetwork

		Arguments:
			caller (MapPoint): The object you wish to add as a node
			args (dict): Dict of args passed along from the JSON object
		"""

		networkToAssociate = caller.properties['associatedNetwork']
		networkToAssociate.addPoint(caller)


class GenerateSimplexTerrain(TriggeredFunction):
	"""[summary]
	
	Arguments:
		TriggeredFunction {[type]} -- [description]
	"""
	# TODO: Consider how to move this to another file most likely
	def test(self): print("test")
	def execute(
			self,
			caller: 'MapTerrain',
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

		simplexTerrain = SimplexTerrain(size, seed, caller, args)

		caller.terrainObject = simplexTerrain