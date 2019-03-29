from typing import Optional
import networkx

from ..layer import *
from ...terrain import generators
from .objects import buildings
from ...utils.coordinate_grid import CartesianPoint

class City(Layer):
	"""
	TODO
	"""

	def __init__(
			self, 
			parentLayer: Optional[Layer] = None, 
			timeController: Optional[temporal.TimeController] = None):
		"""[summary]
		
		Arguments:
			parentLayer {[type]} -- [description]
		"""

		super().__init__(parentLayer=parentLayer, timeController=timeController)
		try:
			terrainAttributes = self.getTerrainAttributes(self.parentLayer)
		except NotImplementedError:
			terrainAttributes = None
		self.terrainObject = CityTerrain(terrainAttributes = terrainAttributes)
		self.roadNetwork = CityRoads(self)

class CityRoads:
	"""
	Builds and manages the road network(s) of a city

	Attributes:
		graph (networkx.Graph): The actual graph of the road network
		parentLayer (Layer): The layer that owns this roadnetwork
	"""

	def __init__(self, city: City):
		self.graph = networkx.Graph()
		self.city = city

	def nearestPointInNetwork(self, location: CartesianPoint):
		"""
		Find the nearest point in the network as the crow flies
		
		Arguments:
			location (CartesianPoint): Location from which to find the nearest point

		Returns:
			(node): The nearest node in the network
		"""

		nodes = list(self.graph.nodes)
		minDist = location.distanceTo(nodes[0].location)
		minDistNode = nodes[0]
		for node in nodes[1:]:
			dist = location.distanceTo(node.location)
			if dist < minDist: 
				minDist = dist
				minDistNode = node
		return (minDistNode)


	def pathfindToNetwork(self, location: CartesianPoint, maxSlope: float = 1.0):
		"""
		Find the route to the nearest point in the network without exceeding maxSlope in slope
		Note that this might not always be the shortest route to the network in general
		
		Arguments:
			location (CartesianPoint): Location from which to pathfind
			maxSlope (float): The maximum slope allowed on the path, given as rise:run ratio.
				Default: 1.0 (i.e. 45 degrees)
		"""

		nearestNode = self.nearestPointInNetwork(location)
		x, y = self.city.terrainObject.nearestXY(location)
		route = [self.city.terrainObject.getPoint(x,y)]
		# TODO

class CityTerrain:
	"""[summary]
	"""

	def __init__(
			self,
			terrainAttributes: Optional[dict] = None):

		# TODO: Support more terrainAttributes
		if terrainAttributes is None: terrainAttributes = {}
		self.terrainAttributes = terrainAttributes
		if 'size' not in self.terrainAttributes: self.terrainAttributes['size'] = (1024,1024)
		self.size = self.terrainAttributes['size']
		self.heightmap = generators.SimplexGenerator2d().generate(self.size)

	def nearestXY(self, location: CartesianPoint) -> tuple:
		"""
		Returns the nearest valid (x, y) tuple for accessing the heighmap
		
		Arguments:
			location (CartesianPoint): Location for which to find the nearest (x, y)

		Returns:
			(x, y) (tuple): The nearest valid (x, y)
		"""

		x, y = location.coords[:2]
		x = int(x+.5)
		y = int(y+.5)
		if x >= self.size[0]: x = self.size[0]-1
		if y >= self.size[1]: y = self.size[1]-1
		if x < 0: x = 0
		if y < 0: y = 0

		return(x,y)
	
	def getHeight(self, location: CartesianPoint) -> float:
		"""
		Returns the height of the nearest point on the map to the location
		
		Arguments:
			location (CartesianPoint): Location to get the height near
		
		Returns:
			(float): Height near that location
		"""

		return(self.heightmap[self.nearestXY(location)])

	def getPoint(self, x: int, y: int) -> CartesianPoint:
		"""
		Returns a point on the heightmap as a CartesianPoint
		
		Arguments:
			x {[type]} -- [description]
			y {[type]} -- [description]
		
		Returns:
			CartesianPoint -- [description]
		"""

		z = self.heightmap[(x,y)]
		return(CartesianPoint((x,y,z)))
	


