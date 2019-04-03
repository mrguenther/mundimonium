from typing import Optional
import networkx
from networkx.algorithms.shortest_paths.astar import astar_path as astar

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
		self.roadNetwork = RoadNetwork(self)

class RoadNetwork:
	"""
	Builds and manages the road network(s) of a city

	Attributes:
		graph (networkx.Graph): The actual graph of the road network
		parentLayer (Layer): The layer that owns this roadnetwork
	"""

	def __init__(self, city: City):
		self.graph = networkx.Graph()
		self.city = city

	def addPoint(self, location: CartesianPoint):
		"""[summary]
		
		Arguments:
			location {CartesianPoint} -- [description]
		"""

		self.graph.add_node(location.coords[:2], location=location)

	def nearestPointInNetwork(self, location: CartesianPoint):
		"""
		Find the nearest point in the network as the crow flies
		
		Arguments:
			location (CartesianPoint): Location from which to find the nearest point

		Returns:
			(node): The nearest node in the network
		"""

		minDist = self.city.terrainObject.size[0]*self.city.terrainObject.size[1] 
		# If the shortest distance is greater than this something's gone terribly wrong
		for node, loc in self.graph.nodes.data('location'):
			dist = location.distanceTo(loc)
			if dist < minDist: 
				minDist = dist
				minDistNode = node
		return (minDistNode)

	def getSlopedVal(self, p1: tuple, p2: tuple, dist: float, args: dict):
		"""[summary]
		
		
		Arguments:
			p1 {[type]} -- [description]
			p2 {[type]} -- [description]
			dist {[type]} -- [description]
			args (dict): Static values passed from the layer file for this selection property. Relevant for this class:
				gradeExp (float): Exponent to which the grade value is raised. Default: 1.5
				gradeMult (Number): Multiplier for the grade value, before subtraction: Default: 50

		Returns:
			value (float): The value of the location
				With default arguments, this varies between 1 (flat ground) and 4.5 (20% grade)
		"""

		assert (abs(p1[0]-p2[0])==1) != (abs(p1[1]-p2[1])==1) # The horizontal distance between points should always be 1
		grade = (dist**2-1)**0.5 # Which simplifies calculating the grade, somewhat (note that grade = rise/run = rise/1)

		gradeMult = 50
		if 'gradeMult' in args: gradeMult = args['gradeMult']
		gradeExp = 1.5
		if 'gradeExp' in args: gradeExp = args['gradeExp']
		slopedVal = 1 + gradeMult * grade**gradeExp
		return(slopedVal)

	def naiveAStarHeuristic(self, p1: tuple, p2: tuple) -> float:
		"""
		Minimal a-star heuristic for pathfinding
		Just returns distance based off taxicab
		TODO: Consider improving this
		
		Arguments:
			p1 (tuple): 2-tuple (x,y). The point at which to check the heuristic
			p2 (tuple): 2-tuple (x,y). The end point to which we are pathfinding

		Returns:
			float: Distance to the endpoint (taxicab).
		"""

		#dist = ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**.5
		dist = abs(p1[0]-p2[0]) + abs(p1[1]-p2[1])
		return(dist)

	def addRoute(self, route: list):
		"""[summary]
		
		Arguments:
			route {list} -- [description]
		"""

		prev = None
		for point in route:
			loc = self.city.terrainObject.getPoint(point[0],point[1])
			self.addPoint(loc)
			if prev is not None:
				self.graph.add_edge(prev, point)
				self.city.terrainObject.graph.edges[prev,point]['slopedWeight'] = 0
			prev = point

	def pathfindToNetwork(self, location: CartesianPoint, gradeMult: Optional[float] = None, gradeExp: Optional[float] = None):
		"""
		Find the route to the nearest point in the network without exceeding maxSlope in slope
		Note that this might not always be the shortest route to the network in general
		
		Arguments:
			location (CartesianPoint): Location from which to pathfind
			maxSlope (float): The maximum slope allowed on the path, given as rise:run ratio.
				Default: 1.0 (i.e. 45 degrees)
		"""
		# TODO: Make this pathfind to the network in general less naively
		# Consider: Make slopedWeight = 0 wherever the RoadNetwork already exists

		slopeArgs = {}
		if gradeMult is not None: slopeArgs['gradeMult'] = gradeMult
		if gradeExp is not None: slopeArgs['gradeExp'] = gradeExp

		if 'slopedWeight' not in self.city.terrainObject.graph.edges[(1,1),(1,2)]:
			print('slopedWeight not yet calculated for this city.terrainObject.')
			self.city.terrainObject.addSlopedWeights(self.getSlopedVal, slopeArgs, attributeName='slopedWeight')

		nearestNode = self.nearestPointInNetwork(location)
		x, y = self.city.terrainObject.nearestXY(location)
		route = astar(self.city.terrainObject.graph, (x, y), nearestNode, heuristic=self.naiveAStarHeuristic, weight='slopedWeight')
		return(route)

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
		print("Generating CityTerrain.heightDict...")
		self.heightDict = generators.SimplexGenerator2d().generate(self.size)
		print("Creating CityTerrain.graph...")
		self.graph = networkx.Graph()
		self.createGraph()
		print("Done creating CityTerrain.")

	def createGraph(self):
		"""
		Creates a graph of the heightDict, with edges between adjacent points.
		Nodes are indexed by (x, y) tuples, and have the following attributes:
			z: Height of the terrain at this location
		Edges have the following attributes:
			dist: Distance between points - As such, they also indicate the slope of the terrain
				For example, a 45-degree angle would result in sqrt(2) weight
		
		Keyword Arguments:
			heightDict {[type]} -- [description] (default: {None})
		"""

		self.graph = networkx.Graph()
		for i in self.heightDict: 
			self.graph.add_node(i,z=self.heightDict[i])

		print("Adding adjacency edges...")
		for x in range(self.size[0]):
			for y in range(self.size[1]):
				loc = self.getPoint(x,y)
				if x<self.size[0]-1:
					dist = loc.distanceTo(self.getPoint(x+1,y))
					self.graph.add_edge((x,y),(x+1,y),dist=dist)
				if y<self.size[1]-1:
					dist = loc.distanceTo(self.getPoint(x,y+1))
					self.graph.add_edge((x,y),(x,y+1),dist=dist)

	
	def addSlopedWeights(self, slopeFunc: callable, args: dict, attributeName: str = 'slopedWeight'):
		"""
		Add an attribute for slope weights to the graph
		These will usually be used for pathfinding later - so "invalid" slopes should probably have very high values
		
		Arguments:
			slopeFunc {callable} -- [description]
			args {dict} -- [description]
		
		Keyword Arguments:
			attributeName {str} -- [description] (default: {'slopedWeight'})
		"""

		print('Calculating %(attributeName)s...' %
			{'attributeName': attributeName})
		for p1, p2, dist in self.graph.edges.data('dist'):
			self.graph.edges[p1, p2][attributeName] = slopeFunc(p1, p2, dist, args)


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

		return((x,y))
	
	def getHeight(self, location: CartesianPoint) -> float:
		"""
		Returns the height of the nearest point on the map to the location
		
		Arguments:
			location (CartesianPoint): Location to get the height near
		
		Returns:
			(float): Height near that location
		"""

		return(self.graph.nodes[self.nearestXY(location)]['z'])

	def getPoint(self, x: int, y: int) -> CartesianPoint:
		"""
		Returns a point in the terrain as a CartesianPoint
		
		Arguments:
			x {[type]} -- [description]
			y {[type]} -- [description]
		
		Returns:
			CartesianPoint -- [description]
		"""

		z = self.getHeight(CartesianPoint((x,y,0)))
		return(CartesianPoint((x,y,z)))
	
def tempRender(heightDict, route, size):
	from PIL import Image
	display = Image.new('RGB', size)
	print('Readying image...')
	# Print a pretty picture
	for x in range(0,size[0]):
		for y in range(0,size[1]):
			height = heightDict[(x,y)]
			rgbVal = int(height*127+128)
			rgb = (0,rgbVal,255-rgbVal)
			if (x,y) in route: rgb=(255,0,0)
			display.putpixel((x,y),rgb)
		print(int(10000.0*x/size[0])/100, '%')
	display.show()

