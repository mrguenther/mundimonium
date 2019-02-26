# TODO: Delete this file, as it is now extraneous (and missed in the last purge)
# It remains for now as it may still have useful bits to pull into other files
# But your days are just as numbered as city_object.py's, road_graph.py

from typing import Tuple, Optional

from utils.coordinate_grid import CartesianPoint

import networkx

class RoadGraph:
	"""Road network object

	Attributes:
		graph {networkx.Graph} -- Graph itself
		origin {CartesianPoint} -- Starting location of this RoadGraph
	"""

	def __init__(self, origin: CartesianPoint):
		"""[summary]
		
		Arguments:
			originPoint {CartesianPoint} -- [description]
		"""

		self.origin = origin
		self.graph = networkx.Graph()
		self.graph.add_node(self.origin)

	





