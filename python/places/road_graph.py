from typing import Tuple, Optional

from utils.coordinate_grid import CartesianPoint
from city_object import CityObject

import networkx


class RoadPoint(CityObject):
    pass


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

    





