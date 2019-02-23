from typing import Tuple, Optional

from utils_win.coordinate_grid import CartesianPoint

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

    





