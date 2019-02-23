from typing import Tuple, List, Optional

from utils_win.coordinate_grid import CartesianPoint
from road_graph import RoadGraph
from selection_properties import *
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
        locationFactors {list [locationFactor {SelectionProperty}]}
            -- The factors to be evaluated when checking a location's suitability
                locationFactor is a generic for a subclass of SelectionProperty
        locationWeights {numpy.array}
            -- The array of weights associated with each locationFactor
    """

    locationFactors = None
    locationWeights = None
    
    def __init__(self):
        """[summary]
        
        Arguments:
            location {CartesianPoint} -- [description]
        """

        self.properties = {}

        if self.locationFactors is None:
            self.locationFactors = {}
        if self.locationWeights is None:
            self.locationWeights = numpy.array([])

    def loadObject(self, jsonObject: type):
        """
        Load an ExtensibleObject from a given JSON Object
        
        Arguments:
            jsonObject {[type]} -- [description]
        """

        # TODO
        raise NotImplementedError("This is not an abstract class, it's just not written yet.")


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

        locationValues = numpy.array([])

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

        return(returnValues)

class NetworkObject(ExtensibleObject): pass

class TerrainObject(ExtensibleObject): pass

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

        self.location = location

    # TODO: Decide if these should just be accessed from .properties instead
    # It seems sensible enough, and simpler

    def getLocation(self) -> CartesianPoint:
        """
        getter for location - Used by SelectionProperty objects.

        Returns:
            location {CartesianPoint}
                -- The location
        """

        return(self.location)

    def getAssociatedNetwork(self) -> NetworkObject:
        """
        getter for associatedNetwork - Used by SelectionProperty objects.

        Returns:
            associatedNetwork {NetworkObject}
                -- The associatedNetwork
        """

        return(self.properties['associatedNetwork'])

