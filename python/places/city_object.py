#!/usr/local/bin/python36

from typing import Tuple, Optional

from utils.coordinate_grid import CartesianPoint
from road_graph import RoadGraph
import numpy


class CityObject(CartesianPoint):
    """Base class for objects in the City layer, such as Buildings
    Contain functions for road-management, location evaluation, etc

    Attributes:
        location {CartesianPoint} -- The object's location
        locationFactors {dict [locationFactor {LocationSelectionProperty}: weight {float}]}
            -- The factors to be evaluated when checking a location's suitability
            -- locationFactor is a generic for a subclass of LocationSelectionProperty
            -- weight is the weight this factor has on the class
    """
    
    locationFactors = None
    locationWeights = None

    
    def __init__(self, location: Optional[CartesianPoint] = None):
        """[summary]
        
        Arguments:
            location {CartesianPoint} -- [description]
        """

        if self.locationFactors is None:
            self.locationFactors = {}
        if self.locationWeights is None:
            self.locationWeights = numpy.array([])

        self.location = location

    def getValues(self, potentialLocation: CartesianPoint):
        # TODO: Save these results for any given factor/arg set, so we don't recalculate all the time
        locationValues = numpy.array([])
        for i in range(self.locationFactors):
            factorValue = self.locationFactors[i].determineValue(potentialLocation) * self.locationWeights[i]
            locationValues.append(factorValue)
        
        total = locationValues.dot(self.locationWeights)

        return(total)



class LocationSelectionProperty:
    """Base class for properties that influence the selection of locations.
    Most objects in the city will have a list of properties that must be considered when determining placement.
    """
    

    def determineValue(
        self, 
        location: CartesianPoint, 
        objectSelecting: Optional[CityObject] = None,
        args: Optional[dict] = None) -> float:
        """Abstract Method: Determine the value of a location for the property.
        
        Arguments:
            location {CartesianPoint} -- The location at which to check
            objectSelecting {CityObject} -- The object determining the value at the location

        Returns:
            valueMultiplier {float} -- The effect of the result on the point's value, as a scalar
                valueMultiplier >= 0
        """

        raise NotImplementedError


class DistanceToRoadSelection(LocationSelectionProperty):
    """Class for determining the influence of the distance to the nearest road for a location.
    This should be weighted significantly more for certain buildings, such as residences.
    """

    def determineValue(
        self, 
        location: CartesianPoint,
        objectSelecting: CityObject,
        roadGraph: RoadGraph) -> float:
        """Determine the value.
        
        Arguments:
            location {CartesianPoint} -- The location at which to check, and from where to determine distance
            objectSelecting {CityObject} -- The object determining the value at the location
            roadGraph {RoadGraph} -- The RoadGraph to check for nearest road within
        
        Returns:
            valueMultiplier {float} -- The effect of the result on the point's value, as a scalar
                valueMultiplier >= 0
        """

        raise NotImplementedError("This is not an abstract class, it's just not written yet.")


