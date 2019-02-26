from typing import Tuple, List, Optional

from utils.coordinate_grid import CartesianPoint

class PropertyFunction:
    """
    Base class for functions that ExtensibleObjects will call on load to determine object property values.
        These are properties that vary by instance of any object, but may be checked by other functions,
        for example while determining placement
    
    Children of this class should implement determineValue (and, optionally, additional helper functions)
        This is the function that's called by ExtensibleObjects for each dynamic property in the object's property list

    Strictly speaking, SelectionProperties could be implemented as PropertyFunctions
        But for now, they're their own, very-similar thing, as they differ just enough to be worth having their own classes

    TODO: Decide if these should just be functions instead
        As classes, they support inheritance, which might be nice
    """
    
    def determineValue(
        self, 
        caller: Optional['ExtensibleObject'] = None) -> any:
        """
        Abstract Method: Evaluate a property for an object.
        
        Arguments:
            caller {ExtensibleObject}
                -- The object you wish to evaluate the property for

        Returns:
            value {any}
                -- The value of the property. This might be anything.
        """

        raise NotImplementedError

class DetermineRoadNetwork(PropertyFunction):
    """
    Class for determining the associatedNetwork property of a building
        Generally speaking, this will just be the first instantiated instance of a RoadNetwork NetworkObject
    """

    def determineValue(self) -> 'NetworkObject':
        """
        Find a RoadNetwork object. The expected usage of the object is to populate the associatedNetwork field.

        Returns:
            roadNetwork {NetworkObject} 
                -- The RoadNetwork object
        """

        raise NotImplementedError("This is not an abstract class, it's just not written yet.")


class DetermineTerrain(PropertyFunction):
    """
    Class for determining the associatedTerrain property of a building
        Generally speaking, this will just be the first instantiated instance of a CityTerrain TerrainObject
    """

    def determineValue(self) -> 'TerrainObject':
        """
        Find a CityTerrain TerrainObject. The expected usage of the object is to populate the associatedTerrain field.

        Returns:
            cityTerrain {TerrainObject} 
                -- The CityTerrain object
        """

        raise NotImplementedError("This is not an abstract class, it's just not written yet.")

class DetermineCreator(PropertyFunction):
    """
    Class for determining what object created the calling object (by calling its CreationTrigger)
        If the object was created by an Ordered trigger, the creator will be its parent layer
        Otherwise, this returns the object that called the OnCalled trigger
    """

    def determineValue(
        self,
        caller: 'ExtensibleObject') -> 'ExtensibleObject':
        """
        Find the object that created caller

        Returns:
            creator {ExtensibleObject}
                -- The object that created caller 
        """

        raise NotImplementedError("This is not an abstract class, it's just not written yet.")


class PropertyFunctionManager:
    """
    Class for associating strings (loaded by ExtensibleObjects from extension files)
        with their respective PropertyFunction classes, allowing them to be
        arbitrarily called without full arbitrary function execution

    Attributes:
        propertyFunctions {dict[str: function]}
            -- The dictionary of PropertyFunction classes, from which determineValue is called
    """

    propertyFunctions = {
        "DetermineRoadNetwork": DetermineRoadNetwork, 
        "DetermineTerrain": DetermineTerrain,
        "DetermineCreator": DetermineCreator}