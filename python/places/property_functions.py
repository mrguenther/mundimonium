from typing import Tuple, List, Optional, Dict

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

	TODO: Decide if these should be read-only
		For now they're not - Notably, FindAndAssociateNetworkByName is the current method of adding to NetworkObjects
		But these types of functions could be moved into universal triggered functions instead
	"""
	
	def determineValue(
		self, 
		caller: 'ExtensibleObject',
		args: dict) -> any:
		"""
		Abstract Method: Evaluate a property for an object.
		
		Arguments:
			caller {ExtensibleObject}
				-- The object you wish to evaluate the property for
			args {dict}
				-- Dict of args passed along from the JSON object

		Returns:
			value {any}
				-- The value of the property. This might be anything.
		"""

		raise NotImplementedError

class FindObjectByName(PropertyFunction):
	"""
	Class for locating an object by name and instance number
		Note that the object to be found must have been created before the object calling this
	"""

	def determineValue(
		self, 
		caller: 'ExtensibleObject',
		args: Dict[str, any]) -> 'ExtensibleObject':
		"""
		Find an object. The expected usage of the object is to populate the associatedNetwork field.

		Arguments:
			caller {ExtensibleObject}
				-- The object you wish to evaluate the property for
			args {dict}
				-- Dict of args passed along from the JSON object
				-- Required Values:
					'ObjectName': str
					'ObjectInstance': int

		Returns:
			namedObject {ExtensibleObject} 
				-- The object named
		"""

		name = args['ObjectName']
		instance = args['ObjectInstance']
		return(caller.parentLayer.objectManager.objectsByName[name][instance])

class FindAndAssociateNetworkByName(FindObjectByName):
	"""
	Class for finding a NetworkObject by name and instance number, 
		then adding the calling PointObject as a node
	"""

	def determineValue(
		self, 
		caller: 'PointObject',
		args: Dict[str, any]) -> 'NetworkObject':
		"""
		Find a NetworkObject and add caller as a node, then return the NetworkObject

		Arguments:
			caller {PointObject}
				-- The object you wish to add as a node
			args {dict}
				-- Dict of args passed along from the JSON object
				-- Required Values:
					'ObjectName': str
					'ObjectInstance': int

		Returns:
			networkToAssociate {NetworkObject} 
				-- The NetworkObject found and added to
		"""

		networkToAssociate = super().determineValue(caller, args)
		networkToAssociate.addPoint(caller)
		return(networkToAssociate)

class DetermineCreator(PropertyFunction):
	"""
	Class for determining what object created the calling object (by calling its CreationTrigger)
		If the object was created by an Ordered trigger, the creator will be its parent layer
		Otherwise, this returns the object that called the OnCalled trigger
	"""

	def determineValue(
		self,
		caller: 'ExtensibleObject',
		args: dict) -> 'ExtensibleObject':
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
		'FindObjectByName': FindObjectByName,
		'DetermineCreator': DetermineCreator,
		'FindAndAssociateNetworkByName': FindAndAssociateNetworkByName}