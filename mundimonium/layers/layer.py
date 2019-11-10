from typing import Optional
from mundimonium.controllers import temporal

class Layer:
	"""
	Base class for layers

	Attributes:
		parentLayer (Optional[Layer]): The parent of the current layer
		timeController (temporal.TimeController): The TimeController that should manage this layer
			Default: parentLayer's timeController, if parentLayer is extant. Otherwise, a new instance.
		children (List[Layer]): Child layers of this layer
		terrainObject (any): An object that manages the layer's terrain
	"""

	def __init__(
			self,
			parentLayer: Optional['Layer'] = None,
			timeController: Optional[temporal.TimeController] = None):

		self.parentLayer = parentLayer
		self.timeController = timeController
		self.children = []
		self.terrainObject = None

		if timeController is None:
			if parentLayer is None:
				self.timeController = temporal.TimeController()
			else:
				self.timeController = parentLayer.timeController

	def getTerrainAttributes(
			self,
			childLayer: Optional['Layer'] = None,
			area: Optional[tuple] = None) -> dict:
		"""
		Abstract method: Provides a function for child layers to get info from their parents about terrain attributes and constraints
			E.G. major terrain features that have been generated on prior layers, climate info, what have you

		Raises:
			NotImplementedError

		Keyword Arguments:
			childLayer {Optional[Layer]} -- [description] (default: {None})
			area {Optional[tuple]} -- [description] (default: {None})

		Returns:
			A dict of attributes
			TODO: Consider changing the return type to a TerrainAttributes object or something
		"""

		raise NotImplementedError

