class ComponentError(Exception):
    """Base class for exceptions in this module."""
    pass

class MapNetworkError(ComponentError):
	"""Base class for exceptions related to MapNetwork objects."""
	pass

class GraphAlreadyConnectedError(MapNetworkError):
	"""Raised when a component function that requires a disconnected graph is called on a connected one."""
	pass



