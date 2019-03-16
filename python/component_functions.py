from .utils.helper_funcs import descendants

class ComponentFunction:
	"""
	Base class for functions executable by MapComponents.
	
	Children of this class should implement execute (and, optionally, additional helper functions).
	"""
	
	memoizable = False


	def execute(
			self, 
			caller: 'MapComponent',
			args: dict):
		"""
		Abstract Method: Execute a function for an object.
		
		Arguments:
			caller (MapComponent): The object executing the function
			args (dict): Arguments for the function
		"""

		raise NotImplementedError

from . import extensible_python
# Import all the descendants now, so ComponentFunctionManager can retrieve them
# This import has to come after the ComponentFunction base class is defined

class ComponentFunctionManager:
	"""
	Class for associating strings (loaded by MapComponents from extension files)
		with their respective ComponentFunction classes, allowing them to be
		arbitrarily called without full arbitrary function execution

	Attributes:
		componentFunctions (dict[str: function]): The dictionary of component functions that can be called by SelectionProperty objects
	"""

	componentFunctions = {descendant.__name__:descendant for descendant in descendants(ComponentFunction)}