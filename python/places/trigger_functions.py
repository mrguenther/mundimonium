from typing import Tuple, List, Optional, Dict

class TriggerThreshold:
	"""
	Base class for functions that will be used to determine if a trigger has been called enough times
		TODO
	"""
	
	def testThreshold(
		self, 
		layer: 'LayerObject',
		args: dict) -> float:
		"""
		[summary]
		"""

		raise NotImplementedError


class TriggerThresholdManager:
	"""
	[summary]
	"""

	triggerThresholds = {
		}


'''class TriggerFunction:
	"""
	[summary]
	"""
	
	def testThreshold(
		self, 
		layer: 'LayerObject',
		args: dict) -> float:
		"""
		[summary]
		"""

		raise NotImplementedError


class TriggerFunctionManager:
	"""
	[summary]
	"""

	triggerFunctions = {
		}'''