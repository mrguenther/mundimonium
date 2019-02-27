from typing import Tuple, List, Optional, Callable
import json

class ObjectManager:
	"""
	Class for managing objects, of course
		Provides utility for loading and keeping track of objects and triggers
		Such as functions to process layer files for CreationTriggers
		Each LayerObject has its own ObjectManager, which in turn has a TriggerManager

	Attributes:
		triggerManager {TriggerManager}
			-- The TriggerManager attached to the ObjectManager, and therefore the layer
		layer {LayerObject}
			-- The LayerObject for which objects are being managed
	"""

	def __init__(self, layer: 'LayerObject', layerFilePath: str):
		"""[summary]
		"""

		self.triggerManager = TriggerManager()
		self.layer = layer
		self.layerFile = None

		self.objects = []

		self.loadLayerFile(layerFilePath)
		self.loadTriggers(self.layerFile)

	def loadLayerFile(self, path: str):
		"""[summary]
		
		Arguments:
			path {str} -- [description]
		"""

		with open(path, 'r') as f:
			contents = json.load(f)
		self.layerFile = contents

	def loadTriggers(self, layerFile: dict):
		"""
		Load callable triggers from a layer file
			For now, this is limited to only Creation Triggers

		Arguments:
			layerFile {dict}
				-- The layer file from which to load triggers
		"""

		for objectName, objectToLoad in layerFile.items():
			if 'CreationTrigger' in objectToLoad:
				creationTrigger = objectToLoad['CreationTrigger']
				if creationTrigger['TriggerType'] == 'OnCalled':
					trigger = Trigger(self.layer, triggerName = creationTrigger['TriggerName'])
				elif creationTrigger['TriggerType'] == 'Ordered':
					trigger = Trigger(self.layer, triggerOrder = creationTrigger['TriggerOrder'])
				else:
					raise KeyError('Invalid TriggerType %(triggerType)s in %(objectName)s' % 
						{'triggerType': str(creationTrigger['TriggerType']),
						'objectName': str(objectName)})


				self.triggerManager.addTrigger(trigger, self.layer.loadObject, (objectName, objectToLoad))
		

class Trigger:
	"""
	Triggers to be used by ExtensibleObjects for controlling object creation etc.
		Managed by TriggerManagers
	
	Attributes:
		parentLayer {ExtensibleObject}
			-- The layer to which the Trigger belongs
				This is also the caller for Ordered Triggers
		triggerName {Optional[str]}
			-- The name of the trigger
				In the case of Ordered Triggers, this will be None
		triggerOrder {Optional[int]}
			-- The order of the trigger
				In the case of Named Triggers, this will be None
	"""

	def __init__(
		self,
		parentLayer: 'LayerObject',
		triggerName: Optional[str] = None,
		triggerOrder: Optional[int] = None):
		"""		
		Arguments:
			parentLayer {LayerObject} 
				-- The layer to which the Trigger belongs
		
		Keyword Arguments:
			triggerName {Optional[str]} 
				-- The name by which the trigger is called. This xor triggerOrder MUST be defined.
			triggerOrder {Optional[int]} 
				-- The order in which the trigger should be called.
		"""

		if (triggerName is None) != (triggerOrder is None):
			raise TypeError("Trigger() requires triggerName xor triggerOrder")

		self.triggerOrder = triggerOrder
		self.triggerName = triggerName
		self.parentLayer = parentLayer

	def __hash__(self):
		"""
		Hash function - allows Trigger objects to be used as valid keys in dicts
		"""

		return(hash((self.triggerName,self.triggerOrder,self.parentLayer)))

	def __eq__(self, other):
		"""
		Equivalence function - allows comparison and usage of Trigger objects as hashables
		"""

		if type(self) == type(other): return self.__hash__() == other.__hash__()
		return False



class TriggerManager:
	"""
	Manages triggers from loaded layers

	Attributes:
		triggers {dict}
			-- The triggers being managed
				Format: {Trigger: [(triggeredFunction, arg), (triggeredFunction, arg)]}
		orderedMax {dict}
			-- The greatest ordered trigger for each layer
				Format: {LayerObject: int}
	"""

	def __init__(self):
		"""[summary]
		"""

		self.triggers = {}
		self.orderedMax = {}

	def addTrigger(
		self,
		trigger: Trigger,
		triggeredFunction: Callable[['ExtensibleObject', any],any],
		arg: any):
		"""
		Add a trigger and the function it should call
		
		Arguments:
			trigger {Trigger} 
				-- The trigger
			triggeredFunction {Callable[[ExtensibleObject, any],any]} 
				-- The function to call. Note that this should take an ExtensibleObject and one other argument
					Most often, this will be to create an ExtensibleObject of some variety
			arg {any}
				-- Arg to pass along to the function
					In the case of creation triggers, this will be a dict of the object to load
					Otherwise it will still probably be a dict, but not necesarrily
		"""

		if trigger.triggerOrder is not None:
			if trigger.parentLayer not in self.orderedMax:
				self.orderedMax[trigger.parentLayer] = 0
			self.orderedMax[trigger.parentLayer] = max(self.orderedMax[trigger.parentLayer], trigger.triggerOrder + 1)
		if trigger not in self.triggers:
			self.triggers[trigger] = []
		self.triggers[trigger].append((triggeredFunction, arg))

	def resolveOrdered(self, layer: 'LayerObject'):
		"""
		Resolve the ordered triggers for a layer

		Arguments:
			layer {LayerObject}
				-- The layer for which to resolve the ordered triggers
		"""

		for order in range(self.orderedMax[layer]):
			trigger = Trigger(layer, triggerOrder = order)
			if trigger in self.triggers:
				self.resolveTrigger(trigger, caller = layer)


	def resolveTrigger(
		self, 
		trigger: Trigger, 
		caller: Optional['ExtensibleObject'] = None):
		"""
		Resolve a given Trigger for objects in objectManager
		
		Arguments:
			objectManager {ObjectManager} 
				-- The ObjectManager who's ExtensibleObjects we will be resolving the trigger for
			trigger {Trigger} 
				-- The trigger to resolve
			caller {Optional[ExtensibleObject]}
				-- The object that called this trigger to be resolved
					If the trigger is Ordered, this should be None
					In which case it will be called with the trigger's parentLayer instead
		"""

		if caller is None:
			caller = trigger.parentLayer

		for (triggeredFunction, arg) in self.triggers[trigger]:
			triggeredFunction(caller, arg)
