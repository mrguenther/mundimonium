from typing import Tuple, List, Optional, Callable
import json
from .component_functions import ComponentFunctionManager
import warnings

class LayerManager:
	"""
	Class for managing a layer's components, of course
		Provides utility for loading and keeping track of objects and triggers
		Such as functions to process layer files for CreationTriggers
		Each MapLayer has its own LayerManager, which in turn has a TriggerManager

	Attributes:
		triggerManager (TriggerManager): The TriggerManager attached to the LayerManager, and therefore the layer
		layer (MapLayer): The MapLayer for which objects are being managed
	"""

	def __init__(self, layer: 'MapLayer', layerFilePath: str):

		self.triggerManager = TriggerManager()
		self.layer = layer
		self.layerFile = None

		self.objects = []
		self.objectsByName = {}

		self.loadLayerFile(layerFilePath)
		self.loadTriggers(self.layerFile)

	def addObject(self, mapObject: 'MapComponent'):
		"""
		Add a MapComponent to the object lists
		
		Arguments:
			objectName {[type]} -- [description]
		"""

		self.objects.append(mapObject)
		if mapObject.objectName not in self.objectsByName:
			self.objectsByName[mapObject.objectName] = []
		self.objectsByName[mapObject.objectName].append(mapObject)

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
			layerFile (dict): The layer file from which to load triggers
		"""

		for objectName, objectToLoad in layerFile['MapComponents'].items():
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

				self.triggerManager.addTrigger(trigger, self.layer.loadOtherObject, (objectName, objectToLoad))
		
		for triggerName, triggerToLoad in layerFile['FunctionTriggers'].items():
			triggerOrder = None
			if 'TriggerOrder' in triggerToLoad: triggerOrder = triggerToLoad['TriggerOrder']
			triggeredFunction = ComponentFunctionManager.componentFunctions[triggerToLoad['TriggeredFunction']]().execute
			triggeredFunctionArgs = triggerToLoad['TriggeredFunctionArgs']

			trigger = Trigger(self.layer, triggerName=triggerName, triggerOrder=triggerOrder)

			self.triggerManager.addTrigger(trigger, triggeredFunction, triggeredFunctionArgs)		

class Trigger:
	"""
	Triggers to be used by MapComponents for controlling object creation etc.
		Managed by TriggerManagers

	TODO: Set this up to deal with expanded trigger types

	Attributes:
		parentLayer (MapLayer): The layer to which the Trigger belongs
					This is also the caller for Ordered Triggers
		repeatCount (Optional[int]): How many times the trigger should be called
		stopThreshold {Optional[float]}
		continueFunction (Optional[Callable[['MapLayer', dict], float]]): Function whose value is to be compared against the stopThreshold
				Trigger will be called until the return of this function exceeds stopThreshold
		continueArgs (Optional[dict]): Dict of args to be sent to continueFunction
		maxCalls {Optional[int]} (default: 10000)
			-- Maximum number of times the trigger can be called
				This should be set to a suitably high number, or left as the default 10k
				It primarily serves as an escape route for continueFunctions that never meet the stopThreshold
		triggerName (Optional[str]): The name by which the trigger is called. This xor triggerOrder MUST be defined.
		triggerOrder (Optional[int]): The order in which the trigger should be called.
	"""

	def __init__(
			self,
			parentLayer: 'MapLayer',
			repeatCount: Optional[int] = 0,
			stopThreshold: Optional[float] = None,
			continueFunction: Optional[Callable[['MapLayer', dict], float]] = None,
			continueArgs: Optional[dict] = None,
			maxCalls: Optional[int] = 10000,
			triggerName: Optional[str] = None,
			triggerOrder: Optional[int] = None):

		if (triggerName is None) == (triggerOrder is None):
			raise TypeError("Trigger() requires triggerName xor triggerOrder")

		self.triggerOrder = triggerOrder
		self.triggerName = triggerName
		self.parentLayer = parentLayer
		self.repeatCount = repeatCount
		self.continueFunction = continueFunction
		self.stopThreshold = stopThreshold
		self.continueArgs = continueArgs
		self.calls = 0
		self.maxCalls = maxCalls

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

	def __str__(self):
		"""
		Pretty-printable(-ish) representation of the trigger.
		"""

		outStr = ('Trigger(triggerName=%(triggerName)s, triggerOrder=%(triggerOrder)s, parentLayer=%(parentLayer)s)' %
			{'triggerName': self.triggerName,
			'triggerOrder': self.triggerOrder,
			'parentLayer': self.parentLayer})
		return(outStr)


class TriggerManager:
	"""
	Manages triggers from loaded layers

	Attributes:
		triggers (dict): The triggers being managed
				Format: {Trigger: [(triggeredFunction, arg), (triggeredFunction, arg)]}
		orderedMax (dict): The greatest ordered trigger for each layer
				Format: {MapLayer: int}
	"""

	def __init__(self):
		
		self.triggers = {}
		self.orderedMax = {}

	def addTrigger(
			self,
			trigger: Trigger,
			triggeredFunction: Callable[['MapComponent', any],any],
			arg: any):
		"""
		Add a trigger and the function it should call
		
		Arguments:
			trigger (Trigger): The trigger
			triggeredFunction (Callable[[MapComponent, any],any]): The function to call. Note that this should take an MapComponent and one other argument
					Most often, this will be to create an MapComponent of some variety
			arg (any): Arg to pass along to the function
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

	def resolveOrdered(self, layer: 'MapLayer'):
		"""
		Resolve the ordered triggers for a layer

		Arguments:
			layer (MapLayer): The layer for which to resolve the ordered triggers
		"""

		for order in range(self.orderedMax[layer]):
			trigger = Trigger(layer, triggerOrder = order)
			print('Trigger called: ' + str(trigger))
			if trigger in self.triggers:
				self.resolveTrigger(trigger, caller = layer)


	def resolveTrigger(
			self, 
			trigger: Trigger, 
			caller: Optional['MapComponent'] = None):
		"""
		Resolve a given Trigger for objects in layerManager
		
		Arguments:
			layerManager (LayerManager): The LayerManager who's MapComponents we will be resolving the trigger for
			trigger (Trigger): The trigger to resolve
			caller (Optional[MapComponent]): The object that called this trigger to be resolved
					If the trigger is Ordered, this should be None
					In which case it will be called with the trigger's parentLayer instead
		"""

		#print('Trigger resolved: ' + str(trigger))

		if caller is None:
			caller = trigger.parentLayer

		try:
			# Call the trigger's triggered function as many times as repeatCount
			for _ in range(trigger.repeatCount):
				self.executeTrigger(trigger, caller)
			# If the trigger has a function threshold, repeat the trigger until it's met
			if trigger.continueFunction is not None and trigger.stopThreshold is not None:
				while trigger.continueFunction(caller, trigger.continueArgs) < trigger.stopThreshold:
					self.executeTrigger(trigger, caller)
			# If there is no repeatCount or function threshold, call the trigger once
			if trigger.continueFunction is None and trigger.repeatCount == 0:
				self.executeTrigger(trigger, caller)
		except (Exception) as e:
			warnings.warn('''%(eType)s raised while executing trigger %(trigger)s from caller %(caller)s: 
			%(eStr)s
			Aborting resolution of this trigger.''' %
				{'eType': type(e),
				'trigger': str(trigger),
				'eStr': str(e),
				'caller': str(caller)})

	def executeTrigger(
			self, 
			trigger: Trigger, 
			caller: 'MapComponent'):
		"""
		Execute a given Trigger for objects in layerManager
		
		Arguments:
			layerManager (LayerManager): The LayerManager who's MapComponents we will be resolving the trigger for
			trigger (Trigger): The trigger to resolve
			caller (MapComponent): The object that called this trigger to be resolved
		"""
		
		if trigger.calls < trigger.maxCalls:
			print('Trigger executed: ' + str(trigger) + ', caller: ' + str(caller))
			for (triggeredFunction, arg) in self.triggers[trigger]:
				triggeredFunction(caller, arg)
			trigger.calls += 1

