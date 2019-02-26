from typing import Tuple, List, Optional, Callable

class ObjectManager: pass

class Trigger:
    """[summary]

    Attributes:
        parentLayer {ExtensibleObject}
            -- The layer to which the Trigger belongs
                This is also the caller for Ordered Triggers
    """

    def __init__(
        self,
        parentLayer: 'LayerObject',
        triggerName: Optional[str] = None,
        triggerOrder: Optional[int] = None):
        """[summary]
        
        Arguments:
            parentLayer {[type]} -- [description]
        
        Keyword Arguments:
            triggerName {Optional[str]} -- [description] (default: {None})
            triggerOrder {Optional[int]} -- [description] (default: {None})
        """

        if triggerName is None and triggerOrder is None:
            raise TypeError("Trigger() requires triggerName or triggerOrder")



class TriggerManager:
    """
    Manages triggers from loaded ExtensibleObjects
    """

    def __init__(self):
        """[summary]
        """

        self.triggers = {}

    def addTrigger(
        self,
        trigger: Trigger,
        triggeredFunction: Callable[['ExtensibleObject'],...,any]):
        """
        Add a trigger and the function it should call
        
        Arguments:
            trigger {Trigger} 
                -- The trigger
            triggeredFunction {Callable[[],...,any]} 
                -- The function to call. Note that this should take only an ExtensibleObject as its arguments
        """


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
            caller {ExtensibleObject}
                -- The object that called this trigger to be resolved
                    If the trigger is Ordered, this should be None
                    In which case it will be called with the trigger's parentLayer instead
        """

        if caller is None:
            caller = trigger.parentLayer

        for triggeredFunction in self.triggers[trigger]:
            triggeredFunction(caller)
