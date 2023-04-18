from __future__ import annotations
from typing import TYPE_CHECKING
from enum import Enum
import abc

import numpy as np

if TYPE_CHECKING:
    from agent import GeneralObs, UnitObs, FactoryObs
    from master_state import MasterState
    from unit_manager import FriendlyUnitManager
    from factory_manager import FactoryManager


class Actions(Enum):
    MINE_ICE = "mine_ice"
    MINE_ORE = "mine_ore"
    CLEAR_RUBBLE = "clear_rubble"
    ATTACK = "attack"
    RUN_AWAY = "run_away"
    NOTHING = "do_nothing"
    CONTINUE_NO_CHANGE = "continue previous action no change"
    CONTINUE_UPDATE = "continue previous action but update"


######################


class HighLevelAction(abc.ABC):
    """
    The commands that will be converted to action queues before returning to the lux engine
    Examples:
        # Simple ones that can basically be passed straight through
        - Build Heavy Unit (maybe just need to wait a turn or two)

        # More complex actions
        - Mine at X,Y for Factory Z
        - Dig Lichen in area near X, Y
    """

    pass


class Recommendation(HighLevelAction):
    role: str = None
