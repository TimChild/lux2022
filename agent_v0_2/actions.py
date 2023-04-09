from __future__ import annotations

import abc

import numpy as np

from .agent_v3 import GeneralObs, UnitObs, FactoryObs
from .factory_manager import FactoryManager
from .master_state import MasterState
from .unit_manager import UnitManager


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
    @abc.abstractmethod
    def to_action_queue(self, plan: MasterState) -> list[np.ndarray]:
        """To action queue recognized by lux engine i.e. list[np.array(6).astype(int)]"""  # TODO: Is that right for action
        pass


def calculate_high_level_unit_action(
        general_obs: GeneralObs, unit_obs: UnitObs
) -> HighLevelAction:
    """Take the processed obs, and return high level actions per unit/factory

    Examples:
        - Mine X Ore for factory X
        - Mine X Ice for factory X
        - Attack area X
        - Defend area X
        - Solar Farm at X
    """
    return unit_obs.recommendations[0]


def calculate_high_level_factory_actions(
        general_obs: GeneralObs, factory_obs: FactoryObs
) -> HighLevelAction:
    """Take the processed obs, and return high level actions per factory

    Examples:
        - Make Light Unit
        - Make Heavy Unit
    """
    return factory_obs.recommendations[0]


def unit_should_consider_acting(unit: UnitManager, plan: MasterState) -> bool:
    """Whether unit should consider acting this turn
    If no, can save the cost of calculating obs/options for that unit
    """
    power = unit.unit.power
    action_queue_cost = unit.unit_config.ACTION_QUEUE_POWER_COST
    if power < action_queue_cost:
        return False

    nearest_enemy_distance = plan.maps.nearest_enemy(unit.unit.pos)
    if nearest_enemy_distance <= 2:
        return True

    current_role = unit.status.role
    if current_role in []:
        return False

    current_actions = unit.unit.action_queue
    if len(current_actions) == 0:
        return True
    return True


def factory_should_consider_acting(factory: FactoryManager, plan: MasterState) -> bool:
    """Whether factory should consider acting this turn
    If no, can save the cost of calculating obs/options for that factory
    """
    center_tile_occupied = True if plan.maps.unit_at_tile(factory.factory.pos) is not None else False
    if center_tile_occupied:
        return False
    return True




