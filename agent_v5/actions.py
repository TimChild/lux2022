from __future__ import annotations
from typing import TYPE_CHECKING
import abc

import numpy as np

if TYPE_CHECKING:
    from agent import GeneralObs, UnitObs, FactoryObs
    from master_state import MasterState
    from unit_manager import FriendlyUnitManger
    from factory_manager import FactoryManager


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


class Recommendation(HighLevelAction):
    role: str = 'not set'
    value: float = 0

    # @abc.abstractmethod
    # def to_array(self):
    #     """Turn recommendation into an array of values with standard size"""
    #     pass


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
    # return factory_obs.recommendations[0]
    return None


def unit_should_consider_acting(unit: FriendlyUnitManger, plan: MasterState) -> bool:
    """Whether unit should consider acting this turn
    If no, can save the cost of calculating obs/options for that unit
    """
    power = unit.unit.power
    action_queue_cost = unit.unit_config.ACTION_QUEUE_POWER_COST
    if power < action_queue_cost:
        # Not enough energy to update action queue, so nothing this unit can do
        return False

    nearest_unit_id, nearest_enemy_distance = plan.units.nearest_unit(
        unit.pos, friendly=False, enemy=True, light=True, heavy=True
    )
    if nearest_enemy_distance <= 2:
        # Nearby enemy, unit might need to consider running away or attacking
        return True

    if unit.factory_id is None:
        # No factory assigned to unit (factory died?), unit should do something else
        return True

    current_role = unit.status.role
    if current_role in []:
        # Certain roles may no require actions
        return False

    current_actions = unit.unit.action_queue
    if len(current_actions) == 0:
        # No current actions, might want to consider doing something
        return True
    return True


def factory_should_consider_acting(factory: FriendlyUnitManger, plan: MasterState) -> bool:
    """Whether factory should consider acting this turn
    If no, can save the cost of calculating obs/options for that factory
    """
    return True
