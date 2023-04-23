from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union, List, TYPE_CHECKING

import numpy as np

from config import get_logger
import util


if TYPE_CHECKING:
    from master_state import Maps
    from unit_action_planner import UnitsToAct
    from collisions import UnitPaths
    from factory_manager import FactoryInfo
    from unit_manager import FriendlyUnitManager

logger = get_logger(__name__)


@dataclass
class SingleActionValid:
    valid: bool
    type: int
    reason: str


@dataclass
class FactoryResources:
    power: int
    ice: int
    water: int
    ore: int
    metal: int


def factory_num_to_id(num: int):
    return f"factory_{num}"


class ValidActionCalculator:
    """Calculate whether friendly units next action is valid or not
    Basics:
        - move blocked
        - not enough resource to pickup (have to check in order of units doing stuff)
            - Note: Unit_id is used by the game engine, but I'd like to avoid that having an impact
        - no unit or factory to transfer to
    """

    def __init__(
        self,
        factory_infos: Dict[str, FactoryInfo],
        maps: Maps,
        unit_paths: UnitPaths,
    ):
        self.factory_infos = factory_infos
        self.maps = maps
        self.unit_paths = unit_paths

        # self.allowed_travel_map = self._get_allowed_travel_map()
        self.factory_resources = {}
        self._init_factory_resources()

    # def _get_allowed_travel_map(self) -> np.ndarray:.factory
    #     map = np.ones_like(self.maps.rubble, dtype=bool)
    #     map[self.maps.factory_maps.enemy >= 0] = False
    #     return map

    def _init_factory_resources(self) -> Dict[str, FactoryResources]:
        """Get the amount of resources each factory will have based on starting amount and other units transfers"""
        # Get the initial values
        self.factory_resources: Dict[str, FactoryResources] = {}
        for factory_id, info in self.factory_infos.items():
            factory = info.factory.factory
            self.factory_resources[factory_id] = FactoryResources(
                power=factory.power,
                ice=factory.cargo.ice,
                water=factory.cargo.water,
                ore=factory.cargo.ore,
                metal=factory.cargo.metal,
            )

        # # Update based on first action of units (pickup or transfer)
        # all_infos = dict(**self.units_to_act.should_not_act, **self.units_to_act.has_updated_actions)
        # for unit_id, act_info in all_infos.items():
        #     self.add_next_action(act_info.unit)
        #
        return self.factory_resources

    def add_next_action(self, unit: FriendlyUnitManager):
        """Apply next action of unit to factories (i.e. update their resources if applicable)"""
        # Are there even actions?
        q = unit.action_queue
        if not len(q) > 0:
            return

        # Get info from first action
        action = q[0]
        act_type = action[util.ACT_TYPE]
        amount = action[util.ACT_AMOUNT]
        resource = action[util.ACT_RESOURCE]
        direction = action[util.ACT_DIRECTION]

        unit_power = unit.start_of_turn_power
        unit_pos = unit.start_of_turn_pos
        # Add or subtract from factory if unit is transferring to or picking up from factory
        if act_type == util.PICKUP:
            f_num = self.maps.factory_maps.friendly[unit_pos[0], unit_pos[1]]
            if not f_num >= 0:
                return
            factory_id = factory_num_to_id(f_num)
            factory_resources = self.factory_resources[factory_id]
            if resource == util.POWER:
                factory_resources.power -= amount
            elif resource == util.WATER:
                factory_resources.water -= amount
            elif resource == util.ICE:
                factory_resources.ice -= amount
            elif resource == util.METAL:
                factory_resources.metal -= amount
            elif resource == util.ORE:
                factory_resources.ore -= amount
            else:
                logger.error(f"{resource} not understood as a resource")
        elif act_type == util.TRANSFER:
            pos = util.add_direction_to_pos(unit_pos, direction)
            f_num = self.maps.factory_maps.friendly[pos[0], pos[1]]
            if not f_num >= 0:
                return
            factory_id = factory_num_to_id(f_num)
            factory_resources = self.factory_resources[factory_id]
            if resource == util.POWER:
                factory_resources.power += amount
            elif resource == util.WATER:
                factory_resources.water += amount
            elif resource == util.ICE:
                factory_resources.ice += amount
            elif resource == util.METAL:
                factory_resources.metal += amount
            elif resource == util.ORE:
                factory_resources.ore += amount
            else:
                logger.error(f"{resource} not understood as a resource")
        return

    def _get_available_factory_resource(self, factory_id: str, resource: int) -> int:
        """Get the amount that will be available after other units have acted
        Note: If total is exceeded for pickup, units with lower unit_id will have priority
        """
        # Get starting resource in factory
        resources = self.factory_resources[factory_id]
        if resource == util.POWER:
            return resources.power
        elif resource == util.WATER:
            return resources.water
        elif resource == util.METAL:
            return resources.metal
        elif resource == util.ICE:
            return resources.ice
        elif resource == util.ORE:
            return resources.ore
        else:
            raise NotImplementedError(f"resource={resource} not implemented")

    def next_action_valid(self, unit: FriendlyUnitManager, action: np.ndarray = None):
        logger.debug(f"Next action valid for {unit.unit_id}")
        if len(unit.action_queue) == 0:
            logger.error(f"No action, for {unit.unit_id}")
            return False

        action = action if action is not None else unit.action_queue[0]
        act_type = action[util.ACT_TYPE]
        amount = action[util.ACT_AMOUNT]
        resource = action[util.ACT_RESOURCE]
        direction = action[util.ACT_DIRECTION]

        unit_power = unit.start_of_turn_power
        unit_pos = unit.start_of_turn_pos

        # Map will allow collision with enemy but not friendly (friendly collision is NOT valid)
        allowed_move_map = self.unit_paths.to_costmap(
            pos=unit_pos,
            start_step=0,
            exclude_id_nums=[unit.id_num],
            friendly_heavy=True,
            friendly_light=True,
            enemy_heavy=False,
            enemy_light=False,
            true_intercept=True,
        )
        if act_type == util.MOVE:
            valid = unit.valid_moving_actions(allowed_move_map, max_len=1, ignore_repeat=True)
            if valid.was_valid is False:
                logger.warning(f"Move not valid because {valid.invalid_reasons[0]}")
                return False
            move_cost = util.power_cost_of_path(
                [unit_pos, util.add_direction_to_pos(unit_pos, direction)],
                self.maps.rubble,
                unit.unit_type,
            )
            if unit_power < move_cost:
                logger.info(f"Move not valid because not enough power {unit_power} < {move_cost}")
                return False
        elif act_type == util.PICKUP:
            f_num = self.maps.factory_maps.friendly[unit_pos[0], unit_pos[1]]
            if not f_num >= 0:
                logger.warning(f"Unit not on factory, cannot pickup")
                return False
            factory_id = factory_num_to_id(f_num)

            available = self._get_available_factory_resource(factory_id=factory_id, resource=resource)
            if amount > available:
                logger.debug(f"False -- amount > available: {amount} > {available}")
                return False
        elif act_type == util.TRANSFER:
            pos = util.add_direction_to_pos(unit_pos, direction)
            f_num = self.maps.factory_maps.friendly[pos[0], pos[1]]
            if not f_num >= 0:
                logger.warning(f"Transfer location {pos} not on factory, cannot transfer")
                return False
            if resource == util.ICE and unit.cargo.ice == 0:
                logger.warning(f"Attempting to transfer ice with 0 ice in cargo")
                return False
            elif resource == util.ORE and unit.cargo.ore == 0:
                logger.warning(f"Attempting to transfer ore with 0 ore in cargo")
                return False
            elif resource == util.METAL and unit.cargo.metal == 0:
                logger.warning(f"Attempting to transfer metal with 0 metal in cargo")
                return False
            elif resource == util.WATER and unit.cargo.water == 0:
                logger.warning(f"Attempting to transfer water with 0 water in cargo")
                return False

        elif act_type == util.DIG:
            if unit_power < unit.unit_config.DIG_COST:
                logger.debug(f"Not enough power to dig")
                return False
            ice = self.maps.ice[unit_pos[0], unit_pos[1]]
            ore = self.maps.ore[unit_pos[0], unit_pos[1]]
            rubble = self.maps.rubble[unit_pos[0], unit_pos[1]]
            lichen = self.maps.lichen[unit_pos[0], unit_pos[1]]
            if not any([ice, ore, rubble, lichen]):
                logger.warning(f"Nothing to dig under unit")
                return False
        elif act_type == util.DESTRUCT:
            if unit_power < unit.unit_config.SELF_DESTRUCT_COST:
                logger.info(
                    f"Self destruct not valid because not enough power {unit_power} < {unit.unit_config.SELF_DESTRUCT_COST}"
                )
                return False
        else:
            raise NotImplementedError(f"Not implemented checking act_type = {act_type}")
        return True


def valid_action_space(actions: Union[np.ndarray, List[np.ndarray]]) -> bool:
    """
    Check if an action or a list of actions is valid based on action space constraints

    Args:
        actions (Union[np.ndarray, List[np.ndarray]]): A 1D numpy array representing a single action or a list of actions.

    Returns:
        bool: True if ALL actions are valid, False otherwise
    """
    low = np.array([0, 0, 0, 0, 0, 1])
    high = np.array([5, 4, 4, 3000 + 1, 9999, 9999])

    if isinstance(actions, list) and len(actions) == 0:
        return True

    # Convert a list of actions to a 2D numpy array
    if isinstance(actions, list):
        actions = np.array(actions)

    # Ensure the input is a 2D numpy array
    if actions.ndim == 1:
        actions = actions[np.newaxis, :]

    # Check dtype
    if not np.issubdtype(actions.dtype, np.integer):
        logger.error(f"Actions had wrong dtype {actions.dtype}")
        return False

    # Check if the actions are valid
    valid = np.logical_and(low <= actions, actions <= high).all()

    return valid
