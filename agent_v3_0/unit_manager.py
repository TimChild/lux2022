from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np
import re
import abc
import copy

from luxai_s2.unit import UnitCargo

from unit_status import Status
from lux.unit import Unit
from lux.config import UnitConfig

import util
from actions_util import Actions

from config import get_logger
from master_state import MasterState

if TYPE_CHECKING:
    from factory_manager import FriendlyFactoryManager

logger = get_logger(__name__)


def get_index(lst, index, default=None):
    """Get the element at the specified index in the list, or return the default value if the index is out of range."""
    return lst[index] if 0 <= index < len(lst) else default


class UnitManager(abc.ABC):
    def __init__(self, unit: Unit):
        self.unit_id = unit.unit_id
        self.unit = unit
        self.unit_config: UnitConfig = unit.unit_cfg
        self.id_num = int(re.search(r"\d+", unit.unit_id).group())

        self.dig = unit.dig
        self.transfer = unit.transfer
        self.pickup = unit.pickup

        # Keep track of pos a start of turn because pos will be updated while planning what to do next
        self.start_of_turn_pos = tuple(unit.pos)
        self.start_of_turn_power = unit.power

    def power_cost_of_actions(self, rubble: np.ndarray, max_actions=None):
        actions = self.action_queue[:max_actions]
        return util.power_cost_of_actions(
            start_pos=self.start_of_turn_pos,
            rubble=rubble,
            unit=self,
            actions=actions,
        )

    def pickup(self, pickup_resource, pickup_amount, repeat=0, n=1):
        return self.unit.pickup(pickup_resource, pickup_amount, repeat, n)

    def _valid_moving_actions(
        self,
        costmap: np.ndarray,
        start_of_turn_pos,
        action_queue,
        max_len=20,
        ignore_repeat=False,
    ) -> util.ValidActionsMoving:
        return util.calculate_valid_move_actions(
            start_of_turn_pos,
            action_queue,
            valid_move_map=costmap,
            max_len=max_len,
            ignore_repeat=ignore_repeat,
        )

    def valid_moving_actions(self, costmap, max_len=20, ignore_repeat=False) -> util.ValidActionsMoving:
        return self._valid_moving_actions(costmap, self.start_of_turn_pos, self.action_queue, max_len, ignore_repeat)

    @property
    def log_prefix(self) -> str:
        return f"{self.unit_type} {self.unit_id}(spos{self.start_of_turn_pos})(pos{self.pos}):"

    def update(self, unit: Unit):
        """Beginning of turn update"""
        self.unit = unit
        # Avoid changing the actual pos of unit.pos (which the env also uses)
        unit.pos = tuple(unit.pos)
        self.start_of_turn_pos = unit.pos
        self.start_of_turn_power = unit.power

    def current_path(self, max_len: int = 10, actions=None) -> np.ndarray:
        """Return current path from start of turn based on current action queue
        Adds a single no-move if not enough power to do the next move (hopefully avoids collisions better?)
        """
        if actions is None:
            actions = self.action_queue
        path = util.actions_to_path(self.start_of_turn_pos, actions, max_len=max_len)
        return path

    @property
    def action_queue(self) -> List[np.ndarray]:
        return self.unit.action_queue

    @action_queue.setter
    def action_queue(self, value):
        self.unit.action_queue = value

    @property
    def pos(self) -> util.POS_TYPE:
        return self.unit.pos

    @pos.setter
    def pos(self, value):
        if value is None or len(value) != 2:
            raise ValueError(f"got {value} with type {type(value)} for pos")
        self.unit.pos = value

    @property
    def pos_slice(self):
        """Can be used for indexing arrays directly
        Examples:
            r = rubble[unit.pos_slice]
        """
        return np.s_[self.pos[0], self.pos[1]]

    @property
    def cargo(self) -> UnitCargo:
        return self.unit.cargo

    @property
    def unit_type(self) -> str:
        return self.unit.unit_type

    @property
    def power(self) -> int:
        return self.unit.power

    @power.setter
    def power(self, value):
        self.unit.power = value

    def actions_to_path(self, actions: [None, List[np.ndarray]] = None, max_len=20) -> np.ndarray:
        """
        Return a list of coordinates of the path the actions represent starting from unit.pos
        (which may have been updated since beginning of turn)
        """
        if actions is None:
            actions = self.unit.action_queue
        return util.actions_to_path(self.unit.pos, actions, max_len=max_len)

    @abc.abstractmethod
    def dead(self):
        """Called when unit dies, should tidy up anything related to unit"""
        pass


class EnemyUnitManager(UnitManager):
    def dead(self):
        logger.info(f"Enemy unit {self.unit_id} dead, nothing more to do")


def step_actions(actions: List[np.ndarray]) -> List[np.ndarray]:
    """Change actions as if an env step has occurred"""


class FriendlyUnitManager(UnitManager):
    def __init__(self, unit: Unit, master_state: MasterState, factory_id: str):
        super().__init__(unit)
        self.factory_id = factory_id
        self.master: MasterState = master_state
        self.status: Status = Status(
            master=self.master,
            current_action=Actions.NOTHING,
            previous_action=Actions.NOTHING,
            last_action_update_step=0,
            last_action_success=True,
            action_queue_valid_after_step=True,
        )
        self.start_of_turn_actions = []

    @property
    def log_prefix(self) -> str:
        log_prefix = super().log_prefix
        log_prefix += f"({self.status.current_action}):\n\t\t\t"
        return log_prefix

    def update(self, unit: Unit):
        """Beginning of turn update"""
        super().update(unit)
        self.start_of_turn_actions = copy.copy(unit.action_queue)
        self.status.step_update_planned_actions(self)

    def current_path(self, max_len: int = 10, actions=None, planned_actions=True) -> np.ndarray:
        """Return current path from start of turn based on current action queue"""
        if actions is None:
            actions = self.status.planned_actions if planned_actions else self.action_queue
        return super().current_path(max_len=max_len, actions=actions)

    def update_status(self, new_action, success: bool):
        """Update unit status with new action"""
        self.status.update_status(new_action, success)

    def update_planned_actions_with_queue(self):
        """Update the planned actions with the current action queue (i.e. after building new action queue)"""
        self.status.update_planned_actions(self.action_queue)

    def valid_moving_actions(
        self, costmap, max_len=20, ignore_repeat=False, planned_actions=True
    ) -> util.ValidActionsMoving:
        """Calculate the moving actions based on the planned actions queue"""
        if planned_actions:
            actions = self.status.planned_actions
        else:
            actions = self.action_queue
        return self._valid_moving_actions(costmap, self.start_of_turn_pos, actions, max_len, ignore_repeat)

    def next_action_is_move(self) -> bool:
        """Bool if next action is a move action"""
        if len(self.start_of_turn_actions) == 0:
            return False
        next_action = self.start_of_turn_actions[0]
        if next_action[util.ACT_TYPE] == util.MOVE and next_action[util.ACT_DIRECTION] != util.CENTER:
            return True
        return False

    def on_own_factory(self) -> bool:
        """Is this unit on its own factory"""
        factory_loc = self.factory_loc
        if factory_loc is not None:
            return self.factory_loc[self.pos[0], self.pos[1]] == 1
        return False
        # return (
        #     self.factory_loc[self.start_of_turn_pos[0], self.start_of_turn_pos[1]] == 1
        # )

    def pickup(self, pickup_resource, pickup_amount, repeat=0, n=1):
        """Keep track of a couple of turns of power pickups"""
        turns_of_actions = util.num_turns_of_actions(self.action_queue)
        # If this pickup is in very near future check the power is available
        if turns_of_actions < 2 and pickup_resource == util.POWER:
            logger.debug(f"Checking power pickup is valid and removing power from factory")
            factory_num = self.master.maps.factory_maps.friendly[self.pos_slice]
            if factory_num >= 0:
                factory_id = f"factory_{factory_num}"
                factory = self.master.factories.friendly.get(factory_id)
                if factory.short_term_power < pickup_amount:
                    logger.warning(
                        f"{self.log_prefix} planning to pickup {pickup_amount} but {factory_id} expects to have {factory.short_term_power}"
                    )
                factory.short_term_power -= pickup_amount
            else:
                logger.warning(f"{self.log_prefix} Did not find factory at {self.pos} for pickup")
        return self.unit.pickup(pickup_resource, pickup_amount, repeat, n)

    @property
    def factory_loc(self) -> [None, np.ndarray]:
        """Shortcut to the factory_loc of this unit or None if not assigned a factory"""
        factory = self.factory
        if factory is not None:
            return factory.factory_loc

    @property
    def factory(self) -> FriendlyFactoryManager:
        if self.factory_id:
            factory = self.master.factories.friendly.get(self.factory_id, None)
            if factory is not None:
                return factory
        logger.error(f"{self.log_prefix}: f_id={self.factory_id} not in factories")
        raise ValueError(f"{self.log_prefix} has no factory")

    def power_remaining(self, rubble: np.ndarray = None) -> int:
        """Return power remaining at final step in actions so far"""
        rubble = self.master.maps.rubble if rubble is None else rubble
        return (
            self.start_of_turn_power
            - self.unit_config.ACTION_QUEUE_POWER_COST
            - self.power_cost_of_actions(rubble=rubble)
        )

    def dead(self):
        """Called when unit is detected as dead"""
        logger.warning(f"{self.log_prefix} Friendly unit dead")
        if self.factory_id:
            logger.info(f"removing from {self.factory_id} units also")
            fkey = "light" if self.unit.unit_type == "LIGHT" else "heavy"
            popped = getattr(self.master.factories.friendly[self.factory_id], f"{fkey}_units").pop(self.unit_id, None)
            if popped is None:
                logger.warning(f"{self.log_prefix}  was not in {self.factory_id} units")
