from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np
import abc
import copy
from dataclasses import dataclass

from luxai_s2.unit import UnitCargo

from lux.unit import Unit
from lux.config import UnitConfig

import util
import actions

from config import get_logger
from master_state import MasterState

if TYPE_CHECKING:
    from factory_manager import FriendlyFactoryManager

logger = get_logger(__name__)


def get_index(lst, index, default=None):
    """Get the element at the specified index in the list, or return the default value if the index is out of range."""
    return lst[index] if 0 <= index < len(lst) else default


@dataclass
class Status:
    current_action: str
    previous_action: str
    last_action_success: bool


class UnitManager(abc.ABC):
    def __init__(self, unit: Unit):
        self.unit_id = unit.unit_id
        self.unit = unit
        self.unit_config: UnitConfig = unit.unit_cfg

        self.dig = unit.dig
        self.transfer = unit.transfer
        self.pickup = unit.pickup

        # Keep track of pos a start of turn because pos will be updated while planning what to do next
        self.start_of_turn_pos = tuple(unit.pos)
        self.start_of_turn_power = unit.power

    def power_cost_of_actions(self, rubble: np.ndarray):
        return util.power_cost_of_actions(
            rubble=rubble, unit=self, actions=self.action_queue
        )

    def valid_moving_actions(
        self, costmap: np.ndarray, max_len=20, ignore_repeat=False
    ) -> util.ValidActionsMoving:
        return util.calculate_valid_move_actions(
            self.start_of_turn_pos,
            self.action_queue,
            valid_move_map=costmap,
            max_len=max_len,
            ignore_repeat=ignore_repeat,
        )

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

    def current_path(self, max_len: int = 10) -> np.ndarray:
        """Return current path from start of turn based on current action queue"""
        return util.actions_to_path(
            self.start_of_turn_pos, self.action_queue, max_len=max_len
        )

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

    def actions_to_path(
        self, actions: [None, List[np.ndarray]] = None, max_len=20
    ) -> np.ndarray:
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


class FriendlyUnitManger(UnitManager):
    def __init__(self, unit: Unit, master_state: MasterState, factory_id: str):
        super().__init__(unit)
        self.factory_id = factory_id
        self.master: MasterState = master_state
        self.status: Status = Status(
            current_action=actions.NOTHING,
            previous_action=actions.NOTHING,
            last_action_success=True,
        )
        self.start_of_turn_actions = []

    def update(self, unit: Unit):
        super().update(unit)
        self.start_of_turn_actions = copy.copy(unit.action_queue)

    def on_own_factory(self) -> bool:
        """Is this unit on its own factory"""
        return self.factory_loc[self.pos[0], self.pos[1]] == 1

    @property
    def factory_loc(self) -> [None, np.ndarray]:
        """Shortcut to the factory_loc of this unit or None if not assigned a factory"""
        factory = self.factory
        if factory is not None:
            return factory.factory_loc

    @property
    def factory(self) -> [None, FriendlyFactoryManager]:
        if self.factory_id:
            factory = self.master.factories.friendly.get(self.factory_id, None)
            return factory
        else:
            logger.error(f"{self.log_prefix}: f_id={self.factory_id} not in factories")
            return None

    def dead(self):
        """Called when unit is detected as dead"""
        logger.warning(f"{self.log_prefix} Friendly unit dead")
        if self.factory_id:
            logger.info(f"removing from {self.factory_id} units also")
            fkey = "light" if self.unit.unit_type == "LIGHT" else "heavy"
            popped = getattr(
                self.master.factories.friendly[self.factory_id], f"{fkey}_units"
            ).pop(self.unit_id, None)
            if popped is None:
                logger.warning(f"{self.log_prefix}  was not in {self.factory_id} units")
