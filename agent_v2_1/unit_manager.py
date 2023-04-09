from typing import Optional, Tuple, List
import numpy as np
import abc
from dataclasses import dataclass
from typing import Union
import logging

from lux.unit import Unit
from lux.config import UnitConfig
import util

from master_state import MasterState
from actions import Recommendation

LOGGING_LEVEL = 3


@dataclass
class Status:
    role: Union[str, None]
    current_action: str
    recommendation: Optional[Recommendation]


class UnitManager(abc.ABC):
    def __init__(self, unit: Unit):
        self.unit_id = unit.unit_id
        self.unit = unit
        self.unit_config: UnitConfig = unit.unit_cfg

        # Keep track of pos a start of turn because pos will be updated while planning what to do next
        self.start_of_turn_pos = unit.pos

    def update(self, unit: Unit):
        """Beginning of turn update"""
        self.unit = unit
        self.start_of_turn_pos = unit.pos

    @property
    def current_path(self) -> np.ndarray:
        """Return current path from start of turn based on current action queue"""
        return util.new_actions_to_path(self.start_of_turn_pos, self.action_queue)

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
    def unit_type(self) -> str:
        return self.unit.unit_type

    @property
    def power(self) -> int:
        return self.unit.power

    @pos.setter
    def power(self, value):
        self.unit.power = value

    def actions_to_path(self, actions: [None, List[np.ndarray]] = None) -> np.ndarray:
        """
        Return a list of coordinates of the path the actions represent starting from unit.pos
        (which may have been updated since beginning of turn)
        """
        if actions is None:
            actions = self.unit.action_queue
        return util.new_actions_to_path(self.unit.pos, actions)

    @abc.abstractmethod
    def dead(self):
        """Called when unit dies, should tidy up anything related to unit"""
        pass


class EnemyUnitManager(UnitManager):
    def dead(self):
        logging.info(f'Enemy unit {self.unit_id} dead, nothing more to do')


class FriendlyUnitManger(UnitManager):
    def __init__(self, unit: Unit, master_state: MasterState, factory_id: str):
        super().__init__(unit)
        self.factory_id = factory_id
        self.master: MasterState = master_state
        self.status: Status = Status(role=None, current_action='', recommendation=None)

    def dead(self):
        """Called when unit is detected as dead"""
        logging.info(
            f'Friendly unit {self.unit_id} dead, removing from {self.factory_id} units also'
        )
        fkey = 'light' if self.unit.unit_type == 'LIGHT' else 'heavy'
        getattr(self.master.factories.friendly[self.factory_id], f'{fkey}_units').pop(
            self.unit_id
        )
