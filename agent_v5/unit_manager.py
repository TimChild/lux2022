from typing import Optional, Tuple
import abc
from dataclasses import dataclass
import logging

from lux.unit import Unit
from lux.config import UnitConfig
from util import (
    actions_to_path,
)

from master_state import MasterState
from actions import Recommendation

LOGGING_LEVEL = 3


@dataclass
class Status:
    role: str
    current_action: str
    recommendation: Optional[Recommendation]


class UnitManager(abc.ABC):
    def __init__(self, unit: Unit):
        self.unit_id = unit.unit_id
        self.unit = unit
        self.unit_config: UnitConfig = unit.unit_cfg

    def update(self, unit: Unit):
        self.unit = unit

    @property
    def pos(self):
        return self.unit.pos

    def actions_to_path(self, actions=None):
        """
        Return a list of coordinates of the path the actions represent
        """
        if actions is None:
            actions = self.unit.action_queue
        return actions_to_path(self.unit, actions)

    @abc.abstractmethod
    def dead(self):
        """Called when unit dies, should tidy up anything related to unit"""
        pass


class EnemyUnitManager(UnitManager):
    def log(self, message, level=logging.INFO):
        logging.log(
            level,
            f"Enemy {self.unit_id}: {message}",
        )

    def dead(self):
        self.log(f'Enemy unit {self.unit_id} dead, nothing more to do')


class FriendlyUnitManger(UnitManager):
    def __init__(self, unit: Unit, master_state: MasterState, factory_id: str):
        super().__init__(unit)
        self.factory_id = factory_id
        self.master: MasterState = master_state
        self.status: Status = Status(
            role='not set', current_action='', recommendation=None
        )

    def log(self, message, level=logging.INFO):
        logging.log(
            level,
            f"Step {self.master.game_state.real_env_steps}, Unit {self.unit_id}: {message}",
        )

    def dead(self):
        """Called when unit is detected as dead"""
        self.log(
            f'Friendly unit {self.unit_id} dead, removing from {self.factory_id} units also'
        )
        fkey = 'light' if self.unit.unit_type == 'LIGHT' else 'heavy'
        getattr(self.master.factories.friendly[self.factory_id], f'{fkey}_units').pop(
            self.unit_id
        )
