from typing import Optional, Tuple

from dataclasses import dataclass
import logging

from lux.unit import Unit
from lux.config import UnitConfig
from util import (
    manhattan,
    ICE,
    ORE,
    CENTER,
    POWER,
    MOVE,
    path_to_actions,
    actions_to_path,
    ACT_REPEAT,
    power_cost_of_actions,
)

from master_state import MasterState
from actions import Recommendation

LOGGING_LEVEL = 3


@dataclass
class Status:
    role: str
    current_action: str
    recommendation: Optional[Recommendation]


class UnitManager:
    def __init__(self, unit: Unit, master_state: MasterState, factory_id: str):
        self.unit_id = unit.unit_id
        self.unit = unit
        self.factory_id = factory_id
        self.unit_config: UnitConfig = unit.unit_cfg
        self.master: MasterState = master_state

        self.status: Status = Status(
            role='not set', current_action='', recommendation=None
        )

    def update(self, unit: Unit):
        self.unit = unit

    @property
    def pos(self):
        return self.unit.pos

    def assign(
        self,
        role: str,
        recommendation: Recommendation,
        resource_pos: Optional[Tuple[int, int]] = None,
        factory_pos: Optional[Tuple[int, int]] = None,
    ):
        self.status.role = role
        self.status.recommendation = recommendation
        # # TODO: Not sure UnitManager should have to deal with master_plan... Think more (should master plan be calling this method?)
        # self.master.deassign_unit_resource(unit_id=self.unit_id)
        # self.master.deassign_unit_factory(unit_id=self.unit_id)
        # if resource_pos is not None:
        #     self.master.assign_unit_resource(self.unit_id, resource_pos)
        # if factory_pos is not None:
        #     self.master.assign_unit_factory(self.unit_id, factory_pos)

    def actions_to_path(self, actions=None):
        """
        Return a list of coordinates of the path the actions represent
        """
        if actions is None:
            actions = self.unit.action_queue
        return actions_to_path(self.unit, actions)

    def log(self, message, level=logging.INFO):
        logging.log(
            level,
            f"Step {self.master.game_state.real_env_steps}, Unit {self.unit_id}: {message}",
        )
