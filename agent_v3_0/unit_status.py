from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Union, Deque, TYPE_CHECKING, Dict
import numpy as np

import util
from actions_util import Actions
from config import get_logger

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManager, EnemyUnitManager

logger = get_logger(__name__)

@dataclass
class AttackValues:
    position: Tuple[int, int] = (0, 0)
    chase_radius: int = 15
    eliminate_threshold: int = 2

    @dataclass
    class Hold:
        distance: int = 10
        radius: int = 5

    hold: Hold = field(default_factory=Hold)

    @dataclass
    class Temporary:
        num_remaining: int = 0

    temp: Temporary = field(default_factory=Temporary)


@dataclass
class MineValues:
    min_resource_value_cutoff: int = 20
    _resource_locations: List[Tuple[int, int]] = field(default_factory=list)
    _resource_types: List[int] = field(default_factory=list)

    @property
    def resource_locations(self) -> List[Tuple[int, int]]:
        return self._resource_locations

    @resource_locations.setter
    def resource_locations(self, new_resource_locations: List[Tuple[int, int]]) -> None:
        self._resource_locations = new_resource_locations
        self._resource_types = self.calculate_resource_types(new_resource_locations)

    @property
    def resource_types(self) -> List[int]:
        return self._resource_types

    @resource_types.setter
    def resource_types(self, new_resource_types: List[int]) -> None:
        self._resource_types = new_resource_types
        self._resource_locations = self.calculate_resource_locations(new_resource_types)

    def calculate_resource_types(self, resource_locations: List[Tuple[int, int]]) -> List[int]:
        # Implement the logic to calculate resource types based on resource locations
        pass

    def calculate_resource_locations(self, resource_types: List[int]) -> List[Tuple[int, int]]:
        # Implement the logic to calculate resource locations based on resource types
        pass


@dataclass
class RubbleValues:
    target_position: Tuple[int, int] = None
    target_path: Optional[List[Tuple[int, int]]] = None
    clear_radius: int = 8


@dataclass
class LichenValues:
    target_position: Tuple[int, int] = None
    clear_radius: int = 7
    clear_direction: str = 'out'
    suicide_after: bool = False


class DestType(enum.Enum):
    FACTORY = 'factory'
    ICE = 'ice'
    ORE = 'ore'
    ENEMY = 'enemy'


@dataclass
class TravelLocation:
    pos: Tuple[int, int] = None
    dest_type: DestType = None
    amount: int = 0


@dataclass
class Status:
    current_action: Actions
    previous_action: Actions
    last_action_update_step: int
    last_action_success: bool
    action_queue_valid_after_step: bool
    planned_actions: List[np.ndarray] = field(default_factory=list)
    _last_beginning_of_step_update: int = 0

    next_dest: TravelLocation = None

    previous_locations: Deque[TravelLocation] = field(default_factory=deque)

    action_queue_empty_ok: bool = False
    replan_required: bool = False

    attack_values: AttackValues = field(default_factory=AttackValues)
    mine_values: MineValues = field(default_factory=MineValues)
    rubble_values: RubbleValues = field(default_factory=RubbleValues)
    lichen_values: LichenValues = field(default_factory=LichenValues)


    def _step_planned_actions(self) -> List[np.ndarray]:
        new_actions = list(self.planned_actions.copy())
        if len(new_actions) == 0:
            return []
        elif new_actions[0][util.ACT_N] > 1:
            new_actions[0][util.ACT_N] -= 1
            return new_actions
        elif new_actions[0][util.ACT_N] == 1:
            if new_actions[0][util.ACT_REPEAT] > 0:
                logger.warning(
                    f"Not implemented adding repeat actions to back of planned actions (might not make sense)"
                )
            new_actions.pop(0)
            return new_actions
        else:
            logger.error(f"failed to update planned actions. First planned action = {new_actions[0]}")
            raise NotImplementedError(f"failed to update planned actions. planned actions = {new_actions}")

    def step_update_planned_actions(self, unit: FriendlyUnitManager):
        """Update the planned actions
        0. if not a real update step return (debugging)
        1. else update planned actions assuming an action took place
        2. If no actions and no planned actions no update
        3. if action_queue and planned actions don't match, replace planned actions and set a flag
        4. else set flag passed
        """
        step = unit.master.step
        # print(unit.master.step, self._last_beginning_of_step_update)
        # Check if this is the same step as last update (i.e. running in my debugging env)
        if step == self._last_beginning_of_step_update:
            logger.warning(f"{unit.log_prefix}: Trying to update for same step again, not updating")
            return True

        valid = False
        new_planned = self._step_planned_actions()
        if len(unit.start_of_turn_actions) == 0:
            if len(new_planned) == 0:
                logger.debug(f"no unit or planned actions, valid")
                valid = True
            else:
                logger.error(f"{unit.log_prefix} unit actions empty, but len planned was {len(new_planned)}")
                new_planned = []
                valid = False
        elif len(unit.start_of_turn_actions) > 0 and len(new_planned) == 0:
            logger.error(
                f"{unit.log_prefix}: len(actions) = {len(unit.start_of_turn_actions)} != len(planned) = {len(new_planned)}"
            )
            valid = False
        elif np.all(unit.start_of_turn_actions[0] == new_planned[0]):
            logger.debug(f"planned_actions still valid")
            valid = True
        else:
            logger.warning(
                f"{unit.log_prefix} planned actions no longer match q1 = {unit.start_of_turn_actions[0]}, p1 = {new_planned[0]}. updating planned actions"
            )
            new_planned = unit.start_of_turn_actions.copy()
            valid = False
        self._last_beginning_of_step_update = step
        self.planned_actions = new_planned
        self.action_queue_valid_after_step = valid
        return valid

    def update_status(self, new_action: Actions, success: bool):
        self.previous_action = self.current_action
        self.current_action = new_action
        self.last_action_success = success
        # Action queue might not actually be getting updated
        # self.status.last_action_update_step = self.master.step

    def update_planned_actions(self, action_queue: List[np.ndarray]):
        self.planned_actions = action_queue
        self.action_queue_valid_after_step = True
