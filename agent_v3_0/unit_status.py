from __future__ import annotations

import enum
from collections import deque
from dataclasses import dataclass, field, InitVar
from typing import Tuple, List, Optional, Union, Deque, TYPE_CHECKING, Dict, NamedTuple, Any
import numpy as np

import util
import actions_util
from collisions import UnitPaths
from config import get_logger

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManager, EnemyUnitManager
    from master_state import MasterState, Maps
    from decide_actions import ActReasons, ShouldActInfo
    from combat_planner import TargetInfo

logger = get_logger(__name__)


@dataclass
class AttackValues:
    # Position to attack/hold
    position: Tuple[int, int] = (0, 0)
    chase_radius: int = 15
    eliminate_threshold: int = 2
    heavy_attack_light: bool = False
    target: Optional[TargetInfo] = None

    @dataclass
    class Hold:
        # How close to hold position is allowed (i.e. allow collision avoidance and not have to move back)
        pos_buffer: int = 2
        # Max dist from position to attack enemy
        attack_dist: int = 10
        # How far to look for units within hold pos
        search_radius: int = 5

    hold: Hold = field(default_factory=Hold)

    @dataclass
    class Temporary:
        num_remaining: int = 0
        # Max dist from position to attack enemy
        attack_radius = 5

    temp: Temporary = field(default_factory=Temporary)


@dataclass
class MineValues:
    ice: np.ndarray
    ore: np.ndarray
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

    def calculate_resource_types(self, resource_locations: List[Tuple[int, int]]) -> List[int]:
        # Implement the logic to calculate resource types based on resource locations
        r_types = []
        for pos in resource_locations:
            if self.ice[pos[0], pos[1]] > 0:
                r_types.append(DestType.ICE)
            elif self.ore[pos[0], pos[1]] > 0:
                r_types.append(DestType.ORE)
        return r_types


@dataclass
class RubbleValues:
    target_position: Tuple[int, int] = None
    target_path: Optional[List[Tuple[int, int]]] = None
    clear_radius: int = 8


@dataclass
class LichenValues:
    target_position: Tuple[int, int] = None
    clear_radius: int = 7
    clear_direction: str = "out"
    suicide_after: bool = False


class DestType(NamedTuple):
    FACTORY: str = "factory"
    FACTORY_QUEUE: str = "factory_queue"
    ICE: int = util.ICE
    ORE: int = util.ORE
    RUBBLE: str = "rubble"
    LICHEN: str = "lichen"
    UNIT: str = "unit"


@dataclass
class DestStatus:
    pos: Tuple[int, int]
    # Manhattan dist to dest
    dist: int
    # step of arrival where 0 is now
    step: int
    # What type of thing is there
    type: DestType = None
    # e.g. rubble, lichen
    amount: int = 0
    # If  a unit or factory there
    id_num: int = -1


class ActCategory(enum.Enum):
    """Which planner should be sent to"""

    COMBAT = "combat"
    MINE = "mine"
    CLEAR = "clear"
    RUN_AWAY = "runaway"
    WAITING = "waiting"  # Waiting outside factory
    DROPOFF = "dropoff"  # Drop off resources at factory
    NOTHING = "nothing"  # Nothing but could be anywhere


class ActSubCategory(enum.Enum):
    pass


class MineActSubCategory(ActSubCategory):
    ORE = "ore"
    ICE = "ice"


class ClearActSubCategory(ActSubCategory):
    RUBBLE = "rubble"
    LICHEN = "lichen"


class CombatActSubCategory(ActSubCategory):
    RETREAT_HOLD = "retreat hold"
    ATTACK_HOLD = "attack hold"
    TEMPORARY = "temporary"
    CONSERVE = "conserve"
    RUN_AWAY = "run away"


@dataclass
class ActStatus:
    category: ActCategory = ActCategory.NOTHING
    sub_category: Optional[ActSubCategory] = None
    step: int = 1


def next_dest(path: np.ndarray, maps: Maps, unit_paths: UnitPaths) -> DestStatus:
    """Find the next destination of this unit and return DestInfo"""
    dest_step = actions_util.find_dest_step_from_step(path, 0, direction="forward")
    pos = path[dest_step]
    dist = util.manhattan(path[0], pos)
    dest = DestStatus(pos, dist, dest_step)
    if maps.ice[pos[0], pos[1]] > 0:
        dest.type = DestType.ICE
    elif maps.ore[pos[0], pos[1]] > 0:
        dest.type = DestType.ORE
    elif maps.factory_maps.all[pos[0], pos[1]] >= 0:
        dest.type = DestType.FACTORY
        dest.id_num = maps.factory_maps.all[pos[0], pos[1]]
    elif maps.rubble[pos[0], pos[1]] > 0:
        dest.type = DestType.RUBBLE
        dest.amount = maps.rubble[pos[0], pos[1]]
    elif maps.lichen[pos[0], pos[1]] > 0:
        dest.type = DestType.LICHEN
        dest.amount = maps.lichen[pos[0], pos[1]]
    elif unit_paths.all[dest_step, pos[0], pos[1]] >= 0:
        dest.type = DestType.UNIT
        dest.id_num = unit_paths.all[dest_step, pos[0], pos[1]]
    return dest


@dataclass
class TurnStatus:
    """Holding things relevant to calculating actions this turn"""

    #  Some useful things that can be used to help decide if actions need updating
    next_dest: DestStatus = None
    should_act_reasons: List[ShouldActInfo] = field(default_factory=list)
    # Whether planned actions are valid after stepping (i.e. do they match the next action in real action queue)
    planned_actions_valid_from_last_step: bool = True
    next_action_valid = False

    # must move next turn
    must_move: bool = False

    # If False, and planned actions are empty at the end of action calculation either error or plan again or something
    action_queue_empty_ok: bool = False
    # If set True, the units next action calculation loop will be started again (e.g. if switch from Mine decides it
    # should switch to attack)
    replan_required: bool = False

    # I.e. next action not valid, plan should update (might be temporary while I get rid of old Actions)
    recommend_plan_update: Optional[bool] = None

    def update(self, unit: FriendlyUnitManager, master: MasterState):
        """Beginning of turn update"""
        self.next_dest = next_dest(
            unit.current_path(max_len=30, planned_actions=True), master.maps, master.pathfinder.unit_paths
        )
        self.should_act_reasons = []
        # Gets set when stepping planned actions
        self.planned_actions_valid_from_last_step = False
        self.next_action_valid = False
        self.must_move = False
        self.action_queue_empty_ok = False
        self.replan_required = False
        self.recommend_plan_update = None


@dataclass
class Status:
    master: InitVar[MasterState]
    # _current_action: ActStatus = ActStatus()
    current_action: ActStatus = ActStatus()
    previous_action: ActStatus = ActStatus()
    last_action_update_step: int = 0
    _planned_action_queue: List[np.ndarray] = field(default_factory=list)
    act_statuses: List[ActStatus] = field(default_factory=list)

    # Storing things about processing of turn for unit (reset at beginning of turn)
    turn_status: TurnStatus = field(default_factory=TurnStatus)

    previous_locations: Deque[DestStatus] = field(default_factory=deque)

    mine_values: MineValues = field(init=False)
    attack_values: AttackValues = field(default_factory=AttackValues)
    rubble_values: RubbleValues = field(default_factory=RubbleValues)
    lichen_values: LichenValues = field(default_factory=LichenValues)

    # For debug use only
    _debug_last_beginning_of_step_update: int = 0

    # @property
    # def current_action(self):
    #     return self._current_action
    #
    # @current_action.setter
    # def current_action(self, value: ActStatus):
    #     if not isinstance(value, ActStatus):
    #         raise TypeError(f'must be ActStatus got type {value}')
    #     self._current_action = value

    @property
    def planned_action_queue(self):
        """Should only be updated at beginning/end of turn (use unit.action_queue when planning etc"""
        return self._planned_action_queue

    def update_planned_action_queue(self, new_queue: List[np.ndarray]):
        if not isinstance(new_queue, list):
            raise TypeError
        self._planned_action_queue = new_queue

    def __post_init__(self, master: MasterState):
        self.mine_values: MineValues = MineValues(ice=master.maps.ice, ore=master.maps.ore)

    def update(self, unit: FriendlyUnitManager, master: MasterState):
        """beginning of turn update"""
        self.turn_status.update(unit, master)
        self._step_update_planned_actions(unit)

    def _step_planned_actions_and_status(self) -> List[np.ndarray]:
        new_actions = list(self.planned_action_queue.copy())
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
            self.act_statuses.pop(0)
            return new_actions
        else:
            logger.error(f"failed to update planned actions. First planned action = {new_actions[0]}")
            raise NotImplementedError(f"failed to update planned actions. planned actions = {new_actions}")

    def _step_update_planned_actions(self, unit: FriendlyUnitManager):
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
        if step == self._debug_last_beginning_of_step_update:
            logger.warning(f"{unit.log_prefix}: Trying to update for same step again, not updating")
            return True

        valid = False
        new_planned = self._step_planned_actions_and_status()
        if len(new_planned) != len(self.act_statuses):
            logger.error(
                f"{unit.log_prefix} Planned actions and ActStatuses not matched in length {len(new_planned)}!={len(self.act_statuses)}"
            )
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
            new_planned = unit.start_of_turn_actions.copy()
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
        self._debug_last_beginning_of_step_update = step
        self.update_planned_action_queue(new_planned)

        self.turn_status.planned_actions_valid_from_last_step = valid
        return valid

    def update_action_status(self, new_action: ActStatus):
        self.previous_action = self.current_action
        self.current_action = new_action
        # Action queue might not actually be getting updated
        # self.status.last_action_update_step = self.master.step
