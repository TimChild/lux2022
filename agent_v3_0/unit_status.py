from __future__ import annotations

import copy
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
    WAITING = "waiting"  # Waiting outside factory
    DROPOFF = "dropoff"  # Drop off resources at factory
    NOTHING = "nothing"  # Nothing but could be anywhere
    TRANSFER = "transfer"  # Transferring resource from one factory to another


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
    previous_action: ActStatus = field(default_factory=lambda: ActStatus())

    # Used when adding actions to unit (they are overwritten, so don't set them here!)
    targeting_enemy: bool = None  # For pathing to enemy (don't raise Close/Collision enemy statuses)
    allow_partial: bool = None  # Allow partial pickup of resources

    def copy(self) -> ActStatus:
        # TODO: Might be able to change this to just copy instead of deepcopy if it's super slow
        return copy.deepcopy(self)


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
    factory_dist: int = 0

    # Todo: not sure I'm using this now
    should_act_reasons: List[ShouldActInfo] = field(default_factory=list)
    # Whether planned actions are valid after stepping (i.e. do they match the next action in real action queue)
    planned_actions_valid_from_last_step: bool = True
    next_action_was_valid = False

    # must move next turn
    must_move: bool = False

    # If False, and planned actions are empty at the end of action calculation either error or plan again or something
    action_queue_empty_ok: bool = False
    # If set True, the units next action calculation loop will be started again (e.g. if switch from Mine decides it
    # should switch to attack)
    replan_required: bool = False

    # I.e. next action not valid, plan should update (might be temporary while I get rid of old Actions)
    # recommend_plan_update: Optional[bool] = None

    def update(self, unit: FriendlyUnitManager, master: MasterState):
        """Beginning of turn update"""
        self.next_dest = next_dest(
            unit.current_path(max_len=30, planned_actions=True), master.maps, master.pathfinder.unit_paths
        )
        self.factory_dist = util.manhattan(unit.start_of_turn_pos, unit.factory.pos)
        self.should_act_reasons = []

        # Gets set when stepping planned actions
        self.planned_actions_valid_from_last_step = False
        # Set when checking if next action of unit is valid, if not, this gets set and status gets reset to zero
        self.next_action_was_valid = False
        self.must_move = False
        self.action_queue_empty_ok = False
        self.replan_required = False
        # self.recommend_plan_update = None


@dataclass
class Status:
    master: InitVar[MasterState]
    # _current_action: ActStatus = ActStatus()
    current_action: ActStatus = ActStatus()
    # When was the action queue last updated
    last_real_action_update_step: int = 0
    _planned_action_queue: List[np.ndarray] = field(default_factory=list)
    _planned_act_statuses: List[ActStatus] = field(default_factory=list)

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
    def planned_action_queue(self) -> List[np.ndarray]:
        """Should only be updated at beginning/end of turn (use unit.action_queue when planning etc"""
        return self._planned_action_queue

    @property
    def planned_act_statuses(self) -> List[ActStatus]:
        """Should only be updated when updating planned actions"""
        return self._planned_act_statuses

    def update_planned_action_queue(self, new_queue: List[np.ndarray], new_act_statuses: List[ActStatus]):
        if not isinstance(new_queue, list):
            raise TypeError
        if not isinstance(new_act_statuses, list):
            raise TypeError
        if not len(new_queue) == len(new_act_statuses):
            raise ValueError(f"{len(new_queue)} == {len(new_act_statuses)} did not evaluate True")

        self._planned_action_queue = new_queue
        self._planned_act_statuses = new_act_statuses

    def update_action_status(self, new_action: ActStatus):
        """For updating with a new act status (keeps track of previous)"""
        new_action.previous_action = self.current_action.copy()
        self.current_action = new_action

    def reset_to_step(self, step: int = 0):
        """For resetting to ActStatus at step from now
        Returns ActStatus and queue before that action was applied (so that action can apply again from where it was)

        Notes:
            - Planned actions start from what will be applied on step 1
            - Act status stored at same index as Planned Actions are the statuses that made those actions
            - So when restoring status for e.g.
                - step 0, want the 0th index status, but no action queue
                - step 1, want the 1st index status, only zeroth action queue step
        """
        if not step < len(self.planned_act_statuses):
            logger.error(
                f"Tried to set act status to {step} but act_statuses only {len(self.planned_act_statuses)}. Setting ActStatus Nothing"
            )
            self.current_action = ActStatus()
            return

        new_status = self.planned_act_statuses[step]
        new_planned = self.planned_action_queue[:step]
        new_statuses = self.planned_act_statuses[:step]
        logger.info(f"Resetting to {step} where action was {new_status}")
        self.current_action = new_status
        self.update_planned_action_queue(new_planned, new_statuses)
        return

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
            self.planned_act_statuses.pop(0)
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
        if len(new_planned) != len(self.planned_act_statuses):
            logger.error(
                f"{unit.log_prefix} Planned actions and ActStatuses not matched in length {len(new_planned)}!={len(self.planned_act_statuses)}"
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
