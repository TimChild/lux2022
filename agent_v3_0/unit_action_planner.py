from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional
import copy

import numpy as np
import pandas as pd

import actions_util
import util
from general_planner import GeneralPlanner
import collisions
from decide_actions import ActionDecider, ActReasons, ConsiderActInfo, ShouldActInfo
from config import get_logger
from factory_action_planner import FactoryDesires
from factory_manager import FactoryInfo
from master_state import MasterState, AllUnits
from mining_planner import MiningPlanner
from new_path_finder import Pather
from rubble_clearing_planner import ClearingPlanner
from combat_planner import CombatPlanner
from unit_manager import FriendlyUnitManager, UnitManager
from action_validation import ValidActionCalculator, valid_action_space
from decide_actions import ActStatus
from unit_status import ActCategory, MineActSubCategory, ClearActSubCategory, CombatActSubCategory, DestType

logger = get_logger(__name__)


@dataclass
class UnitInfo:
    unit: FriendlyUnitManager
    act_info: ConsiderActInfo
    unit_id: str
    last_action_update_step: int
    len_action_queue: int
    distance_to_factory: Optional[float]
    is_heavy: bool
    unit_type: str
    enough_power_to_move: bool
    power: int
    ice: int
    ore: int
    power_over_20_percent: bool

    @classmethod
    def from_data(
        cls,
        unit: FriendlyUnitManager,
        act_info: ConsiderActInfo,
    ):
        unit_info = cls(
            unit=unit,
            act_info=act_info,
            unit_id=unit.unit_id,
            last_action_update_step=unit.status.last_real_action_update_step,
            len_action_queue=len(unit.action_queue),
            distance_to_factory=util.manhattan(unit.start_of_turn_pos, unit.factory.pos)
            if unit.factory_id is not None
            else None,
            is_heavy=unit.unit_type == "HEAVY",
            unit_type=unit.unit_type,
            enough_power_to_move=(unit.power > unit.unit_config.MOVE_COST + unit.unit_config.ACTION_QUEUE_POWER_COST),
            power=unit.power,
            ice=unit.cargo.ice,
            ore=unit.cargo.ore,
            power_over_20_percent=unit.start_of_turn_power > unit.unit_config.BATTERY_CAPACITY * 0.2,
        )
        return unit_info


@dataclass
class UnitInfos:
    infos: Dict[str, UnitInfo]

    def sort_by_priority(self):
        """
        Sorts units by priority by first converting to a dataframe and then doing some ordered sorting
        """
        logger.info(f"sort_units_by_priority called")
        if len(self.infos) == 0:
            logger.debug("No unit_infos data to sort")
            return None

        df = self.to_df()
        """
        Sort Order:
            - Heavy before light
            - Not enough power to move -- so others can units position
            - Power over 20% -- Higher power has priority
            - Last action update step -- Older has priority
            
        Note: 
            False == High/True values first
            True == Low/False values first
        """
        sorted_df = df.sort_values(
            by=[
                "is_heavy",
                "enough_power_to_move",
                "power_over_20_percent",
                "last_action_update_step",
            ],
            ascending=[False, True, False, True],
        )
        highest = sorted_df.iloc[0]
        lowest = sorted_df.iloc[-1]

        for series, priority in zip([highest, lowest], ["higheset", "lowest"]):
            logger.debug(
                f"Unit with {priority} priority: {series.unit_id}  ({series.unit.pos}), is_heavy={series.is_heavy}, "
                f"last_acted_step={series.last_real_action_update_step}, power={series.power}, ice={series.ice}, ore={series.ore}, len_acts={series.len_action_queue}"
            )
        ordered_infos = OrderedDict()
        for unit_id in sorted_df.index:
            ordered_infos[unit_id] = self.infos[unit_id]
        self.infos = ordered_infos
        logger.debug(f"Done sorting units")
        return None

    def to_df(self) -> pd.DataFrame:
        # Convert the list of UnitInfo instances to a list of dictionaries
        unit_info_dicts = [unit_info.__dict__ for unit_info in self.infos.values()]

        # Create a DataFrame from the list of dictionaries
        df = pd.DataFrame(unit_info_dicts)

        # Set the 'unit_id' column as the DataFrame index
        df.index = df["unit_id"]
        return df


# @dataclass
# class UnitsToAct:
#     needs_to_act: dict[str, ConsiderActInfo]
#     should_not_act: dict[str, ConsiderActInfo]
#     has_updated_actions: dict[str, ConsiderActInfo] = field(default_factory=dict)
#
#     def get_act_info(self, unit_id: str) -> ConsiderActInfo:
#         for d in [self.needs_to_act, self.should_not_act, self.has_updated_actions]:
#             if unit_id in d:
#                 return d[unit_id]
#         raise KeyError(f"{unit_id} not in UnitsToAct")


@dataclass
class NewUnitsToAct:
    needs_to_act: dict[str, FriendlyUnitManager]
    should_not_act: dict[str, FriendlyUnitManager]
    has_updated_actions: dict[str, FriendlyUnitManager] = field(default_factory=dict)

    def get_unit(self, unit_id: str) -> FriendlyUnitManager:
        for d in [self.needs_to_act, self.should_not_act, self.has_updated_actions]:
            if unit_id in d:
                return d[unit_id]
        raise KeyError(f"{unit_id} not in UnitsToAct")


@dataclass
class CloseUnits:
    """Record nearby units"""

    unit_id: str
    unit_pos: Tuple[int, int]
    other_unit_ids: List[str] = field(default_factory=list)
    other_units: List[UnitManager] = field(default_factory=list)
    other_unit_positions: List[Tuple[int, int]] = field(default_factory=list)
    other_unit_distances: List[int] = field(default_factory=list)
    other_unit_types: List[str] = field(default_factory=list)
    other_unit_powers: List[int] = field(default_factory=list)

    def closest(self) -> Optional[UnitManager]:
        if len(self.other_unit_distances) == 0:
            return None
        return self.other_units[self.other_unit_distances.index(min(self.other_unit_distances))]


@dataclass
class AllCloseUnits:
    """Collection of close unit info for all units,
    Their id will only be in the dicts if they are close to other units
    """

    close_to_friendly: Dict[str, CloseUnits]
    close_to_enemy: Dict[str, CloseUnits]
    close_threshold: int

    @classmethod
    def from_info(cls, all_units: AllUnits, close_threshold: int, map_shape: Tuple[int, int]):
        """Calculates which friendly units are close to any other unit"""
        friendly = {}
        enemy = {}
        # Keep track of being close to friendly and enemy separately
        for all_close, other_units in zip(
            [friendly, enemy],
            [all_units.friendly.all, all_units.enemy.all],
        ):
            # For all friendly units, figure out which friendly and enemy they are near
            for unit_id, unit in all_units.friendly.all.items():
                # print(f'For {unit_id}:')
                unit_distance_map = cls.unit_distance_map(unit, map_shape)
                close = CloseUnits(unit_id=unit_id, unit_pos=unit.pos)
                for other_id, other_unit in other_units.items():
                    # print(f'checking {other_id}')
                    if other_id == unit_id:  # Don't compare to self
                        continue
                    # print(f'{other_unit.unit_id} pos = {other_unit.pos}')
                    dist = unit_distance_map[other_unit.pos[0], other_unit.pos[1]]
                    # print(f'dist {dist}')

                    if dist <= close_threshold:
                        # print(f'adding {other_id} as close')
                        close.other_unit_ids.append(other_id)
                        close.other_units.append(other_unit)
                        close.other_unit_positions.append(other_unit.pos)
                        close.other_unit_distances.append(dist)
                        close.other_unit_types.append(other_unit.unit_type)
                        close.other_unit_powers.append(other_unit.power)
                if len(close.other_unit_ids) > 0:
                    # print(f'Adding to dict for {unit_id}')
                    all_close[unit_id] = close
        all_close_units = cls(
            close_to_friendly=friendly,
            close_to_enemy=enemy,
            close_threshold=close_threshold,
        )
        return all_close_units

    @staticmethod
    def unit_distance_map(unit: UnitManager, map_shape):
        unit_distance_map = util.pad_and_crop(
            util.manhattan_kernel(30),
            large_arr_shape=map_shape,
            x1=unit.start_of_turn_pos[0],
            y1=unit.start_of_turn_pos[1],
            fill_value=35,
        )
        return unit_distance_map


class SingleUnitActionPlanner:
    # If unit has fewer than this many planned steps, replan
    min_planned_steps = 10
    # TODO: Not sure what happens if these numbers aren't equal to each other!
    # Check this many steps are valid in the planned queue
    max_check_steps = 10

    def __init__(
        self,
        unit: FriendlyUnitManager,
        master: MasterState,
        multi_planner: MultipleUnitActionPlanner,
        current_paths: collisions.UnitPaths,
        all_close_units: AllCloseUnits,
        collision_info: collisions.AllCollisionsForUnit,
        action_validator: ValidActionCalculator,
        collision_resolve_max_step: int,
    ):
        self.unit = unit
        self.master = master
        self.multi_planner = multi_planner
        self.current_paths = current_paths
        self.all_close_units = all_close_units
        self.collision_info = collision_info
        self.action_validator = action_validator
        self.collision_resolve_max_step = collision_resolve_max_step

    def _unit_must_move(self) -> bool:
        """Must move if current location will be occupied at step 1 (not zero which is now)"""
        start_costmap = self.master.pathfinder.generate_costmap(self.unit, override_step=1, collision_only=True)

        unit_must_move = False
        # If current location will be occupied
        if start_costmap[self.unit.start_of_turn_pos[0], self.unit.start_of_turn_pos[1]] <= 0:
            logger.info(f"{self.unit.unit_id} MUST move first turn to avoid collision at current pos {self.unit.pos}")
            unit_must_move = True

        # If very close to enemy that can kill us
        if self.unit.unit_id in self.all_close_units.close_to_enemy:
            close = self.all_close_units.close_to_enemy[self.unit.unit_id]
            close_dists = close.other_unit_distances
            # Dist 1 == adjacent
            if len(close_dists) > 0 and min(close_dists) <= 1:
                # Only matters if other is Heavy or both lights
                for utype, dist in zip(close.other_unit_types, close_dists):
                    if dist <= 1 and (utype == self.unit.unit_type or utype == "HEAVY"):
                        unit_must_move = True

        return unit_must_move

    # def _attempt_resolve_continue_actions(self) -> bool:
    #     """
    #     Note: probably remove this once action planners are working
    #
    #     Returns:
    #         bool: unti.status.turn_status updated
    #     """
    #     if self.unit_info.act_info.reason == ActReasons.COLLISION_WITH_FRIENDLY:
    #         collision_resolver = CollisionResolver(
    #             self.unit,
    #             pathfinder=self.master.pathfinder,
    #             maps=self.master.maps,
    #             unit_paths=self.current_paths,
    #             collisions=self.collision_info,
    #             max_step=self.collision_resolve_max_step,
    #         )
    #         status_updated = collision_resolver.resolve()
    #         return status_updated
    #     logger.info(f"Don't know how to resolve {self.unit_info.act_info.reason} without calling the planner again")
    #     self.unit.status.turn_status.recommend_plan_update = True
    #     return True

    def _force_moving_if_necessary(self, unit_must_move: bool) -> bool:
        success = True
        if unit_must_move:
            logger.debug(f"{self.unit.unit_id} checking this unit is moving first turn")
            q = self.unit.status.planned_action_queue
            if len(q) == 0 or q[0][util.ACT_TYPE] != util.MOVE or q[0][util.ACT_DIRECTION] == util.CENTER:
                logger.warning(
                    f"{self.unit.log_prefix} was required to move first turn, but actions are {q}, trying to move unit"
                )
                self.unit.reset_unit_to_start_of_turn_empty_queue()
                self.unit.action_handler.add_cheapest_move()
                status = self.unit.status.update_planned_action_queue(self.unit.action_queue.copy(), self.unit.act_statuses.copy())
                if status == self.unit.action_handler.HandleStatus.SUCCESS:
                    logger.warning(f"successfully forced move out of the way")
                else:
                    logger.error(
                        f"{self.unit.log_prefix} was required to move first turn, but did not, and failed to force a "
                        f"move to empty adjacent cell with status {status}"
                    )
                success = False
        return success

    def _run_actions_to_step(self, max_check_steps):
        """If any of first few actions are not valid, reset unit to that point so repathing can happen"""
        # First action valid?
        if len(self.unit.status.planned_action_queue) == 0:
            logger.debug(f'Actions valid because 0 len')
            # Done, current actions are valid
            valid = True
        else:
            action = self.unit.status.planned_action_queue[0]
            # check action is valid
            # TODO: Can improve validation to use the up to date factory values
            valid = self.action_validator.next_action_valid(self.unit, action)
        self.unit.status.turn_status.next_action_was_valid = valid
        if not valid:
            logger.warning(f"{self.unit.log_prefix} First action invalid, resetting and returning")
            self.unit.status.reset_to_step(step=0)
            return self.unit.action_handler.HandleStatus.INVALID_FIRST_STEP
        logger.debug(f'First action valid')

        # Try running units actions
        actions_to_check = actions_util.split_actions_at_step(self.unit.status.planned_action_queue, max_check_steps)[0]
        act_statuses = self.unit.status.planned_act_statuses[: len(actions_to_check)]
        status = self.unit.run_actions(actions_to_check, act_statuses)
        if status != self.unit.action_handler.HandleStatus.SUCCESS:
            num_success = util.num_turns_of_actions(self.unit.action_queue)
            logger.warning(
                f"{self.unit.log_prefix} Actions failed after {num_success} with status {status}, resetting to last successful"
            )
            # TODO: maybe don't want to actually reset here...
            self.unit.status.reset_to_step(step=num_success)
        logger.debug(f'passed initial validation')
        return status

    def calculate_actions_for_unit(self) -> bool:
        # Will be using this a lot in here
        unit = self.unit
        HS = self.unit.action_handler.HandleStatus

        logger.info(
            f"\nBeginning calculating action for {unit.unit_id}: power = {unit.power}, pos = {unit.pos}, \n"
            f"\tlen(actions) = {len(unit.action_queue)}, \n"
            f"\tcurrent_action = {unit.status.current_action.category}:{unit.status.current_action.sub_category}\n"
        )

        # Is current location in existing paths for next step or equal or higher enemy adjacent
        unit_must_move = self._unit_must_move()
        self.unit.status.turn_status.must_move = unit_must_move
        logger.debug(f'Unit must move = {unit_must_move}')

        # Check next few actions are valid
        # TODO: Should I change the check value?, lower than max leaves potentially invalid queues after X steps, on othe other hand
        # TODO: they may become valid by the time they are closer to occurring
        max_check_steps = min(self.max_check_steps, unit.max_queue_step_length)
        status = self._run_actions_to_step(max_check_steps)
        logger.debug(f'status of checks = {status}')

        # If status success and no more updates needed
        if (
            status == HS.SUCCESS
            and util.num_turns_of_actions(self.unit.status.planned_action_queue) > self.min_planned_steps
        ):
            # Done, no need to do more
            logger.info(f'No need to update this units planned actions')
            return status

        for i in range(5):
            logger.debug(f'Updating plans round {i}')
            # Otherwise get updates from general planners
            status = None
            if self.unit.status.current_action.category in [ActCategory.NOTHING, ActCategory.WAITING, ActCategory.DROPOFF, ActCategory.TRANSFER]:
                logger.debug(f'updating with general planner')
                status = self.multi_planner.general_planner.get_unit_planner(self.unit).update_planned_actions()
            if self.unit.status.current_action.category == ActCategory.COMBAT:
                logger.debug(f'updating with combat planner')
                status = self.multi_planner.combat_planner.get_unit_planner(self.unit).update_planned_actions()
            if self.unit.status.current_action.category == ActCategory.MINE:
                logger.debug(f'updating with mine planner')
                status = self.multi_planner.mining_planner.get_unit_planner(self.unit).update_planned_actions()
            if self.unit.status.current_action.category == ActCategory.CLEAR:
                logger.debug(f'updating with clear planner')
                status = self.multi_planner.clearing_planner.get_unit_planner(self.unit).update_planned_actions()
            if status is None:
                raise ValueError(f"{self.unit.log_prefix} Failed to get updates")

            if self.unit.status.turn_status.planned_actions_require_update:
                logger.debug(f'copying action_queue to planned_action_queue (and statuses)')
                self.unit.status.update_planned_action_queue(self.unit.action_queue.copy(), self.unit.act_statuses.copy())

            # Check again if unit must move
            if status == HS.SUCCESS:
                if (
                    util.num_turns_of_actions(self.unit.status.planned_action_queue) > 0
                    or self.unit.status.turn_status.action_queue_empty_ok
                ):
                    logger.debug(f'planning successful')
                    # Good break here
                    break
                else:
                    logger.error(
                        f"{self.unit.log_prefix}, status SUCCESS, but empty planned actions and empty_actions_ok False"
                    )
        else:
            logger.error(f"{self.unit.log_prefix}, after X attempts at planning, status = {status}")

            mm_success = self._force_moving_if_necessary(unit_must_move)
            if not mm_success:
                logger.warning(f"{self.unit.log_prefix} status was must_move, but first step was not a move")
                status = HS.INVALID_FIRST_STEP

        return status


# class ActionImplementer:
#     def __init__(
#         self,
#         master,
#         unit_paths: UnitPaths,
#         unit_info: UnitInfo,
#         action_validator: ValidActionCalculator,
#         close_units: AllCloseUnits,
#         factory_desires: FactoryDesires,
#         factory_info: FactoryInfo,
#         mining_planner: MiningPlanner,
#         clearing_planner: ClearingPlanner,
#         combat_planner: CombatPlanner,
#     ):
#         self.master = master
#         self.unit_paths = unit_paths
#         self.unit_info = unit_info
#         self.action_validator = action_validator
#         self.close_units = close_units
#         self.factory_desires = factory_desires
#         self.factory_info = factory_info
#         self.mining_planner = mining_planner
#         self.clearing_planner = clearing_planner
#         self.combat_planner = combat_planner
#
#     def implement_desired_action(
#         self,
#         unit: FriendlyUnitManager,
#         desired_action: ActStatus,
#         unit_must_move: bool,
#     ):
#         if desired_action.category == ActCategory.COMBAT:
#             success = self.combat_planner.get_unit_planner(unit).update_planned_actions()
#         elif desired_action.category == ActCategory.MINE:
#             success = self.mining_planner.get_unit_planner(unit).update_planned_actions()
#         elif desired_action.category == ActCategory.CLEAR:
#             success = self.clearing_planner.get_unit_planner(unit).update_planned_actions()
#         elif desired_action.category == ActCategory.NOTHING:
#             success = self._do_nothing(unit, unit_must_move)
#         else:
#             logger.error(f"{desired_action} not understood as an action")
#             success = False
#
#         if success:
#             pass
#         else:
#             unit.status.update_action_status(new_action=ActStatus())
#         return success
#
#     def _do_nothing(self, unit, unit_must_move) -> bool:
#         logger.debug(f"Setting action queue to empty to do action {ActCategory.NOTHING}")
#         unit.action_queue = []
#         success = True
#         if unit_must_move:
#             if not unit.factory_id:
#                 logger.error(f"Unit must move, but has action {ActCategory.NOTHING} and no factory assigned")
#             else:
#                 success = self._handle_nothing_with_must_move(unit)
#         return success
#
#     def _handle_nothing_with_must_move(self, unit) -> bool:
#         if unit.on_own_factory():
#             success = util.move_to_new_spot_on_factory(
#                 self.master.pathfinder,
#                 unit,
#                 self.master.factories.friendly[unit.factory_id],
#             )
#             if not success:
#                 util.move_to_cheapest_adjacent_space(self.master.pathfinder, unit)
#         else:
#             util.move_to_cheapest_adjacent_space(self.master.pathfinder, unit)
#         return True


class MultipleUnitActionPlanner:
    # How far should distance map extend (padded with max value after that)
    max_distance_map_dist = 20
    # What is considered a close unit when considering future paths
    close_threshold = 4
    # If there will be a collision within this many steps consider acting
    check_enemy_collision_steps = 3
    # If there will be a collision within this many steps consider acting
    check_friendly_collision_steps = 30
    # Increase cost to travel near units based on kernel with this dist
    kernel_dist = 5
    # If this many actions the same, don't update unit (2 so that next action in queue matches planned actions)
    actions_same_check = 2
    # Number of steps to block other unit path locations for
    max_enemy_path_length = 50
    # Max pathing steps when calculating paths of all units (and for collision resolution)
    max_pathing_steps = 50

    def __init__(
        self,
        master: MasterState,
        general_planner: GeneralPlanner,
        mining_planner: MiningPlanner,
        clearing_planner: ClearingPlanner,
        combat_planner: CombatPlanner,
    ):
        """Assuming this is called after beginning of turn update"""
        self.master = master
        self.general_planner = general_planner
        self.mining_planner = mining_planner
        self.clearing_planner = clearing_planner
        self.combat_planner = combat_planner

        # Will be filled on update
        self.factory_desires: Dict[str, FactoryDesires] = None
        self.factory_infos: Dict[str, FactoryInfo] = None
        self.base_costmap: np.ndarray = None
        self.start_of_turn_paths: collisions.UnitPaths = None  # Enemy queue and Friendly planned queue
        self.current_paths: collisions.UnitPaths = None  # Enemy queue and Friendly that have already acted
        self.all_upcoming_collisions: Dict[str, collisions.AllCollisionsForUnit] = None
        self.all_close_units: AllCloseUnits = None
        self.ordered_units: Dict[str, FriendlyUnitManager] = None
        self.units_to_act: NewUnitsToAct = None

        # Stored for ease of access debugging
        self.debug_units_to_act_start = None
        self.debug_units_to_act = None
        self.debug_unit_infos = None
        self.debug_action_validator = None
        self.debug_single_action_planners = {}
        self.debug_existing_paths = None
        self.debug_actions_returned = None

    def update(
        self,
        # factory_infos: Dict[str, FactoryInfo],
        # factory_desires: Dict[str, FactoryDesires],
    ):
        """Beginning of turn update"""
        # self.factory_infos = factory_infos
        # self.factory_desires = factory_desires

        # Validate and replace enemy actions so that their move path is correct (i.e. cannot path through friendly
        # factories or off edge of map, so replace those moves with move.CENTER)
        self._replace_invalid_enemy_moves()

        # Update calculated things
        # Base travel costmap (i.e. with factories)
        self.base_costmap = self._calculate_base_costmap()

        # self.start_of_turn_infos = self._collect_unit_data(units_to_act.needs_to_act)
        # unit_infos.sort_by_priority()

        # Calculate collisions
        self.all_upcoming_collisions = self._calculate_collisions()

        # Calculate close units
        self.all_close_units = self._calculate_close_units()

        # Calculate start of turn unit paths
        self.start_of_turn_paths = self._get_unit_paths(
            friendly_units=self.master.units.friendly.all, enemy=True, max_step=10
        )

        # Calculate start of turn unit paths
        self.current_paths = self._get_unit_paths(friendly_units={}, enemy=True)
        self.master.pathfinder = Pather(self.base_costmap, self.current_paths)

        # Set up the NextActionValidCalculator
        self.action_validator = ValidActionCalculator(
            friendly_factories=self.master.factories.friendly,
            maps=self.master.maps,
            unit_paths=self.current_paths,
        )

        # Order units by which should act first
        self.ordered_units = self._order_units()

        # Keep track of which units have acted or updated actions with this
        self.units_to_act = NewUnitsToAct(
            needs_to_act=self.ordered_units.copy(), should_not_act={}, has_updated_actions={}
        )

        # Clear things that are only stored for ease of access debugging
        self.debug_units_to_act_start = None
        self.debug_units_to_act = None
        self.debug_unit_infos = None
        self.debug_action_validator = None
        self.debug_single_action_planners = {}
        self.debug_existing_paths = None
        self.debug_actions_returned = None

    def _get_unit_paths(self, friendly_units, enemy: bool = True, max_step=None) -> collisions.UnitPaths:
        if enemy:
            enemy_units = self.master.units.enemy.all
        else:
            enemy_units = {}
        max_step = max_step if max_step is not None else self.max_pathing_steps
        paths = collisions.UnitPaths(
            friendly=friendly_units,
            enemy=enemy_units,
            friendly_valid_move_map=self.master.maps.valid_friendly_move,
            enemy_valid_move_map=self.master.maps.valid_enemy_move,
            max_step=max_step,
            rubble=self.master.maps.rubble,
        )
        return paths

    def _calculate_close_units(self) -> AllCloseUnits:
        return AllCloseUnits.from_info(
            self.master.units,
            close_threshold=self.close_threshold,
            map_shape=self.master.maps.rubble.shape,
        )

    def _replace_invalid_enemy_moves(self):
        """Replace invalid (move) actions in enemy unit so invalid enemy paths don't mess up my stuff
        E.g. if enemy is pathing over a friendly factory or outside of map
        """
        friendly_factory_map = self.master.maps.factory_maps.friendly
        valid_move_map = np.ones_like(friendly_factory_map, dtype=bool)
        valid_move_map[friendly_factory_map >= 0] = False

        for unit_id, unit in self.master.units.enemy.all.items():
            valid_actions = unit.valid_moving_actions(
                costmap=valid_move_map,
                max_len=self.max_enemy_path_length,
                ignore_repeat=False,
            )
            if valid_actions.was_valid is False:
                logger.warning(
                    f"Enemy {unit_id} actions were invalid. First invalid at step {valid_actions.invalid_steps[0]}"
                )
                unit.action_queue = valid_actions.valid_actions

    # def _get_units_to_act(self, units: Dict[str, FriendlyUnitManager], close_units: AllCloseUnits) -> UnitsToAct:
    #     """
    #     Determines which units should potentially act this turn, and which should continue with current actions
    #     Does this based on:
    #         - collisions in next couple of turns
    #         - enemies nearby
    #         - empty action queue
    #
    #     Args:
    #         units: list of friendly units
    #
    #     Returns:
    #         Instance of UnitsToAct
    #     """
    #     logger.info(f"units_should_consider_acting called with len(units): {len(units)}")
    #
    #     all_unit_collisions = self._calculate_collisions()
    #     all_unit_close_to_enemy = close_units.close_to_enemy
    #     needs_to_act = {}
    #     should_not_act = {}
    #     for unit_id, unit in units.items():
    #         # Todo: this actually updates list of should_act_reasons in unit now
    #         should_act = should_unit_consider_acting(
    #             unit,
    #             upcoming_collisions=all_unit_collisions,
    #             close_enemies=all_unit_close_to_enemy,
    #         )
    #
    #         if should_act.should_act:
    #             needs_to_act[unit_id] = should_act
    #         else:
    #             should_not_act[unit_id] = should_act
    #     return UnitsToAct(needs_to_act=needs_to_act, should_not_act=should_not_act)

    def _calculate_collisions(self) -> Dict[str, collisions.AllCollisionsForUnit]:
        """Calculates the upcoming collisions based on action queues of all units"""
        all_collisions = collisions.calculate_collisions(
            self.master.units,
            rubble=self.master.maps.rubble,
            check_steps_enemy=self.check_enemy_collision_steps,
            check_steps_friendly=self.check_friendly_collision_steps,
        )
        return all_collisions

    # def _collect_unit_data(self, act_infos: Dict[str, ConsiderActInfo]) -> UnitInfos:
    #     """
    #     Collects data from units and stores it in a pandas dataframe.
    #
    #     Args:
    #         act_infos: List of ActInfo objects.
    #
    #     Returns:
    #         A pandas dataframe containing the unit data.
    #     """
    #     data = {}
    #     for unit_id, act_info in act_infos.items():
    #         unit = act_info.unit
    #         unit_info = UnitInfo.from_data(unit=unit, act_info=act_info)
    #         data[unit_id] = unit_info
    #     return UnitInfos(infos=data)

    def _calculate_base_costmap(self) -> np.ndarray:
        """
        Calculates the base costmap based on:
            - rubble array
            - Enemy factories impassible
            - Center of friendly factories (in case a unit is built there)

        Returns:
            A numpy array representing the costmap.
        """
        # Turn rubble into costmap
        costmap = self.master.maps.rubble.copy() * 0.1  # was 0.05
        costmap += 1  # Zeros aren't traversable

        # Block enemy factories
        enemy_factory_map = self.master.maps.factory_maps.enemy
        costmap[enemy_factory_map >= 0] = -1  # Not traversable

        # Add cost to factory waiting areas
        for factory in self.master.factories.friendly.values():
            costmap[factory.queue_array > 0] += 10

        # Make center of factories impassible (in case unit is built there)
        # TODO: Only block center if unit actually being built
        for factory_id, factory in self.master.factories.friendly.items():
            pos = factory.pos
            costmap[pos[0], pos[1]] = -1
        return costmap

    def _collect_changed_actions(self, units_to_act: NewUnitsToAct):
        unit_actions = {}
        for unit_id, unit in units_to_act.has_updated_actions.items():
            if len(unit.status.planned_action_queue) > 0:
                unit_actions[unit_id] = unit.status.planned_action_queue[:20]
            else:
                logger.warning(
                    f"Updating {unit_id} with empty actions (previous action len = "
                    f"{len(unit.start_of_turn_actions)}) previous_status = {unit.status.previous_action}"
                    f"(could be on purpose, but probably should figure out a better thing for this unit to do (even if stay still for a while first))"
                )
                if len(unit.start_of_turn_actions) == 0:
                    # no need to actually send empty as a command if already empty
                    continue
                unit_actions[unit_id] = []
        return unit_actions

    def _validate_changed_actions_against_action_space(self, unit_actions):
        for unit_id, actions in unit_actions.items():
            validated_actions = []
            for i, action in enumerate(actions):
                if valid_action_space(action):
                    validated_actions.append(action)
                else:
                    logger.error(f"Invalid action was {action} at position {i}")
                    break

            if len(validated_actions) == 0:
                validated_actions = [np.array([0, 0, 0, 0, 0, 1], dtype=int)]

            unit_actions[unit_id] = validated_actions

        return unit_actions

    def _assign_new_factory_if_necessary(self, unit: FriendlyUnitManager, factory_infos: Dict[str, FactoryInfo]):
        """If doesn't have a factory, assign it to an existing one"""
        if not unit.factory_id:
            best_factory = None
            best_space = -1
            for f_info in factory_infos.values():
                if f_info.connected_growable_space > best_space:
                    best_space = f_info.connected_growable_space
                    best_factory = f_info.factory
            unit.factory_id = best_factory.unit_id
            best_factory.assign_unit(unit)
            logger.warning(f"Re-assigning to {best_factory.unit_id} because no factory assigned")

    def _real_action_update(self, unit: FriendlyUnitManager, units_to_act: NewUnitsToAct):
        """If real unit actions should update, update with these, otherwise this will return None (i.e. if
        real action queue still matches planned actions for now)"""
        current_unit_actions = unit.start_of_turn_actions
        planned_actions = unit.status.planned_action_queue

        update_required = False
        # If first X actions are the same, don't update (unnecessary cost for unit)
        if np.all(
            np.array(current_unit_actions[: self.actions_same_check])
            == np.array(planned_actions[: self.actions_same_check])
        ):
            first_act = unit.start_of_turn_actions[0] if len(unit.start_of_turn_actions) > 0 else []
            logger.info(
                f"First {self.actions_same_check} real actions same ({first_act}), not updating unit action queue"
            )

            # Set the action_queue to what it will be (don't think this will actually get used again)
            update_required = False
            # unit.action_queue = planned_actions[:20]
            # units_to_act.should_not_act[unit.unit_id] = unit_info.act_info
        else:
            last_updated = self.master.step - unit.status.last_real_action_update_step
            logger.info(
                f"{unit.log_prefix} has updated actions "
                f"(last updated {last_updated} ago),"
                f"was {unit.status.current_action.previous_action.category}:{unit.status.current_action.previous_action.sub_category}, now {unit.status.current_action.category}:{unit.status.current_action.sub_category}"
                f" first few new actions are {planned_actions[:3]}, first few old actions were {unit.start_of_turn_actions[:3]}"
            )
            update_required = True

        units_to_act.needs_to_act.pop(unit.unit_id)
        if update_required:
            unit.status.last_real_action_update_step = self.master.step
            units_to_act.has_updated_actions[unit.unit_id] = unit
            return True
        else:
            units_to_act.should_not_act[unit.unit_id] = unit
            return False
            # unit.action_queue = planned_actions[:20]
            # unit.status.last_action_update_step = self.master.step
            # units_to_act.has_updated_actions[unit.unit_id] = unit_info.act_info

    def _check_additional_collisions(self, unit: FriendlyUnitManager, units_to_act: collisions.UnitsToAct):
        collisions_ = collisions.find_collisions(
            unit,
            [info.unit for info in units_to_act.has_updated_actions.values()],
            max_step=self.check_friendly_collision_steps,
            other_is_enemy=False,
            rubble=self.master.maps.rubble,
        )
        if len(collisions_) > 0:
            unit.status.turn_status.should_act_reasons.append(
                ShouldActInfo(reason=ActReasons.COLLISION_WITH_FRIENDLY, requires_action=True)
            )
        return collisions_

    def _order_units(self) -> Dict[str, FriendlyUnitManager]:
        """Return units in order of
        - Heavy first (even non acting)
            - Nothing units first  (they may have issues or are at center of factory)
            - Acting first
                - Below X power  (need more direct routes)
                - last step update (older have  higher priority so generally don't need to update)
            - Waiting units last
                - Above 90% power first
                - Then nearest to factory outward
        """
        units = self.master.units.friendly.all
        units_datas = []

        for unit_id, unit in units.items():
            current_action = (
                unit.status.planned_act_statuses[0] if len(unit.status.planned_act_statuses) > 0 else ActStatus()
            )
            currently_acting = (
                True if current_action.category not in [ActCategory.NOTHING, ActCategory.WAITING] else False
            )
            turns_left_in_plan = util.num_turns_of_actions(unit.status.planned_action_queue)
            turns_left_in_real = util.num_turns_of_actions(unit.start_of_turn_actions)
            action_is_nothing = True if current_action.category == ActCategory.NOTHING else False

            unit_data = {
                "unit_id": unit_id,
                "is_heavy": True if unit.unit_type == "HEAVY" else False,
                "nothing_action": action_is_nothing,
                "currently_acting": currently_acting,
                "below_15_power": unit.start_of_turn_power < unit.unit_config.BATTERY_CAPACITY * 0.15,
                "turns_left_in_plan": turns_left_in_plan,
                "turns_left_in_real": turns_left_in_real,
                "power": unit.start_of_turn_power,
                "last_real_update": unit.status.last_real_action_update_step,
                "above_90_power": unit.start_of_turn_power > unit.unit_config.BATTERY_CAPACITY * 0.9,
                "factory_dist": util.manhattan(unit.start_of_turn_pos, unit.factory.pos),
            }

            units_datas.append(unit_data)

        if units_datas:
            units_df = pd.DataFrame(units_datas)

            HIGHEST = False
            LOWEST = True

            # Sorting conditions as a dictionary
            sort_conditions = {
                "is_heavy": HIGHEST,
                "nothing_action": HIGHEST,
                "currently_acting": HIGHEST,
                "below_15_power": HIGHEST,
                "turns_left_in_plan": HIGHEST,
                "turns_left_in_real": LOWEST,
                "above_90_power": HIGHEST,
                "factory_dist": LOWEST,
            }

            # Sort the DataFrame based on the desired order
            units_df.sort_values(by=list(sort_conditions.keys()), ascending=list(sort_conditions.values()), inplace=True)

            # Create an ordered dictionary with the sorted unit IDs
            ordered_units = OrderedDict()
            for unit_id in units_df["unit_id"]:
                ordered_units[unit_id] = units[unit_id]
            return ordered_units
        else:
            return {}


    def decide_unit_actions(
        self,
    ) -> Dict[str, List[np.ndarray]]:
        logger.info(f"deciding all unit actions")

        # For each unit
        for unit_id, unit in self.ordered_units.items():
            if unit_id not in self.units_to_act.needs_to_act:
                logger.warning(f"{unit_id} not in needs_to_act, skipping")
                continue

            # Update planned actions in here
            unit_action_planner = SingleUnitActionPlanner(
                unit=unit,
                master=self.master,
                multi_planner=self,
                current_paths=self.current_paths,
                all_close_units=self.all_close_units,
                action_validator=self.action_validator,
                collision_resolve_max_step=self.check_friendly_collision_steps,
                collision_info=self.all_upcoming_collisions[unit_id],
            )
            status = unit_action_planner.calculate_actions_for_unit()
            self.debug_single_action_planners[unit_id] = unit_action_planner

            # update actual action queue if necessary and move to relevant place in units_to_act
            self._real_action_update(unit, self.units_to_act)

            # Make sure the next action is taken into account for next validation (e.g. if this unit is taking power)
            self.action_validator.add_next_action(unit)

            # And for next pathing
            self.current_paths.add_unit(unit, is_enemy=False)

        # Collect the actions send back to env
        unit_actions = self._collect_changed_actions(self.units_to_act)
        # Make sure they are at least valid against the action space
        unit_actions = self._validate_changed_actions_against_action_space(unit_actions)
        self.debug_actions_returned = unit_actions
        logger.info(f"Updating actions of {list(unit_actions.keys())}")
        return unit_actions
