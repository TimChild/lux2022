from __future__ import annotations
from enum import Enum

import abc
from collections import OrderedDict
import functools
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Union, Optional, Iterable

import numpy as np
import pandas as pd

import util
from actions_util import Actions
import actions_util
from config import get_logger
from factory_action_planner import FactoryDesires, FactoryInfo
from master_state import MasterState, AllUnits, Maps
from mining_planner import MiningPlanner
from new_path_finder import Pather
from rubble_clearing_planner import RubbleClearingPlanner
from combat_planner import CombatPlanner
from unit_manager import FriendlyUnitManager, UnitManager, EnemyUnitManager
from action_validation import ValidActionCalculator, valid_action_space

logger = get_logger(__name__)


def find_collisions(
    this_unit: UnitManager,
    other_units: Iterable[UnitManager],
    max_step: int,
    other_is_enemy: bool,
) -> Dict[str, Collision]:
    """Find the first collision point between this_unit and each other_unit
    I.e. Only first collision coordinate when comparing the two paths
    """
    this_path = this_unit.current_path(max_len=max_step)

    collisions = {}
    for other in other_units:
        # Don't include collisions with self
        if this_unit.unit_id == other.unit_id:
            continue
        other_path = other.current_path(max_len=max_step)

        # Note: zip stops iterating when end of a path is reached
        for i, (this_pos, other_pos) in enumerate(zip(this_path, other_path)):
            if np.array_equal(this_pos, other_pos):
                logger.debug(
                    f"Collision found at step {i} at pos {this_pos} between {this_unit.unit_id} and {other.unit_id}"
                )
                collision = Collision(
                    unit_id=this_unit.unit_id,
                    other_unit_id=other.unit_id,
                    other_unit_is_enemy=other_is_enemy,
                    pos=this_pos,
                    step=i,
                )
                collisions[other.unit_id] = collision
                # Don't find any more collisions for this path
                break
    return collisions


class CollisionResolver:
    # + extra so I can check where unit is heading
    extra_steps = 5

    def __init__(
        self,
        unit: FriendlyUnitManager,
        pathfinder: Pather,
        maps: Maps,
        unit_paths: UnitPaths,
        collisions: AllCollisionsForUnit,
        max_step: int,
    ):
        self.unit = unit
        self.pathfinder = pathfinder
        self.maps = maps
        self.unit_paths = unit_paths
        if collisions is None:
            raise NotImplementedError(f"Expected collisions, got None")
        self.collisions = collisions
        self.max_step = max_step

        # Calculate some things
        self.unit_actions = unit.status.planned_actions
        self.unit_path = self.unit.current_path(max_len=max_step, planned_actions=True)

    def _collision_while_travelling(self, collision: Collision) -> bool:
        """Is the collision part way through travelling a-b
        Basically is the step after somewhere else and not final dest?
        """
        was_moving = actions_util.was_unit_moving_at_step(
            self.unit_actions, collision.step
        )
        will_be_moving = actions_util.will_unit_move_at_step(
            self.unit_actions, collision.step
        )
        travelling = was_moving and will_be_moving
        if travelling:
            logger.debug(
                f"{self.unit.unit_id} is travelling at collision step {collision.step}, at {collision.pos}"
            )
        return travelling

    def _collision_on_factory(self, collision: Collision) -> bool:
        """Is the collision on a friendly factory tile?"""
        at_factory = (
            self.maps.factory_maps.friendly[collision.pos[0], collision.pos[1]] >= 0
        )
        if at_factory:
            logger.debug(
                f"{self.unit.unit_id} is at factory at collision step {collision.step}, at {collision.pos}"
            )
        return False

    def _collision_at_destination(self, collision: Collision) -> bool:
        """Is the collision at the destination of unit (i.e. doesn't move next step)"""
        will_be_moving = actions_util.will_unit_move_at_step(
            self.unit_actions, collision.step
        )
        at_destination = not will_be_moving
        if at_destination:
            logger.debug(
                f"{self.unit.unit_id} is at destination at collision step {collision.step}, at {collision.pos}"
            )
        return at_destination

    def _next_dest_or_last_step(self, step: int) -> Tuple[int, util.POS_TYPE]:
        """Return the pos of the next dest or last move step after step
        Note: index returned is step of path (accounting for start step) (i.e. 0 == now, 1 == first action in list)
        """
        index = actions_util.find_dest_step_from_step(
            self.unit_path, step, direction="forward"
        )
        return index, self.unit_path[index]

    def _previous_dest_or_start_step(self, step: int) -> Tuple[int, util.POS_TYPE]:
        """Return the pos of the previous dest before step, or start pos if no previous dest"""
        index = actions_util.find_dest_step_from_step(
            self.unit_path, step, direction="backward"
        )
        return index, self.unit_path[index]

    def _replace_unit_actions(self, start_step, end_step, new_actions):
        """Replace the actions from start step to end step with new actions
        Note: New actions may have different length"""
        #  update actions
        existing_actions = self.unit_actions
        replaced_actions = actions_util.replace_actions(
            existing_actions, start_step, end_step, new_actions
        )
        self.unit.action_queue = replaced_actions
        self.unit_actions = replaced_actions

    def _resolve_travel_collision(self, collision: Collision) -> Actions:
        """Repath to later point on path (ideally destination)"""
        last_step, next_dest_or_last_step = self._next_dest_or_last_step(collision.step)
        first_step, prev_dest_or_first_step = self._previous_dest_or_start_step(
            collision.step
        )
        if np.all(next_dest_or_last_step == collision.pos) or np.all(
            prev_dest_or_first_step == collision.step
        ):
            logger.error(
                f"first or last dest was same as collision pos next={next_dest_or_last_step} prev={prev_dest_or_first_step} collision step={collision.step}"
            )
            return Actions.CONTINUE_UPDATE
        cm = self.pathfinder.generate_costmap(self.unit, override_step=first_step)
        new_path = self.pathfinder.fast_path(
            prev_dest_or_first_step, next_dest_or_last_step, costmap=cm
        )
        if len(new_path) == 0:
            logger.warning(
                f"failed to find new path from {prev_dest_or_first_step} to {next_dest_or_last_step} starting step {first_step}"
            )
            return Actions.CONTINUE_UPDATE

        new_actions = util.path_to_actions(new_path)
        self._replace_unit_actions(first_step, last_step, new_actions)
        return Actions.CONTINUE_RESOLVED

    def _resolve_factory_collision(self, collision: Collision) -> Actions:
        """Repath to new spot on factory if possible"""
        first_step, prev_dest_or_first_step = self._previous_dest_or_start_step(
            collision.step
        )
        factory_num = self.maps.factory_maps.all[collision.pos[0], collision.pos[1]]
        if factory_num < 0:
            raise ValueError(f"collision not on factory")

        factory_loc = (self.maps.factory_maps.all == factory_num).astype(int)
        cm = self.pathfinder.generate_costmap(self.unit, override_step=first_step)
        new_path = util.path_to_factory_edge_nearest_pos(
            self.pathfinder,
            factory_loc,
            prev_dest_or_first_step,
            prev_dest_or_first_step,
            cm,
        )
        if len(new_path) == 0:
            logger.warning(
                f"failed to find new path from {prev_dest_or_first_step} to factory_{factory_num} starting step {first_step}"
            )
            return Actions.CONTINUE_UPDATE

        new_actions = util.path_to_actions(new_path)
        self._replace_unit_actions(first_step, collision.step + 1, new_actions)
        return Actions.CONTINUE_RESOLVED

    def _resolve_destination_collision(self, collision: Collision) -> Actions:
        """Ideally repath to new nearby resource or something"""
        logger.info(f"resolving conflict at destination required but not implemented")
        return Actions.CONTINUE_UPDATE

    def resolve(self) -> Actions:
        """If there is a collision with a friendly or enemy coming up, re-path just the part that avoids the collision
        1. Moving from a-b
        2. Collision on factory
        3. Collision on destination (resource/rubble)
        """
        # Note: Can only resolve 1 collision at the moment, because the change to the action queue will screw up the
        # collision.step etc... So just solving  the nearest collision (if there are multiple, this can get called a
        # few turns in a row

        # Find nearest collision
        nearest_collision = None
        nearest_step = 999
        for other_id, collision in self.collisions.with_friendly.all.items():
            if collision.step < nearest_step and collision.step < self.max_step:
                nearest_collision = collision
                nearest_step = collision.step

        if nearest_collision is None:
            logger.error(f"No collisions to solve in {self.max_step} step")
            return Actions.CONTINUE_UPDATE

        logger.info(
            f"resolving collision between {self.unit.unit_id} and {nearest_collision.unit_id} at {nearest_collision.pos} at step {nearest_collision.step}"
        )
        suggested_action = Actions.CONTINUE_UPDATE
        if self._collision_while_travelling(nearest_collision):
            suggested_action = self._resolve_travel_collision(nearest_collision)
        elif self._collision_on_factory(nearest_collision):
            suggested_action = self._resolve_factory_collision(nearest_collision)
        elif self._collision_at_destination(nearest_collision):
            suggested_action = self._resolve_destination_collision(nearest_collision)
        else:
            suggested_action = Actions.CONTINUE_UPDATE
        if suggested_action != Actions.CONTINUE_RESOLVED:
            logger.info(
                f"Failed to solve collision {nearest_collision}, route still needs updating"
            )
            return Actions.CONTINUE_UPDATE
        logger.info(f"Nearest collisions solved, can continue")
        return Actions.CONTINUE_RESOLVED


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
            last_action_update_step=unit.status.last_action_update_step,
            len_action_queue=len(unit.action_queue),
            distance_to_factory=util.manhattan(unit.start_of_turn_pos, unit.factory.pos)
            if unit.factory
            else None,
            is_heavy=unit.unit_type == "HEAVY",
            unit_type=unit.unit_type,
            enough_power_to_move=(
                unit.power
                > unit.unit_config.MOVE_COST + unit.unit_config.ACTION_QUEUE_POWER_COST
            ),
            power=unit.power,
            ice=unit.cargo.ice,
            ore=unit.cargo.ore,
            power_over_20_percent=unit.start_of_turn_power
            > unit.unit_config.BATTERY_CAPACITY * 0.2,
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
                f"last_acted_step={series.last_action_update_step}, power={series.power}, ice={series.ice}, ore={series.ore}, len_acts={series.len_action_queue}"
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


@dataclass
class UnitsToAct:
    needs_to_act: dict[str, ConsiderActInfo]
    should_not_act: dict[str, ConsiderActInfo]
    has_updated_actions: dict[str, ConsiderActInfo] = field(default_factory=dict)

    def get_act_info(self, unit_id: str) -> ConsiderActInfo:
        for d in [self.needs_to_act, self.should_not_act, self.has_updated_actions]:
            if unit_id in d:
                return d[unit_id]
        raise KeyError(f"{unit_id} not in UnitsToAct")


@dataclass
class Collision:
    """First collision only between units"""

    unit_id: str
    other_unit_id: str
    other_unit_is_enemy: bool
    pos: Tuple[int, int]
    step: int


@dataclass
class CollisionsForUnit:
    """Collection of collisions with other units for given unit"""

    # Dict[other_id: Collision]
    light: Dict[str, Collision]
    heavy: Dict[str, Collision]

    @property
    def all(self) -> Dict[str, Collision]:
        return dict(**self.light, **self.heavy)


@dataclass
class AllCollisionsForUnit:
    """All collisions  with other units for given unit"""

    with_friendly: CollisionsForUnit
    with_enemy: CollisionsForUnit

    @property
    def all(self) -> Dict[str, Collision]:
        return dict(**self.with_friendly.all, **self.with_enemy.all)

    def next_collision(self) -> Collision:
        nearest_collision = None
        nearest_step = 999
        for enemy_id, collision in self.all.items():
            if collision.step < nearest_step:
                nearest_collision = collision
                nearest_step = collision.step
        if nearest_collision is None:
            nearest_collision = Collision('none', 'none', False, (-1,  -1),  -1)
        return nearest_collision

    def num_collisions(self, friendly=True, enemy=True):
        num = 0
        if enemy:
            num += len(self.with_enemy.light)
            num += len(self.with_enemy.heavy)
        if friendly:
            num += len(self.with_friendly.light)
            num += len(self.with_friendly.heavy)
        return num


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
        return self.other_units[
            self.other_unit_distances.index(min(self.other_unit_distances))
        ]


@dataclass
class AllCloseUnits:
    """Collection of close unit info for all units"""

    close_to_friendly: Dict[str, CloseUnits]
    close_to_enemy: Dict[str, CloseUnits]
    close_threshold: int

    @classmethod
    def from_info(
        cls, all_units: AllUnits, close_threshold: int, map_shape: Tuple[int, int]
    ):
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
        large_array = np.zeros(map_shape)
        unit_distance_map = util.pad_and_crop(
            util.manhattan_kernel(30),
            large_arr=large_array,
            x1=unit.start_of_turn_pos[0],
            y1=unit.start_of_turn_pos[1],
            fill_value=35,
        )
        return unit_distance_map


@dataclass
class UnitPaths:
    """Unit paths stored in a 3D array (step, x, y) where value is id_num (otherwise -1)"""

    friendly_valid_move_map: np.ndarray
    enemy_valid_move_map: np.ndarray
    friendly_light: np.ndarray
    friendly_heavy: np.ndarray
    enemy_light: np.ndarray
    enemy_heavy: np.ndarray
    max_step: int

    def __post_init__(self):
        # For use when indexing to create costmaps
        x, y = np.meshgrid(
            np.arange(self.friendly_valid_move_map.shape[0]),
            np.arange(self.friendly_valid_move_map.shape[1]),
            indexing="ij",
        )
        self._x, self._y = x, y
        self._friendly_blur_kernel = self._friendly_blur_kernel()
        self._enemy_blur_kernel = self._enemy_blur_kernel()
        self._friendly_collision_kernel = self._friendly_collision_blur()
        self._enemy_collision_kernel = self._enemy_collision_blur()

    @property
    def all(self):
        """
        There *SHOULD* only be one unit in any place at any time, so this *SHOULD* be ok to do
        """
        return np.sum(
            [
                self.friendly_light,
                self.friendly_heavy,
                self.enemy_light,
                self.enemy_heavy,
            ],
            axis=0,
        )

    def _enemy_blur_kernel(self):
        blur_size = 3
        blur_kernel = (2 / 0.5) * 0.5 ** util.manhattan_kernel(blur_size)
        # blur_kernel = (
        #     util.pad_and_crop(
        #         adj_kernel, blur_kernel, blur_size, blur_size, fill_value=1
        #     )
        #     * blur_kernel
        # )
        # blur_kernel[blur_kernel < 0] = -2
        return blur_kernel

    def _enemy_collision_blur(self):
        adj_kernel = (util.manhattan_kernel(1) <= 1).astype(int) * -1
        adj_kernel[adj_kernel >= 0] = 0
        return adj_kernel

    def _friendly_blur_kernel(self):
        blur_size = 3
        blur_kernel = (1 / 0.5) * 0.5 ** util.manhattan_kernel(blur_size)
        blur_kernel[blur_size, blur_size] = -1
        return blur_kernel

    def _friendly_collision_blur(self):
        adj_kernel = (util.manhattan_kernel(0) <= 1).astype(int) * -1
        adj_kernel[adj_kernel >= 0] = 0
        return adj_kernel

    @classmethod
    def from_units(
        cls,
        friendly: Dict[str, FriendlyUnitManager],
        enemy: Dict[str, EnemyUnitManager],
        friendly_valid_move_map,
        enemy_valid_move_map,
        max_step,
    ):
        friendly_light = {
            unit.id_num: unit for unit in friendly.values() if unit.unit_type == "LIGHT"
        }
        friendly_heavy = {
            unit.id_num: unit for unit in friendly.values() if unit.unit_type == "HEAVY"
        }
        enemy_light = {
            unit.id_num: unit for unit in enemy.values() if unit.unit_type == "LIGHT"
        }
        enemy_heavy = {
            unit.id_num: unit for unit in enemy.values() if unit.unit_type == "HEAVY"
        }

        map_shape = friendly_valid_move_map.shape

        # +1 so that last layer is always left empty of units (for longer pathing)
        arrays = [
            np.full(
                (max_step + 1, map_shape[0], map_shape[1]), fill_value=-1, dtype=int
            )
            for _ in range(4)
        ]
        for array, dict_, move_map, is_enemy in zip(
            arrays,
            [friendly_light, friendly_heavy, enemy_light, enemy_heavy],
            [
                friendly_valid_move_map,
                friendly_valid_move_map,
                enemy_valid_move_map,
                enemy_valid_move_map,
            ],
            [False, False, True, True],
        ):
            for unit_num, unit in dict_.items():
                valid_path = unit.valid_moving_actions(
                    costmap=move_map, max_len=max_step
                )
                path = util.actions_to_path(
                    unit.start_of_turn_pos,
                    actions=valid_path.valid_actions,
                    max_len=max_step,
                )

                cls._add_path_to_array(unit, path, array, max_step, is_enemy=is_enemy)

        return cls(
            friendly_valid_move_map=friendly_valid_move_map,
            enemy_valid_move_map=enemy_valid_move_map,
            friendly_light=arrays[0],
            friendly_heavy=arrays[1],
            enemy_light=arrays[2],
            enemy_heavy=arrays[3],
            max_step=max_step,
        )

    @staticmethod
    def _add_path_to_array(
        unit: UnitManager, path, arr: np.ndarray, max_step: int, is_enemy: bool
    ):
        x, y = unit.start_of_turn_pos
        # # If first coord of path is x, y (which it should be), remove it (next pathing step will be step 1 in paths)
        # if np.all(path[0] == (x, y)):
        #     path = path[1:]

        # At least leave friendly units on map for one more turn (otherwise they get walked over while on factory)
        max_extra = 10 if is_enemy else 1
        extra = 0
        for step in range(max_step):
            if step < len(path):
                x, y = path[step]
                arr[step, x, y] = unit.id_num

            # Use last enemy position as position for a few extra turns (so they can't hide by standing still)
            elif x is not None and extra < max_extra:
                arr[step, x, y] = unit.id_num
                extra += 1
            # Don't do that for too long (probably not true) or for friendly (friendly will act again)
            else:
                break

    def add_unit(self, unit: UnitManager, is_enemy=False):
        """Add a new unit to the path arrays"""
        if is_enemy:
            move_map = self.enemy_valid_move_map
            if unit.unit_type == "LIGHT":
                array = self.enemy_light
            else:
                array = self.enemy_heavy
        else:
            move_map = self.friendly_valid_move_map
            if unit.unit_type == "LIGHT":
                array = self.friendly_light
            else:
                array = self.friendly_heavy

        unit_num = unit.id_num
        max_step = self.friendly_light.shape[0]

        # Calculate the valid path (i.e. can't walk of edge of map or through enemy factory, doesn't consider energy)
        # Especially important for enemy units... Don't want to deal with invalid paths later
        valid_path = unit.valid_moving_actions(costmap=move_map, max_len=max_step)

        # Get the valid path coords (first value is current position)
        path = util.actions_to_path(
            unit.start_of_turn_pos, actions=valid_path.valid_actions, max_len=max_step
        )

        # Add that path to the 3D path array
        self._add_path_to_array(unit, path, array, max_step, is_enemy=is_enemy)

    def to_costmap(
        self,
        pos: util.POS_TYPE,
        start_step: int,
        exclude_id_nums: Union[int, List[int]],
        friendly_light: bool,
        friendly_heavy: bool,
        enemy_light: bool,
        enemy_heavy: bool,
        enemy_collision_cost_value=-1,
        friendly_collision_cost_value=-1,
        enemy_nearby_start_cost: Optional[float] = 2,
        friendly_nearby_start_cost: Optional[float] = 1,
        step_dropoff_multiplier=0.92,
        true_intercept=False,
    ):
        """Create a costmap from a specific position at a specific step
        Args:
            true_intercept: True disables multi-step and only adds cost where collisions would occur if travelling shortest manhattan dist
            nearby_start_cost: None disables mutistep checking, 0 enables mutistep, but no extra cost from being near

        """
        if (
            friendly_nearby_start_cost is not None and friendly_nearby_start_cost < 0
        ) or (
            friendly_nearby_start_cost is not None and friendly_nearby_start_cost < 0
        ):
            raise ValueError(f"Nearby start cost must be positive, zero, or None")
        close_encounters_dicts = [
            self.calculate_likely_unit_collisions(
                pos=pos,
                start_step=start_step,
                exclude_id_nums=exclude_id_nums,
                friendly_light=friendly_light,
                friendly_heavy=friendly_heavy,
                enemy_light=enemy_light,
                enemy_heavy=enemy_heavy,
            )
        ]
        if (
            enemy_nearby_start_cost or friendly_nearby_start_cost
        ) and not true_intercept:
            close_encounters_dicts.extend(
                [
                    self.calculate_likely_unit_collisions(
                        pos=pos,
                        start_step=start_step + i,
                        exclude_id_nums=exclude_id_nums,
                        friendly_light=friendly_light,
                        friendly_heavy=friendly_heavy,
                        enemy_light=enemy_light,
                        enemy_heavy=enemy_heavy,
                    )
                    for i in range(1, 5)
                ]
            )

        # setup map
        cm = np.ones_like(self.friendly_valid_move_map, dtype=float)

        # collect blurs and collision kernels
        enemy_blur = self._enemy_blur_kernel
        enemy_collision = self._enemy_collision_kernel
        friendly_blur = self._friendly_blur_kernel
        friendly_collision = self._friendly_collision_kernel
        for i, close_dict in enumerate(close_encounters_dicts):
            for k, arr in close_dict.items():
                # Don't blur or make collision mask as big for friendly
                if k.startswith("friendly"):
                    collision_cost = friendly_collision_cost_value
                    nearby_start_cost = friendly_nearby_start_cost
                    blur = friendly_blur.copy()
                    collision_kernel = friendly_collision
                # For enemy it depends if we are trying to collide or avoid
                else:
                    collision_cost = enemy_collision_cost_value
                    nearby_start_cost = enemy_nearby_start_cost
                    blur = enemy_blur.copy()
                    # Trying to avoid
                    if enemy_collision_cost_value <= 0:
                        # blocks adjacent
                        collision_kernel = enemy_collision
                    # Trying to collide
                    else:
                        # only selects intercepts
                        collision_kernel = friendly_collision

                # Make 1s wherever units present
                arr = (arr >= 0).astype(float)

                # Do nearby blur if not None or 0
                if (
                    enemy_nearby_start_cost is not None
                    and enemy_nearby_start_cost
                    and not true_intercept
                ):
                    blur *= nearby_start_cost * step_dropoff_multiplier ** (
                        start_step + i
                    )
                    # Blur that with kernel
                    add_cm = util.convolve_array_kernel(arr, blur, fill=0)
                    # Add non-blocking costs
                    cm[cm > 0] += add_cm[cm > 0]

                # Calculate likely collisions
                col_cm = util.convolve_array_kernel(arr, collision_kernel, fill=0)

                # Set blocking (or targeting) costs
                if i > 2 and collision_cost < 0:
                    # already expect to be off by a few steps, just make collision spaces expensive, not blocked
                    # unless already blocked!
                    cm[(col_cm < 0) & (cm > 0)] = 5
                else:
                    cm[col_cm < 0] = collision_cost
        return cm

    def calculate_likely_unit_collisions(
        self,
        pos: util.POS_TYPE,
        start_step: int,
        exclude_id_nums: Union[int, List[int]],
        friendly_light: bool,
        friendly_heavy: bool,
        enemy_light: bool,
        enemy_heavy: bool,
    ) -> Dict[str, np.ndarray]:
        """
        Creates 2D array of unit ID nums where their path is manhattan distance from pos, only keeps first unit_id
        at location (so all valid, but can't see where things cross paths)
        """
        if isinstance(exclude_id_nums, int):
            exclude_id_nums = [exclude_id_nums]

        do_calcs = True
        if not 0 <= start_step < self.max_step:
            if start_step < 0:
                logger.error(f"{start_step} must be between 0 and {self.max_step}")
                start_step = 0
            elif start_step >= self.max_step:
                logger.debug(
                    f"Requesting map outside of max_steps {self.max_step} returning empty"
                )
                do_calcs = False

        if do_calcs:
            # Distance kernel
            kernel = util.manhattan_kernel(self.max_step - start_step) + start_step
            kernel[kernel > self.max_step] = self.max_step

            # Place that in correct spot
            index_array = util.pad_and_crop(
                kernel,
                self.friendly_valid_move_map,
                pos[0],
                pos[1],
                fill_value=self.max_step - start_step,
            )

        # Start with empty
        x, y = self._x, self._y
        likely_collision_id_maps = OrderedDict()
        teams, utypes = self._get_teams_and_utypes(
            friendly_light, friendly_heavy, enemy_light, enemy_heavy
        )
        for team, utype in zip(teams, utypes):
            key = f"{team}_{utype}"
            arr = getattr(self, key)
            if do_calcs:
                cm = arr[index_array, x, y]
                cm = np.where(np.isin(cm, exclude_id_nums), -1, cm)
            else:
                cm = arr[-1, x, y]
            likely_collision_id_maps[key] = cm

        return likely_collision_id_maps

    def _get_teams_and_utypes(
        self,
        friendly_light: bool,
        friendly_heavy: bool,
        enemy_light: bool,
        enemy_heavy: bool,
    ) -> Tuple[List[str], List[str]]:
        teams = ["friendly"] * (friendly_light + friendly_heavy)
        teams.extend(["enemy"] * (enemy_light + enemy_heavy))
        utypes = []
        if friendly_light:
            utypes.append("light")
        if friendly_heavy:
            utypes.append("heavy")
        if enemy_light:
            utypes.append("light")
        if enemy_heavy:
            utypes.append("heavy")
        return teams, utypes


def store_all_paths_to_array(
    paths: Dict[int, np.ndarray], map_shape, max_step=30, fill_value: int = -1
) -> np.ndarray:
    """
    Fill a 3D numpy array with keys corresponding to each path.

    Args:
        paths (Dict[Any, np.ndarray]): A dictionary of paths, where the key is the unit_id num and the value is an (N, 2) numpy array representing the path.
        map_shape: Shape of the map the paths are in (e.g. rubble.shape)
        max_step: How many steps in the paths to include
        fill_value (int, optional): The value to fill the array initially. Defaults to -1.

    Returns:
        np.ndarray: A 3D numpy array with shape (max_step, x_size, y_size) containing the path keys.
    """
    # Initialize a 3D array filled with fill_value (or any other invalid key) with shape (max_step, x_size, y_size)
    filled_array = np.full((max_step, map_shape[0], map_shape[1]), fill_value)

    # Fill the 3D array with the keys corresponding to each path
    for key, path in paths.items():
        for step, (x, y) in enumerate(path):
            filled_array[step, x, y] = key

    return filled_array


def calculate_collisions(
    all_units: AllUnits, check_steps_enemy: int = 5, check_steps_friendly: int = 20
) -> Dict[str, AllCollisionsForUnit]:
    """Calculate first collisions in the next <check_steps> for all units"""
    all_unit_collisions = {}
    for unit_id, unit in all_units.friendly.all.items():
        # print(f'checking {unit_id}')
        collisions_for_unit = AllCollisionsForUnit(
            with_friendly=CollisionsForUnit(
                light=find_collisions(
                    unit,
                    all_units.friendly.light.values(),
                    max_step=check_steps_friendly,
                    other_is_enemy=False,
                ),
                heavy=find_collisions(
                    unit,
                    all_units.friendly.heavy.values(),
                    max_step=check_steps_friendly,
                    other_is_enemy=False,
                ),
            ),
            with_enemy=CollisionsForUnit(
                light=find_collisions(
                    unit,
                    all_units.enemy.light.values(),
                    max_step=check_steps_enemy,
                    other_is_enemy=True,
                ),
                heavy=find_collisions(
                    unit,
                    all_units.enemy.heavy.values(),
                    max_step=check_steps_enemy,
                    other_is_enemy=True,
                ),
            ),
        )
        if collisions_for_unit.num_collisions(friendly=True, enemy=True) > 0:
            all_unit_collisions[unit_id] = collisions_for_unit
    return all_unit_collisions


class ActionDecider:
    def __init__(
        self,
        unit: FriendlyUnitManager,
        unit_info: UnitInfo,
        action_validator: ValidActionCalculator,
        factory_desires: FactoryDesires,
        factory_info: FactoryInfo,
        close_enemy_units: Union[None, CloseUnits],
    ):
        self.unit = unit
        self.unit_info = unit_info
        self.action_validator = action_validator
        self.factory_desires = factory_desires
        self.factory_info = factory_info
        self.close_units = close_enemy_units

    def _possible_switch_to_attack(self) -> Optional[Actions]:
        """Potentially switch to attacking if an enemy wanders by"""
        # Only if something nearby
        if self.close_units is not None:
            # And not doing something important
            if not self.unit.status.current_action in [
                Actions.MINE_ICE,
                Actions.MINE_ORE,
                Actions.ATTACK,
            ]:
                # And has a reasonable amount of energy
                if (
                    self.unit.start_of_turn_power
                    > self.unit.unit_config.BATTERY_CAPACITY * 0.3
                ):
                    # And did not change actions in the last few turns
                    if (
                        self.unit.master.step - self.unit.status.last_action_update_step
                        > 5
                    ):
                        # And nearby enemy is equal type with lower power
                        closest = self.close_units.closest()
                        if (
                            closest.unit_type == self.unit.unit_type
                            and closest.power < self.unit.start_of_turn_power
                        ):
                            logger.warning(
                                f"{self.unit.log_prefix} switching to attack"
                            )
                            return Actions.ATTACK
        return None

    def _decide_noops(self, unit_must_move: bool):
        act_reason = self.unit_info.act_info.reason

        # Validation only
        if act_reason in [
            ActReasons.NEXT_ACTION_PICKUP,
            ActReasons.NEXT_ACTION_TRANSFER,
            ActReasons.NEXT_ACTION_DIG,
        ]:
            if self.action_validator.next_action_valid(self.unit):
                logger.debug("Next pickup, transfer, dig is valid, do not update")
                return Actions.CONTINUE_NO_CHANGE
            else:
                logger.debug(
                    "Next pickup, transfer, dig not valid, continue but update plan"
                )
                return Actions.CONTINUE_UPDATE

        # Avoid collision with friendly
        if act_reason in [
            ActReasons.COLLISION_WITH_FRIENDLY,
        ]:
            logger.debug("Need to avoid collision with friendly")
            return Actions.CONTINUE_UPDATE

        # Close enemies don't matter if running away (as long as not colliding)
        if (
            self.unit.status.current_action == Actions.RUN_AWAY
            and act_reason == ActReasons.CLOSE_TO_ENEMY
        ):
            logger.debug("Already running away, no change necessary")
            return Actions.CONTINUE_NO_CHANGE

        # Heavy doesn't care about enemy light
        if (
            act_reason == ActReasons.CLOSE_TO_ENEMY
            and self.unit.unit_type == "HEAVY"
            and all([t == "LIGHT" for t in self.close_units.other_unit_types])
        ):
            logger.debug("Heavy doesn't care about light enemies, continuing path")
            return Actions.CONTINUE_NO_CHANGE

        # Previous action invalid can probably just update plan with current action
        if act_reason == ActReasons.PREVIOUS_ACTION_INVALID:
            logger.debug(
                "Previous action was invalid, may need to update plan of current role"
            )
            return Actions.CONTINUE_UPDATE

        # If already attacking, just update in case new path to enemy
        if (
            act_reason in [ActReasons.CLOSE_TO_ENEMY, ActReasons.COLLISION_WITH_ENEMY]
            and self.unit.status.current_action == Actions.ATTACK
        ):
            return Actions.CONTINUE_UPDATE

        if act_reason in [
            ActReasons.NEXT_ACTION_INVALID_MOVE,
            ActReasons.NEXT_ACTION_INVALID,
        ]:
            # If must move, is next action a move anyway (no move not checked in validator, other moves are)
            condition = self.unit.next_action_is_move() if unit_must_move else True
            if self.action_validator.next_action_valid(self.unit) and condition:
                logger.debug(
                    "Next action passed validation, suggesting no action update"
                )
                return Actions.CONTINUE_NO_CHANGE
            else:
                logger.debug("Next action not valid, suggesting keep role but update")
                return Actions.CONTINUE_UPDATE
        return None

    def _decide_light_unit_action(self, unit_must_move: bool) -> Actions:
        logger.debug(f"Deciding between light unit actions")
        action = self._possible_switch_to_attack()
        if action is not None:
            return action
        action = self._decide_noops(unit_must_move)
        if action is not None:
            return action
        action = self._decide_unit_action_based_on_factory_needs(
            self.factory_desires.light_mining_ore,
            self.factory_info.light_mining_ore,
            self.factory_desires.light_clearing_rubble,
            self.factory_info.light_clearing_rubble,
            self.factory_desires.light_mining_ice,
            self.factory_info.light_mining_ice,
            self.factory_desires.light_attacking,
            self.factory_info.light_attacking,
        )
        return action

    def _decide_heavy_unit_action(self, unit_must_move: bool) -> Actions:
        logger.debug(f"Deciding between heavy unit actions")
        action = self._possible_switch_to_attack()
        if action is not None:
            return action
        action = self._decide_noops(unit_must_move)
        if action is not None:
            return action
        if action is None:
            action = self._decide_unit_action_based_on_factory_needs(
                self.factory_desires.heavy_mining_ore,
                self.factory_info.heavy_mining_ore,
                self.factory_desires.heavy_clearing_rubble,
                self.factory_info.heavy_clearing_rubble,
                self.factory_desires.heavy_mining_ice,
                self.factory_info.heavy_mining_ice,
                self.factory_desires.heavy_attacking,
                self.factory_info.heavy_attacking,
            )
        return action

    def _decide_unit_action_based_on_factory_needs(
        self,
        desired_mining_ore: int,
        current_mining_ore: int,
        desired_clearing_rubble: int,
        current_clearing_rubble: int,
        desired_mining_ice: int,
        current_mining_ice: int,
        desired_attacking: int,
        current_attacking: int,
    ) -> Actions:
        if (
            not self.unit_info.unit.on_own_factory()
            and self.unit_info.unit.status.current_action != Actions.NOTHING
        ):
            action = self.unit_info.unit.status.current_action
            logger.debug(
                f"Unit NOT on factory and currently assigned, should continue same job ({action})"
            )
        else:
            logger.debug(
                f"Unit on factory, can decide a new type of action depending on factory needs"
            )

            self.factory_info.remove_unit_from_current_count(self.unit_info.unit)
            if current_mining_ice < desired_mining_ice:
                action = Actions.MINE_ICE
            elif current_mining_ore < desired_mining_ore:
                action = Actions.MINE_ORE
            elif current_clearing_rubble < desired_clearing_rubble:
                action = Actions.CLEAR_RUBBLE
            elif current_attacking < desired_attacking:
                action = Actions.ATTACK
            else:
                action = Actions.NOTHING
        return action

    def decide_action(self, unit_must_move: bool) -> Actions:
        logger.info(f"Deciding action for {self.unit_info.unit_id}")
        if self.unit_info.unit_type == "LIGHT":
            action = self._decide_light_unit_action(unit_must_move)
        else:  # unit_type == "HEAVY"
            action = self._decide_heavy_unit_action(unit_must_move)

        logger.debug(f"action should be {action}")
        return action


class ActReasons(Enum):
    NOT_ENOUGH_POWER = "not enough power"
    NO_ACTION_QUEUE = "no action queue"
    CURRENT_STATUS_NOTHING = "currently no action"
    COLLISION_WITH_ENEMY = "collision with enemy"
    COLLISION_WITH_FRIENDLY = "collision with friendly"
    CLOSE_TO_ENEMY = "close to enemy"
    NEXT_ACTION_INVALID = "next action invalid"
    NEXT_ACTION_INVALID_MOVE = "next action invalid move"
    NEXT_ACTION_PICKUP = "next action pickup"
    NEXT_ACTION_DIG = "next action dig"
    NEXT_ACTION_TRANSFER = "next action transfer"
    NO_REASON_TO_ACT = "no reason to act"
    ATTACKING = "attacking"
    PREVIOUS_ACTION_INVALID = "previous action was invalid"


@dataclass
class ConsiderActInfo:
    unit: FriendlyUnitManager
    should_act: bool = False
    reason: ActReasons = ActReasons.NO_REASON_TO_ACT


def should_unit_consider_acting(
    unit: FriendlyUnitManager,
    upcoming_collisions: Dict[str, AllCollisionsForUnit],
    close_enemies: Dict[str, CloseUnits],
) -> ConsiderActInfo:
    unit_id = unit.unit_id
    # If not enough power to do something meaningful
    should_act = ConsiderActInfo(unit=unit)

    move_valid = unit.valid_moving_actions(
        unit.master.maps.valid_friendly_move, max_len=1
    )
    if unit.power < (
        unit.unit_config.ACTION_QUEUE_POWER_COST + unit.unit_config.MOVE_COST
    ):
        should_act.should_act = False
        should_act.reason = ActReasons.NOT_ENOUGH_POWER
    elif unit.status.action_queue_valid_after_step is False:
        should_act.should_act = True
        should_act.reason = ActReasons.PREVIOUS_ACTION_INVALID
    elif move_valid.was_valid is False:
        should_act.should_act = True
        should_act.reason = ActReasons.NEXT_ACTION_INVALID_MOVE
        logger.debug(
            f"Move from {unit.start_of_turn_pos} was invalid for reason {move_valid.invalid_reasons[0]}, action={unit.action_queue[0]}"
        )
    # If no queue
    elif len(unit.action_queue) == 0:
        should_act.should_act = True
        should_act.reason = ActReasons.NO_ACTION_QUEUE
    # If colliding with friendly
    elif (
        unit_id in upcoming_collisions
        and upcoming_collisions[unit_id].num_collisions(friendly=True, enemy=False) > 0
    ):
        should_act.should_act = True
        should_act.reason = ActReasons.COLLISION_WITH_FRIENDLY
    # If colliding with enemy
    elif (
        unit_id in upcoming_collisions
        and upcoming_collisions[unit_id].num_collisions(friendly=False, enemy=True) > 0
    ):
        should_act.should_act = True
        should_act.reason = ActReasons.COLLISION_WITH_ENEMY
    # If close to enemy
    elif unit_id in close_enemies:
        should_act.should_act = True
        should_act.reason = ActReasons.CLOSE_TO_ENEMY
    elif unit.status.current_action == Actions.ATTACK:
        should_act.should_act = True
        should_act.reason = ActReasons.ATTACKING
    elif unit.action_queue[0][util.ACT_TYPE] == util.PICKUP:
        should_act.should_act = True
        should_act.reason = ActReasons.NEXT_ACTION_PICKUP
    elif unit.action_queue[0][util.ACT_TYPE] == util.TRANSFER:
        should_act.should_act = True
        should_act.reason = ActReasons.NEXT_ACTION_TRANSFER
    elif unit.action_queue[0][util.ACT_TYPE] == util.DIG:
        should_act.should_act = True
        should_act.reason = ActReasons.NEXT_ACTION_DIG
    elif unit.status.current_action == Actions.NOTHING:
        should_act.should_act = True
        should_act.reason = ActReasons.CURRENT_STATUS_NOTHING
    else:
        should_act.should_act = False
        should_act.reason = ActReasons.NO_REASON_TO_ACT

    if should_act.should_act:
        logger.info(f"{unit_id} should consider acting -- {should_act.reason}")
    else:
        logger.info(f"{unit_id} should not consider acting -- {should_act.reason}")
    return should_act


def _decide_collision_avoidance(
    unit, other_unit, is_enemy, power_threshold_low, power_threshold_high
) -> Tuple[float, bool]:
    """
    Handle collision cases based on unit types and their friendly or enemy status.

    Args: unit: A Unit object representing the primary unit. other_unit: A Unit object representing the other
    unit to compare against. power_threshold_low: A numeric value representing the lower threshold for the
    power difference between the two units. power_threshold_high: A numeric value representing the upper
    threshold for the power difference between the two units.

    Returns: tuple of float, bool for weighting and allowing collisions (weighting means prefer move towards
    -ve or away +ve)
    """
    unit_type = unit.unit_type  # "LIGHT" or "HEAVY"
    other_unit_type = other_unit.unit_type  # "LIGHT" or "HEAVY"

    power_difference = unit.power - other_unit.power

    if unit_type == "HEAVY":
        if other_unit_type == "HEAVY":
            if is_enemy:
                if power_difference > power_threshold_high:
                    # Path toward and try to collide
                    return -1, True
                elif power_difference < power_threshold_low:
                    # Path away and avoid colliding
                    return 1, False
                else:
                    # Just avoid colliding
                    return 0, False
            else:  # other_unit is friendly
                # Just avoid colliding
                return 0, False
        elif other_unit_type == "LIGHT":
            if is_enemy:
                # Ignore the other unit completely
                return 0, True
            else:  # other_unit is friendly
                # Avoid colliding
                return 0, False
    elif unit_type == "LIGHT":
        if other_unit_type == "HEAVY" and is_enemy:
            # Path away and avoid colliding
            return 1, False
        elif other_unit_type == "LIGHT" and is_enemy:
            if power_difference > power_threshold_high:
                # Path toward and try to collide
                return -1, True
            elif power_difference < power_threshold_low:
                # Path away and avoid colliding
                return 1, False
            else:
                # Just avoid colliding
                return 0, False
        else:  # other_unit is friendly
            # Just avoid colliding
            return 0, False
    raise RuntimeError(f"Shouldn't reach here")


class SingleUnitActionPlanner:
    def __init__(
        self,
        unit: FriendlyUnitManager,
        master: MasterState,
        base_costmap: np.ndarray,
        unit_paths: UnitPaths,
        unit_info: UnitInfo,
        close_units: Optional[AllCloseUnits],
        collision_info: Optional[AllCollisionsForUnit],
        factory_desires: FactoryDesires,
        factory_info: FactoryInfo,
        action_validator: ValidActionCalculator,
        mining_planner: MiningPlanner,
        rubble_clearing_planner: RubbleClearingPlanner,
        combat_planner: CombatPlanner,
        collision_resolve_max_step: int,
    ):
        self.unit = unit
        self.master = master
        self.base_costmap = base_costmap
        self.unit_paths = unit_paths
        self.unit_info = unit_info
        self.close_units = close_units
        self.collision_info = collision_info
        self.factory_desires = factory_desires
        self.factory_info = factory_info
        self.action_validator = action_validator
        self.mining_planner = mining_planner
        self.rubble_clearing_planner = rubble_clearing_planner
        self.combat_planner = combat_planner
        self.collision_resolve_max_step = collision_resolve_max_step

        self.action_decider = ActionDecider(
            unit,
            unit_info,
            action_validator,
            factory_desires,
            factory_info,
            close_units.close_to_enemy.get(unit_info.unit_id, None),
        )
        self.action_implementer = ActionImplementer(
            master=master,
            unit_paths=unit_paths,
            unit_info=unit_info,
            action_validator=action_validator,
            close_units=close_units,
            factory_desires=factory_desires,
            factory_info=factory_info,
            mining_planner=mining_planner,
            rubble_clearing_planner=rubble_clearing_planner,
            combat_planner=combat_planner,
        )

    def _unit_must_move(self):
        """Must move if current location will be occupied at step 1 (not zero which is now)"""
        start_costmap = self.master.pathfinder.generate_costmap(
            self.unit, override_step=1, collision_only=True
        )

        unit_must_move = False
        # If current location will be occupied
        if (
            start_costmap[
                self.unit.start_of_turn_pos[0], self.unit.start_of_turn_pos[1]
            ]
            <= 0
        ):
            logger.info(
                f"{self.unit.unit_id} MUST move first turn to avoid collision at current pos {self.unit.pos}"
            )
            unit_must_move = True

        # If very close to enemy that can kill us
        elif self.unit.unit_id in self.close_units.close_to_enemy:
            close = self.close_units.close_to_enemy[self.unit.unit_id]
            close_dists = close.other_unit_distances
            # Dist 1 == adjacent
            if len(close_dists) > 0 and min(close_dists) <= 1:
                # Only matters if other is Heavy or both lights
                for utype, dist in zip(close.other_unit_types, close_dists):
                    if dist <= 1 and (utype == self.unit.unit_type or utype == "HEAVY"):
                        unit_must_move = True
        return unit_must_move

    def _resolve_continue_actions(self) -> Actions:
        if self.unit_info.act_info.reason == ActReasons.COLLISION_WITH_FRIENDLY:
            collision_resolver = CollisionResolver(
                self.unit,
                pathfinder=self.master.pathfinder,
                maps=self.master.maps,
                unit_paths=self.unit_paths,
                collisions=self.collision_info,
                max_step=self.collision_resolve_max_step,
            )
            resolve_action_status = collision_resolver.resolve()
            return resolve_action_status
        logger.info(
            f"Don't know how to resolve {self.unit_info.act_info.reason} without calling the planner again"
        )
        return Actions.CONTINUE_UPDATE

    def _force_moving_if_necessary(self, unit_must_move: bool) -> bool:
        success = True
        if unit_must_move:
            q = self.unit.action_queue
            if (
                len(q) == 0
                or q[0][util.ACT_TYPE] != util.MOVE
                or q[0][util.ACT_DIRECTION] == util.CENTER
            ):
                logger.error(
                    f"{self.unit.log_prefix} was required to move first turn, but actions are {q}, trying to move unit"
                )
                self.unit.pos = self.unit.start_of_turn_pos
                self.unit.action_queue = []
                util.move_to_cheapest_adjacent_space(self.master.pathfinder, self.unit)
                success = False
        return success

    def calculate_actions_for_unit(self) -> bool:
        # Will be using this a lot in here
        unit = self.unit

        logger.info(
            f"Beginning calculating action for {unit.unit_id}: power = {unit.power}, pos = {unit.pos}, len(actions) = {len(unit.action_queue)}, current_action = {unit.status.current_action}"
        )

        # Is current location in existing paths for next step or equal or higher enemy adjacent
        unit_must_move = self._unit_must_move()

        # Decide what action needs to be taken next (may be to continue with current plan)
        desired_action = self.action_decider.decide_action(unit_must_move)

        success = False
        do_update = True

        # If no update required, return now (queue not updated, so no changes will happen)
        if desired_action in [Actions.CONTINUE_NO_CHANGE]:
            logger.info(f"No update of actions necessary")
            return True
        elif desired_action == Actions.CONTINUE_UPDATE:
            resolve_action_status = self._resolve_continue_actions()
            desired_action = unit.status.current_action
            if resolve_action_status == Actions.CONTINUE_UPDATE:
                do_update = True
            elif resolve_action_status == Actions.CONTINUE_RESOLVED:
                do_update = False

        # Do or redo the planning for this unit if necessary
        if do_update:
            # Clear queue to build a new one (old is in unit.start_of_...)
            unit.action_queue = []

            # Build the new action queue
            success = self.action_implementer.implement_desired_action(
                unit,
                desired_action,
                unit_must_move=unit_must_move,
            )

        # Check again if unit must move
        mm_success = self._force_moving_if_necessary(unit_must_move)

        # Update the planned actions
        self.unit.update_planned_actions_with_queue()
        return success and mm_success


class ActionImplementer:
    def __init__(
        self,
        master,
        unit_paths: UnitPaths,
        unit_info: UnitInfo,
        action_validator: ValidActionCalculator,
        close_units: AllCloseUnits,
        factory_desires: FactoryDesires,
        factory_info: FactoryInfo,
        mining_planner: MiningPlanner,
        rubble_clearing_planner: RubbleClearingPlanner,
        combat_planner: CombatPlanner,
    ):
        self.master = master
        self.unit_paths = unit_paths
        self.unit_info = unit_info
        self.action_validator = action_validator
        self.close_units = close_units
        self.factory_desires = factory_desires
        self.factory_info = factory_info
        self.mining_planner = mining_planner
        self.rubble_clearing_planner = rubble_clearing_planner
        self.combat_planner = combat_planner

    def implement_desired_action(
        self,
        unit: FriendlyUnitManager,
        desired_action: Actions,
        unit_must_move: bool,
    ):
        if desired_action == Actions.ATTACK:
            success = self.combat_planner.attack(unit)
        elif desired_action == Actions.RUN_AWAY:
            success = self.combat_planner.run_away(unit)
        elif desired_action == Actions.MINE_ORE:
            success = self._mine_ore(unit, unit_must_move)
        elif desired_action == Actions.MINE_ICE:
            success = self._mine_ice(unit, unit_must_move)
        elif desired_action == Actions.CLEAR_RUBBLE:
            success = self._clear_rubble(unit, unit_must_move)
        elif desired_action == Actions.NOTHING:
            success = self._do_nothing(unit, unit_must_move)
        else:
            logger.error(f"{desired_action} not understood as an action")
            success = False

        if success:
            unit.update_status(new_action=desired_action, success=success)
        else:
            unit.update_status(new_action=Actions.NOTHING, success=False)
        return success

    def _mine_ore(self, unit, unit_must_move) -> bool:
        rec = self.mining_planner.recommend(
            unit, util.ORE, unit_must_move=unit_must_move
        )
        if rec is not None:
            success = self.mining_planner.carry_out(
                unit, rec, unit_must_move=unit_must_move
            )
        else:
            success = False
        return success

    def _mine_ice(self, unit, unit_must_move) -> bool:
        rec = self.mining_planner.recommend(
            unit, util.ICE, unit_must_move=unit_must_move
        )
        if rec is not None:
            success = self.mining_planner.carry_out(
                unit, rec, unit_must_move=unit_must_move
            )
        else:
            success = False
        return success

    def _clear_rubble(self, unit, unit_must_move) -> bool:
        rec = self.rubble_clearing_planner.recommend(unit)
        success = self.rubble_clearing_planner.carry_out(
            unit, rec, unit_must_move=unit_must_move
        )
        return success

    def _do_nothing(self, unit, unit_must_move) -> bool:
        logger.debug(f"Setting action queue to empty to do action {Actions.NOTHING}")
        unit.action_queue = []
        success = True
        if unit_must_move:
            if not unit.factory_id:
                logger.error(
                    f"Unit must move, but has action {Actions.NOTHING} and no factory assigned"
                )
            else:
                success = self._handle_nothing_with_must_move(unit)
        return success

    def _handle_nothing_with_must_move(self, unit) -> bool:
        if unit.on_own_factory():
            success = util.move_to_new_spot_on_factory(
                self.master.pathfinder,
                unit,
                self.master.factories.friendly[unit.factory_id],
            )
            if not success:
                util.move_to_cheapest_adjacent_space(self.master.pathfinder, unit)
        else:
            util.move_to_cheapest_adjacent_space(self.master.pathfinder, unit)
        return True


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
    ):
        """Assuming this is called after beginning of turn update"""
        self.master = master

        # Will be filled on update
        self.factory_desires: Dict[str, FactoryDesires] = None
        self.factory_infos: Dict[str, FactoryInfo] = None

        # Caching
        self.base_costmap: np.ndarray = None
        self.all_upcoming_collisions: Dict[str, AllCollisionsForUnit] = None
        self.all_close_units: AllCloseUnits = None

    def update(
        self,
        factory_infos: Dict[str, FactoryInfo],
        factory_desires: Dict[str, FactoryDesires],
    ):
        """Beginning of turn update"""
        self.factory_infos = factory_infos
        self.factory_desires = factory_desires

        # Validate and replace enemy actions so that their move path is correct (i.e. cannot path through friendly
        # factories or off edge of map, so replace those moves with move.CENTER)
        self._replace_invalid_enemy_moves()

        # Update calculated things
        # Base travel costmap (i.e. with factories)
        self.base_costmap = self._calculate_base_costmap()

        # Calculate collisions
        self.all_upcoming_collisions = self._calculate_collisions()

        # Calculate close units
        self.all_close_units = self._calculate_close_units()

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

    def _get_units_to_act(
        self, units: Dict[str, FriendlyUnitManager], close_units: AllCloseUnits
    ) -> UnitsToAct:
        """
        Determines which units should potentially act this turn, and which should continue with current actions
        Does this based on:
            - collisions in next couple of turns
            - enemies nearby
            - empty action queue

        Args:
            units: list of friendly units

        Returns:
            Instance of UnitsToAct
        """
        logger.info(
            f"units_should_consider_acting called with len(units): {len(units)}"
        )

        all_unit_collisions = self._calculate_collisions()
        all_unit_close_to_enemy = close_units.close_to_enemy
        needs_to_act = {}
        should_not_act = {}
        for unit_id, unit in units.items():
            should_act = should_unit_consider_acting(
                unit,
                upcoming_collisions=all_unit_collisions,
                close_enemies=all_unit_close_to_enemy,
            )
            if should_act.should_act:
                needs_to_act[unit_id] = should_act
            else:
                should_not_act[unit_id] = should_act
        return UnitsToAct(needs_to_act=needs_to_act, should_not_act=should_not_act)

    def _calculate_collisions(self) -> Dict[str, AllCollisionsForUnit]:
        """Calculates the upcoming collisions based on action queues of all units"""
        all_collisions = calculate_collisions(
            self.master.units,
            check_steps_enemy=self.check_enemy_collision_steps,
            check_steps_friendly=self.check_friendly_collision_steps,
        )
        return all_collisions

    def _collect_unit_data(self, act_infos: Dict[str, ConsiderActInfo]) -> UnitInfos:
        """
        Collects data from units and stores it in a pandas dataframe.

        Args:
            act_infos: List of ActInfo objects.

        Returns:
            A pandas dataframe containing the unit data.
        """
        data = {}
        for unit_id, act_info in act_infos.items():
            unit = act_info.unit
            unit_info = UnitInfo.from_data(unit=unit, act_info=act_info)
            data[unit_id] = unit_info
        return UnitInfos(infos=data)

    def _calculate_base_costmap(self) -> np.ndarray:
        """
        Calculates the base costmap based on:
            - rubble array
            - Enemy factories impassible
            - Center of friendly factories (in case a unit is built there)

        Returns:
            A numpy array representing the costmap.
        """
        costmap = self.master.maps.rubble.copy() * 0.1  # was 0.05
        costmap += 1  # Zeros aren't traversable
        enemy_factory_map = self.master.maps.factory_maps.enemy
        costmap[enemy_factory_map >= 0] = -1  # Not traversable

        # Make center of factories impassible (in case unit is built there)
        # TODO: Only block center if unit actually being built
        for factory_id, factory in self.master.factories.friendly.items():
            pos = factory.pos
            costmap[pos[0], pos[1]] = -1
        return costmap

    def _collect_changed_actions(self, units_to_act):
        unit_actions = {}
        for unit_id, act_info in units_to_act.has_updated_actions.items():
            if len(act_info.unit.action_queue) > 0:
                unit_actions[unit_id] = act_info.unit.action_queue[:20]
            else:
                logger.warning(
                    f"Updating {unit_id} with empty actions (previous action len = "
                    f"{len(act_info.unit.start_of_turn_actions)}) previous_status = {act_info.unit.status.previous_action}"
                    f"(could be on purpose, but probably should figure out a better thing for this unit to do (even if stay still for a while first))"
                )
                if len(act_info.unit.start_of_turn_actions) == 0:
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

    def _assign_new_factory_if_necessary(
        self, unit: FriendlyUnitManager, factory_infos: Dict[str, FactoryInfo]
    ):
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
            logger.warning(
                f"Re-assigning to {best_factory.unit_id} because no factory assigned"
            )

    def _should_real_actions_update(
        self, unit: FriendlyUnitManager, unit_info: UnitInfo, units_to_act: UnitsToAct
    ):
        current_unit_actions = unit.start_of_turn_actions
        planned_actions = unit.status.planned_actions

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
            unit.action_queue = planned_actions[:20]
            units_to_act.should_not_act[unit.unit_id] = unit_info.act_info
        else:
            last_updated = self.master.step - unit.status.last_action_update_step
            logger.info(
                f"{unit.log_prefix} has updated actions "
                f"(last updated {last_updated} ago),"
                f"was {unit.status.previous_action}, now {unit.status.current_action}"
                f" first few new actions are {planned_actions[:3]}, first few old actions were {unit.start_of_turn_actions[:3]}"
            )
            if last_updated < 3 or last_updated > 30 and unit.status.previous_action != Actions.NOTHING:
                logger.info(
                    f"{unit.log_prefix} updated {last_updated} ago <<< This is just a note to see how things are going"
                )

            unit.action_queue = planned_actions[:20]
            unit.status.last_action_update_step = self.master.step
            units_to_act.has_updated_actions[unit.unit_id] = unit_info.act_info

    def decide_unit_actions(
        self,
        mining_planner: MiningPlanner,
        rubble_clearing_planner: RubbleClearingPlanner,
        combat_planner: CombatPlanner,
        factory_desires: Dict[str, FactoryDesires],
        factory_infos: Dict[str, FactoryInfo],
    ) -> Dict[str, List[np.ndarray]]:
        logger.info(f"deciding all unit actions")

        units_to_act = self._get_units_to_act(
            self.master.units.friendly.all, self.all_close_units
        )
        unit_infos = self._collect_unit_data(units_to_act.needs_to_act)
        unit_infos.sort_by_priority()

        # Calculate 3D path arrays to use for calculating costmaps later
        existing_paths = UnitPaths.from_units(
            friendly={k: act.unit for k, act in units_to_act.should_not_act.items()},
            enemy=self.master.units.enemy.all,
            friendly_valid_move_map=self.master.maps.valid_friendly_move,
            enemy_valid_move_map=self.master.maps.valid_enemy_move,
            max_step=self.max_pathing_steps,
        )

        # Update the pathfinder now that we know which units are acting
        base_costmap = self.base_costmap
        self.master.pathfinder = Pather(
            base_costmap=base_costmap,
            unit_paths=existing_paths,
        )

        # Setup validator for next unit actions
        action_validator = ValidActionCalculator(
            units_to_act=units_to_act,
            factory_infos=factory_infos,
            maps=self.master.maps,
            unit_paths=existing_paths,
        )

        # for unit_id, act_info in units_to_act.needs_to_act.items():
        # Go through units in order of priority (unit_infos is sorted)
        for unit_id, unit_info in unit_infos.infos.items():
            unit = unit_info.unit
            self._assign_new_factory_if_necessary(unit, factory_infos)
            unit_info = unit_infos.infos[unit_id]
            unit_action_planner = SingleUnitActionPlanner(
                unit=unit,
                master=self.master,
                base_costmap=base_costmap,
                unit_paths=existing_paths,
                unit_info=unit_info,
                close_units=self.all_close_units,
                collision_info=self.all_upcoming_collisions.get(unit_id, None),
                factory_desires=factory_desires[unit.factory_id],
                factory_info=factory_infos[unit.factory_id],
                action_validator=action_validator,
                mining_planner=mining_planner,
                rubble_clearing_planner=rubble_clearing_planner,
                combat_planner=combat_planner,
                collision_resolve_max_step=self.max_pathing_steps,
            )
            unit_action_planner.calculate_actions_for_unit()

            # Check if planned actions differ from existing actions on next turn
            # TODO:  Currently checks whole action, can instead check next step only
            self._should_real_actions_update(unit, unit_info, units_to_act)

            # Make sure the next action is taken into account for next validation
            action_validator.add_next_action(unit)
            # And for next pathing
            existing_paths.add_unit(unit, is_enemy=False)

        unit_actions = self._collect_changed_actions(units_to_act)
        unit_actions = self._validate_changed_actions_against_action_space(unit_actions)
        logger.info(f'Updating actions of {list(unit_actions.keys())}')
        return unit_actions
