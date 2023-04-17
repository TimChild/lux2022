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
import actions
from config import get_logger
from factory_action_planner import FactoryDesires, FactoryInfo
from master_state import MasterState, AllUnits
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
        distance_map: np.ndarray,
    ):
        unit_factory = unit.factory

        unit_info = cls(
            unit=unit,
            act_info=act_info,
            unit_id=unit.unit_id,
            last_action_update_step=unit.status.last_action_update_step,
            len_action_queue=len(unit.action_queue),
            distance_to_factory=(
                distance_map[unit_factory.factory.pos[0], unit_factory.factory.pos[1]]
                if unit_factory
                else np.nan
            ),
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
    """First collision only"""

    unit_id: str
    other_unit_id: str
    other_unit_is_enemy: bool
    pos: Tuple[int, int]
    step: int


@dataclass
class CollisionsForUnit:
    light: Dict[str, Collision]
    heavy: Dict[str, Collision]


@dataclass
class AllCollisionsForUnit:
    with_friendly: CollisionsForUnit
    with_enemy: CollisionsForUnit

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


@dataclass
class AllCloseUnits:
    close_to_friendly: Dict[str, CloseUnits]
    close_to_enemy: Dict[str, CloseUnits]


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
        max_step=30,
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
                # x, y = unit.pos
                # for step in range(max_step):
                #     if step < len(path):
                #         x, y = path[step]
                #     if x is not None:
                #         array[step, x, y] = unit_num

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
        extra = 0
        for step in range(max_step):
            if step < len(path):
                x, y = path[step]
                arr[step, x, y] = unit.id_num
            # Continue to fill last enemy position as position for a few extra turns
            elif x is not None and is_enemy and extra < 5:
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

        valid_path = unit.valid_moving_actions(costmap=move_map, max_len=max_step)
        path = util.actions_to_path(
            unit.start_of_turn_pos, actions=valid_path.valid_actions, max_len=max_step
        )

        self._add_path_to_array(unit, path, array, max_step, is_enemy=is_enemy)
        # x, y = unit.pos
        # for step in range(self.max_step):
        #     if step < len(path):
        #         x, y = path[step]
        #     if x is not None:
        #         array[step, x, y] = unit_num

        # for step, (x, y) in enumerate(path):
        #     array[step, x, y] = unit_num

    def to_costmap(
        self,
        pos: util.POS_TYPE,
        start_step: int,
        exclude_id_nums: Union[int, List[int]],
        friendly_light: bool,
        friendly_heavy: bool,
        enemy_light: bool,
        enemy_heavy: bool,
        collision_cost_value=-1,
        nearby_start_cost: Optional[int] = 5,
        step_dropoff_multiplier=0.92,
        true_intercept=False,
    ):
        """Create a costmap from a specific position at a specific step"""
        if nearby_start_cost is not None and nearby_start_cost < 0:
            raise ValueError(f"Nearby start cost must be positive or None")
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
        if nearby_start_cost and not true_intercept:
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
                    blur = friendly_blur.copy()
                    collision = friendly_collision
                # For enemy it depends if we are trying to collide or avoid
                else:
                    blur = enemy_blur.copy()
                    # Trying to avoid
                    if collision_cost_value <= 0:
                        # blocks adjacent
                        collision = enemy_collision
                    # Trying to collide
                    else:
                        # only selects intercepts
                        collision = friendly_collision

                # Make 1s wherever units present
                arr = (arr >= 0).astype(float)

                # Do nearby blur if not None or 0
                if nearby_start_cost is not None and nearby_start_cost and not true_intercept:
                    blur *= nearby_start_cost * step_dropoff_multiplier ** (start_step + i)
                    # Blur that with kernel
                    add_cm = util.convolve_array_kernel(arr, blur, fill=0)
                    # Add non-blocking costs
                    cm[cm > 0] += add_cm[cm > 0]

                # Calculate likely collisions
                col_cm = util.convolve_array_kernel(arr, collision, fill=0)

                # Set blocking (or targeting) costs
                cm[col_cm < 0] = collision_cost_value
        return cm
        #######

        # cm = np.ones_like(self.friendly_valid_move_map, dtype=float)
        # collisions = self.calculate_likely_unit_collisions(
        #     pos=pos,
        #     start_step=start_step,
        #     exclude_id_nums=exclude_id_nums,
        #     friendly_light=friendly_light,
        #     friendly_heavy=friendly_heavy,
        #     enemy_light=enemy_light,
        #     enemy_heavy=enemy_heavy,
        # )
        #
        # if nearby_start_cost is not None:
        #     for i in range(1, 5):
        #         close_encounters = self.calculate_likely_unit_collisions(
        #             pos=pos,
        #             start_step=start_step + i,
        #             exclude_id_nums=exclude_id_nums,
        #             friendly_light=friendly_light,
        #             friendly_heavy=friendly_heavy,
        #             enemy_light=enemy_light,
        #             enemy_heavy=enemy_heavy,
        #         )
        #         cm[close_encounters >= 0] = (
        #             nearby_start_cost * nearby_dropoff_multiplier**i
        #         )
        # cm[collisions >= 0] = collision_cost_value
        # return cm

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
            if start_step >= self.max_step:
                logger.info(
                    f"Requesting map outside of max_steps {self.max_step} returning empty"
                )
                do_calcs =  False
            logger.error(f"{start_step} must be between 0 and {self.max_step}")
            if start_step < 0:
                start_step = 0
            else:
                start_step = self.max_step

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
            utypes.append('light')
        if friendly_heavy:
            utypes.append('heavy')
        if enemy_light:
            utypes.append('light')
        if enemy_heavy:
            utypes.append('heavy')
        return teams, utypes
        # # Start with empty
        # unit_id_map = np.full_like(self.friendly_valid_move_map, -1, dtype=int)
        # x, y = self._x, self._y
        # # Keep first unit at location only
        # if friendly_light:
        #     cm = self.friendly_light[index_array, x, y]
        #     # Replace excluded with -1 (same as ignoring)
        #     cm = np.where(np.isin(cm, exclude_id_nums), -1, cm)
        #     unit_id_map = np.where(unit_id_map < 0, cm, unit_id_map)
        # if friendly_heavy:
        #     cm = self.friendly_heavy[index_array, x, y]
        #     cm = np.where(np.isin(cm, exclude_id_nums), -1, cm)
        #     unit_id_map = np.where(unit_id_map < 0, cm, unit_id_map)
        # if enemy_light:
        #     cm = self.enemy_light[index_array, x, y]
        #     cm = np.where(np.isin(cm, exclude_id_nums), -1, cm)
        #     unit_id_map = np.where(unit_id_map < 0, cm, unit_id_map)
        # if enemy_heavy:
        #     cm = self.enemy_heavy[index_array, x, y]
        #     cm = np.where(np.isin(cm, exclude_id_nums), -1, cm)
        #     unit_id_map = np.where(unit_id_map < 0, cm, unit_id_map)
        # return unit_id_map


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
    all_units: AllUnits, check_steps: int = 2
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
                    max_step=check_steps,
                    other_is_enemy=False,
                ),
                heavy=find_collisions(
                    unit,
                    all_units.friendly.heavy.values(),
                    max_step=check_steps,
                    other_is_enemy=False,
                ),
            ),
            with_enemy=CollisionsForUnit(
                light=find_collisions(
                    unit,
                    all_units.enemy.light.values(),
                    max_step=check_steps,
                    other_is_enemy=True,
                ),
                heavy=find_collisions(
                    unit,
                    all_units.enemy.heavy.values(),
                    max_step=check_steps,
                    other_is_enemy=True,
                ),
            ),
        )
        if collisions_for_unit.num_collisions(friendly=True, enemy=True) > 0:
            all_unit_collisions[unit_id] = collisions_for_unit
    return all_unit_collisions


@dataclass
class ActionDecisionData:
    unit_type: str
    power: int
    ore_desired: bool
    rubble_clearing_desired: bool


def decide_action(
    unit_info: UnitInfo,
    factory_desires: FactoryDesires,
    factory_info: FactoryInfo,
    close_units: Union[None, CloseUnits],
) -> str:
    """
    Decide what type of action unit should be doing based on Factory preferences

    Note: Should only switch from current job to attack/run_away or if it back at the factory

    """
    logger.info(f"Deciding action for {unit_info.unit_id}")

    def attack_or_run_away(
        close_units: CloseUnits,
        unit_type: str,
        power_threshold: int,
    ) -> [None, str]:
        # Only if close to other units
        if close_units is not None:
            close_enemies_within_2 = [
                enemy
                for enemy, distance in zip(
                    close_units.other_unit_types, close_units.other_unit_distances
                )
                if distance <= 2 and enemy == unit_type
            ]
            if len(close_enemies_within_2) == 1:
                enemy_power = close_units.other_unit_powers[
                    close_units.other_unit_types.index(unit_type)
                ]
                if enemy_power < unit_info.power - power_threshold:
                    return actions.ATTACK
            elif len(close_enemies_within_2) > 1:
                return actions.RUN_AWAY
        return None

    action = None
    if unit_info.unit_type == "LIGHT":
        logger.debug(f"Deciding between light unit actions")
        action = attack_or_run_away(close_units, "LIGHT", 10)
        if action is None:
            logger.debug(f"should not attack or run_away, deciding what next")

            # If not on factory, continue doing whatever it was doing before
            if (
                not unit_info.unit.on_own_factory()
                and unit_info.unit.status.current_action != actions.NOTHING
            ):
                action = unit_info.unit.status.current_action
                logger.debug(
                    f"Unit NOT on factory and currently assigned, should continue same job ({action})"
                )
            # If on factory, it can switch jobs to whatever is necessary
            else:
                logger.debug(
                    f"Unit on factory, can decide a new type of action depending on factory needs"
                )

                # Make sure factory info is updated (i.e. if this unit stopped mining_ice, then num needs decreasing)
                factory_info.remove_unit_from_current_count(unit_info.unit)
                if factory_info.light_mining_ore < factory_desires.light_mining_ore:
                    logger.debug(
                        f"{factory_info.factory_id}: Adding light ore miner, light_ore={factory_info.light_mining_ore}, desires={factory_desires.light_mining_ore}"
                    )
                    action = actions.MINE_ORE
                elif (
                    factory_info.light_clearing_rubble
                    < factory_desires.light_clearing_rubble
                ):
                    action = actions.CLEAR_RUBBLE
                elif factory_info.light_mining_ice < factory_desires.light_mining_ice:
                    action = actions.MINE_ICE
                elif factory_info.light_attacking < factory_desires.light_attacking:
                    action = actions.ATTACK
                else:
                    action = actions.NOTHING
    else:  # unit_type == "HEAVY"
        logger.debug(f"Deciding between heavy unit actions")
        action = attack_or_run_away(close_units, "HEAVY", 100)
        if action is None:
            logger.debug(f"should not attack or run_away, deciding what next")
            # If not on factory, continue doing whatever it was doing before
            if (
                not unit_info.unit.on_own_factory()
                and unit_info.unit.status.current_action != actions.NOTHING
            ):
                action = unit_info.unit.status.current_action
                logger.debug(
                    f"Unit NOT on factory and currently assigned, should continue same job ({action})"
                )
            # If on factory, it can switch jobs to whatever is necessary
            else:
                logger.debug(
                    f"Unit on factory, can decide a new type of action depending on factory needs"
                )
                # Make sure factory info is updated (i.e. if this unit stopped mining_ice, then num needs decreasing)
                factory_info.remove_unit_from_current_count(unit_info.unit)
                if factory_info.heavy_mining_ice < factory_desires.heavy_mining_ice:
                    action = actions.MINE_ICE
                elif factory_info.heavy_mining_ore < factory_desires.heavy_mining_ore:
                    action = actions.MINE_ORE
                elif (
                    factory_info.heavy_clearing_rubble
                    < factory_desires.heavy_clearing_rubble
                ):
                    action = actions.CLEAR_RUBBLE
                elif factory_info.heavy_attacking < factory_desires.heavy_attacking:
                    action = actions.ATTACK
                else:
                    action = actions.NOTHING
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
    elif unit.status.current_action == actions.ATTACK:
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
    elif unit.status.current_action == actions.NOTHING:
        should_act.should_act = True
        should_act.reason = ActReasons.CURRENT_STATUS_NOTHING
    else:
        should_act.should_act = False
        should_act.reason = ActReasons.NO_REASON_TO_ACT
    if should_act.should_act:
        logger.debug(f"{unit_id} should consider acting -- {should_act.reason}")
    else:
        logger.debug(f"{unit_id} should not consider acting -- {should_act.reason}")
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


class UnitActionPlanner:
    # How far should distance map extend (padded with max value after that)
    max_distance_map_dist = 20
    # What is considered a close unit when considering future paths
    close_threshold = 4
    # If there will be a collision within this many steps consider acting
    check_collision_steps = 3
    # Increase cost to travel near units based on kernel with this dist
    kernel_dist = 5
    # If this many actions the same, don't update unit
    actions_same_check = 3
    # Number of steps to block other unit path locations for
    avoid_collision_steps = 20

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
        self._costmap: np.ndarray = None
        self._upcoming_collisions: AllCollisionsForUnit = None
        self._close_units: AllCloseUnits = None
        self._start_distance_maps = {}

    def update(
        self,
        factory_infos: Dict[str, FactoryInfo],
        factory_desires: Dict[str, FactoryDesires],
    ):
        """Beginning of turn update"""
        self.factory_infos = factory_infos
        self.factory_desires = factory_desires

        # Clear caches
        self._costmap = None
        self._upcoming_collisions = None
        self._close_units = None
        self._start_distance_maps = {}

        # Validate and replace enemy actions so that their move path is correct (i.e. cannot path through friendly
        # factories or off edge of map, so replace those moves with move.CENTER)
        self._replace_invalid_enemy_moves()

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
                max_len=self.avoid_collision_steps,
                ignore_repeat=False,
            )
            if valid_actions.was_valid is False:
                logger.warning(
                    f"Enemy {unit_id} actions were invalid. First invalid at step {valid_actions.invalid_steps[0]}"
                )
                unit.action_queue = valid_actions.valid_actions

    def _get_units_to_act(self, units: Dict[str, FriendlyUnitManager]) -> UnitsToAct:
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
        all_unit_close_to_enemy = self._calculate_close_enemies()
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
            self.master.units, check_steps=self.check_collision_steps
        )
        return all_collisions

    def _calculate_close_units(self) -> AllCloseUnits:
        """Calculates which friendly units are close to any other unit"""
        if self._close_units is None:
            friendly = {}
            enemy = {}
            # Keep track of being close to friendly and enemy separately
            for all_close, other_units in zip(
                [friendly, enemy],
                [self.master.units.friendly.all, self.master.units.enemy.all],
            ):
                # For all friendly units, figure out which friendly and enemy they are near
                for unit_id, unit in self.master.units.friendly.all.items():
                    # print(f'For {unit_id}:')
                    unit_distance_map = self._unit_start_distance_map(unit_id)
                    close = CloseUnits(unit_id=unit_id, unit_pos=unit.pos)
                    for other_id, other_unit in other_units.items():
                        # print(f'checking {other_id}')
                        if other_id == unit_id:  # Don't compare to self
                            continue
                        # print(f'{other_unit.unit_id} pos = {other_unit.pos}')
                        dist = unit_distance_map[other_unit.pos[0], other_unit.pos[1]]
                        # print(f'dist {dist}')

                        if dist <= self.close_threshold:
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
            all_close_units = AllCloseUnits(
                close_to_friendly=friendly, close_to_enemy=enemy
            )
            self._close_units = all_close_units
        return self._close_units

    def _calculate_close_enemies(self) -> Dict[str, CloseUnits]:
        """Calculate the close enemy units to all friendly units"""
        close_units = self._calculate_close_units()
        return close_units.close_to_enemy

    def _unit_start_distance_map(self, unit_id: str) -> np.ndarray:
        """Calculate the distance map for the given unit at the start of turn, this will be used to determine how close other units are"""
        if unit_id not in self._start_distance_maps:
            unit = self.master.units.get_unit(unit_id)
            unit_distance_map = util.pad_and_crop(
                util.manhattan_kernel(self.max_distance_map_dist),
                large_arr=self.master.maps.rubble,
                x1=unit.pos[0],
                y1=unit.pos[1],
                fill_value=self.max_distance_map_dist,
            )
            self._start_distance_maps[unit_id] = unit_distance_map
        return self._start_distance_maps[unit_id]

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
            unit_distance_map = self._unit_start_distance_map(unit_id)

            unit_info = UnitInfo.from_data(
                unit=unit, act_info=act_info, distance_map=unit_distance_map
            )

            data[unit_id] = unit_info
        return UnitInfos(infos=data)

    def _get_base_costmap(self) -> np.ndarray:
        """
        Calculates the base costmap based on:
            - rubble array
            - Enemy factories impassible
            - Center of friendly factories (in case a unit is built there)

        Returns:
            A numpy array representing the costmap.
        """
        if self._costmap is None:
            costmap = self.master.maps.rubble.copy() * 0.1  # was 0.05
            costmap += 1  # Zeros aren't traversable
            enemy_factory_map = self.master.maps.factory_maps.enemy
            costmap[enemy_factory_map >= 0] = -1  # Not traversable

            # Make center of factories impassible (in case unit is built there)
            # TODO: Only block center if unit actually being built
            for factory_id, factory in self.master.factories.friendly.items():
                pos = factory.pos
                costmap[pos[0], pos[1]] = -1
            self._costmap = costmap
        return self._costmap

    def _update_costmap_with_path(
        self,
        costmap: np.ndarray,
        unit_pos: util.POS_TYPE,
        other_path: util.PATH_TYPE,
        is_enemy: bool,
        avoidance: float = 0,
        allow_collision: bool = False,
    ) -> np.ndarray:
        """
        Add additional cost to travelling near path of other unit with avoidance, and prevent collisions (unless
        allowed with allow_collision)

        Args:
            costmap: The base costmap for travel (i.e. rubble)
            unit_pos: This units position
            other_path: Other units path
            is_enemy: If enemy, their next turn may change (so avoid being 1 step away!)
            avoidance: How  much extra cost to be near other_poth (this would be added at collision points, then decreasing amount added near collision points)
            allow_collision: If False, coordinates likely to result in collision are make impassable (likely means the manhattan distance to the path coord is the same or slightly lower)
        """

        def generate_collision_likelihood_array(distances: np.ndarray) -> np.ndarray:
            index_positions = np.arange(len(distances))
            distance_diffs = np.abs(index_positions - distances)

            # Gaussian-like function (you can adjust the scale and exponent as needed)
            likelihood_array = np.exp(-0.5 * (distance_diffs**1))

            return likelihood_array

        logger.info(f"Updating costmap with other_path[0:2] {other_path[0:2]}")
        other_path = other_path
        # Figure out distance to other_units path at each point
        other_path_distance = [
            util.manhattan(p, unit_pos)
            for p in other_path[: self.avoid_collision_steps + 2]
        ]

        # Separate out current position and actual path going forward
        other_pos_now = other_path[0]
        other_dist_now = other_path_distance[0]
        other_path = other_path[1:]
        other_path_distance = other_path_distance[1:]

        # If action queue is about to end, then it's going to stay in the same position
        if len(other_path) == 0:
            other_path = [other_pos_now]
            other_path_distance = [other_dist_now]

        if allow_collision is False:
            # Enemy may change plans, on first step don't allow being even 1 dist away
            if is_enemy and other_dist_now <= 2:
                logger.info(
                    f"Enemy is close at {other_pos_now}, blocking adjacent cells"
                )
                # Block all adjacent cells to enemy (and current enemy pos)
                for delta in util.MOVE_DELTAS:
                    pos = np.array(other_pos_now) + delta
                    try:
                        costmap[pos[0], pos[1]] = -1
                    except IndexError:
                        # near edge of map
                        continue

            logger.debug(f"--------------------- Starting path blocking")

            # Block next X steps in other path that are equal in distance
            max_x, max_y = costmap.shape
            for i, (p, d) in enumerate(
                zip(other_path[: self.avoid_collision_steps], other_path_distance)
            ):
                # logger.debug(f"--------------------- {i}, {p}, {d}")
                if (
                    # Prevents moving to collision manhattan dist
                    d == i + 1
                    or
                    # Prevent sitting in path of collision or reaching in manhattan dist+1 (also likely)
                    d == i
                ):  # I.e. if distance to point on path is same as no. steps it would take to get there
                    logger.info(f"making {p} impassable")
                    x, y = p
                    if 0 <= x < max_x and 0 <= y < max_y:
                        costmap[x, y] = -1
                    else:
                        logger.info(
                            f"{p} cannot be made impassable because outside of map"
                        )

        # If current location becomes blocked, warn that should be unblocked elsewhere
        if costmap[unit_pos[0], unit_pos[1]] == -1:
            logger.info(
                f"{unit_pos} got blocked even though that is the units current position. If cost not changed > 0 pathing will fail"
            )

        # if np.all(unit_pos == (35, 10)):
        #     print(other_path, other_path_distance)
        # # If need to encourage moving away from other path
        # raise NotImplementedError
        # if avoidance != 0:
        #     if avoidance > 1 or avoidance < -1:
        #         raise ValueError(f'got {avoidance}. weighting must be between -1 and 1')
        #     avoidance *= 0.9  # So don't end up multiplying by 0
        #     avoidance += 1  # So can just multiply arrays by this
        #
        #     amplitudes = generate_collision_likelihood_array(
        #         np.array(other_path_distance)
        #     )
        #     kernels = [
        #         # decreasing away from middle (and with distance) * weighting
        #         amp ** util.manhattan_kernel(max_dist=self.kernel_dist) * avoidance
        #         for amp in amplitudes
        #     ]
        #     masks = [
        #         util.pad_and_crop(
        #             kernel,
        #             costmap,
        #             p[0],
        #             p[1],
        #             fill_value=1,
        #         )
        #         for kernel, p in zip(kernels, other_path)
        #     ]
        #     mask = np.mean(masks, axis=0)
        #     costmap *= mask
        return costmap

    def _add_other_unit_to_costmap(
        self,
        costmap: np.ndarray,
        this_unit: FriendlyUnitManager,
        other_unit: [FriendlyUnitManager, EnemyUnitManager],
        other_is_enemy: bool,
    ) -> np.ndarray:
        """Add or removes cost from cost map based on distance and path of nearby unit"""

        if this_unit.unit_type == "LIGHT":
            # If we have 10 more energy, prefer moving toward
            low_power_diff, high_power_diff = -1, 0
        else:
            low_power_diff, high_power_diff = -1, 0

        # TODO: Actually use avoidance or remove it completely
        avoidance, allow_collision = _decide_collision_avoidance(
            this_unit,
            other_unit,
            is_enemy=other_is_enemy,
            power_threshold_low=low_power_diff,
            power_threshold_high=high_power_diff,
        )
        logger.info(
            f"For this {this_unit.unit_id} and other {other_unit.unit_id} - travel avoidance = {avoidance} and allow_collision = {allow_collision}"
        )
        if avoidance == 0 and allow_collision is True:
            # Ignore unit
            pass
        else:
            other_path = other_unit.current_path(max_len=self.avoid_collision_steps)
            costmap = self._update_costmap_with_path(
                costmap,
                this_unit.pos,
                other_path,
                other_is_enemy,
                avoidance=avoidance,
                allow_collision=allow_collision,
            )
        return costmap

    def _get_travel_costmap_for_unit(
        self,
        base_costmap: np.ndarray,
        units_to_act: UnitsToAct,
        unit: FriendlyUnitManager,
    ) -> np.ndarray:
        """
        Updates the costmap with the paths of the units that have determined paths this turn (not acting, done acting, or enemy)

        Args:
            base_costmap: A numpy array representing the costmap.
            units_to_act: UnitsToAct instance containing units that need to act and units that should not act.
            unit: Unit to get the costmap for (i.e. distances calculated relative to this unit)
        """
        logger.info(f"Calculating costmap with paths for unit {unit.unit_id}")
        new_cost = base_costmap.copy()

        all_close_units = self._calculate_close_units()
        units_yet_to_act = units_to_act.needs_to_act.keys()

        # If close to enemy, add those paths
        if unit.unit_id in all_close_units.close_to_enemy:
            logger.debug(f"Close to at least one enemy, adding those to costmap")
            close_units = all_close_units.close_to_enemy[unit.unit_id]
            # For each nearby enemy unit
            for other_id in close_units.other_unit_ids:
                other_unit = self.master.units.enemy.get_unit(other_id)
                if other_unit is None:
                    logger.error(
                        f"{self.master.player} step {self.master.step}: {other_id} does not exist in  master.units.enemy"
                    )
                    continue
                new_cost = self._add_other_unit_to_costmap(
                    new_cost, this_unit=unit, other_unit=other_unit, other_is_enemy=True
                )

        # If close to friendly, add those paths
        if unit.unit_id in all_close_units.close_to_friendly:
            logger.debug(f"Close to at least one friendly, adding those to costmap")
            close_units = all_close_units.close_to_friendly[unit.unit_id]

            # For each friendly unit if it has already acted or is not acting this turn (others can get out of the way)
            for other_id in close_units.other_unit_ids:
                if other_id in units_yet_to_act:
                    # That other unit can get out of the way
                    logger.debug(
                        f"Not adding friendly {other_id}'s path to costmap, assuming it will get out of the way"
                    )
                    continue
                other_unit = self.master.units.friendly.get_unit(other_id)
                new_cost = self._add_other_unit_to_costmap(
                    new_cost,
                    this_unit=unit,
                    other_unit=other_unit,
                    other_is_enemy=False,
                )

        logger.debug(f"Done calculating costmap")
        return new_cost

    def _calculate_actions_for_unit(
        self,
        base_costmap: np.ndarray,
        # travel_costmap: np.ndarray,
        unit_paths: UnitPaths,
        unit_info: UnitInfo,
        unit: FriendlyUnitManager,
        factory_desires: FactoryDesires,
        factory_info: FactoryInfo,
        mining_planner: MiningPlanner,
        rubble_clearing_planner: RubbleClearingPlanner,
        combat_planner: CombatPlanner,
    ) -> bool:
        """Calculate new actions for this unit"""
        logger.info(
            f"Beginning calculating action for {unit.unit_id}: power = {unit.power}, pos = {unit.pos}, len(actions) = {len(unit.action_queue)}, current_action = {unit.status.current_action}"
        )
        # Update the master pathfinder with newest Pather (full_costmap changes for each unit)
        self.master.pathfinder = Pather(
            base_costmap=base_costmap,
            unit_paths=unit_paths,
            # full_costmap=travel_costmap,
        )
        start_costmap = self.master.pathfinder.generate_costmap(unit)

        unit_must_move = False
        # If current location is blocked, unit MUST move first turn
        # if travel_costmap[unit.pos[0], unit.pos[1]] <= 0:
        if start_costmap[unit.pos[0], unit.pos[1]] <= 0:
            logger.warning(
                f"{unit.unit_id} MUST move first turn to avoid collision at {unit.pos}"
            )
            unit_must_move = True

        unit.action_queue = []

        close_units = self._calculate_close_units()
        possible_close_enemy: [None, CloseUnits] = close_units.close_to_enemy.get(
            unit.unit_id, None
        )

        desired_action = decide_action(
            unit_info=unit_info,
            factory_desires=factory_desires,
            factory_info=factory_info,
            close_units=possible_close_enemy,
        )
        if desired_action == actions.NOTHING:
            logger.warning(
                f"{unit.log_prefix} desired action is {desired_action}. Note to self: What should it be doing? Attacking?"
            )
        else:
            logger.info(f"desired action is {desired_action}")

        success = self._calculate_unit_actions(
            unit,
            desired_action=desired_action,
            unit_must_move=unit_must_move,
            possible_close_enemy=possible_close_enemy,
            mining_planner=mining_planner,
            rubble_planner=rubble_clearing_planner,
            combat_planner=combat_planner,
        )

        # If current location is going to be occupied by another unit, the first action must be to move!
        if unit_must_move:
            q = unit.action_queue
            if (
                len(q) == 0
                or q[0][util.ACT_TYPE] != util.MOVE
                or q[0][util.ACT_DIRECTION] == util.CENTER
            ):
                logger.error(
                    f"{unit.unit_id} was required to move first turn, but actions are {q}"
                )
                # TODO: Then just let it happen?
                # TODO: Or force a move and rerun calculate unit_actions?

        return success

    def _calculate_unit_actions(
        self,
        unit: FriendlyUnitManager,
        desired_action: str,
        unit_must_move: bool,
        possible_close_enemy: [None, CloseUnits],
        mining_planner: MiningPlanner,
        rubble_planner: RubbleClearingPlanner,
        combat_planner: CombatPlanner,
    ):
        success = False
        if desired_action == actions.ATTACK:
            success = combat_planner.attack(unit, possible_close_enemy)
        elif desired_action == actions.RUN_AWAY:
            success = combat_planner.run_away(unit)
        elif desired_action == actions.MINE_ORE:
            rec = mining_planner.recommend(
                unit, util.ORE, unit_must_move=unit_must_move
            )
            if rec is not None:
                success = mining_planner.carry_out(
                    unit, rec, unit_must_move=unit_must_move
                )
            else:
                success = False
        elif desired_action == actions.MINE_ICE:
            rec = mining_planner.recommend(
                unit, util.ICE, unit_must_move=unit_must_move
            )
            if rec is not None:
                success = mining_planner.carry_out(
                    unit, rec, unit_must_move=unit_must_move
                )
            else:
                success = False
        elif desired_action == actions.CLEAR_RUBBLE:
            rec = rubble_planner.recommend(unit)
            success = rubble_planner.carry_out(unit, rec, unit_must_move=unit_must_move)
        elif desired_action == actions.NOTHING:
            logger.debug(
                f"Setting action queue to empty to do action {actions.NOTHING}"
            )
            unit.action_queue = []
            if unit_must_move:
                if not unit.factory_id:
                    logger.error(
                        f"Unit must move, but has action {actions.NOTHING} and no factory assigned"
                    )
                else:
                    # If on factory, stay on factory
                    if unit.on_own_factory():
                        success = util.move_to_new_spot_on_factory(
                            self.master.pathfinder,
                            unit,
                            self.master.factories.friendly[unit.factory_id],
                        )
                        # Just try move
                        if not success:
                            util.move_to_cheapest_adjacent_space(
                                self.master.pathfinder, unit
                            )
                    # Otherwise just move
                    else:
                        util.move_to_cheapest_adjacent_space(
                            self.master.pathfinder, unit
                        )
            success = True
        else:
            logger.error(f"{desired_action} not understood as an action")
            success = False

        if success:
            unit.update_status(new_action=desired_action, success=success)
        else:
            unit.update_status(new_action=actions.NOTHING, success=False)
        return success

    def decide_unit_actions(
        self,
        mining_planner: MiningPlanner,
        rubble_clearing_planner: RubbleClearingPlanner,
        combat_planner: CombatPlanner,
        factory_desires: Dict[str, FactoryDesires],
        factory_infos: Dict[str, FactoryInfo],
    ) -> Dict[str, List[np.ndarray]]:
        """
        Processes the units by choosing the actions the units should take this turn in order of priority

        Returns:
            Actions to update units with
        """
        logger.info(
            f"process_units called with mining_planner: {mining_planner}, rubble_clearing_planner: {rubble_clearing_planner}"
        )

        # Which units should even think about acting
        units_to_act = self._get_units_to_act(self.master.units.friendly.all)

        # Get some info on those units and determine what order they should act
        unit_infos = self._collect_unit_data(units_to_act.needs_to_act)
        unit_infos.sort_by_priority()

        # Get the base costmap for travel
        # (basically rubble converted to cost and enemy factories impassible)
        base_costmap = self._get_base_costmap()

        # Calculate 3D path arrays to use for calculating costmaps later
        existing_paths = UnitPaths.from_units(
            friendly={k: act.unit for k, act in units_to_act.should_not_act.items()},
            enemy=self.master.units.enemy.all,
            friendly_valid_move_map=self.master.maps.valid_friendly_move,
            enemy_valid_move_map=self.master.maps.valid_enemy_move,
            max_step=50,
        )

        # Create the action validator for this turn
        action_validator = ValidActionCalculator(
            units_to_act=units_to_act,
            factory_infos=factory_infos,
            maps=self.master.maps,
        )

        # For each unit, decide to keep same or update actions
        for unit_id, unit_info in unit_infos.infos.items():
            # Remove from needs_to_act queue since we are calculating these actions now
            unit = unit_info.unit
            units_to_act.needs_to_act.pop(unit.unit_id)

            # So I can keep track of units in logs
            logger.info(
                f"\n\nProcessing unit {unit.unit_id}({unit.pos}): "
                f"is_heavy={unit_info.is_heavy}, enough_power_to_move={unit_info.enough_power_to_move}, "
                f"power={unit_info.power}, ice={unit_info.ice}, ore={unit_info.ore}"
            )

            # Re-assign unit to a factory if necessary
            if not unit.factory_id:
                # TODO: pick factory better
                factory_id = next(iter(self.master.factories.friendly.keys()))
                factory = self.master.factories.friendly[factory_id]
                unit.factory_id = factory_id
                factory.assign_unit(unit)
                logger.warning(
                    f"Re-assigning to {factory_id} because no factory assigned"
                )

            # Get the specific costmap for this unit (i.e. blocked enemies and friendly paths)
            # travel_costmap = self._get_travel_costmap_for_unit(
            #     unit=unit,
            #     base_costmap=base_costmap,
            #     units_to_act=units_to_act,
            # )

            # If only considering because action *might* be invalid, check now
            if unit_info.act_info.reason in [
                ActReasons.NEXT_ACTION_PICKUP,
                ActReasons.NEXT_ACTION_TRANSFER,
                ActReasons.NEXT_ACTION_DIG,
            ]:
                if (
                    action_validator.next_action_valid(unit)
                    # and not colliding with friendly next turn
                    and unit.valid_moving_actions(
                        costmap=existing_paths.to_costmap(
                            unit.start_of_turn_pos,
                            start_step=0,
                            exclude_id_nums=[unit.id_num],
                            friendly_light=True,
                            friendly_heavy=True,
                            enemy_heavy=False,
                            enemy_light=False,
                            collision_cost_value=-1,
                            nearby_start_cost=None,
                        ),
                        max_len=1,
                    ).was_valid
                ):
                    logger.debug(f"Next action IS valid, no need to update")
                    # Make sure the next action is taken into account for next validation
                    action_validator.apply_next_action(unit)
                    existing_paths.add_unit(unit, is_enemy=False)
                    units_to_act.should_not_act[unit_id] = unit_info.act_info
                    continue
                else:
                    logger.debug(f"Next action NOT valid, will recalculate actions")

            # Figure out new actions for unit  (i.e. RoutePlanners)
            unit.action_queue = []
            success = self._calculate_actions_for_unit(
                base_costmap=base_costmap,
                unit_paths=existing_paths,
                # travel_costmap=travel_costmap,
                unit_info=unit_info,
                unit=unit,
                factory_desires=factory_desires[unit.factory_id],
                factory_info=factory_infos[unit.factory_id],
                mining_planner=mining_planner,
                rubble_clearing_planner=rubble_clearing_planner,
                combat_planner=combat_planner,
            )

            # If first X actions are the same, don't update (unnecessary cost for unit)
            if np.all(
                np.array(unit.action_queue[: self.actions_same_check])
                == np.array(unit.start_of_turn_actions[: self.actions_same_check])
            ):
                logger.debug(
                    f"First {self.actions_same_check} actions same, not updating unit action queue"
                )
                # Put the action queue back to what it was since we are not updating it
                unit.action_queue = unit.start_of_turn_actions

                # Note: some other things about unit may be wrong, e.g. pos, power. But probably not important from here on (and slow to copy)
                units_to_act.should_not_act[unit.unit_id] = unit_info.act_info
            else:
                logger.info(
                    f"{unit.log_prefix} has updated actions "
                    f"(last updated {self.master.step - unit.status.last_action_update_step} ago),"
                    f"was {unit.status.previous_action}, now {unit.status.current_action}"
                    f" first few actions are {unit.action_queue[:3]}"
                )
                unit.status.last_action_update_step = self.master.step
                units_to_act.has_updated_actions[unit.unit_id] = unit_info.act_info

            # Make sure the next action is taken into account for next validation
            action_validator.apply_next_action(unit)
            # And for next pathing
            existing_paths.add_unit(unit, is_enemy=False)

        # Collect the unit actions for returning to Env
        unit_actions = {}
        for unit_id, act_info in units_to_act.has_updated_actions.items():
            if len(act_info.unit.action_queue) > 0:
                unit_actions[unit_id] = act_info.unit.action_queue[:20]
            else:
                logger.error(
                    f"Updating {unit_id} with empty actions (could be on purpose, but probably should figure out a better thing for this unit to do (even if stay still for a while first))"
                )
                unit_actions[unit_id] = []

        # Quick validation of actions
        for unit_id, actions in unit_actions.items():
            if not valid_action_space(actions):
                logger.error(
                    f"Invalid action (action space) in actions ({actions}) for unit {unit_id}, returning earlier valid actions"
                )
                actions = []
                for i, action in enumerate(actions):
                    logger.debug(f"Checking action {action}")
                    if valid_action_space(action):
                        actions.append(action)
                    else:
                        logger.error(f"Invalid action was {action} at position {i}")
                        break
                if len(actions) == 0:
                    # Move center
                    actions = [np.array([0, 0, 0, 0, 0, 1], dtype=int)]
                unit_actions[unit_id] = actions
        return unit_actions


#######################################


# def find_collisions2(all_unit_paths: AllUnitPaths) -> List[Collision]:
#     """
#     Find collisions between friendly units and all units (friendly and enemy) in the given paths.
#
#     Args:
#         all_unit_paths: AllUnitPaths object containing friendly and enemy unit paths.
#
#     Returns:
#         A list of Collision objects containing information about each detected collision.
#     """
#     raise NotImplementedError('Need to make some modifications first')
#     collisions = []
#
#     friendly_units = {**all_unit_paths.friendly.light, **all_unit_paths.friendly.heavy}
#     enemy_units = {**all_unit_paths.enemy.light, **all_unit_paths.enemy.heavy}
#     all_units = {**friendly_units, **enemy_units}
#
#     for unit_id, unit_path in friendly_units.items():
#         for other_unit_id, other_unit_path in all_units.items():
#             # Skip self-comparison
#             if unit_id == other_unit_id:
#                 continue
#
#             # Broadcast and compare the paths to find collisions
#             unit_path_broadcasted = unit_path[:, np.newaxis]
#             other_unit_path_broadcasted = other_unit_path[np.newaxis, :]
#
#             # Calculate the differences in positions at each step
#             diff = np.abs(unit_path_broadcasted - other_unit_path_broadcasted)
#
#             # Find the indices where both x and y differences are zero (i.e., collisions)
#             collision_indices = np.argwhere(np.all(diff == 0, axis=-1))
#
#             # Create Collision objects for each detected collision
#             for index in collision_indices:
#                 collision = Collision(
#                     unit_id=unit_id,
#                     other_unit_id=other_unit_id,
#                     pos=tuple(unit_path[index[0]]),
#                     step=index[0],
#                 )
#                 collisions.append(collision)
#
#     return collisions
#
#
# def find_collisions3(all_unit_paths: AllUnitPaths) -> List[Collision]:
#     """
#     Find collisions between friendly units and all units (friendly and enemy) in the given paths.
#
#     Args:
#         all_unit_paths: AllUnitPaths object containing friendly and enemy unit paths.
#
#     Returns:
#         A list of Collision objects containing information about each detected collision.
#     """
#
#     def pad_paths(paths_dict: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str]]:
#         max_path_length = max([path.shape[0] for path in paths_dict.values()])
#         padded_paths = []
#         unit_ids = []
#         for unit_id, path in paths_dict.items():
#             padding_length = max_path_length - path.shape[0]
#             padded_path = np.pad(
#                 path,
#                 ((0, padding_length), (0, 0)),
#                 mode='constant',
#                 constant_values=np.nan,
#             )
#             padded_paths.append(padded_path)
#             unit_ids.append(unit_id)
#
#         return np.array(padded_paths), unit_ids
#
#     raise NotImplementedError('Need to make some modifications first')
#     collisions = []
#
#     friendly_units = {**all_unit_paths.friendly.light, **all_unit_paths.friendly.heavy}
#     enemy_units = {**all_unit_paths.enemy.light, **all_unit_paths.enemy.heavy}
#     all_units = {**friendly_units, **enemy_units}
#
#     friendly_paths, friendly_ids = pad_paths(friendly_units)
#     enemy_paths, enemy_ids = pad_paths(enemy_units)
#     all_paths, all_ids = pad_paths(all_units)
#
#     # Broadcast and compare the friendly paths with all paths to find collisions
#     diff = np.abs(friendly_paths[:, :, np.newaxis] - all_paths[np.newaxis, :, :])
#
#     # Find the indices where both x and y differences are zero (i.e., collisions)
#     collision_indices = np.argwhere(np.all(diff == 0, axis=-1))
#
#     # Create Collision objects for each detected collision
#     for index in collision_indices:
#         unit_id = friendly_ids[index[0]]
#         other_unit_id = all_ids[index[2]]
#
#         # Skip self-comparison
#         if unit_id == other_unit_id:
#             continue
#
#         collision = Collision(
#             unit_id=unit_id,
#             other_unit_id=other_unit_id,
#             pos=tuple(friendly_paths[index[0], index[1]]),
#             step=index[1],
#         )
#         collisions.append(collision)
#
#     return collisions
