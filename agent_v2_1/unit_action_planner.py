from __future__ import annotations

import abc
from collections import OrderedDict
import copy
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
from unit_manager import FriendlyUnitManger, UnitManager, EnemyUnitManager

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


# def find_collisions1(
#     all_unit_paths: AllUnitPaths, check_num_steps: int = None
# ) -> List[Collision]:
#     """
#     Find collisions between friendly units and all units (friendly and enemy) in the given paths.
#
#     Args:
#         all_unit_paths: AllUnitPaths object containing friendly and enemy unit paths.
#
#     Returns:
#         A list of Collision objects containing information about each detected collision.
#     """
#     collisions = []
#
#     friendly_units = {**all_unit_paths.friendly.light, **all_unit_paths.friendly.heavy}
#     enemy_units = {**all_unit_paths.enemy.light, **all_unit_paths.enemy.heavy}
#
#     for unit_id, unit_path in friendly_units.items():
#         # if unit_id == 'unit_19' and other_unit_id == 'unit_23':
#         #     print('looking at unit_19')
#         logger.debug(f"Checking collisions for {unit_id}")
#         for other_unit_id, other_unit_path in {**friendly_units, **enemy_units}.items():
#             # if unit_id == 'unit_19' and other_unit_id == 'unit_23':
#             #     print('other', other_unit_id)
#             # Skip self-comparison
#             if unit_id == other_unit_id:
#                 continue
#
#             # Find the minimum path length to avoid index out of range errors
#             min_path_length = min(len(unit_path), len(other_unit_path))
#             # if unit_id == 'unit_19' and other_unit_id == 'unit_23':
#             #     print('min', min_path_length)
#
#             # Optionally only check fewer steps
#             _check = (
#                 min_path_length
#                 if check_num_steps is None
#                 else min(min_path_length, check_num_steps)
#             )
#             # if unit_id == 'unit_19' and other_unit_id == 'unit_23':
#             #     print('check', _check)
#             #     print('other_path', other_unit_path)
#
#             # Check if there's a collision at any step up to check_num_steps
#             for step in range(_check):
#                 # if unit_id == 'unit_19' and other_unit_id == 'unit_23':
#                 #     print('other_path at step',step,  other_unit_path[step])
#                 if np.array_equal(unit_path[step], other_unit_path[step]):
#                     logger.debug(
#                         f"Collision found at step {step} pos {unit_path[step]} with {other_unit_id}"
#                     )
#                     collision = Collision(
#                         unit_id=unit_id,
#                         other_unit_id=other_unit_id,
#                         other_unit_is_enemy=False
#                         if other_unit_id in friendly_units
#                         else True,
#                         pos=tuple(unit_path[step]),
#                         step=step,
#                     )
#                     collisions.append(collision)
#
#     return collisions


@dataclass
class UnitInfo:
    unit: FriendlyUnitManger
    unit_id: str
    len_action_queue: int
    distance_to_factory: Optional[float]
    is_heavy: bool
    unit_type: str
    enough_power_to_move: bool
    power: int
    ice: int
    ore: int


def unit_infos_to_df(unit_infos: Dict[str, UnitInfo]) -> pd.DataFrame:
    # Convert the list of UnitInfo instances to a list of dictionaries
    unit_info_dicts = [unit_info.__dict__ for unit_info in unit_infos.values()]

    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(unit_info_dicts)

    # Set the 'unit_id' column as the DataFrame index
    df.index = df["unit_id"]
    # df.set_index("unit_id", inplace=True)
    return df


@dataclass
class UnitsToAct:
    needs_to_act: dict[str, FriendlyUnitManger]
    should_not_act: dict[str, FriendlyUnitManger]
    has_updated_actions: dict[str, FriendlyUnitManger] = field(default_factory=dict)

    def get_unit(self, unit_id: str) -> FriendlyUnitManger:
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

    def num_collisions(self, friendly=True, enemy=False):
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
    other_unit_positions: List[Tuple[int, int]] = field(default_factory=list)
    other_unit_distances: List[int] = field(default_factory=list)
    other_unit_types: List[str] = field(default_factory=list)
    other_unit_powers: List[int] = field(default_factory=list)


@dataclass
class AllCloseUnits:
    close_to_friendly: Dict[str, CloseUnits]
    close_to_enemy: Dict[str, CloseUnits]


@dataclass
class UnitPaths(abc.ABC):
    light: Dict[str, np.ndarray] = dict
    heavy: Dict[str, np.ndarray] = dict

    @property
    def all(self):
        return dict(**self.light, **self.heavy)


def calculate_collisions(
    all_units: AllUnits, check_steps: int = 2
) -> Dict[str, AllCollisionsForUnit]:
    """Calculate first collisions in the next <check_steps> for all units"""
    all_unit_collisions = {}
    for unit_id, unit in all_units.friendly.all.items():
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
        if collisions_for_unit.num_collisions() > 0:
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
    def attack_or_run_away(
        close_units: CloseUnits,
        unit_type: str,
        power_threshold: int,
    ) -> str:
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
                    return "attack"
            elif len(close_enemies_within_2) > 1:
                return "run_away"
        return None

    action = None
    if unit_info.unit_type == "LIGHT":
        action = attack_or_run_away(close_units, "LIGHT", 10)
        if action is None:
            if factory_info.light_mining_ore < factory_desires.light_mining_ore:
                action = "mine_ore"
            elif (
                factory_info.light_clearing_rubble
                < factory_desires.light_clearing_rubble
            ):
                action = "clear_rubble"
            else:
                action = "do_nothing"
    else:  # unit_type == "HEAVY"
        action = attack_or_run_away(close_units, "HEAVY", 100)
        if action is None:
            if factory_info.heavy_mining_ice < factory_desires.heavy_mining_ice:
                action = "mine_ice"
            elif factory_info.heavy_mining_ore < factory_desires.heavy_mining_ore:
                action = "mine_ore"
            elif factory_info.heavy_attacking < factory_desires.heavy_attacking:
                action = "attack"

    return action


def should_unit_act(
    unit: FriendlyUnitManger,
    upcoming_collisions: Dict[str, AllCollisionsForUnit],
    close_enemies: Dict[str, CloseUnits],
):
    unit_id = unit.unit_id
    should_act = False
    # If not enough power to do something meaningful
    if unit.power < (
        unit.unit_config.ACTION_QUEUE_POWER_COST + unit.unit_config.MOVE_COST
    ):
        logger.debug(f"Not enough power -- {unit_id} should not consider acting")
        should_act = False
    # If no queue
    elif len(unit.action_queue) == 0:
        logger.debug(f"No actions -- {unit_id} should consider acting")
        should_act = True
    # If colliding with friendly
    elif (
        unit_id in upcoming_collisions
        and upcoming_collisions[unit_id].num_collisions(friendly=True, enemy=False) > 0
    ):
        logger.debug(f"Collision with friendly -- {unit_id} should consider acting")
        should_act = True
    # If colliding with enemy
    elif (
        unit_id in upcoming_collisions
        and upcoming_collisions[unit_id].num_collisions(friendly=False, enemy=True) > 0
    ):
        logger.debug(f"Collision with enemy -- {unit_id} should consider acting")
        should_act = True
    # If close to enemy
    elif unit_id in close_enemies:
        logger.debug(f"Close to enemy -- {unit_id} should consider acting")
        should_act = True
    # TODO: If about to do invalid action:
    # TODO: pickup more power than available, dig where no resource, transfer to unoccupied location
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
    # Look for close units within this distance
    search_dist = 4
    # What is considered a close unit when considering future paths
    close_threshold = 3
    # If there will be a collision within this many steps consider acting
    check_collision_steps = 2
    # Increase cost to travel near units based on kernel with this dist
    kernel_dist = 5
    # If this many actions the same, don't update unit
    actions_same_check = 3
    # Number of steps to block other unit path locations for
    avoid_collision_steps = 10

    def __init__(
        self,
        master: MasterState,
        factory_desires: Dict[str, FactoryDesires],
        factory_infos: Dict[str, FactoryInfo],
    ):
        """Assuming this is called after beginning of turn update"""
        self.master = master
        self.factory_desires = factory_desires
        self.factory_infos = factory_infos

        # Caching
        self._costmap: np.ndarray = None
        self._upcoming_collisions: AllCollisionsForUnit = None
        self._close_units: AllCloseUnits = None

    def _get_units_to_act(self, units: Dict[str, FriendlyUnitManger]) -> UnitsToAct:
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
        logger.function_call(
            f"units_should_consider_acting called with len(units): {len(units)}"
        )

        all_unit_collisions = self._calculate_collisions()
        all_unit_close_to_enemy = self._calculate_close_enemies()
        needs_to_act = {}
        should_not_act = {}
        for unit_id, unit in units.items():
            should_act = should_unit_act(
                unit,
                upcoming_collisions=all_unit_collisions,
                close_enemies=all_unit_close_to_enemy,
            )
            if should_act:
                needs_to_act[unit_id] = unit
            else:
                should_not_act[unit_id] = unit
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
                    unit_distance_map = self._unit_distance_map(unit_id)
                    close = CloseUnits(unit_id=unit_id, unit_pos=unit.pos)
                    for other_id, other_unit in other_units.items():
                        if other_id == unit_id:  # Don't compare to self
                            continue
                        dist = unit_distance_map[other_unit.pos[0], other_unit.pos[1]]
                        if dist <= self.close_threshold:
                            close.other_unit_ids.append(other_id)
                            close.other_unit_positions.append(other_unit.pos)
                            close.other_unit_distances.append(dist)
                            close.other_unit_types.append(other_unit.unit_type)
                            close.other_unit_powers.append(other_unit.power)
                    if len(close.other_unit_ids) > 0:
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

    @functools.lru_cache(maxsize=128)
    def _unit_distance_map(self, unit_id: str) -> np.ndarray:
        """Calculate the distance map for the given unit, this will be used to determine how close other units are"""
        unit = self.master.units.get_unit(unit_id)
        unit_distance_map = util.pad_and_crop(
            util.manhattan_kernel(self.search_dist),
            large_arr=self.master.maps.rubble,
            x1=unit.pos[0],
            y1=unit.pos[1],
            fill_value=self.search_dist,
        )
        return unit_distance_map

    def _collect_unit_data(
        self, units: Dict[str, FriendlyUnitManger]
    ) -> Dict[str, UnitInfo]:
        """
        Collects data from units and stores it in a pandas dataframe.

        Args:
            units: List of FriendlyUnitManger objects.

        Returns:
            A pandas dataframe containing the unit data.
        """
        data = {}
        for unit_id, unit in units.items():
            unit_factory = self.master.factories.friendly.get(unit.factory_id, None)
            unit_distance_map = self._unit_distance_map(unit_id)

            unit_info = UnitInfo(
                unit=unit,
                unit_id=unit.unit_id,
                len_action_queue=len(unit.action_queue),
                distance_to_factory=(
                    unit_distance_map[
                        unit_factory.factory.pos[0], unit_factory.factory.pos[1]
                    ]
                    if unit_factory
                    else np.nan
                ),
                is_heavy=unit.unit_type == "HEAVY",
                unit_type=unit.unit_type,
                enough_power_to_move=(
                    unit.power
                    > unit.unit_config.MOVE_COST
                    + unit.unit_config.ACTION_QUEUE_POWER_COST
                ),
                power=unit.power,
                ice=unit.cargo.ice,
                ore=unit.cargo.ore,
            )
            data[unit_id] = unit_info
        return data

    def _sort_units_by_priority(
        self, unit_infos: Dict[str, UnitInfo]
    ) -> OrderedDict[str, UnitInfo]:
        """
        Sorts units by priority based on the provided dataframe.

        Args:
            df: A pandas dataframe containing the unit data.

        Returns:
            A sorted pandas dataframe with units ordered by priority.
        """
        logger.function_call(f"sort_units_by_priority called")
        if len(unit_infos) == 0:
            logger.debug("No unit_infos data to sort, returning as is")
            return unit_infos

        df = unit_infos_to_df(unit_infos)

        sorted_df = df.sort_values(
            by=["is_heavy", "enough_power_to_move", "power", "ice", "ore"],
            ascending=[False, False, True, False, True],
        )
        logger.debug(f"Sorted units by priority")
        highest = sorted_df.iloc[0]
        lowest = sorted_df.iloc[-1]
        logger.debug(
            f"Unit with highest priority: {highest.unit_id}  ({highest.unit.pos}), is_heavy={highest.is_heavy}, power={highest.power}, ice={highest.ice}, ore={highest.ore}, len_acts={highest.len_action_queue}"
        )
        logger.debug(
            f"Unit with lowest priority: {lowest.unit_id}  ({lowest.unit.pos}), is_heavy={lowest.is_heavy}, power={lowest.power}, ice={lowest.ice}, ore={lowest.ore}, len_acts={lowest.len_action_queue}"
        )
        ordered_infos = OrderedDict()
        for unit_id in sorted_df.index:
            ordered_infos[unit_id] = unit_infos[unit_id]
        return ordered_infos

    def _get_base_costmap(self) -> np.ndarray:
        """
        Calculates the base costmap based on:
            - rubble array
            - Enemy factories impassible

        Returns:
            A numpy array representing the costmap.
        """
        if self._costmap is None:
            costmap = self.master.maps.rubble.copy() * 0.1  # was 0.05
            costmap += 1  # Zeros aren't traversable
            enemy_factory_map = self.master.maps.factory_maps.enemy
            costmap[enemy_factory_map >= 0] = -1  # Not traversable
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

        if allow_collision is False:
            # Enemy may change plans, on first step don't allow being even 1 dist away
            if is_enemy and other_dist_now <= 2:
                logger.info(
                    f"Enemy is close at {other_pos_now}, blocking adjacent cells"
                )
                # Block all adjacent cells to enemy (and current enemy pos)
                for delta in util.MOVE_DELTAS:
                    pos = np.array(other_pos_now) + delta
                    costmap[pos[0], pos[1]] = -1

            # Block next X steps in other path that are equal in distance
            for i, (p, d) in enumerate(
                zip(other_path[: self.avoid_collision_steps], other_path_distance)
            ):
                if (
                    d == i + 1
                ):  # I.e. if distance to point on path is same as no. steps it would take to get there
                    logger.info(f"making {p} impassable")
                    costmap[p[0], p[1]] = -1

        # If current location becomes blocked, warn that should be unblocked elsewhere
        if costmap[unit_pos[0], unit_pos[1]] == -1:
            logger.warning(
                f"{unit_pos} got blocked even though that is the units current position. If cost not changed > 0 pathing will fail"
            )

        return costmap

    def _add_other_unit_to_costmap(
        self,
        costmap: np.ndarray,
        this_unit: FriendlyUnitManger,
        other_unit: [FriendlyUnitManger, EnemyUnitManager],
        other_is_enemy: bool,
    ) -> np.ndarray:
        """Add or removes cost from cost map based on distance and path of nearby unit"""

        if this_unit.unit_type == "LIGHT":
            # If we have 10 more energy, prefer moving toward
            low_power_diff, high_power_diff = -1, 10
        else:
            low_power_diff, high_power_diff = -1, 100

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
        unit: FriendlyUnitManger,
    ) -> np.ndarray:
        """
        Updates the costmap with the paths of the units that have determined paths this turn (not acting, done acting, or enemy)

        Args:
            base_costmap: A numpy array representing the costmap.
            units_to_act: UnitsToAct instance containing units that need to act and units that should not act.
            unit: Unit to get the costmap for (i.e. distances calculated relative to this unit)
        """
        logger.function_call(f"Calculating costmap with paths for unit {unit.unit_id}")
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
        travel_costmap: np.ndarray,
        unit_info: UnitInfo,
        unit: FriendlyUnitManger,
        factory_desires: FactoryDesires,
        factory_info: FactoryInfo,
        mining_planner: MiningPlanner,
        rubble_clearing_planner: RubbleClearingPlanner,
        combat_planner: CombatPlanner,
    ) -> bool:
        """Calculate new actions for this unit"""
        logger.function_call(
            f"Beginning calculating action for {unit.unit_id}: power = {unit.power}, pos = {unit.pos}, len(actions) = {len(unit.action_queue)}, current_action = {unit.status.current_action}"
        )
        # def calculate_unit_actions(unit: FriendlyUnitManger, unit_must_move: bool, poss_close_enemy: [None, CloseUnits], row: pd.Series):
        #     _success = False
        #     if unit.unit_type == 'HEAVY':
        #         _success = mining_planner.update_actions_of(
        #             unit, unit_must_move=unit_must_move, resource_type=util.ICE
        #         )
        #     elif unit.unit_type == 'LIGHT':
        #         _success = rubble_clearing_planner.update_actions_of(
        #             unit, unit_must_move=unit_must_move
        #         )
        #     return _success

        # Update the master pathfinder with newest Pather (full_costmap changes for each unit)
        self.master.pathfinder = Pather(
            base_costmap=base_costmap,
            full_costmap=travel_costmap,
        )

        unit_must_move = False
        # If current location is blocked, unit MUST move first turn
        if travel_costmap[unit.pos[0], unit.pos[1]] <= 0:
            logger.warning(
                f"{unit.unit_id} MUST move first turn to avoid collision at {unit.pos}"
            )
            unit_must_move = True
            travel_costmap[
                unit.pos[0], unit.pos[1]
            ] = 100  # <= 0 breaks pathing, 100 will make unit avoid this position for future travel

        # unit_before = copy.deepcopy(unit)
        unit.action_queue = []

        close_units = self._calculate_close_units()
        possible_close_enemy: [None, CloseUnits] = close_units.close_to_enemy.get(
            unit.unit_id, None
        )
        # action_type = decide_action(df_row, possible_close_enemy)

        # TODO: If close to enemy and should attack - do it
        # TODO: If close to enemy and run away - do it
        # TODO: Collect some factory obs to help decide what to do

        desired_action = decide_action(
            unit_info=unit_info,
            factory_desires=factory_desires,
            factory_info=factory_info,
            close_units=possible_close_enemy,
        )
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
        logger.function_call(
            f"process_units called with mining_planner: {mining_planner}, rubble_clearing_planner: {rubble_clearing_planner}"
        )

        units_to_act = self._get_units_to_act(self.master.units.friendly.all)
        unit_infos = self._collect_unit_data(units_to_act.needs_to_act)
        sorted_unit_infos = self._sort_units_by_priority(unit_infos)

        base_costmap = self._get_base_costmap()

        for unit_id, unit_info in sorted_unit_infos.items():
            # Note: Row is a pd.Series of the unit_data_df
            # Remove from needs_to_act queue since we are calculating these actions now
            unit = unit_info.unit
            units_to_act.needs_to_act.pop(unit.unit_id)

            # Re-assign unit to a factory if necessary
            if not unit.factory_id:
                # TODO: pick factory better
                factory_id = next(iter(self.master.factories.friendly.keys()))
                factory = self.master.factories.friendly[factory_id]
                unit.factory_id = factory_id
                factory.assign_unit(unit)

            logger.info(
                f"\n\nProcessing unit {unit.unit_id}({unit.pos}): "
                f"is_heavy={unit_info.is_heavy}, enough_power_to_move={unit_info.enough_power_to_move}, "
                f"power={unit_info.power}, ice={unit_info.ice}, ore={unit_info.ore}"
            )
            travel_costmap = self._get_travel_costmap_for_unit(
                unit=unit,
                base_costmap=base_costmap,
                units_to_act=units_to_act,
            )

            unit_before = copy.deepcopy(unit)
            # Figure out new actions for unit  (i.e. RoutePlanners)
            success = self._calculate_actions_for_unit(
                base_costmap=base_costmap,
                travel_costmap=travel_costmap,
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
                == np.array(unit_before.action_queue[: self.actions_same_check])
            ):
                logger.debug(
                    f"First {self.actions_same_check} actions same, not updating unit action queue"
                )
                #  Store the unit_before (i.e. not updated at all since it's not changing it's actions)
                units_to_act.should_not_act[unit.unit_id] = unit_before
                self.master.units.friendly.replace_unit(unit.unit_id, unit_before)
            else:
                units_to_act.has_updated_actions[unit.unit_id] = unit

        actions = {}

        # TODO: One last check for collisions?

        for unit_id, unit in units_to_act.has_updated_actions.items():
            if len(unit.action_queue) > 0:
                actions[unit_id] = unit.action_queue[:20]
        return actions

    def _calculate_unit_actions(
        self,
        unit: FriendlyUnitManger,
        desired_action: str,
        unit_must_move: bool,
        possible_close_enemy: [None, CloseUnits],
        mining_planner: MiningPlanner,
        rubble_planner: RubbleClearingPlanner,
        combat_planner: CombatPlanner,
    ):
        unit.status.previous_action = unit.status.current_action
        unit.status.current_action = actions.NOTHING
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
            success = True
            pass
        else:
            logger.error(f"{desired_action} not understood as an action")

        unit.status.current_action = desired_action
        unit.status.last_action_success = success
        return success


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
