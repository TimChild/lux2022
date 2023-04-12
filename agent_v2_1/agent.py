from __future__ import annotations
import abc
import copy
import functools
from typing import TYPE_CHECKING, List, Dict, Tuple
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
import logging

from lux.kit import obs_to_game_state
from lux.config import EnvConfig
from lux.utils import my_turn_to_place_factory

from unit_manager import UnitManager, FriendlyUnitManger, EnemyUnitManager
from master_state import MasterState
from factory_manager import FriendlyFactoryManager
from actions import (
    factory_should_consider_acting,
)
from new_path_finder import Pather
from mining_planner import MiningPlanner
from rubble_clearing_planner import RubbleClearingPlanner

import util

logging.basicConfig(level=logging.INFO)
logging.info('Starting Log')

if TYPE_CHECKING:
    from .master_state import Recommendation


def find_collisions1(
    all_unit_paths: AllUnitPaths, check_num_steps: int = None
) -> List[Collision]:
    """
    Find collisions between friendly units and all units (friendly and enemy) in the given paths.

    Args:
        all_unit_paths: AllUnitPaths object containing friendly and enemy unit paths.

    Returns:
        A list of Collision objects containing information about each detected collision.
    """
    collisions = []

    friendly_units = {**all_unit_paths.friendly.light, **all_unit_paths.friendly.heavy}
    enemy_units = {**all_unit_paths.enemy.light, **all_unit_paths.enemy.heavy}

    for unit_id, unit_path in friendly_units.items():
        for other_unit_id, other_unit_path in {**friendly_units, **enemy_units}.items():
            # Skip self-comparison
            if unit_id == other_unit_id:
                continue

            # Find the minimum path length to avoid index out of range errors
            min_path_length = min(len(unit_path), len(other_unit_path))

            # Optionally only check fewer steps
            check_num_steps = (
                min_path_length
                if check_num_steps is None
                else min(min_path_length, check_num_steps)
            )

            # Check if there's a collision at any step up to check_num_steps
            for step in range(check_num_steps):
                if np.array_equal(unit_path[step], other_unit_path[step]):
                    collision = Collision(
                        unit_id=unit_id,
                        other_unit_id=other_unit_id,
                        other_unit_is_enemy=False
                        if other_unit_id in friendly_units
                        else True,
                        pos=tuple(unit_path[step]),
                        step=step,
                    )
                    collisions.append(collision)

    return collisions


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


@dataclass
class UnitsToAct:
    needs_to_act: dict[str, FriendlyUnitManger]
    should_not_act: dict[str, FriendlyUnitManger]
    has_updated_actions: dict[str, FriendlyUnitManger] = field(default_factory=dict)

    def get_unit(self, unit_id: str) -> FriendlyUnitManger:
        for d in [self.needs_to_act, self.should_not_act, self.has_updated_actions]:
            if unit_id in d:
                return d[unit_id]
        raise KeyError(f'{unit_id} not in UnitsToAct')


@dataclass
class Collision:
    """First collision only"""

    unit_id: str
    other_unit_id: str
    other_unit_is_enemy: bool
    pos: Tuple[int, int]
    step: int


@dataclass
class Collisions:
    friendly: Dict[str, Collision]  # Collisions with friendly unit
    enemy: Dict[str, Collision]  # Collisions with enemy units


@dataclass
class CloseUnits:
    """Record nearby units"""

    unit_id: str
    unit_pos: Tuple[int, int]
    other_unit_ids: List[str] = field(default_factory=list)
    other_unit_positions: List[Tuple[int, int]] = field(default_factory=list)
    other_unit_distances: List[int] = field(default_factory=list)


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


@dataclass
class FriendlyUnitPaths(UnitPaths):
    pass


@dataclass
class EnemyUnitPaths(UnitPaths):
    pass


@dataclass
class AllUnitPaths:
    friendly: FriendlyUnitPaths = FriendlyUnitPaths
    enemy: EnemyUnitPaths = EnemyUnitPaths

    def __post_init__(self):
        self.unit_location_dict = {
            unit_id: d
            for d in [
                self.friendly.light,
                self.friendly.heavy,
                self.enemy.light,
                self.enemy.heavy,
            ]
            for unit_id in d
        }

    def get_unit(self, unit_id: str):
        for paths in [self.friendly, self.enemy]:
            if unit_id in paths.all:
                return paths.all[unit_id]
        raise ValueError(f'{unit_id} not in AllUnitPaths')

    def update_path(self, unit: UnitManager):
        """Update the path of a unit that is already in AllUnitPaths"""
        unit_id, path = unit.unit_id, unit.current_path
        if unit_id not in self.unit_location_dict:
            raise KeyError(
                f'{unit_id} is not in the AllUnitPaths. Only have {self.unit_location_dict.keys()}'
            )
        self.unit_location_dict[unit_id][unit_id] = path

    def calculate_collisions(self, check_steps: int = 2) -> Collisions:
        """Calculate first collisions in the next <check_steps> for all units"""
        collisions = find_collisions1(self, check_num_steps=check_steps)
        friendly = []
        enemy = []
        for collision in collisions:
            if collision.other_unit_is_enemy:
                enemy.append(collision)
            else:
                friendly.append(collision)
        collisions = Collisions(
            friendly={collision.unit_id: collision for collision in friendly},
            enemy={collision.unit_id: collision for collision in enemy},
        )
        return collisions


class TurnPlanner:
    # Look for close units within this distance
    search_dist = 6
    # What is considered a close unit when considering future paths
    close_threshold = 5
    # If there will be a collision within this many steps consider acting
    check_collision_steps = 2
    # Increase cost to travel near units based on kernel with this dist
    kernel_dist = 5
    # If this many actions the same, don't update unit
    actions_same_check = 3
    # Number of steps to block other unit path locations for
    avoid_collision_steps = 5

    def __init__(self, master: MasterState):
        """Assuming this is called after beginning of turn update"""
        self.master = master

        # Caching
        self._costmap: np.ndarray = None
        self._upcoming_collisions: Collisions = None
        self._close_units: AllCloseUnits = None

    def units_should_consider_acting(
        self, units: Dict[str, FriendlyUnitManger]
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
        upcoming_collisions = self.calculate_collisions()
        close_to_enemy = self.calculate_close_enemies()
        needs_to_act = {}
        should_not_act = {}
        for unit_id, unit in units.items():
            should_act = False
            # If not enough power to do something meaningful
            if unit.power < (
                unit.unit_config.ACTION_QUEUE_POWER_COST + unit.unit_config.MOVE_COST
            ):
                logging.info(
                    f'not enough power -- {unit_id} should not consider acting'
                )
                should_act = False
            # If no queue
            elif len(unit.action_queue) == 0:
                logging.info(f'no actions -- {unit_id} should consider acting')
                should_act = True
            # If colliding with friendly
            elif unit_id in upcoming_collisions.friendly:
                logging.info(
                    f'collision with friendly -- {unit_id} should consider acting'
                )
                should_act = True
            # If colliding with enemy
            elif unit_id in upcoming_collisions.enemy:
                logging.info(
                    f'collision with enemy -- {unit_id} should consider acting'
                )
                should_act = True
            # If close to enemy
            elif unit_id in close_to_enemy:
                logging.info(f'close to enemy -- {unit_id} should consider acting')
                should_act = True
            # TODO: If about to do invalid action:
            # TODO: pickup more power than available, dig where no resource, transfer to unoccupied location

            if should_act:
                needs_to_act[unit_id] = unit
            else:
                should_not_act[unit_id] = unit
        return UnitsToAct(needs_to_act=needs_to_act, should_not_act=should_not_act)

    def get_unit_paths(self) -> AllUnitPaths:
        """Gets the current unit paths"""
        units = self.master.units
        # Collect all current paths of units
        friendly_paths = FriendlyUnitPaths(
            light={
                unit_id: unit.current_path
                for unit_id, unit in units.friendly.light.items()
            },
            heavy={
                unit_id: unit.current_path
                for unit_id, unit in units.friendly.heavy.items()
            },
        )
        enemy_paths = EnemyUnitPaths(
            light={
                unit_id: unit.current_path
                for unit_id, unit in units.enemy.light.items()
            },
            heavy={
                unit_id: unit.current_path
                for unit_id, unit in units.enemy.heavy.items()
            },
        )
        all_unit_paths = AllUnitPaths(friendly=friendly_paths, enemy=enemy_paths)
        return all_unit_paths

    def calculate_collisions(self) -> Collisions:
        """Calculates the upcoming collisions based on action queues of all units"""
        # if self._upcoming_collisions is None:
        if True:  # TODO: Can I cache this?
            all_unit_paths = self.get_unit_paths()
            collisions = all_unit_paths.calculate_collisions(
                check_steps=self.check_collision_steps
            )
            self._upcoming_collisions = collisions
        return self._upcoming_collisions

    def calculate_close_units(self) -> AllCloseUnits:
        """Calculates which units are close to enemies"""
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
                    if len(close.other_unit_ids) > 0:
                        all_close[unit_id] = close
            all_close_units = AllCloseUnits(
                close_to_friendly=friendly, close_to_enemy=enemy
            )
            self._close_units = all_close_units
        return self._close_units

    def calculate_close_enemies(self) -> Dict[str, CloseUnits]:
        close_units = self.calculate_close_units()
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

    def collect_unit_data(self, units: Dict[str, FriendlyUnitManger]) -> pd.DataFrame:
        """
        Collects data from units and stores it in a pandas dataframe.

        Args:
            units: List of FriendlyUnitManger objects.

        Returns:
            A pandas dataframe containing the unit data.
        """
        data = []
        for unit_id, unit in units.items():
            unit_factory = self.master.factories.friendly.get(unit.factory_id, None)
            unit_distance_map = self._unit_distance_map(unit_id)

            data.append(
                {
                    'unit': unit,
                    'distance_to_factory': unit_distance_map[
                        unit_factory.factory.pos[0], unit_factory.factory.pos[1]
                    ]
                    if unit_factory
                    else np.nan,
                    'is_heavy': unit.unit_type == 'HEAVY',
                    'enough_power_to_move': unit.power
                    > unit.unit_config.MOVE_COST
                    + unit.unit_config.ACTION_QUEUE_POWER_COST,
                    'power': unit.power,
                    'ice': unit.cargo.ice,
                    'ore': unit.cargo.ore,
                }
            )

        df = pd.DataFrame(data)
        return df

    def sort_units_by_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts units by priority based on the provided dataframe.

        Args:
            df: A pandas dataframe containing the unit data.

        Returns:
            A sorted pandas dataframe with units ordered by priority.
        """
        if not df.empty:
            sorted_df = df.sort_values(
                by=['is_heavy', 'enough_power_to_move', 'power', 'ice', 'ore'],
                ascending=[True, False, True, False, True],
            )
            return sorted_df
        else:
            return df

    def base_costmap(self) -> np.ndarray:
        """
        Calculates the base costmap based on:
            - rubble array
            - most travelled?

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

    def update_costmap_with_path(
        self,
        costmap: np.ndarray,
        unit_pos: util.POS_TYPE,
        other_path: util.PATH_TYPE,
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
            avoidance: How  much extra cost to be near other_poth (this would be added at collision points, then decreasing amount added near collision points)
            allow_collision: If False, coordinates likely to result in collision are make impassable (likely means the manhattan distance to the path coord is the same or slightly lower)
        """

        def generate_collision_likelihood_array(distances: np.ndarray) -> np.ndarray:
            index_positions = np.arange(len(distances))
            distance_diffs = np.abs(index_positions - distances)

            # Gaussian-like function (you can adjust the scale and exponent as needed)
            likelihood_array = np.exp(-0.5 * (distance_diffs**1))

            return likelihood_array
        logging.info(f'Updating costmap with path unit at {unit_pos} with other_path {other_path} (first value in '
                     f'other_path is other units CURRENT location that will be removed before calculating collisions '
                     f'etc)')
        other_path = other_path[1:]
        # Figure out distance to other_units path at each point
        other_path_distance = [util.manhattan(p, unit_pos) for p in other_path]

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
            # Block next X steps in other path that are equal in distance
            for i, (p, d) in enumerate(zip(other_path[:self.avoid_collision_steps], other_path_distance)):
                if d == i:  # I.e. if distance to point on path is same as no. steps it would take to get there
                    logging.info(f'making {p} impassable')
                    costmap[p[0], p[1]] = -1

        # If current location becomes blocked, warn that should be unblocked elsewhere
        if costmap[unit_pos[0], unit_pos[1]] == -1:
            logging.warning(
                f'{unit_pos} got blocked even though that is the units current position. If cost not changed > 0 pathing will fail'
            )

        return costmap

    def update_costmap_with_unit(
        self,
        costmap: np.ndarray,
        this_unit: FriendlyUnitManger,
        other_unit: [FriendlyUnitManger, EnemyUnitManager],
        other_is_enemy: bool,
    ) -> np.ndarray:
        """Add or removes cost from cost map based on distance and path of nearby unit"""

        def handle_collision_case(
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

        if this_unit.unit_type == 'LIGHT':
            # If we have 10 more energy, prefer moving toward
            low_power_diff, high_power_diff = -1, 10
        else:
            low_power_diff, high_power_diff = -1, 100

        # TODO: Actually use avoidance or remove it completely
        avoidance, allow_collision = handle_collision_case(
            this_unit,
            other_unit,
            is_enemy=other_is_enemy,
            power_threshold_low=low_power_diff,
            power_threshold_high=high_power_diff,
        )
        logging.info(
            f'For this {this_unit.unit_id} and other {other_unit.unit_id} - travel avoidance = {avoidance} and allow_collision = {allow_collision}'
        )
        if avoidance == 0 and allow_collision is True:
            # Ignore unit
            pass
        else:
            other_path = other_unit.current_path
            costmap = self.update_costmap_with_path(
                costmap,
                this_unit.pos,
                other_path,
                avoidance=avoidance,
                allow_collision=allow_collision,
            )
        return costmap

    def get_travel_costmap(
        self,
        base_costmap: np.ndarray,
        units_to_act: UnitsToAct,
        unit: FriendlyUnitManger,
    ) -> np.ndarray:
        """
        Updates the costmap with the paths of the units that have determined paths this turn (not acting, done acting, or enemy)

        Args:
            base_costmap: A numpy array representing the costmap.
            unit: Unit to get the costmap for (i.e. distances calculated relative to this unit)
        """
        logging.info(f'For {unit.unit_id}, calculating costmap with paths')
        new_cost = base_costmap.copy()

        all_close_units = self.calculate_close_units()
        units_yet_to_act = units_to_act.needs_to_act.keys()

        # If close to enemy, add those paths
        if unit.unit_id in all_close_units.close_to_enemy:
            logging.info(
                f'{unit.unit_id} is close to at least one enemy, adding those to costmap'
            )
            close_units = all_close_units.close_to_enemy[unit.unit_id]
            # For each nearby enemy unit
            for other_id in close_units.other_unit_ids:
                other_unit = self.master.units.enemy.get_unit(other_id)
                new_cost = self.update_costmap_with_unit(
                    new_cost, this_unit=unit, other_unit=other_unit, other_is_enemy=True
                )

        # If close to friendly, add those paths
        if unit.unit_id in all_close_units.close_to_friendly:
            logging.info(
                f'{unit.unit_id} is close to at least one friendly, adding those to costmap'
            )
            close_units = all_close_units.close_to_friendly[unit.unit_id]

            # For each friendly unit if it has already acted or is not acting this turn (others can get out of the way)
            for other_id in close_units.other_unit_ids:
                if other_id in units_yet_to_act:
                    # That other unit can get out of the way
                    logging.info(
                        f'For {unit.unit_id}, Not adding friendly {other_id}s path to costmap, assuming it will get out of the way'
                    )
                    continue
                other_unit = self.master.units.friendly.get_unit(other_id)
                new_cost = self.update_costmap_with_unit(
                    new_cost,
                    this_unit=unit,
                    other_unit=other_unit,
                    other_is_enemy=False,
                )

        logging.info(f'For {unit.unit_id}, done calculating costmap with paths')
        return new_cost

    def calculate_actions_for_unit(
        self,
        base_costmap: np.ndarray,
        travel_costmap: np.ndarray,
        df_row: pd.Series,
        unit: FriendlyUnitManger,
        mining_planner: MiningPlanner,
        rubble_clearing_planner: RubbleClearingPlanner,
    ) -> bool:
        """Calculate new actions for this unit"""
        # Update the master pathfinder with newest Pather (full_costmap changes for each unit)
        self.master.pathfinder = Pather(
            base_costmap=base_costmap,
            full_costmap=travel_costmap,
        )

        def calculate_unit_actions(unit, unit_must_move):
            _success = False
            if unit.unit_type == 'HEAVY':
                _success = mining_planner.update_actions_of(
                    unit, unit_must_move=unit_must_move, resource_type=util.ICE
                )
            elif unit.unit_type == 'LIGHT':
                _success = rubble_clearing_planner.update_actions_of(
                    unit, unit_must_move=unit_must_move
                )
            return _success

        unit_must_move = False
        # If current location is blocked, unit MUST move first turn
        if travel_costmap[unit.pos[0], unit.pos[1]] <= 0:
            logging.warning(
                f'{unit.unit_id} MUST move first turn to avoid collision at {unit.pos}'
            )
            unit_must_move = True
            travel_costmap[
                unit.pos[0], unit.pos[1]
            ] = 100  # <= 0 breaks pathing, 100 will make unit avoid this position for future travel

        # unit_before = copy.deepcopy(unit)
        unit.action_queue = []

        # TODO: If close to enemy and should attack - do it
        # TODO: If close to enemy and run away - do it
        # TODO: Collect some factory obs to help decide what to do
        # TEMPORARY
        logging.info(f'Deciding actions for {unit.unit_id}')
        success = calculate_unit_actions(unit, unit_must_move)
        # If current location is going to be occupied by another unit, the first action must be to move!
        if unit_must_move:
            q = unit.action_queue
            if (
                len(q) == 0
                or q[0][util.ACT_TYPE] != util.MOVE
                or q[0][util.ACT_DIRECTION] == util.CENTER
            ):
                logging.error(
                    f'{unit.unit_id} was required to move first turn, but actions are {q}'
                )
                # TODO: Then just let it happen?
                # TODO: Or force a move and rerun calculate unit_actions?

        return success

    def process_units(
        self,
        mining_planner: MiningPlanner,
        rubble_clearing_planner: RubbleClearingPlanner,
    ) -> Dict[str, List[np.ndarray]]:
        """
        Processes the units by choosing the actions the units should take this turn in order of priority

        Returns:
            Actions to update units with
        """
        units_to_act = self.units_should_consider_acting(self.master.units.friendly.all)
        unit_data_df = self.collect_unit_data(units_to_act.needs_to_act)
        sorted_data_df = self.sort_units_by_priority(unit_data_df)

        base_costmap = self.base_costmap()


        for index, row in sorted_data_df.iterrows():
            # Note: Row is a pd.Series of the unit_data_df
            # Remove from needs_to_act queue since we are calculating these actions now
            unit = row.unit
            unit: FriendlyUnitManger
            units_to_act.needs_to_act.pop(unit.unit_id)
            travel_costmap = self.get_travel_costmap(
                unit=unit,
                base_costmap=base_costmap,
                units_to_act=units_to_act,
            )

            unit_before = copy.deepcopy(unit)
            # Figure out new actions for unit  (i.e. RoutePlanners)
            success = self.calculate_actions_for_unit(
                base_costmap=base_costmap,
                travel_costmap=travel_costmap,
                df_row=row,
                unit=unit,
                mining_planner=mining_planner,
                rubble_clearing_planner=rubble_clearing_planner,
            )

            # If first X actions are the same, don't update (unnecessary cost for unit)
            if np.all(
                np.array(unit.action_queue[: self.actions_same_check])
                == np.array(unit_before.action_queue[: self.actions_same_check])
            ):
                logging.info(f'For {unit.unit_id}, first {self.actions_same_check} actions same, not update units action queue')
                #  Store the unit_before (i.e. not updated at all since it's not changing it's actions)
                units_to_act.should_not_act[unit.unit_id] = unit_before
                self.master.units.friendly.replace_unit(unit.unit_id, unit_before)
            else:
                units_to_act.has_updated_actions[unit.unit_id] = unit

        actions = {}

        # TODO: One last check for collisions?

        for unit_id, unit in units_to_act.has_updated_actions.items():
            if len(unit.action_queue) > 0:
                actions[unit_id] = unit.action_queue
        return actions


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig):
        logging.info(f'initializing agent for player {player}')
        self.player = player
        self.env_cfg: EnvConfig = env_cfg
        np.random.seed(0)

        # Additional initialization
        self.last_obs = None
        self.master: MasterState = MasterState(
            player=self.player,
            env_cfg=env_cfg,
        )

        self.mining_planner = MiningPlanner(self.master)
        self.rubble_clearing_planner = RubbleClearingPlanner(self.master)

    def bid(self, obs):
        """Bid for starting factory (default to 0)"""
        return dict(faction="TheBuilders", bid=0)

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        """Required API for Agent. This is called until all factories are placed"""
        self._beginning_of_step_update(step, obs, remainingOverageTime)

        action = dict()
        if step == 0:
            action = self.bid(obs)
        else:
            # factory placement period
            my_turn_to_place = my_turn_to_place_factory(
                self.master.game_state.teams[self.player].place_first, step
            )
            factories_to_place = self.master.game_state.teams[
                self.player
            ].factories_to_place
            if factories_to_place > 0 and my_turn_to_place:
                action = FriendlyFactoryManager.place_factory(
                    self.master.game_state, self.player
                )
        logging.info(f'Early setup action {action}')
        return action

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """Required API for Agent. This is called every turn after early_setup is complete"""
        logging.warning(
            f'======== Start of turn {self.master.game_state.real_env_steps} for {self.player} ============'
        )
        self._beginning_of_step_update(step, obs, remainingOverageTime)

        self.mining_planner.update()
        self.rubble_clearing_planner.update()

        tp = TurnPlanner(self.master)
        unit_actions = tp.process_units(
            mining_planner=self.mining_planner,
            rubble_clearing_planner=self.rubble_clearing_planner,
        )

        #
        # # Get processed observations (i.e. the obs that I will use to train a PPO agent)
        # # First - general observations of the full game state (i.e. not factory/unit specific)
        # general_obs = self._get_general_processed_obs()
        #
        # Then additional observations for units (not things they already store)
        # unit_obs: dict[str, UnitObs] = {}
        # for unit_id, unit in self.master.units.friendly.all.items():
        #     if unit_should_consider_acting(unit, self.master):
        #         uobs = calculate_unit_obs(unit, self.master)
        #         unit_obs[unit_id] = uobs

        # Then additional observations for factories (not things they already store)
        factory_obs = {}
        for factory_id, factory in self.master.factories.friendly.items():
            if factory_should_consider_acting(factory, self.master):
                fobs = calculate_factory_obs(factory, self.master)
                factory_obs[factory_id] = fobs

        # Factory Actions
        factory_actions = {}
        for factory_id in factory_obs.keys():
            factory = self.master.factories.friendly[factory_id]
            fobs = factory_obs[factory_id]
            f_action = calculate_factory_action(
                factory=factory, fobs=fobs, master=self.master
            )
            if f_action is not None:
                factory_actions[factory_id] = f_action

        logging.debug(f'{self.player} Unit actions: {unit_actions}')
        logging.info(f'{self.player} Factory actions: {factory_actions}')

        return dict(**unit_actions, **factory_actions)
        #
        # # Unit Recommendations
        # unit_recommendations = {}
        # for unit_id in unit_obs.keys():
        #     unit = self.master.units.friendly.get_unit(unit_id)
        #     # TODO: maybe add some logic here as to whether to get recommendations??
        #     mining_rec = self.mining_planner.recommend(unit_manager=unit)
        #     rubble_clearing_rec = self.rubble_clearing_planner.recommend(unit=unit)
        #     unit_recommendations[unit_id] = {
        #         'mine_ice': mining_rec,
        #         'clear_rubble': rubble_clearing_rec,
        #     }
        #
        # # Unit Actions
        # unit_actions = {}
        # for unit_id in unit_obs.keys():
        #     unit = self.master.units.friendly.get_unit(unit_id)
        #     uobs = unit_obs[unit_id]
        #     if unit.factory_id:
        #         fobs = factory_obs[unit.factory_id]
        #         factory = self.master.factories.friendly[unit.factory_id]
        #     else:
        #         fobs = None
        #         factory = None
        #     u_action = self.calculate_unit_actions(
        #         unit=unit,
        #         uobs=uobs,
        #         factory=factory,
        #         fobs=fobs,
        #         unit_recommendations=unit_recommendations[unit_id],
        #     )
        #     if u_action is not None:
        #         unit_actions[unit_id] = u_action

    # def calculate_unit_actions(
    #     self,
    #     unit: FriendlyUnitManger,
    #     uobs: UnitObs,
    #     factory: [None, FriendlyFactoryManager],
    #     fobs: [None, FactoryObs],
    #     unit_recommendations: dict[str, Recommendation],
    # ) -> [list[np.ndarray], None]:
    #     def factory_has_heavy_ice_miner(factory: FriendlyFactoryManager):
    #         units = factory.heavy_units
    #         for unit_id, unit in units.items():
    #             if unit.status.role == 'mine_ice':
    #                 return True
    #         return False
    #
    #     if factory is None:
    #         logging.info(f'{unit.unit_id} has no factory. Doing nothing')
    #         return None
    #
    #     # Make at least 1 heavy mine ice
    #     if (
    #         not factory_has_heavy_ice_miner(factory)
    #         and unit.unit_type == 'HEAVY'
    #         and unit.status.role != 'mine_ice'
    #     ) or (unit.status.role == 'mine_ice' and len(unit.action_queue) == 0):
    #         logging.info(f'{unit.unit_id} assigned to mine_ice (for {unit.factory_id})')
    #         mining_rec = unit_recommendations.pop('mine_ice', None)
    #         if mining_rec is not None:
    #             unit.status.role = 'mine_ice'
    #             return self.mining_planner.carry_out(unit, recommendation=mining_rec)
    #         else:
    #             raise RuntimeError(f'no `mine_ice` recommendation for {unit.unit_id}')
    #
    #     # Make at least one light mine rubble
    #     if (unit.unit_type == 'LIGHT' and not unit.status.role) or (
    #         unit.status.role == 'clear_rubble' and len(unit.action_queue) == 0
    #     ):
    #         logging.info(
    #             f'{unit.unit_id} assigned to clear_rubble (for {unit.factory_id})'
    #         )
    #
    #         rubble_clearing_rec = unit_recommendations.pop('clear_rubble', None)
    #         if rubble_clearing_rec is not None:
    #             unit.status.role = 'clear_rubble'
    #             return self.rubble_clearing_planner.carry_out(
    #                 unit, recommendation=rubble_clearing_rec
    #             )
    #         pass
    #     return None

    def _beginning_of_step_update(
        self, step: int, obs: dict, remainingOverageTime: int
    ):
        """Use the step and obs to update any turn based info (e.g. map changes)"""
        logging.info(f'Beginning of step update for step {step}')
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        # TODO: Use last obs to see what has changed to optimize update? Or master does this?
        self.master.update(game_state)
        self.last_obs = obs

    def _get_general_processed_obs(self) -> GeneralObs:
        """Get a fixed length DF/array of obs that can be passed into an ML agent

        Thoughts:
            - Mostly generated from MasterPlan
            - Some additional info added based on metadata?
        """
        obs = GeneralObs(
            num_friendly_heavy=len(self.master.units.friendly.heavy.keys()),
        )
        return obs


class MyObs(abc.ABC):
    """General form of my observations (i.e. some general requirements)"""

    pass


@dataclass
class GeneralObs(MyObs):
    num_friendly_heavy: int


@dataclass
class UnitObs(MyObs):
    """Object for holding recommendations and other observations relevant to unit on a per-turn basis"""

    id: str
    nearest_enemy_light_distance: int
    nearest_enemy_heavy_distance: int
    current_role: str
    current_action: str


@dataclass
class FactoryObs(MyObs):
    id: str
    center_occupied: bool  # Center tile (i.e. where units are built)


def calculate_unit_obs(unit: FriendlyUnitManger, plan: MasterState) -> UnitObs:
    """Calculate observations that are specific to a particular unit

    Include all the basic stuff like id etc

    Something like, get some recommended actions for the unit given the current game state?
    Those recommendations can include some standard information (values etc.) that can be used to make an ML interpretable observation along with some extra information that identifies what action to take if this action is recommended

    """
    id = unit.unit_id
    nearest_enemy_light_distance = plan.units.nearest_unit(
        pos=unit.pos, friendly=False, enemy=True, light=True, heavy=False
    )
    nearest_enemy_heavy_distance = plan.units.nearest_unit(
        pos=unit.pos, friendly=False, enemy=True, light=False, heavy=True
    )
    uobs = UnitObs(
        id=id,
        nearest_enemy_light_distance=nearest_enemy_light_distance,
        nearest_enemy_heavy_distance=nearest_enemy_heavy_distance,
        current_role=unit.status.role,
        current_action=unit.status.current_action,
    )

    return uobs


def calculate_factory_obs(
    factory: FriendlyFactoryManager, master: MasterState
) -> FactoryObs:
    """Calculate observations that are specific to a particular factory"""

    center_tile_occupied = (
        # True if plan.maps.unit_at_tile(factory.factory.pos) is not None else False
        True
        if master.units.unit_at_position(factory.factory.pos) is not None
        else False
    )

    return FactoryObs(id=factory.unit_id, center_occupied=center_tile_occupied)


def calculate_factory_action(
    factory: FriendlyFactoryManager, fobs: FactoryObs, master: MasterState
) -> [np.ndarray, None]:
    # Building Units
    if (
        fobs.center_occupied is False and master.step < 800
    ):  # Only consider if middle is free and not near end of game
        # Want at least one heavy mining ice
        if (
            len(factory.heavy_units) < 1
            and factory.factory.cargo.metal > master.env_cfg.ROBOTS['HEAVY'].METAL_COST
            and factory.factory.power > master.env_cfg.ROBOTS['HEAVY'].POWER_COST
        ):
            logging.info(f'{factory.factory.unit_id} building Heavy')
            return factory.factory.build_heavy()

        # Want at least one light to do other things
        if (
            len(factory.light_units) < 1
            and factory.factory.cargo.metal > master.env_cfg.ROBOTS['LIGHT'].METAL_COST
            and factory.factory.power > master.env_cfg.ROBOTS['LIGHT'].POWER_COST
        ):
            logging.info(f'{factory.factory.unit_id} building Light')
            return factory.factory.build_light()

    # Watering Lichen
    water_cost = factory.factory.water_cost(master.game_state)
    if (
        factory.factory.cargo.water > 1000 or master.step > 800
    ):  # Either excess water or near end game
        water_cost = factory.factory.water_cost(master.game_state)
        if factory.factory.cargo.water - water_cost > min(
            100, 1000 - master.game_state.real_env_steps
        ):
            logging.info(
                f'{factory.factory.unit_id} watering with water={factory.factory.cargo.water} and water_cost={water_cost}'
            )
            return factory.factory.water()

    return None


"""
PPO implementation ideas:
Actions:
- Makes decision per unit (units/factories treated equal)
    - Does action space include both unit and factory actions, then mask invalid?
    - Or can I somehow use say 6 outputs and just change the meaning for Factory actions?
        - Probably there is something wrong with this
    - How to make fixed length of actions?
        - Mine Ore, Mine Ice, Attack, Defend
            - But how to specify Defend Factory 1 or Factory 2 etc?
            - Then actions aren't fixed length
- Per Factory?
    - I.e. factory decides if it wants more ice, ore, defence, attack etc.
    - Then units are completely algorithmic
- Other variables
    - There are also many other variables (e.g. thresholds) for when algorithmic things should happen
    - How can these be fine tuned? Maybe separate to the PPO?
        - I.e. train PPO, then tune params, then train PPO, then tune params?


Observations:
- Some sort of gauss peaks for beginning, middle, end game (i.e. some flag that might change strategy for different
parts of game)
- Some calculated states of the game (i.e. total resources, resources per factory, how many light, how many heavy)
- How to give information on what units/factories are doing? 
    - Not fixed length...
    - Could give positions at least as an array of values size of map (but that is a lot!)
"""

# if __name__ == '__main__':
#     obs = GeneralObs()

if __name__ == '__main__':
    pass
    # run_type = 'start'
    # import time
    #
    # start = time.time()
    # ########## PyCharm ############
    # from util import get_test_env, show_env, run
    # import json
    #
    # if run_type == 'test':
    #     pass
    #
    # elif run_type == 'start':
    #     ### From start running X steps
    #     # fig = run(Agent, Agent, map_seed=1, max_steps=40, return_type='figure')
    #     # fig.show()
    #
    #     replay = run(
    #         Agent,
    #         # BasicAgent,
    #         Agent,
    #         map_seed=1,
    #         max_steps=1000,
    #         save_state_at=None,
    #         return_type='replay',
    #     )
    #     with open('replay.json', 'w') as f:
    #         json.dump(replay, f)
    #
    # elif run_type == 'checkpoint':
    #     ### From checkpoint
    #     env = get_test_env('test_state.pkl')
    #     fig = show_env(env)
    #     fig.show()
    #
    # print(f'Finished in {time.time() - start:.3g} s')
    # ####################################
    #
    # ####### JUPYTER ONLY  ########
    # # from lux_eye import run_agents
    # # run_agents(Agent, Agent, map_seed=1, save_state_at=None)
    # #######################
    #
    # # env = get_test_env()
    # # show_env(env)
    # #
    # # game_state = game_state_from_env(env)
    # # unit = game_state.units["player_0"]["unit_4"]
    # #
    # # # First move unit to start on Factory tile -- down, down, left
    # # empty_actions = {
    # #     "player_0": {},
    # #     "player_1": {},
    # # }
    # #
    # # actions = {
    # #     "player_0": {unit.unit_id: [unit.move(DOWN, repeat=1), unit.move(LEFT)]},
    # #     "player_1": {},
    # # }
    # #
    # # obs, rews, dones, infos = env.step(actions)
    # # fig = initialize_step_fig(env)
    # # for i in range(2):
    # #     obs, rews, dones, infos = env.step(empty_actions)
    # #     add_env_step(fig, env)
    # #
    # # # Set work flow -- pickup, right, up, up, up, dig, down, down, down, left, transfer
    # # actions = {
    # #     "player_0": {
    # #         unit.unit_id: [
    # #             unit.pickup(POWER, 200, repeat=-1),
    # #             unit.move(RIGHT, repeat=-1),
    # #             unit.move(UP, repeat=-1),
    # #             unit.move(UP, repeat=-1),
    # #             unit.move(UP, repeat=-1),
    # #             unit.dig(repeat=-1),
    # #             unit.dig(repeat=-1),
    # #             unit.move(DOWN, repeat=-1),
    # #             unit.move(DOWN, repeat=-1),
    # #             unit.move(DOWN, repeat=-1),
    # #             unit.move(LEFT, repeat=-1),
    # #             unit.transfer(CENTER, ICE, unit.cargo.ice, repeat=-1),
    # #         ]
    # #     },
    # #     "player_1": {},
    # # }
    # #
    # # obs, rews, dones, infos = env.step(actions)
    # # add_env_step(fig, env)
    # # for i in range(30):
    # #     obs, rews, dones, infos = env.step(empty_actions)
    # #     add_env_step(fig, env)
    # # fig.show()
