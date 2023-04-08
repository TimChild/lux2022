from __future__ import annotations
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
import logging
import math
from typing import TYPE_CHECKING, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from luxai_s2.unit import UnitType

from agent_v5.unit_manager import UnitManager
from master_state import MasterState, Planner
from actions import Recommendation
from path_finder import CollisionParams
from util import (
    ICE,
    ORE,
    nearest_non_zero,
    power_cost_of_actions,
    path_to_actions,
    HEAVY_UNIT,
    LIGHT_UNIT,
    ACT_REPEAT,
    ACT_START_N,
    POWER,
    CENTER,
    manhattan_kernel,
    SubsetExtractor,
    stretch_middle_of_factory_array,
    connected_factory_zeros,
    create_boundary_array,
    pad_and_crop,
    manhattan_distance_between_values,
    convolve_array_kernel,
    MOVE_DELTAS,
    MOVE_DIRECTIONS,
)

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManger


class RubbleDigValue:
    def __init__(
        self,
        rubble: np.ndarray,
        full_factory_map: np.ndarray,
        factory_pos: Tuple[int, int],
        factory_dist: int = 10,
        factory_dist_dropoff: float = 0.8,
        boundary_kernel_size: int = 3,
        boundary_kernel_dropoff: float = 0.7,
    ):
        """
        Calculate value of rubble digging near a factory
        """
        self.rubble = rubble
        self.full_factory_map = full_factory_map
        self.factory_pos = factory_pos
        self.factory_dist = factory_dist
        self.factory_dist_dropoff = factory_dist_dropoff
        self.boundary_kernel_size = boundary_kernel_size
        self.boundary_kernel_dropoff = boundary_kernel_dropoff

        self._rubble_subset = None
        self._new_factory_pos = None
        self._factory_weighting = None
        self._manhattan_dist_to_zeroes = None
        self._boundary_array = None
        self._conv_boundary_array = None
        self._low_rubble_value = None

    def _get_rubble_subset(self):
        """
        Only look at a small area around the factory for speed
        Note:
            - fills outside with 100 rubble
            - fills all factory locations with 100 rubble
        """
        if self._rubble_subset is None or self._new_factory_pos is None:
            rubble = self.rubble.copy()
            if self.full_factory_map is not None:
                rubble[self.full_factory_map >= 0] = 100
            subsetter = SubsetExtractor(
                rubble, self.factory_pos, radius=self.factory_dist, fill_value=100
            )
            self._rubble_subset = subsetter.get_subset()
            self._new_factory_pos = subsetter.convert_coordinate(self.factory_pos)
        return self._rubble_subset, self._new_factory_pos

    def _get_factory_weighting(self):
        """Make factory weighting (decreasing value away from factory)"""
        if self._factory_weighting is None:
            factory_weighting = self.factory_dist_dropoff ** manhattan_kernel(
                self.factory_dist
            )
            # Stretch the middle to be 3x3
            factory_weighting = stretch_middle_of_factory_array(factory_weighting)[
                1:-1, 1:-1
            ]
            self._factory_weighting = factory_weighting
        return self._factory_weighting

    def _get_manhattan_dist_to_zeros(self):
        """Get distance to zeros for everywhere excluding zeros already connected to factory"""
        if self._manhattan_dist_to_zeroes is None:
            rubble_subset, new_factory_pos = self._get_rubble_subset()
            rubble_factory_non_zero = rubble_subset.copy()
            # Set zeros under the specific factory are looking at again (middle 9 values)
            rubble_factory_non_zero[
                self.factory_dist - 1 : self.factory_dist + 2,
                self.factory_dist - 1 : self.factory_dist + 2,
            ] = 0
            factory_zeroes = connected_factory_zeros(rubble_subset, new_factory_pos)
            rubble_factory_non_zero[factory_zeroes == 1] = 999  # Anything non-zero
            # TODO: Make this faster, or remove... this is the bottleneck for the whole thing
            manhattan_dist_to_zeros = manhattan_distance_between_values(
                rubble_factory_non_zero
            )
            # Invert so that value is higher for lower distance
            manhattan_dist_to_zeros = np.abs(
                manhattan_dist_to_zeros - np.max(manhattan_dist_to_zeros)
            )
            self._manhattan_dist_to_zeros = manhattan_dist_to_zeros
        return self._manhattan_dist_to_zeros

    def _get_boundary_array(self):
        """Create array where boundary of zero areas are valued by how large the zero area is"""
        if self._boundary_array is None:
            rubble_subset, new_factory_pos = self._get_rubble_subset()
            boundary_array = create_boundary_array(rubble_subset)
            self._boundary_array = boundary_array
        return self._boundary_array

    def _get_conv_boundary_array(self):
        """Smooth that with a manhattan dist convolution"""
        if self._conv_boundary_array is None:
            boundary_array = self._get_boundary_array()
            conv_boundary_array = convolve_array_kernel(
                boundary_array,
                self.boundary_kernel_dropoff
                ** manhattan_kernel(self.boundary_kernel_size),
            )
            self._conv_boundary_array = conv_boundary_array
        return self._conv_boundary_array

    def _get_low_rubble_value(self):
        """Calculate value of low rubble areas"""
        if self._low_rubble_value is None:
            rubble_subset, _ = self._get_rubble_subset()
            low_rubble_value = np.ceil(
                np.abs(rubble_subset - 100) / 2
            )  # Over 2 because light can dig 2 at a time
            self._low_rubble_value = low_rubble_value
        return self._low_rubble_value

    def calculate_final_value(self):
        self._get_rubble_subset()
        self._get_factory_weighting()
        self._get_manhattan_dist_to_zeros()
        self._get_boundary_array()
        self._get_conv_boundary_array()
        self._get_low_rubble_value()

        conv_boundary_array = self._get_conv_boundary_array()
        low_rubble_value = self._get_low_rubble_value()
        manhattan_dist_to_zeroes = self._get_manhattan_dist_to_zeros()
        factory_weighting = self._get_factory_weighting()
        rubble_subset, _ = self._get_rubble_subset()

        # Make a final map
        final_value = (
            conv_boundary_array
            * low_rubble_value
            * manhattan_dist_to_zeroes
            * factory_weighting
        )
        final_value[rubble_subset == 0] = 0

        final_value = pad_and_crop(
            final_value, self.rubble, self.factory_pos[0], self.factory_pos[1]
        )
        return final_value / np.nanmax(final_value)


def calc_path_to_factory(
    pathfinder, pos: Tuple[int, int], factory_loc: np.ndarray
) -> list:
    nearest_factory = nearest_non_zero(factory_loc, pos)
    path = pathfinder.path_fast(
        pos,
        nearest_factory,
        rubble=False,  # Fastest path (ignore rubble that may be changing anyway)
        margin=2,
        collision_params=None,
    )
    return path


def power_cost_of_path(path, rubble: np.ndarray, unit_type="LIGHT") -> int:
    """Cost to move along path including rubble
    Note: Assumes first path is current location (i.e. not part of cost)
    """
    assert unit_type in ["LIGHT", "HEAVY"]
    unit_cfg = LIGHT_UNIT.unit_cfg if unit_type == "LIGHT" else HEAVY_UNIT.unit_cfg

    if len(path) == 0:
        return 0
    cost = 0
    for pos in path[1:]:
        rubble_at_pos = rubble[pos[0], pos[1]]
        cost += math.ceil(
            unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_pos
        )
        # cost -= unit_cfg.CHARGE  # Charging only happens during DAY
    return cost


def calc_value_to_move(pos: Tuple[int, int], value_array: np.ndarray) -> float:
    """Return maximum value of moving in any direction"""
    move_deltas = MOVE_DELTAS[1:]  # Exclude center
    vals = []
    pos = np.array(pos)
    for move in move_deltas:
        new_pos = pos + move
        vals.append(value_array[new_pos[0], new_pos[1]])

    return max(vals)


def calc_best_direction(pos: Tuple[int, int], value_array: np.ndarray) -> int:
    """Return direction to highest adjacent value"""
    # (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    move_deltas = MOVE_DELTAS[1:]  # Exclude center
    move_directions = MOVE_DIRECTIONS[1:]
    pos = np.array(pos)

    best_direction = 0  # Center
    best_value = 0
    for move, direction in zip(move_deltas, move_directions):
        new_pos = pos + move
        new_val = value_array[new_pos[0], new_pos[1]]
        if new_val > best_value:
            best_value = new_val
            best_direction = direction
    return best_direction


class RubbleClearingRecommendation(Recommendation):
    """Recommend Rubble Clearing near factory"""

    role = 'rubble_clearer'

    def __init__(self):
        pass


class RubbleClearingPlanner(Planner):
    def recommend(self, unit: FriendlyUnitManger):
        """
        Make recommendation for this unit to clear rubble around factory

        TODO:
        - Play with ideas for this in a notebook before writing any code here
        - Calculate best area near factory to clear rubble to increase total area of zeros connected to factory
        - Convert rubble to something rounded into units of light/heavy mine quantity (i.e. rubble 1 and 17 are equally bad for heavy that can mine 20 at a time)
        - weight toward location of unit
        - Consider that other units may be able to bring power (more important for carrying out?)
        -
        """
        pass

    def carry_out(self, unit: FriendlyUnitManger, recommendation: Recommendation):
        pass

    def update(self, *args, **kwargs):
        """Called at beginning of turn, may want to clear caches"""
        # For now do nothing
        pass
