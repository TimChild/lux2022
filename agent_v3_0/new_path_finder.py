from __future__ import annotations
from typing import Tuple, List, TYPE_CHECKING, Union, Dict, Optional
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import numpy as np

from config import get_logger
import util

if TYPE_CHECKING:
    from unit_manager import UnitManager, FriendlyUnitManager
    from agent_v3_0.collisions import UnitPaths

logger = get_logger(__name__)


def _get_sub_area(costmap: np.ndarray, lowers, uppers):
    """Reduces area of map"""
    # Ranges
    x_range, y_range = [(lowers[i], uppers[i]) for i in range(2)]

    # Reduced size cost map
    new_cost = costmap[range(*x_range), :][:, range(*y_range)]
    return new_cost


def _adjust_coords(start, end, lowers) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Adjust coords to new reduced area map"""
    new_start = [c - l for c, l in zip(start, lowers)]
    new_end = [c - l for c, l in zip(end, lowers)]
    return tuple(new_start), tuple(new_end)  # Complaining about unknown length of tuples


def _adjust_path_back(path: np.ndarray, lowers):
    """Adjust path back to the original array coords"""
    if len(path) > 0:
        path = np.array(path, dtype=int) + np.array(lowers, dtype=int)
    return path


def _get_bounds(start: util.POS_TYPE, end: util.POS_TYPE, margin: int, map_shape):
    """Get bound of reduced area map"""
    # Convert to coords array
    start, end = np.array(start), np.array(end)
    if start.shape != (2,) or end.shape != (2,):
        raise ValueError(f"One of {start}, or {end} is not a correct position")
    coords = np.array([start, end])

    # Bounds of start, end (x, y)
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)

    # Bounds including margin (x, y)
    lowers = [max(0, v - margin) for v in mins]
    uppers = [min(s - 1, v + margin) + 1 for s, v in zip(reversed(map_shape), maxs)]  # +1 for range
    return lowers, uppers


class Pather:
    """Calculates paths and generates actions for paths, updating the current paths when actions are generated"""

    def __init__(
        self,
        base_costmap: np.ndarray,
        # full_costmap: np.ndarray = None,
        unit_paths: UnitPaths = None,
    ):
        self.base_costmap = base_costmap
        # self.full_costmap = full_costmap if full_costmap is not None else base_costmap
        self.unit_paths = unit_paths

    def generate_costmap(
        self,
        unit: FriendlyUnitManager,
        ignore_id_nums: Optional[List[int]] = None,
        friendly_light=True,
        friendly_heavy=True,
        enemy_light=None,
        enemy_heavy=True,
        collision_only=False,
        enemy_collision_value=-1,
        friendly_collision_value=-1,
        enemy_nearby_start_cost=2,
        friendly_nearby_start_cost=0.5,
        override_step=None,
    ):
        """Generate the costmap for the current step (i.e. taking into account pos of unit and all other unit paths)
        Note: Step 0 are current locations of units, correct to use when pathing from current position.
        Note: Step 1 should be used for checking if unit needs to move this turn (i.e. the spot will become occupied)
        """
        # By default, have friendly heavy ignore enemy light units
        if enemy_light is None:
            enemy_light = False if unit.unit_type == "HEAVY" else True

        if override_step is None:
            step = util.num_turns_of_actions(unit.action_queue)
        else:
            step = override_step
        ignore_id_nums = ignore_id_nums if ignore_id_nums is not None else []
        if unit.id_num not in ignore_id_nums:
            ignore_id_nums.append(unit.id_num)

        cm = self.unit_paths.to_costmap(
            pos=unit.pos,
            start_step=step,
            exclude_id_nums=ignore_id_nums,
            friendly_light=friendly_light,
            friendly_heavy=friendly_heavy,
            enemy_light=enemy_light,
            enemy_heavy=enemy_heavy,
            enemy_collision_cost_value=enemy_collision_value,
            friendly_collision_cost_value=friendly_collision_value,
            enemy_nearby_start_cost=enemy_nearby_start_cost,
            friendly_nearby_start_cost=friendly_nearby_start_cost,
            true_intercept=collision_only,
            # step_dropoff_multiplier=0.92,
        )

        # Note: Be careful not to make -1s or 0s become positive (blocked becomes unblocked)
        blocked = np.logical_or(cm <= 0, self.base_costmap <= 0)
        cm += self.base_costmap
        cm[blocked == 1] = -1
        return cm

    def fast_path(
        self,
        start_pos: util.POS_TYPE,
        end_pos: util.POS_TYPE,
        costmap=None,
        margin=2,
    ):
        """Faster A* pathing by only considering a box around the start/end coord (with additional margin)

        If collision_params passed in, this will try to avoid collisions.
        Note: If collisions cannot be avoided, this will still return the path

        Good for evaluating value of positions etc

        Example:
            # # # # # #
            # # # # # #
            # s # # # #
            # # # e # #
            # # # # # #
            # # # # # #

            only searches (margin = 1)
            # # # # #
            # s # # #
            # # # e #
            # # # # #
        """
        # costmap = costmap if costmap is not None else self.full_costmap
        costmap = costmap if costmap is not None else self.base_costmap

        lowers, uppers = _get_bounds(start_pos, end_pos, margin, costmap.shape)
        sub_costmap = _get_sub_area(costmap, lowers, uppers)
        new_start, new_end = _adjust_coords(start_pos, end_pos, lowers)
        sub_path = self.slow_path(new_start, new_end, sub_costmap)
        path = _adjust_path_back(sub_path, lowers)
        return path

    def append_path_to_actions(self, unit: UnitManager, path: Union[List[Tuple[int, int]], np.ndarray]) -> None:
        """
        Turns the path into actions that are appended to unit. This is how path should ALWAYS be updated
        """
        # TODO: Modify previous actions n if first new action is same direction (just a slight optimization)
        actions = util.path_to_actions(path)
        if isinstance(unit.action_queue, np.ndarray):
            logger.warning(f"Converting action queue from np.array to list")
            unit.action_queue = list(unit.action_queue)
        unit.action_queue.extend(actions)
        if len(path) > 0 and len(path[-1] == 2):
            unit.pos = path[-1]

    def append_direction_to_actions(self, unit: UnitManager, direction: int):
        """Turn the direction into actions that are appended to unit"""
        path = [unit.pos, util.add_direction_to_pos(unit.pos, direction)]
        self.append_path_to_actions(unit, path)

    def slow_path(self, start_pos: Tuple[int, int], end_pos: Tuple[int, int], costmap=None) -> np.ndarray:
        """
        Find A* path from start to end (does not update anything)
        Note: self.full_costmap (or provided costmap) should be such that collisions impossible for first 1 or 2 turns

        Args:
            start_pos: start of path coord
            end_pos: end of path coord
            costmap: override the costmap for pathing

        Returns:
            array shape (len, 2) for path
            Note: If start==end path has len 1
            Note: If fails to find path, len 0
        """
        # costmap = costmap if costmap is not None else self.full_costmap
        costmap = costmap if costmap is not None else self.base_costmap

        # Unblock current position (should be valid to path away from there)
        if costmap[start_pos[0], start_pos[1]] <= 0:
            costmap = costmap.copy()
            costmap[start_pos[0], start_pos[1]] = 10

        # Run A* pathfinder
        finder = AStarFinder()
        costmap = costmap.T  # Required for finder
        grid = Grid(matrix=costmap)
        start = grid.node(*start_pos)
        end = grid.node(*end_pos)
        path, runs = finder.find_path(start, end, grid)
        path = np.array(path, dtype=int)
        return path
