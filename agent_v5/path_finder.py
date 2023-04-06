from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Tuple, List, TYPE_CHECKING, Iterable, Dict, Optional
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
import numpy as np
from lux.utils import direction_to

if TYPE_CHECKING:
    from unit_manager import UnitManager
    from factory_manager import FactoryManager
    from lux.factory import Factory


@dataclass
class CollisionParams:
    turns: int
    friendly_light: bool = True
    friendly_heavy: bool = True
    enemy_light: bool = True
    enemy_heavy: bool = True
    ignore_ids: List[str] = field(default_factory=list)


class PathFinder:
    def __init__(self):
        self.rubble: np.ndarray = None
        self.friendly_light_paths: dict = None
        self.friendly_heavy_paths: dict = None
        self.enemy_light_paths: dict = None
        self.enemy_heavy_paths: dict = None

        self.enemy_factories: dict = None  # Not traversable
        self.enemy_factories_array: np.array = None

    def log(self, message: str, level=logging.INFO):
        logging.log(level, f'PathFinding: {message}')

    def get_costmap(self, rubble=False):
        """Power cost of travelling (not taking into account collisions, but taking into account enemy factories)

        Note: This is in same orientation as other maps (needs to be Transposed before passing to Pathfinder)
        """
        if rubble:
            cost = np.ones(self.rubble.shape) * 1
            cost += self.rubble * 0.05
            # TODO: Technically this is for light units, heavy units should be 20x higher
            # TODO: which will make a slight difference when np.floor is executed
            # TODO: For now I'll slightly overestimate movement cost of units so that light/heavy are similar
            # TODO: cost = np.floor(cost)
            cost[self.enemy_factories_array == 1] = -1  # Not traversable
        else:
            cost = np.ones(self.rubble.shape) * 10
            cost[self.enemy_factories_array == 1] = -1  # Not traversable
        return cost

    def update_unit_path(self, unit: UnitManager, path: np.ndarray):
        path = np.asanyarray(path, dtype=int)
        if unit.unit.unit_type == 'LIGHT':
            self.friendly_light_paths[unit.unit_id] = path
        elif unit.unit.unit_type == 'HEAVY':
            self.friendly_heavy_paths[unit.unit_id] = path
        else:
            raise TypeError(f'unit not correct type {unit}')

    def update(
        self,
        rubble: np.ndarray,
        friendly_units: Dict[str, UnitManager],
        enemy_units: Dict[str, UnitManager],
        enemy_factories: Dict[str, FactoryManager],
    ):
        self.rubble = rubble

        # Update Units
        data = {
            'friendly': {'light': {}, 'heavy': {}},
            'enemy': {'light': {}, 'heavy': {}},
        }

        for units, player in zip(
            [friendly_units.values(), enemy_units.values()], data.keys()
        ):
            for unit in units:
                if unit.unit.unit_type == 'LIGHT':
                    data[player]['light'][unit.unit_id] = unit.actions_to_path()
                elif unit.unit.unit_type == 'HEAVY':
                    data[player]['heavy'][unit.unit_id] = unit.actions_to_path()
                else:
                    raise RuntimeError

        self.all_paths = data
        self.friendly_light_paths = data['friendly']['light']
        self.friendly_heavy_paths = data['friendly']['heavy']
        self.enemy_light_paths = data['enemy']['light']
        self.enemy_heavy_paths = data['enemy']['heavy']

        # Update Enemy Factories
        self.enemy_factories = {
            factory_id: factory.factory.pos_slice
            for factory_id, factory in enemy_factories.items()
        }
        arr = np.zeros(rubble.shape)
        for s in self.enemy_factories.values():
            arr[s] = 1.0
        self.enemy_factories_array = arr

    def path(self, start, end, step=0, rubble=False):
        """
        Full A* pathing using whole grid, but slow
        Note: no check of unit collisions here, only avoids enemy factories
        """
        finder = AStarFinder()
        cost_map = self.get_costmap(rubble=rubble)
        cost_map = cost_map.T  # Required for finder
        grid = Grid(matrix=cost_map)
        start = grid.node(*start)
        end = grid.node(*end)
        path, runs = finder.find_path(start, end, grid)
        path = np.array(path)
        return path

    def get_existing_paths(self, collision_params: CollisionParams):
        """Return the existing paths of other units taking into account collision_params"""
        # TODO: Do some sort of caching of this to make things faster
        # TODO: Note: still needs to be easy to update specific paths (add/remove) when doing this...
        paths = []
        if collision_params.friendly_light:
            paths.extend(
                [
                    p
                    for k, p in self.friendly_light_paths.items()
                    if k not in collision_params.ignore_ids
                ]
            )
        if collision_params.friendly_heavy:
            paths.extend(
                [
                    p
                    for k, p in self.friendly_heavy_paths.items()
                    if k not in collision_params.ignore_ids
                ]
            )
        if collision_params.enemy_light:
            paths.extend(
                [
                    p
                    for k, p in self.enemy_light_paths.items()
                    if k not in collision_params.ignore_ids
                ]
            )
        if collision_params.enemy_heavy:
            paths.extend(
                [
                    p
                    for k, p in self.enemy_heavy_paths.items()
                    if k not in collision_params.ignore_ids
                ]
            )
        return paths

    def check_collisions(self, path: np.ndarray, collision_params: CollisionParams):
        """Returns first collision coordinate if collision, else None"""
        path = np.asanyarray(path, dtype=int)
        paths = self.get_existing_paths(collision_params)

        for i, coord in enumerate(path):
            if i >= collision_params.turns:
                return None
            # TODO: Is this checking for step position?
            for p in paths:
                if np.all(coord == p):  # Collision
                    return p
        return None

    def path_fast(
        self,
        start,
        end,
        step=0,
        rubble=False,
        margin=1,
        collision_params: Optional[CollisionParams] = None,
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

        def _path(additional_blocked_cells=None):
            nonlocal start, end
            additional_blocked_cells = (
                additional_blocked_cells if additional_blocked_cells else []
            )

            cost_map = self.get_costmap(rubble=rubble)
            for x, y in additional_blocked_cells:
                cost_map[x, y] = -1

            coords = np.array([start, end])

            # x, y
            mins = np.min(coords, axis=0)
            maxs = np.max(coords, axis=0)

            # x, y
            lowers = [max(0, v - margin) for v in mins]
            uppers = [
                min(s - 1, v + margin) + 1
                for s, v in zip(reversed(cost_map.shape), maxs)
            ]  # +1 for range

            x_range, y_range = [(lowers[i], uppers[i]) for i in range(2)]

            # x, y
            new_cost = cost_map[range(*x_range), :][:, range(*y_range)]

            # Make small grid and set start/end
            new_cost = new_cost.T  # Required other way around for Grid
            grid = Grid(matrix=new_cost)
            grid_start = grid.node(*[c - l for c, l in zip(start, lowers)])
            grid_end = grid.node(*[c - l for c, l in zip(end, lowers)])

            # Find Path
            pathfinder = AStarFinder(diagonal_movement=-1)
            path, runs = pathfinder.find_path(grid_start, grid_end, grid)

            # Readjust for original map
            if len(path) > 0:
                path = np.array(path, dtype=int) + np.array(lowers, dtype=int)
            return path

        path = []
        if collision_params is None:
            path = _path()
        else:
            blocked_cells = []
            attempts = 0
            while attempts < 20:
                path = _path(blocked_cells)
                if len(path) == 0:
                    self.log(f'No paths found without collisions')
                    break
                pos = self.check_collisions(path, collision_params=collision_params)
                if pos is None:
                    break
                blocked_cells.append(pos)
            else:
                self.log(f'No paths found without collisions')
        return path
