from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Tuple, List
import numpy as np

from util import calc_path_to_factory, power_cost_of_path, num_turns_of_actions
from master_state import MasterState, Planner
from actions import Recommendation
from path_finder import CollisionParams, PathFinder
from util import (
    power_cost_of_actions,
    path_to_actions,
    HEAVY_UNIT,
    LIGHT_UNIT,
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
    add_direction_to_pos,
)

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManger
    from factory_manager import FriendlyFactoryManager


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


def calc_value_to_move(pos: Tuple[int, int], value_array: np.ndarray) -> float:
    """Return maximum value of moving in any direction"""
    move_deltas = MOVE_DELTAS[1:]  # Exclude center
    xmax, ymax = value_array.shape

    vals = []
    pos = np.array(pos)
    for move in move_deltas:
        new_pos = pos + move
        try:
            vals.append(value_array[new_pos[0], new_pos[1]])
        except IndexError:
            logging.info(f'IndexError with pos {pos}, probably near edge?')
            pass

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
        try:
            new_val = value_array[new_pos[0], new_pos[1]]
        except IndexError:
            logging.info(f'IndexError with pos {pos}, probably near edge?')
            continue
        if new_val > best_value:
            best_value = new_val
            best_direction = direction
    return best_direction


class RubbleRoutePlanner:
    move_lookahead = 10
    target_queue_length = 20

    def __init__(
        self,
        pathfinder: PathFinder,
        rubble: np.ndarray,
        rubble_value_map: np.ndarray,
        factory: FriendlyFactoryManager,
        unit_pos: Tuple[int, int],
        unit_id: str,
        unit_power: int,
        unit_type: str,
    ):
        """
        Args:
            rubble: full rubble map
            rubble_value_map: Value map generated by RubbleDigValue.calculate_final_value()
            factory: The factory the rubble is being cleared for
        """
        assert unit_type in ['LIGHT', 'HEAVY']
        self.pathfinder = pathfinder
        # Maps
        self.rubble = rubble
        self.rubble_value_map = rubble_value_map
        # Unit
        self.unit_start_pos = unit_pos
        self.factory = factory

        # These will be changed during route planning
        self.unit = LIGHT_UNIT if unit_type == 'LIGHT' else HEAVY_UNIT
        self.unit.pos = unit_pos
        self.unit.power = unit_power
        self.unit.unit_id = unit_id
        self.unit.action_queue = []
        self._future_rubble = self.rubble.copy()
        self._future_value = self.rubble_value_map.copy()

    def log(self, message, level=logging.INFO):
        return logging.log(level=level, msg=f'RubbleRoutePlanner: {message}')

    def make_route(self):
        if self._unit_starting_on_factory():
            success = self._from_factory_actions()
            if not success:
                return self.unit.action_queue[: self.target_queue_length]

        while len(self.unit.action_queue) < self.target_queue_length:
            queue_cost = self._cost_of_actions(self.unit.action_queue)
            path_to_factory = self._path_to_factory()
            cost_to_factory = power_cost_of_path(
                path_to_factory, self._future_rubble, self.unit.unit_type
            )
            power_remaining = self.unit.power - queue_cost - cost_to_factory

            unit_multiplier = pad_and_crop(
                0.8 ** manhattan_kernel(5),
                self._future_value,
                self.unit.pos[0],
                self.unit.pos[1],
                fill_value=0,
            )
            value_array = self._get_boundary_values() * unit_multiplier
            value_at_pos = value_array[self.unit.pos[0], self.unit.pos[1]]
            value_to_move = calc_value_to_move(self.unit.pos, value_array)

            # TODO: Change value based on light/heavy unit
            if power_remaining > 5 and (value_at_pos > 0 or value_to_move > 0):
                self._calculate_next_action(
                    power_remaining=power_remaining,
                    value_array=value_array,
                    value_at_pos=value_at_pos,
                    value_to_move=value_to_move,
                )
            else:
                if len(path_to_factory) > 0:
                    self.unit.action_queue.extend(path_to_actions(path_to_factory))
                    self.unit.pos = path_to_factory[-1]
                    break
        return self.unit.action_queue[: self.target_queue_length]

    def _unit_starting_on_factory(self) -> bool:
        if (
            self.factory.factory_loc[self.unit_start_pos[0], self.unit_start_pos[1]]
            == 1
        ):
            return True
        return False

    def _calculate_next_action(
        self,
        power_remaining: int,
        value_array: np.ndarray,
        value_at_pos: float,
        value_to_move: float,
    ):
        # If better to mine in current location, mine as much as power allows
        self.log(f"Value at pos={value_at_pos}, value to move={value_to_move}")
        if value_at_pos >= value_to_move and value_at_pos > 0:
            pos_rubble = self._future_rubble[self.unit.pos[0], self.unit.pos[1]]
            digs_required = np.ceil(
                pos_rubble / self.unit.unit_cfg.DIG_RUBBLE_REMOVED
            ).astype(int)
            n = min(
                digs_required,
                np.floor(power_remaining / self.unit.unit_cfg.DIG_COST).astype(int),
            )
            rubble_after = int(
                max(0, pos_rubble - n * self.unit.unit_cfg.DIG_RUBBLE_REMOVED)
            )
            self.log(f"digs_required={digs_required}, digs planned (n)={n}")
            self._future_rubble[self.unit.pos[0], self.unit.pos[1]] = rubble_after
            if rubble_after == 0:
                self._future_value[
                    self.unit.pos[0], self.unit.pos[1]
                ] = 0  # No more value there
            self.unit.action_queue.append(self.unit.dig(n=n))

        # Otherwise move to next best spot
        elif value_to_move > 0:
            self.log(f'adding move to better location')
            best_direction = calc_best_direction(self.unit.pos, value_array)
            self.unit.action_queue.append(self.unit.move(best_direction, n=1))
            self.unit.pos = add_direction_to_pos(self.unit.pos, best_direction)

        # Not near any high value, shouldn't get here
        else:
            self.log(
                f'While calculating next action, values were all zero. (adding move center)',
                level=logging.ERROR,
            )
            self.unit.action_queue.append(self.unit.move(CENTER, n=1))  # Do nothing

    def _from_factory_actions(self) -> bool:
        """Generate starting actions assuming starting on factory"""
        if self.unit.power < self.unit.unit_cfg.BATTERY_CAPACITY:
            self.log(f"topping up battery")
            power_to_pickup = self.unit.unit_cfg.BATTERY_CAPACITY - self.unit.power
            if power_to_pickup > 0:
                self.unit.action_queue.append(self.unit.pickup(POWER, power_to_pickup))

        boundary_values = self._get_boundary_values()
        max_value_coord = np.unravel_index(
            np.argmax(boundary_values), boundary_values.shape
        )
        self.log(f"unit moving to {max_value_coord}")
        path = self.pathfinder.path_fast(
            self.unit.pos,
            max_value_coord,
            rubble=True,
            margin=2,
            collision_params=CollisionParams(
                look_ahead_turns=self.move_lookahead,
                ignore_ids=(self.unit.unit_id,),
                enemy_light=True if self.unit.unit_type == 'LIGHT' else False,
                starting_step=num_turns_of_actions(self.unit.action_queue),
            ),
        )
        if len(path) > 0:
            self.unit.action_queue.extend(path_to_actions(path))
            self.unit.pos = path[-1]  # Update position of unit
            return True
        else:
            self.log(
                f'No path to {max_value_coord} from {self.unit.pos}, moving center',
                level=logging.WARNING,
            )
            self.unit.action_queue.append(self.unit.move(CENTER))
            return False

    def _get_boundary_values(self):
        """Get the array of values on the boundary of the area connected to the factory
        That way, we can be sure we are always expanding the factories lichen area
        """

        factory_zeros = connected_factory_zeros(self.rubble, self.factory.factory.pos)
        boundary = create_boundary_array(factory_zeros, boundary_num=1)
        boundary[boundary != 0] = 1
        return self._future_value * boundary

    def _cost_of_actions(self, actions: List[np.ndarray], rubble=None):
        # TODO: could this use future_rubble? Problem is that rubble may not yet be cleared
        if rubble is None:
            rubble = self.rubble
        return power_cost_of_actions(rubble, self.unit, actions)

    def _path_to_factory(self) -> np.ndarray:
        return calc_path_to_factory(
            self.pathfinder,
            self.unit.pos,
            self.factory.factory_loc,
            rubble=self._future_rubble,
            margin=2,
            collision_params=CollisionParams(
                look_ahead_turns=3,
                ignore_ids=(self.unit.unit_id,),
                starting_step=num_turns_of_actions(self.unit.action_queue),
            ),
        )


class RubbleClearingRecommendation(Recommendation):
    """Recommend Rubble Clearing near factory"""

    role = 'rubble_clearer'

    def __init__(self, best_coord: Tuple[int, int]):
        self.best_coord = best_coord


class RubbleClearingPlanner(Planner):
    def __init__(self, master: MasterState):
        self.master = master

        self._factory_value_maps = {}

    def log(self, message, level=logging.INFO):
        return logging.log(
            level=level,
            msg=f'{self.master.player}, Step{self.master.game_state.real_env_steps}, '
            f'RubbleClearingPlanner: {message}',
        )

    50

    def recommend(self, unit: FriendlyUnitManger):
        """
        Make recommendation for this unit to clear rubble around factory
        """
        unit_factory = unit.factory_id
        if unit_factory is not None:
            value_map = self._factory_value_maps[unit_factory]
            max_coord = np.unravel_index(np.argmax(value_map), value_map.shape)
            return RubbleClearingRecommendation(
                best_coord=tuple(max_coord),
            )
        return None

    def carry_out(
        self, unit: FriendlyUnitManger, recommendation: RubbleClearingRecommendation
    ) -> List[np.ndarray]:
        if unit.factory_id is not None:
            factory = self.master.factories.friendly[unit.factory_id]
            route_planner = RubbleRoutePlanner(
                pathfinder=self.master.pathfinder,
                rubble=self.master.maps.rubble,
                rubble_value_map=self._factory_value_maps[factory.factory.unit_id],
                factory=factory,
                unit_pos=unit.pos,
                unit_id=unit.unit_id,
                unit_power=unit.unit.power,
                unit_type=unit.unit.unit_type,
            )
            actions = route_planner.make_route()
            return actions[:20]
        else:
            self.log(
                f'in carry out, {unit.unit_id} has not factory_id', level=logging.ERROR
            )
            return [unit.unit.move(CENTER)]

    def update(self, *args, **kwargs):
        """Called at beginning of turn, may want to clear caches"""
        # Remove old (in case factories have died)
        self._factory_value_maps = {}

        rubble = self.master.maps.rubble
        all_factory_map = self.master.maps.factory_maps.all

        for factory_manager in self.master.factories.friendly.values():
            # Calculated
            factory_pos = factory_manager.factory.pos

            rubble_value = RubbleDigValue(
                rubble=rubble,
                full_factory_map=all_factory_map,
                factory_pos=factory_pos,
                factory_dist=20,
                factory_dist_dropoff=0.8,
                boundary_kernel_size=3,
                boundary_kernel_dropoff=0.7,
            )
            self._factory_value_maps[
                factory_manager.unit_id
            ] = rubble_value.calculate_final_value()
