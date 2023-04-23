from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Tuple, List
import numpy as np

from config import get_logger
from new_path_finder import Pather
from util import (
    calc_path_to_factory,
    power_cost_of_path,
    move_to_new_spot_on_factory,
    add_direction_to_pos,
)
from master_state import MasterState, Planner, Maps
from actions_util import Recommendation
from util import (
    power_cost_of_actions,
    POWER,
    CENTER,
    manhattan_kernel,
    SubsetExtractor,
    stretch_middle_of_factory_array,
    connected_array_values_from_pos,
    create_boundary_array,
    pad_and_crop,
    manhattan_distance_between_values,
    convolve_array_kernel,
    MOVE_DELTAS,
    MOVE_DIRECTIONS,
)
import util

from unit_status import MineValues, MineActSubCategory
from base_planners import BaseGeneralPlanner, BaseUnitPlanner

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManager
    from factory_manager import FriendlyFactoryManager

logger = get_logger(__name__)


class RubbleDigValue:
    def __init__(
        self,
        rubble: np.ndarray,
        maps: Maps,
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
        self.maps = maps
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

            # Don't clear under ice or ore (lichen can't grow there)
            rubble[self.maps.ice > 0] = 0
            rubble[self.maps.ore > 0] = 0

            if self.full_factory_map is not None:
                rubble[self.full_factory_map >= 0] = 100
            subsetter = SubsetExtractor(rubble, self.factory_pos, radius=self.factory_dist, fill_value=100)
            self._rubble_subset = subsetter.get_subset()
            self._new_factory_pos = subsetter.convert_coordinate(self.factory_pos)
        return self._rubble_subset, self._new_factory_pos

    def _get_factory_weighting(self):
        """Make factory weighting (decreasing value away from factory)"""
        if self._factory_weighting is None:
            factory_weighting = self.factory_dist_dropoff ** manhattan_kernel(self.factory_dist)
            # Stretch the middle to be 3x3
            factory_weighting = stretch_middle_of_factory_array(factory_weighting)[1:-1, 1:-1]
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
            factory_zeroes = connected_array_values_from_pos(rubble_subset, new_factory_pos)
            rubble_factory_non_zero[factory_zeroes == 1] = 999  # Anything non-zero
            manhattan_dist_to_zeros = manhattan_distance_between_values(rubble_factory_non_zero)
            # Invert so that value is higher for lower distance
            manhattan_dist_to_zeros = np.abs(manhattan_dist_to_zeros - np.max(manhattan_dist_to_zeros))
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
                self.boundary_kernel_dropoff ** manhattan_kernel(self.boundary_kernel_size),
            )
            self._conv_boundary_array = conv_boundary_array
        return self._conv_boundary_array

    def _get_low_rubble_value(self):
        """Calculate value of low rubble areas"""
        if self._low_rubble_value is None:
            rubble_subset, _ = self._get_rubble_subset()
            low_rubble_value = np.ceil(np.abs(rubble_subset - 100) / 2)  # Over 2 because light can dig 2 at a time
            self._low_rubble_value = low_rubble_value
        return self._low_rubble_value

    def calculate_final_value(self):
        # self._get_rubble_subset()
        # self._get_factory_weighting()
        # self._get_manhattan_dist_to_zeros()
        # self._get_boundary_array()
        # self._get_conv_boundary_array()
        # self._get_low_rubble_value()

        conv_boundary_array = self._get_conv_boundary_array()
        low_rubble_value = self._get_low_rubble_value()
        manhattan_dist_to_zeroes = self._get_manhattan_dist_to_zeros()
        factory_weighting = self._get_factory_weighting()
        rubble_subset, _ = self._get_rubble_subset()

        # Make a final map
        final_value = conv_boundary_array * low_rubble_value * manhattan_dist_to_zeroes * factory_weighting
        final_value[rubble_subset == 0] = 0

        final_value = pad_and_crop(final_value, self.rubble.shape, self.factory_pos[0], self.factory_pos[1])
        return final_value / np.nanmax(final_value)


def calc_value_to_move(
    pos: Tuple[int, int], value_array: np.ndarray, costmap: np.ndarray
) -> Tuple[float, util.POS_TYPE]:
    """Return maximum value of moving in any allowed direction"""
    best_direction = calc_best_direction(pos, value_array, costmap)
    if best_direction != CENTER:
        new_pos = add_direction_to_pos(pos, best_direction)
        return value_array[new_pos[0], new_pos[1]], new_pos
    return 0, pos


def calc_best_direction(pos: Tuple[int, int], value_array: np.ndarray, costmap: np.ndarray) -> int:
    """Return direction to highest allowed adjacent value"""
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
            logger.info(f"IndexError with pos {pos}, probably near edge?")
            continue
        if new_val > best_value and costmap[new_pos[0], new_pos[1]] > 0:
            best_value = new_val
            best_direction = direction
    return best_direction


class RubbleRoutePlanner:
    target_queue_length = 20

    def __init__(
        self,
        pathfinder: Pather,
        rubble: np.ndarray,
        rubble_value_map: np.ndarray,
        factory: FriendlyFactoryManager,
        unit: FriendlyUnitManager,
    ):
        """
        Args:
            rubble: full rubble map
            rubble_value_map: Value map generated by RubbleDigValue.calculate_final_value()
            factory: The factory the rubble is being cleared for
        """
        self.pathfinder = pathfinder
        # Maps
        self.rubble = rubble
        self.rubble_value_map = rubble_value_map
        # Unit
        self.unit_start_pos = unit.pos
        self.factory = factory

        # These will be changed during route planning
        self.unit = unit
        # self.unit.action_queue = []
        self._future_rubble = self.rubble.copy()
        self._future_value = self.rubble_value_map.copy()

    def make_route(self, unit_must_move: bool) -> bool:
        if unit_must_move:
            logger.info(f"Acknowledged must move, setting rubble to zero under unit at pos {self.unit.pos}")
            # Don't count rubble under current position (will ensure move from this location)
            self._future_rubble[self.unit.pos_slice] = 0
            # If on factory, move to new spot first
            if self._unit_starting_on_factory():
                logger.debug(f"Also moving on factory to get out of the way")
                #  TODO: make the move toward the best area to rubble clear instead of just anywhere on factory
                success = move_to_new_spot_on_factory(self.pathfinder, self.unit, self.factory)
                if not success:
                    logger.warning(f"Need to move, but no other location on factory to move to")
                    return False

        # Calculate actions from factory (pickup power and move to good starting location)
        if self._unit_starting_on_factory():
            success = self._from_factory_actions()
            logger.debug(f"from factory action success = {success}")
            if not success:
                return False

        # Calculate rubble route near good location until queue length reached
        kernel = 0.8 ** manhattan_kernel(5)
        for i in range(20):  # max no. loops (should break out before this)
            if len(self.unit.action_queue) < self.target_queue_length:
                logger.debug(f"Adding rubble_clear_action")
                costmap = self.pathfinder.generate_costmap(self.unit)

                # How much power left to do rubble clearing
                path_to_factory = self._path_to_factory(costmap)
                cost_to_factory = (
                    power_cost_of_path(
                        path=path_to_factory,
                        rubble=self.rubble,
                        unit_type=self.unit.unit_type,
                    )
                    + 1
                    * self.unit.unit_config.MOVE_COST
                    * self._future_rubble[self.unit.pos_slice]
                    * self.unit.unit_config.RUBBLE_MOVEMENT_COST
                )
                if cost_to_factory > 0.5 * self.unit.unit_config.BATTERY_CAPACITY:
                    logger.warning(
                        f"{self.unit.log_prefix} cost back to factory high {cost_to_factory}. path = {path_to_factory}"
                    )
                power_remaining = self.unit.power_remaining() - cost_to_factory

                # Get values to clear nearby (allowed movements only)
                unit_multiplier = pad_and_crop(
                    kernel,
                    self._future_value.shape,
                    self.unit.pos[0],
                    self.unit.pos[1],
                    fill_value=0,
                )
                value_array = self._get_boundary_values() * unit_multiplier
                value_at_pos = value_array[self.unit.pos[0], self.unit.pos[1]]
                value_to_move, new_pos = calc_value_to_move(
                    self.unit.pos,
                    value_array,
                    costmap,
                )

                # Decide what to do based on values
                if power_remaining > self.unit.unit_config.DIG_COST * 3 + self.unit.unit_config.MOVE_COST * self.rubble[
                    new_pos[0], new_pos[1]
                ] * self.unit.unit_config.RUBBLE_MOVEMENT_COST and (value_at_pos > 0 or value_to_move > 0):
                    logger.debug(f"Enough power to add another action. power_remaining = {power_remaining}")
                    # If enough power, get next action
                    success = self._calculate_next_action(
                        power_remaining=power_remaining,
                        value_at_pos=value_at_pos,
                        value_to_move=value_to_move,
                        new_pos=new_pos,
                    )
                    if not success:
                        logger.warning(f"Next action failed, at pos {self.unit.pos}")
                        return False
                else:
                    logger.debug(
                        f"Not enough power remaining = {power_remaining}, adding path to factory (cost={cost_to_factory})"
                    )
                    # Otherwise path to factory and break out of loop (done)
                    if len(path_to_factory) > 0:
                        self.pathfinder.append_path_to_actions(self.unit, path_to_factory)
                    else:
                        logger.warning(f"{self.unit.log_prefix} No path back to factory")
                    break
        else:
            logger.error(f"Got stuck in loop, breaking out now")
        return True

    def _unit_starting_on_factory(self) -> bool:
        if self.factory.factory_loc[self.unit_start_pos[0], self.unit_start_pos[1]] == 1:
            return True
        return False

    def _calculate_next_action(
        self,
        power_remaining: int,
        value_at_pos: float,
        value_to_move: float,
        new_pos: util.POS_TYPE,
    ) -> bool:
        # If better to mine in current location, mine as much as power allows
        logger.info(f"Value at pos={value_at_pos}, value to move={value_to_move}")
        if value_at_pos >= value_to_move and value_at_pos > 0:
            pos_rubble = self._future_rubble[self.unit.pos[0], self.unit.pos[1]]
            digs_required = np.ceil(pos_rubble / self.unit.unit_config.DIG_RUBBLE_REMOVED).astype(int)
            n = min(
                digs_required,
                np.floor(power_remaining / self.unit.unit_config.DIG_COST).astype(int),
            )
            rubble_after = int(max(0, pos_rubble - n * self.unit.unit_config.DIG_RUBBLE_REMOVED))
            if n <= 0:
                logger.error(f"digs_required={digs_required}, digs planned (n)={n}, << MUST BE POSITIVE")
                return False
            logger.info(f"digs_required={digs_required}, digs planned (n)={n}")
            self.unit.action_queue.append(self.unit.dig(n=n))
            self._future_rubble[self.unit.pos[0], self.unit.pos[1]] = rubble_after
            if rubble_after == 0:
                self._future_value[self.unit.pos[0], self.unit.pos[1]] = 0  # No more value there

        # Otherwise move to next best spot
        elif value_to_move > 0:
            logger.info(f"adding move to better location")
            # Should only be pathing to adjacent cell, but safer to use this
            cm = self.pathfinder.generate_costmap(self.unit, collision_only=True)
            path = self.pathfinder.fast_path(self.unit.pos, new_pos, costmap=cm)
            if len(path) > 0:
                self.pathfinder.append_path_to_actions(self.unit, path)
            else:
                logger.error(f"{self.unit.log_prefix} No path from {self.unit.pos} to {new_pos}")
                return False

        # Not near any high value, shouldn't get here
        else:
            logger.error(
                f"While calculating next action, values were all zero. (adding move center)",
            )
            return False
        return True

    def _from_factory_actions(self) -> bool:
        """Generate starting actions assuming starting on factory"""
        logger.debug(f"from_factory_actions")

        # Only top up if need a significant amount of power
        min_power = self.unit.unit_config.BATTERY_CAPACITY * 0.85
        if self.unit.power_remaining() < min_power:
            power_to_pickup = self.unit.unit_config.BATTERY_CAPACITY - self.unit.power_remaining()
            logger.debug(f"topping up power from {self.unit.power} with {power_to_pickup}")
            if self.factory.short_term_power < power_to_pickup:
                logger.warning(
                    f"{self.unit.unit_id} would like to pickup {power_to_pickup} but factory has short term power {self.factory.short_term_power}. Not doing rubble clearing this turn"
                )
                return False
            if power_to_pickup > 0:
                self.unit.action_queue.append(self.unit.pickup(POWER, power_to_pickup))

        # Find next best boundary
        logger.debug(f"Finding best boundary location")
        cm = self.pathfinder.generate_costmap(self.unit)
        for i in range(5):
            boundary_values = self._get_boundary_values()
            max_value_coord = np.unravel_index(np.argmax(boundary_values), boundary_values.shape)
            path = self.pathfinder.fast_path(
                self.unit.pos,
                end_pos=max_value_coord,
                costmap=cm,
                margin=2,
            )
            if len(path) > 0:
                logger.info(f"unit moving to {max_value_coord}")
                self.pathfinder.append_path_to_actions(self.unit, path)
                return True
            else:
                logger.debug(f"no path to {max_value_coord}, blocking and trying again")
                self._future_value[max_value_coord[0], max_value_coord[1]] = 0
        else:
            logger.error(
                f"{self.unit.log_prefix} No path to {max_value_coord} from {self.unit.pos}, moving center",
            )
            self.pathfinder.append_direction_to_actions(self.unit, CENTER)
            return False

    def _get_boundary_values(self):
        """Get the array of values on the boundary of the area connected to the factory
        That way, we can be sure we are always expanding the factories lichen area
        """

        # Get array of 1s where rubble is zero connected to factory
        factory_zeros = connected_array_values_from_pos(self._future_rubble, self.factory.factory.pos)
        # Get only the boundary of that area
        boundary = create_boundary_array(factory_zeros, boundary_num=1)
        boundary[boundary != 0] = 1
        # Look at value only where that boundary is
        return self._future_value * boundary

    def _cost_of_actions(self, actions: List[np.ndarray], rubble=None):
        # TODO: could this use future_rubble? Problem is that rubble may not yet be cleared
        if rubble is None:
            rubble = self.rubble
        return power_cost_of_actions(self.unit.start_of_turn_pos, rubble, self.unit, actions)

    def _path_to_factory(self, costmap: np.ndarray = None) -> np.ndarray:
        if costmap is None:
            costmap = self.pathfinder.generate_costmap(self.unit)
        return calc_path_to_factory(
            pathfinder=self.pathfinder,
            pos=self.unit.pos,
            costmap=costmap,
            factory_loc=self.factory.factory_loc,
            margin=2,
        )


class RubbleClearingRecommendation(Recommendation):
    """Recommend Rubble Clearing near factory"""

    role = "rubble_clearer"

    def __init__(self, best_coord: Tuple[int, int]):
        self.best_coord = best_coord


class ClearingUnitPlanner(BaseUnitPlanner, abc.ABC):
    def update_planned_actions(self):
        # print(self.unit.status.current_action.sub_category)

        # 1. Find the target clearing rea
        if self.unit.status.rubble_values.plan_step == 1:
            target = self._get_best_area()
            if target is None:
                logger.warning(f'{self.unit.log_prefix} failed to find available target')
                return
            self.unit.status.rubble_values.plan_step = 2

        # 2. Get at least a min amount of power from factory
        if self.unit.status.rubble_values.plan_step == 2:
            if self.unit.power < min_power:
                status = self.unit.action_handler.add_pickup(allow_partial=True)
                if self._check_and_handle_action_flags(status):
                    return
            if self.unit.power < min_power:
                # remove pickup
                return donothingfornow
            self.unit.status.rubble_values.plan_step = 3

        # 3. Path to target area
        if self.unit.status.rubble_values.plan_step == 3:
            status = self.action_handler.add_path(self.unit, target_array=target)
            if self._check_and_handle_action_flags(status):
                return
            self.unit.status.rubble_values.plan_step = 4

        # 4. Add dig actions
        # TODO: Here I should maybe use the existing rubble clearing stuff? Or make new?
        if self.unit.status.rubble_values.plan_step == 4:
            available_power = self.unit.power_remaining()
            power_to_facory = self._calculate_power_to_factory()
            for _ in range(10):
                n_digs = (available_power - power_to_facory) // self.unit.unit_config.DIG_COST
                if n_digs > 0:
                    status = self.action_handler.add_dig(self.unit, n_digs=n_digs)
                else:
                    self.unit.status.update_action_status(ActStatus(ActCategory.WAITING))
                    self.unit.status.rubble_values.plan_step = 1
                    return
            self.unit.status['mining_step'] = 5
            if self._check_and_handle_action_flags():
                return
            self.unit.status.rubble_values.plan_step = 6

        # 5. Return to queue
        if self.unit.status.rubble_values.plan_step == 6:
            self.unit.status.update_action_status(ActStatus(ActCategory.WAITING))
            self.unit.status.rubble_values.plan_step = 1
            self.unit.status.turn_status.replan_required = True
            return

        logger.error(f'{self.unit.log_prefix} somehow plan_step not valid {self.unit.status.rubble_values.plan_step}')
        self.unit.status.rubble_values.plan_step = 1
        self.unit.status.update_action_status(ActStatus(category=ActCategory.DROPOFF))
        return
        pass


class LichenUnitPlanner(ClearingUnitPlanner):
    def update_planned_actions(self):
        pass

    def add_new_actions(self):
        pass


class RubbleUnitPlanner(ClearingUnitPlanner):
    def update_planned_actions(self):
        return self.add_new_actions()

    def add_new_actions(self):
        rec = self.recommend(self.unit)
        return self.carry_out(self.unit, rec, self.unit.status.turn_status.must_move)

    def recommend(self, unit: FriendlyUnitManager, *args, **kwargs):
        """
        Make recommendation for this unit to clear rubble around factory
        """
        unit_factory = unit.factory_id
        if unit_factory is not None:
            value_map = self.planner._factory_value_maps[unit_factory]
            max_coord = np.unravel_index(np.argmax(value_map), value_map.shape)
            return RubbleClearingRecommendation(
                best_coord=tuple(max_coord),
            )
        return None

    def carry_out(
        self,
        unit: FriendlyUnitManager,
        recommendation: RubbleClearingRecommendation,
        unit_must_move: bool,
    ) -> bool:
        if unit.factory_id is not None:
            factory = self.master.factories.friendly[unit.factory_id]
            route_planner = RubbleRoutePlanner(
                pathfinder=self.master.pathfinder,
                rubble=self.master.maps.rubble,
                rubble_value_map=self.planner._factory_value_maps[factory.factory.unit_id],
                factory=factory,
                unit=unit,
            )
            success = route_planner.make_route(unit_must_move=unit_must_move)
            return success
        else:
            logger.error(f"in carry out, {unit.unit_id} has no factory_id")
            return False


class ClearingPlanner(BaseGeneralPlanner):
    def __init__(self, master: MasterState):
        super().__init__(master)
        self.master = master

        self._factory_value_maps = {}

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
                maps=self.master.maps,
                full_factory_map=all_factory_map,
                factory_pos=factory_pos,
                factory_dist=20,
                factory_dist_dropoff=0.8,
                boundary_kernel_size=3,
                boundary_kernel_dropoff=0.7,
            )
            self._factory_value_maps[factory_manager.unit_id] = rubble_value.calculate_final_value()

    # new
    def get_unit_planner(self, unit: FriendlyUnitManager) -> RubbleUnitPlanner:
        """Return a subclass of BaseUnitPlanner to actually update or create new actions for a single Unit"""
        if unit.unit_id not in self.unit_planners:
            unit_planner = RubbleUnitPlanner(self.master, self, unit)
            self.unit_planners[unit.unit_id] = unit_planner
        return self.unit_planners[unit.unit_id]
