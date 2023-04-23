from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterable, Dict, Tuple, Union, List, Optional, TYPE_CHECKING

import numpy as np

import actions_util
import util
from master_state import Maps, AllUnits
from new_path_finder import Pather
from config import get_logger

logger = get_logger(__name__)


if TYPE_CHECKING:
    from unit_manager import UnitManager, FriendlyUnitManager, EnemyUnitManager


def ensure_path_includes_at_least_next_step(path):
    """
    If path only 1 in length, probably unit has no action queue and will stay still (for at least 1 turn)
    (i.e. if path is only [current_pos], next pos will probably be [current_pos] too)
    """
    if len(path) == 1:
        path = [path[0], path[0]]
    return path


def check_next_move_will_happen(path, unit: UnitManager, rubble: np.ndarray):
    """
    Does the unit actually have enough energy to do first move if first action is to move
    If not, add a not move to path so next step collisions can be found
    """
    # Is next action supposed to be move
    if not np.all(path[0] == path[1]):
        move_cost = unit.unit_config.MOVE_COST + unit.unit_config.RUBBLE_MOVEMENT_COST * rubble[path[1][0], path[1][1]]

        # Unit isn't really going to move next turn
        if move_cost > unit.start_of_turn_power:
            path = list(path)
            path.insert(0, path[0])
    return path


def sanitize_path_start(path, unit: UnitManager, rubble: np.ndarray):
    """Make sure the beginning of the path is accurate (for next step collisions)"""
    path = ensure_path_includes_at_least_next_step(path)
    path = check_next_move_will_happen(path, unit, rubble)
    return path


def find_collisions(
    this_unit: UnitManager,
    other_units: Iterable[UnitManager],
    max_step: int,
    other_is_enemy: bool,
    rubble: np.ndarray,
) -> Dict[str, Collision]:
    """Find the first collision point between this_unit and each other_unit
    I.e. Only first collision coordinate when comparing the two paths
    """

    # Note: this defaults to planned path for friendly units
    this_path = this_unit.current_path(max_len=max_step)
    this_path = sanitize_path_start(this_path, this_unit, rubble)

    collisions = {}
    for other in other_units:
        # Don't include collisions with self
        if this_unit.unit_id == other.unit_id:
            continue
        # Note: this defaults to planned path for friendly units
        other_path = other.current_path(max_len=max_step)
        other_path = sanitize_path_start(other_path, other, rubble)

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
        self.unit_actions = unit.status.planned_action_queue
        self.unit_path = self.unit.current_path(max_len=max_step, planned_actions=True)

    def _collision_while_travelling(self, collision: Collision) -> bool:
        """Is the collision part way through travelling a-b
        Basically is the step after somewhere else and not final dest?
        """
        # was_moving = actions_util.was_unit_moving_at_step(
        #     self.unit_actions, collision.step
        # )
        moving_this_turn = actions_util.will_unit_move_at_step(self.unit_actions, collision.step)
        moving_next_turn = actions_util.will_unit_move_at_step(self.unit_actions, collision.step + 1)

        travelling = moving_next_turn and moving_this_turn
        if travelling:
            logger.debug(f"{self.unit.unit_id} is travelling at collision step {collision.step}, at {collision.pos}")
        return travelling

    def _collision_on_factory(self, collision: Collision) -> bool:
        """Is the collision on a friendly factory tile?"""
        at_factory = self.maps.factory_maps.friendly[collision.pos[0], collision.pos[1]] >= 0
        if at_factory:
            logger.debug(f"{self.unit.unit_id} is at factory at collision step {collision.step}, at {collision.pos}")
        return False

    def _collision_at_destination(self, collision: Collision) -> bool:
        """Is the collision at the destination of unit (i.e. doesn't move next step)"""
        will_be_moving = actions_util.will_unit_move_at_step(self.unit_actions, collision.step)
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
        index = actions_util.find_dest_step_from_step(self.unit_path, step, direction="forward")
        return index, self.unit_path[index]

    def _previous_dest_or_start_step(self, step: int) -> Tuple[int, util.POS_TYPE]:
        """Return the pos of the previous dest before step, or start pos if no previous dest"""
        index = actions_util.find_dest_step_from_step(self.unit_path, step, direction="backward")
        return index, self.unit_path[index]

    def _replace_unit_actions(self, start_step, end_step, new_actions):
        """Replace the actions from start step to end step with new actions
        Note: New actions may have different length"""
        #  update actions
        existing_actions = self.unit_actions
        replaced_actions = actions_util.replace_actions(existing_actions, start_step, end_step, new_actions)
        self.unit.action_queue = replaced_actions
        self.unit_actions = replaced_actions

    def _make_path(self, start_step: int, start_pos: util.POS_TYPE, end_pos: util.POS_TYPE):
        cm = self.pathfinder.generate_costmap(self.unit, override_step=start_step)
        new_path = self.pathfinder.fast_path(start_pos, end_pos, costmap=cm)
        # if self.unit.unit_id == 'unit_6':
        #     print(start_pos, end_pos, start_step)
        #     fig=util.show_map_array(cm)
        #     util.plotly_plot_path(fig, new_path)
        #     fig.show()
        if len(new_path) == 0:
            logger.info(f"Default pathing failed, Attempting to resolve path avoiding collisions only")
            cm = self.pathfinder.generate_costmap(self.unit, override_step=start_step, collision_only=True)
            new_path = self.pathfinder.fast_path(start_pos, end_pos, costmap=cm)
        return new_path

    def _make_path_to_factory_edge(self, start_step, factory_loc, start_pos, pos_to_be_near):
        cm = self.pathfinder.generate_costmap(self.unit, override_step=start_step)
        new_path = util.path_to_factory_edge_nearest_pos(
            self.pathfinder,
            factory_loc=factory_loc,
            pos=start_pos,
            pos_to_be_near=pos_to_be_near,
            costmap=cm,
        )
        if len(new_path) == 0:
            logger.info(f"Default pathing failed, Attempting to resolve path avoiding collisions only")
            cm = self.pathfinder.generate_costmap(self.unit, override_step=start_step, collision_only=True)
            new_path = util.path_to_factory_edge_nearest_pos(
                self.pathfinder,
                factory_loc=factory_loc,
                pos=start_pos,
                pos_to_be_near=pos_to_be_near,
                costmap=cm,
            )
        return new_path

    def _resolve_travel_collision(self, collision: Collision) -> bool:
        """Repath to later point on path (ideally destination)

        Note: Might want to replace this after planners are working better

        Returns:
            bool: whether unit.status.turn_status.recommend_plan_update has been updated
        """
        last_step, next_dest_or_last_step = self._next_dest_or_last_step(collision.step)
        first_step, prev_dest_or_first_step = self._previous_dest_or_start_step(collision.step)
        logger.debug(
            f"repathing from {prev_dest_or_first_step} to {next_dest_or_last_step} (starting step {first_step})"
        )
        if np.all(next_dest_or_last_step == collision.pos) or np.all(prev_dest_or_first_step == collision.step):
            logger.error(
                f"first or last dest was same as collision pos next={next_dest_or_last_step} prev={prev_dest_or_first_step} collision step={collision.step}"
            )
            self.unit.status.turn_status.recommend_plan_update = True
            return True
        new_path = self._make_path(first_step, prev_dest_or_first_step, next_dest_or_last_step)

        if len(new_path) == 0:
            logger.warning(
                f"failed to find new path from {prev_dest_or_first_step} to {next_dest_or_last_step} starting step {first_step}"
            )
            self.unit.status.turn_status.recommend_plan_update = True
            return True

        new_actions = util.path_to_actions(new_path)
        self._replace_unit_actions(first_step, last_step, new_actions)
        self.unit.status.turn_status.recommend_plan_update = False
        return True

    def _resolve_factory_collision(self, collision: Collision) -> bool:
        """Repath to new spot on factory if possible

        Note: Might want to replace this after planners are working better

        Returns:
            bool: whether unit.status.turn_status.recommend_plan_update has been updated
        """

        first_step, prev_dest_or_first_step = self._previous_dest_or_start_step(collision.step)
        factory_num = self.maps.factory_maps.all[collision.pos[0], collision.pos[1]]
        if factory_num < 0:
            raise ValueError(f"collision not on factory")

        factory_loc = (self.maps.factory_maps.all == factory_num).astype(int)
        new_path = self._make_path_to_factory_edge(
            first_step, factory_loc, prev_dest_or_first_step, prev_dest_or_first_step
        )
        if len(new_path) == 0:
            logger.warning(
                f"failed to find new path from {prev_dest_or_first_step} to factory_{factory_num} starting step {first_step}"
            )
            self.unit.status.turn_status.recommend_plan_update = True
            return True

        new_actions = util.path_to_actions(new_path)
        self._replace_unit_actions(first_step, collision.step + 1, new_actions)
        self.unit.status.turn_status.recommend_plan_update = False
        return True

    def _resolve_destination_collision(self, collision: Collision) -> bool:
        """Ideally repath to new nearby resource or something"""
        logger.info(f"resolving conflict at destination required but not implemented")
        self.unit.status.turn_status.recommend_plan_update = True
        return True

    def resolve(self) -> bool:
        """If there is a collision with a friendly or enemy coming up, re-path just the part that avoids the collision
        1. Moving from a-b
        2. Collision on factory
        3. Collision on destination (resource/rubble)

        Note: Might want to remove this once planners are working

        Returns:
            bool: Has unit.status.turn_status been updated
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
            self.unit.status.turn_status.recommend_plan_update = True
            return True

        logger.info(
            f"resolving collision between {self.unit.unit_id} and {nearest_collision.unit_id} at {nearest_collision.pos} at step {nearest_collision.step}"
        )
        if self._collision_while_travelling(nearest_collision):
            status_updated = self._resolve_travel_collision(nearest_collision)
        elif self._collision_on_factory(nearest_collision):
            status_updated = self._resolve_factory_collision(nearest_collision)
        elif self._collision_at_destination(nearest_collision):
            status_updated = self._resolve_destination_collision(nearest_collision)
        else:
            self.unit.status.turn_status.recommend_plan_update = True
            status_updated = True
        if status_updated is False:
            logger.info(f"Failed to solve collision {nearest_collision}, route still needs updating")
            self.unit.status.turn_status.recommend_plan_update = True
            return True
        logger.info(f"Nearest collisions solved, can continue")
        self.unit.status.turn_status.recommend_plan_update = False
        return True


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
            nearest_collision = Collision("none", "none", False, (-1, -1), -1)
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


class UnitPaths:
    """Unit paths stored in a 3D array (step, x, y) where value is id_num (otherwise -1)"""

    # How many extra layers to use in pathing (i.e. including taking an extra X turns of collisions in case it takes X
    # turns more to get there)
    additional_lag_steps = 3

    def __init__(
        self,
        friendly: Dict[str, FriendlyUnitManager],
        enemy: Dict[str, EnemyUnitManager],
        friendly_valid_move_map: np.ndarray,
        enemy_valid_move_map: np.ndarray,
        max_step: int,
        rubble: np.ndarray,
    ):
        self.friendly_valid_move_map = friendly_valid_move_map
        self.enemy_valid_move_map = enemy_valid_move_map
        self.rubble = rubble  # For calculating if enough energy to actually take next step
        self.max_step = max_step
        self.friendly_light = np.full((max_step + 1, rubble.shape[0], rubble.shape[1]), fill_value=-1, dtype=int)
        self.friendly_heavy = np.full((max_step + 1, rubble.shape[0], rubble.shape[1]), fill_value=-1, dtype=int)
        self.enemy_light = np.full((max_step + 1, rubble.shape[0], rubble.shape[1]), fill_value=-1, dtype=int)
        self.enemy_heavy = np.full((max_step + 1, rubble.shape[0], rubble.shape[1]), fill_value=-1, dtype=int)
        self._init_arrays(friendly_units=friendly, enemy_units=enemy)

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

    def _init_arrays(self, friendly_units, enemy_units):
        friendly_light = {unit.id_num: unit for unit in friendly_units.values() if unit.unit_type == "LIGHT"}
        friendly_heavy = {unit.id_num: unit for unit in friendly_units.values() if unit.unit_type == "HEAVY"}
        enemy_light = {unit.id_num: unit for unit in enemy_units.values() if unit.unit_type == "LIGHT"}
        enemy_heavy = {unit.id_num: unit for unit in enemy_units.values() if unit.unit_type == "HEAVY"}

        map_shape = self.friendly_valid_move_map.shape

        # +1 so that last layer is always left empty of units (for longer pathing)
        arrays = [
            np.full(
                (self.max_step + 1, map_shape[0], map_shape[1]),
                fill_value=-1,
                dtype=int,
            )
            for _ in range(4)
        ]
        for array, dict_, move_map, is_enemy in zip(
            arrays,
            [friendly_light, friendly_heavy, enemy_light, enemy_heavy],
            [
                self.friendly_valid_move_map,
                self.friendly_valid_move_map,
                self.enemy_valid_move_map,
                self.enemy_valid_move_map,
            ],
            [False, False, True, True],
        ):
            for unit_num, unit in dict_.items():
                self.add_unit(unit, is_enemy=is_enemy)

    def get_unit_nums_near(
        self,
        pos: util.POS_TYPE,
        step: int,
        radius: int = 5,
        friendly_light=True,
        friendly_heavy=True,
        enemy_light=True,
        enemy_heavy=True,
    ) -> np.ndarray:
        """Return the ID_NUM of enemies near to pos at given step"""
        near_nums = np.full_like(self.friendly_valid_move_map, fill_value=-1)
        if step > self.max_step:
            logger.warning(f"Requesting unit_near_nums for step {step} > max step {self.max_step}, returning empty")
            return near_nums

        mask = util.pad_and_crop(util.manhattan_kernel(radius), near_nums.shape, pos[0], pos[1])

        # Start with empty
        x, y = self._x, self._y
        near_nums = np.full_like(self.friendly_valid_move_map, fill_value=-1)
        teams, utypes = self._get_teams_and_utypes(friendly_light, friendly_heavy, enemy_light, enemy_heavy)
        for team, utype in zip(teams, utypes):
            key = f"{team}_{utype}"
            arr = getattr(self, key)
            arr = arr[step]
            # Makes all outside of radius zero
            masked = (arr + 1) * mask  # +1 to make id_0 become 1 (and -1s become 0)
            # But then be sure to copy in the actual id_nums
            near_nums[masked >= 0] = arr
        return near_nums

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

    @staticmethod
    def _add_path_to_array(unit: UnitManager, path, arr: np.ndarray, max_step: int, is_enemy: bool):
        x, y = unit.start_of_turn_pos

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

        # If low energy, add current position as next step as well (just in case unit doesn't move)
        if unit.start_of_turn_power < (
            unit.unit_config.MOVE_COST
            + 100 * unit.unit_config.RUBBLE_MOVEMENT_COST
            + unit.unit_config.ACTION_QUEUE_POWER_COST
        ):
            arr[1, path[0][0], path[0][1]] = unit.id_num

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

        # Calculate the valid path (i.e. can't walk of edge of map or through enemy factory)
        # NOTE: does NOT include power considerations (i.e. if enough power to do first move)
        # Especially important for enemy units... Don't want to deal with invalid paths later
        # NOTE: Friendly uses planned_actions by default
        valid_path = unit.valid_moving_actions(costmap=move_map, max_len=self.max_step)

        # Get the valid path coords (first value is current position)
        path = util.actions_to_path(unit.start_of_turn_pos, actions=valid_path.valid_actions, max_len=self.max_step)

        # Account for low energy or no path to at least get next position correct (assuming no action queue update cost)
        path = sanitize_path_start(path, unit, self.rubble)

        # Add that path to the 3D path array
        self._add_path_to_array(unit, path, array, self.max_step, is_enemy=is_enemy)

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
            true_intercept: True disables multi-step and only adds cost where collisions would occur if travelling
                shortest manhattan dist. Note: Still blocks tiles adjacent to enemy unless
                collision cost > 0 (i.e. not blocked)
            nearby_start_cost: None disables mutistep checking, 0 enables mutistep, but no extra cost from being near

        """
        if (friendly_nearby_start_cost is not None and friendly_nearby_start_cost < 0) or (
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
        if (enemy_nearby_start_cost or friendly_nearby_start_cost) and not true_intercept:
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
                    for i in range(1, self.additional_lag_steps)
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
                if enemy_nearby_start_cost is not None and enemy_nearby_start_cost and not true_intercept:
                    blur *= nearby_start_cost * step_dropoff_multiplier ** (start_step + i)
                    # Blur that with kernel
                    add_cm = util.convolve_array_kernel(arr, blur, fill=0)
                    # Add non-blocking costs
                    cm[cm > 0] += add_cm[cm > 0]

                # Calculate likely collisions
                if np.all(collision_kernel.shape == (1, 1)):
                    col_cm = collision_kernel[0, 0] * arr
                else:
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
                logger.debug(f"Requesting map outside of max_steps {self.max_step} returning empty")
                do_calcs = False

        if do_calcs:
            # Distance kernel
            kernel = util.manhattan_kernel(self.max_step - start_step) + start_step
            kernel[kernel > self.max_step] = self.max_step

            # Place that in correct spot
            index_array = util.pad_and_crop(
                kernel,
                self.friendly_valid_move_map.shape,
                pos[0],
                pos[1],
                fill_value=self.max_step - start_step,
            )

        # Start with empty
        x, y = self._x, self._y
        likely_collision_id_maps = OrderedDict()
        teams, utypes = self._get_teams_and_utypes(friendly_light, friendly_heavy, enemy_light, enemy_heavy)
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


def store_all_paths_to_array(paths: Dict[int, np.ndarray], map_shape, max_step=30, fill_value: int = -1) -> np.ndarray:
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
    all_units: AllUnits,
    rubble: np.ndarray,
    check_steps_enemy: int = 5,
    check_steps_friendly: int = 20,
) -> Dict[str, AllCollisionsForUnit]:
    """Calculate first collisions in the next <check_steps> for all units"""
    all_unit_collisions = {}
    for unit_id, unit in all_units.friendly.all.items():
        collisions_for_unit = AllCollisionsForUnit(
            with_friendly=CollisionsForUnit(
                light=find_collisions(
                    unit,
                    all_units.friendly.light.values(),
                    max_step=check_steps_friendly,
                    other_is_enemy=False,
                    rubble=rubble,
                ),
                heavy=find_collisions(
                    unit,
                    all_units.friendly.heavy.values(),
                    max_step=check_steps_friendly,
                    other_is_enemy=False,
                    rubble=rubble,
                ),
            ),
            with_enemy=CollisionsForUnit(
                light=find_collisions(
                    unit,
                    all_units.enemy.light.values(),
                    max_step=check_steps_enemy,
                    other_is_enemy=True,
                    rubble=rubble,
                ),
                heavy=find_collisions(
                    unit,
                    all_units.enemy.heavy.values(),
                    max_step=check_steps_enemy,
                    other_is_enemy=True,
                    rubble=rubble,
                ),
            ),
        )
        if collisions_for_unit.num_collisions(friendly=True, enemy=True) > 0:
            all_unit_collisions[unit_id] = collisions_for_unit
    return all_unit_collisions


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
