from __future__ import annotations
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
import logging
from typing import TYPE_CHECKING, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np

from new_path_finder import Pather
from master_state import MasterState, Planner
from actions import Recommendation
import util
from util import (
    ICE,
    ORE,
    nearest_non_zero,
    power_cost_of_actions,
    POWER,
    CENTER,
    power_cost_of_path,
    calc_path_to_factory,
    path_to_factory_edge_nearest_pos,
)

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManger
    from factory_manager import FriendlyFactoryManager


@dataclass
class MiningRoutes:
    paths: list
    costs: list
    values: list


class MiningRecommendation(Recommendation):
    """Recommended mining action between a resource and factory"""

    role = 'miner'

    def __init__(
        self, distance_from_factory: float, resource_pos, factory_id, resource_type: int
    ):
        self.distance_from_factory = distance_from_factory
        self.resource_pos: Tuple[int, int] = resource_pos
        self.factory_id: str = factory_id
        self.resource_type = resource_type


# class MiningHelperRecommendation(Recommendation):
#     """Recommend unit helps a mining unit (i.e. provide power and take away resources)"""
#     role = 'miner_helper'
#
#     def __init__(self, resource_pos, factory_id, resource_type: str):
#         self.resource_pos: Tuple[int, int] = resource_pos
#         self.factory_id: str = factory_id
#         self.resource_type = resource_type


class MiningRoutePlanner:
    move_lookahead = 10
    target_queue_length = 20

    def __init__(
        self,
        pathfinder: Pather,
        rubble: np.ndarray,
        resource_pos: Tuple[int, int],
        resource_type: int,
        factory: FriendlyFactoryManager,
        unit: FriendlyUnitManger,
    ):
        """
        Args:
            pathfinder: Agents pathfinding instance
            rubble: full rubble map
            resource_pos: position of resource to mine
            resource_type: i.e. ICE, ORE, etc
            factory: factory being mined for
            unit: unit to make route for
        """
        self.pathfinder = pathfinder

        # Map related
        self.rubble = rubble
        self.resource_pos = resource_pos
        self.resource_type = resource_type
        # Unit related
        self.unit_start_pos = unit.pos
        self.factory = factory

        # This will be changed during route planning
        self.unit = unit
        self.unit.action_queue = []

    def _path_to_and_from_resource(self):
        path_to_resource = self._path_to_resource()
        cost_to_resource = power_cost_of_path(
            path_to_resource, self.rubble, self.unit.unit_type
        )

        path_from_resource_to_factory = self._path_to_factory(
            from_pos=self.resource_pos
        )
        cost_from_resource_to_factory = power_cost_of_path(
            path_from_resource_to_factory, self.rubble, self.unit.unit_type
        )
        return path_to_resource, cost_to_resource, cost_from_resource_to_factory

    def make_route(self, unit_must_move: bool) -> bool:
        """
        - If on factory and needs to move, move first
        - Then check how to get to factory if necessary (resource first or factory first)
        - Then mining route path from factory back to factory
        """
        success = True
        if self._unit_starting_on_factory() and unit_must_move:
            success = util.move_to_new_spot_on_factory(
                self.pathfinder, self.unit, self.factory
            )
            if not success:
                return False

        # If not on factory, route until at factory (possibly resource first)
        if not self._unit_starting_on_factory():
            # collect info to decide if we should move towards resource or factory first
            (
                path_to_resource,
                cost_to_resource,
                cost_from_resource_to_factory,
            ) = self._path_to_and_from_resource()
            power_remaining = (
                self.unit.power - cost_to_resource - cost_from_resource_to_factory
            )

            if len(path_to_resource) == 0:
                # No path to resource
                return False

            # Decide which to do
            if power_remaining > 3 * self.unit.unit_config.DIG_COST:
                # Go to resource first
                success = self._resource_then_factory(path_to_resource, power_remaining)
            else:
                # Go to factory first
                logging.info('pathing to factory first')
                direct_path_to_factory = self._path_to_factory(
                    from_pos=self.resource_pos,
                )
                if len(direct_path_to_factory) > 0:
                    self.pathfinder.append_path_to_actions(
                        self.unit, direct_path_to_factory
                    )
                else:  # No path to factory
                    success = False

        # Then route from factory (if successful up to this point)
        if success:
            if len(self.unit.action_queue) < self.target_queue_length:
                # Then loop from factory
                success = self._from_factory_actions()
        return success

    def _resource_then_factory(
        self, path_to_resource, power_remaining_after_moves
    ) -> bool:
        success = True
        logging.info('pathing to resource first')
        # Move to resource
        self.pathfinder.append_path_to_actions(self.unit, path_to_resource)

        # Dig as many times as possible
        n_digs = int(
            np.floor(power_remaining_after_moves / self.unit.unit_config.DIG_COST)
        )
        if n_digs >= 1:
            self.unit.action_queue.append(self.unit.dig(n=n_digs))
        else:
            logging.error(f'n_digs = {n_digs}, should always be greater than 1')
            success = False

        # Move to factory
        path_from_resource_to_factory = self._path_to_factory(
            from_pos=self.resource_pos,
        )
        if len(path_from_resource_to_factory) > 0:
            self.pathfinder.append_path_to_actions(
                self.unit, path_from_resource_to_factory
            )
        else:  # No path to factory
            success = False

        # Transfer resources to factory (only if successful up to now)
        if success:
            self.unit.action_queue.append(
                self.unit.transfer(
                    CENTER, self.resource_type, self.unit.unit_config.CARGO_SPACE
                )
            )
        return success

    def _unit_starting_on_factory(self) -> bool:
        if (
            self.factory.factory_loc[self.unit_start_pos[0], self.unit_start_pos[1]]
            == 1
        ):
            return True
        return False

    def _from_factory_actions(
        self,
    ) -> bool:
        """Generate actions for mining route assuming starting on factory
        Note: No use of repeat any more for ease of updating at any time
        """
        # Move to outside edge of factory (if necessary)
        success = self._move_to_edge_of_factory()
        if not success:
            return False

        # Calculate travel costs
        (
            path_to_resource,
            cost_to_resource,
            cost_from_resource_to_factory,
        ) = self._path_to_and_from_resource()
        travel_cost = cost_to_resource + cost_from_resource_to_factory

        # Aim for as many digs as battery capacity would allow
        available_power = self.unit.power - power_cost_of_actions(
            self.rubble, self.unit, self.unit.action_queue
        )
        target_digs = int(
            (self.unit.unit_config.BATTERY_CAPACITY - travel_cost)
            / self.unit.unit_config.DIG_COST,
        )
        target_power = travel_cost + target_digs * self.unit.unit_config.DIG_COST

        # If action queue is short, assume factory won't have more energy, if long, assume it will
        if len(self.unit.action_queue) < 10:
            factory_power = self.factory.factory.power
        else:
            # Assume enough to do anything (will be checking if this is true before unit tries to do it)
            factory_power = 3000

        # Pickup power and update n_digs
        if factory_power > target_power - available_power:
            logging.info(
                f'picking up desired power to achieve target of {target_power}'
            )
            power_to_pickup = target_power - available_power
            if power_to_pickup > 0:
                self.unit.action_queue.append(self.unit.pickup(POWER, power_to_pickup))
            n_digs = target_digs
        elif factory_power + available_power > self.unit.unit_config.DIG_COST * 3:
            logging.info(f'picking up available power {factory_power}')
            self.unit.action_queue.append(self.unit.pickup(POWER, factory_power))
            n_digs = int(
                np.floor(
                    (factory_power + available_power - travel_cost)
                    / self.unit.unit_config.DIG_COST
                )
            )
        else:
            # Not enough energy to do a mining run, return now (everything up to this point is still useful to do)
            logging.warning(
                f"{self.factory.unit_id} doesn't have enough energy for {self.unit.unit_id} to do a "
                f"mining run to {self.resource_pos} from {self.unit.pos}"
            )
            return False

        # Add journey out
        if len(path_to_resource) > 0:
            self.pathfinder.append_path_to_actions(self.unit, path_to_resource)
        else:
            logging.warning(
                f"{self.unit.unit_id} has no path to {self.resource_pos} from {self.unit.pos}"
            )
            return False

        # Add digs
        if n_digs >= 1:
            self.unit.action_queue.append(self.unit.dig(n=n_digs))
        else:
            logging.error(
                f'n_digs = {n_digs}, unit heading off to not mine anything. should always be greater than 1'
            )
            return False

        # Add return journey
        return_path = self._path_to_factory(
            self.unit.pos,
        )
        if len(return_path) > 0:
            self.pathfinder.append_path_to_actions(self.unit, return_path)
        else:
            logging.warning(
                f"{self.unit.unit_id} has no path to {self.factory.unit_id} at {self.factory.factory.pos} "
                f"from {self.unit.pos}"
            )
            return False

        # Add transfer
        self.unit.action_queue.append(
            self.unit.transfer(
                CENTER, self.resource_type, self.unit.unit_config.CARGO_SPACE
            )
        )
        return True

    def _move_to_edge_of_factory(self) -> bool:
        path = path_to_factory_edge_nearest_pos(
            pathfinder=self.pathfinder,
            factory_loc=self.factory.factory_loc,
            pos=self.unit.pos,
            pos_to_be_near=self.resource_pos,
            margin=2,
        )
        if len(path) == 0:
            logging.error(
                f'Apparently no way to get to the edge of the factory without colliding from {self.unit.pos}',
            )
            return False
        elif len(path) > 0:
            self.pathfinder.append_path_to_actions(self.unit, path)
            return True

    def _path_to_factory(self, from_pos: Tuple[int, int]) -> np.ndarray:
        return calc_path_to_factory(
            self.pathfinder,
            from_pos,
            self.factory.factory_loc,
            margin=2,
        )

    def _path_to_resource(
        self,
        from_pos: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        if from_pos is None:
            from_pos = self.unit.pos
        return self.pathfinder.fast_path(
            start_pos=from_pos,
            end_pos=self.resource_pos,
            margin=2,
        )

    def _cost_of_actions(self, actions: List[np.ndarray], rubble=None):
        # TODO: could this use future_rubble? Problem is that rubble may not yet be cleared
        if rubble is None:
            rubble = self.rubble
        return power_cost_of_actions(rubble, self.unit, actions)


class MiningPlanner(Planner):
    def __init__(self, master_state: MasterState):
        self.master: MasterState = master_state
        self._mining_routes = None

    @property
    def ice(self):
        return self.master.maps.ice

    @property
    def ore(self):
        return self.master.maps.ore

    @property
    def friendly_factories(self):
        """Map of friendly factories (where -1 is non-factory, otherwise factory id)"""
        return self.master.maps.factory_maps.friendly

    def update(self):
        pass

    def update_actions_of(
        self, unit: FriendlyUnitManger, resource_type: int, unit_must_move: bool
    ):
        """Figure out what the next actions for unit should be

        If must move:
            - If already mining, recommendation will change (current position will now be blocked)
            - If moving to or from recommendation, no problem, already moving
            - If at factory, unit needs to move before picking up power
        """
        rec = self.recommend(
            unit, resource_type=resource_type, unit_must_move=unit_must_move
        )
        success = False
        if rec is not None:
            self.carry_out(unit, rec, unit_must_move=unit_must_move)
            success = True
        else:
            logging.error(f'Mining planner returned None for {unit.unit_id}')
        return success

    def recommend(
        self, unit: FriendlyUnitManger, resource_type: int = ICE, **kwargs
    ) -> [None, MiningRecommendation]:
        # Which resource are we looking for?
        if resource_type == ICE:
            resource_map = self.ice.copy()
        elif resource_type == ORE:
            resource_map = self.ore.copy()
        else:
            raise NotImplementedError(
                f'{resource_type} not recognized, should be {ICE} or {ORE}'
            )

        # If unit must move, make sure not to recommend resource under unit
        resource_map[unit.pos_slice] = 0

        # Where is the unit and where is the factory
        unit_pos = unit.pos
        unit_factory = self.master.factories.friendly.get(unit.factory_id, None)
        if unit_factory is None:
            logging.warning(
                f"Factory doesn't exist for {unit.unit_id} with factory_id {unit.factory_id}"
            )
            return None

        # Find resource nearest factory that doesn't have an impassible cost
        for attempt in range(5):
            nearest_resource = nearest_non_zero(resource_map, unit_factory.pos)
            if nearest_resource is None:
                logging.warning(
                    f"No nearest resource ({resource_type}) to {unit_factory.unit_id} after {attempt} attempts"
                )
                return None
            path_to_resource = self.master.pathfinder.fast_path(
                unit_pos, nearest_resource
            )
            if len(path_to_resource) > 0:
                break

            # Blank out that resource and try again
            resource_map[nearest_resource[0], nearest_resource[1]] = 0
        else:
            logging.warning(
                f"No free resources ({resource_type}) for {unit_factory.unit_id} after a few attempts"
            )
            return None

        return MiningRecommendation(
            distance_from_factory=util.manhattan(unit_factory.pos, nearest_resource),
            resource_pos=nearest_resource,
            factory_id=unit_factory.unit_id,
            resource_type=resource_type,
        )

    def carry_out(
        self,
        unit_manager: FriendlyUnitManger,
        recommendation: MiningRecommendation,
        unit_must_move: bool,
    ) -> bool:
        factory = self.master.factories.friendly[recommendation.factory_id]
        route_planner = MiningRoutePlanner(
            pathfinder=self.master.pathfinder,
            rubble=self.master.maps.rubble,
            resource_pos=recommendation.resource_pos,
            resource_type=recommendation.resource_type,
            factory=factory,
            unit=unit_manager,
        )
        success = route_planner.make_route(unit_must_move=unit_must_move)
        return success
