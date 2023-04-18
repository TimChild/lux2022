from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
from deprecation import deprecated

from config import get_logger
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
    from unit_manager import FriendlyUnitManager
    from factory_manager import FriendlyFactoryManager

logger = get_logger(__name__)


@dataclass
class MiningRoutes:
    paths: list
    costs: list
    values: list


class MiningRecommendation(Recommendation):
    """Recommended mining action between a resource and factory"""

    role = "miner"

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


# class MiningRoutePlanner:
#     move_lookahead = 10
#     target_queue_length = 20
#
#     def __init__(
#         self,
#         pathfinder: Pather,
#         rubble: np.ndarray,
#         resource_pos: Tuple[int, int],
#         resource_type: int,
#         factory: FriendlyFactoryManager,
#         unit: FriendlyUnitManager,
#     ):
#         """
#         Args:
#             pathfinder: Agents pathfinding instance
#             rubble: full rubble map
#             resource_pos: position of resource to mine
#             resource_type: i.e. ICE, ORE, etc
#             factory: factory being mined for
#             unit: unit to make route for
#         """
#         self.pathfinder = pathfinder
#
#         # Map related
#         self.rubble = rubble
#         self.resource_pos = resource_pos
#         self.resource_type = resource_type
#         # Unit related
#         self.unit_start_pos = unit.pos
#         self.factory = factory
#
#         # This will be changed during route planning
#         self.unit = unit
#         self.unit.action_queue = []
#
#     def _path_to_and_from_resource(self):
#         path_to_resource = self._path_to_resource()
#         cost_to_resource = power_cost_of_path(
#             path_to_resource, self.rubble, self.unit.unit_type
#         )
#
#         path_from_resource_to_factory = self._path_to_factory(
#             from_pos=self.resource_pos
#         )
#         cost_from_resource_to_factory = power_cost_of_path(
#             path_from_resource_to_factory, self.rubble, self.unit.unit_type
#         )
#         return path_to_resource, cost_to_resource, cost_from_resource_to_factory
#
#     def make_route(self, unit_must_move: bool) -> bool:
#         """
#         - If on factory and needs to move, move first
#         - Then check how to get to factory if necessary (resource first or factory first)
#         - Then mining route path from factory back to factory
#         """
#         success = True
#         if self.unit.on_own_factory() and unit_must_move:
#             logger.debug(f"Acknowledged that unit must move")
#             cm = self.pathfinder.generate_costmap(self.unit)
#             path = util.path_to_factory_edge_nearest_pos(
#                 self.pathfinder,
#                 self.factory.factory_loc,
#                 self.unit.pos,
#                 self.resource_pos,
#                 costmap=cm,
#             )
#             if len(path) > 0:
#                 self.pathfinder.append_path_to_actions(self.unit, path)
#             else:
#                 logger.error(f'{self.unit.log_prefix}: Failed to path to edge of factory')
#                 return False
#
#         # If not on factory, route until at factory (possibly resource first)
#         if not self.unit.on_own_factory():
#             # collect info to decide if we should move towards resource or factory first
#             logger.debug(f"not on factory, deciding resource or factory first")
#             (
#                 path_to_resource,
#                 cost_to_resource,
#                 cost_from_resource_to_factory,
#             ) = self._path_to_and_from_resource()
#
#             power_remaining = (
#                 self.unit.power_remaining()
#                 - cost_to_resource
#                 - cost_from_resource_to_factory
#             )
#
#             if len(path_to_resource) == 0:
#                 # No path to resource
#                 return False
#
#             # Decide which to do
#             if power_remaining > max(
#                 2 * self.unit.unit_config.DIG_COST,
#                 len(path_to_resource) // 2 * self.unit.unit_config.DIG_COST,
#             ):
#                 # Go to resource first
#                 success = self.resource_then_factory_route(path_to_resource, power_remaining)
#             else:
#                 # Go to factory first
#                 logger.info(
#                     f"pathing to {self.factory.factory.unit_id} at {self.factory.pos} first"
#                 )
#                 direct_path_to_factory = self._path_to_factory(
#                     from_pos=self.unit.pos,
#                 )
#                 if len(direct_path_to_factory) > 0:
#                     self.pathfinder.append_path_to_actions(
#                         self.unit, direct_path_to_factory
#                     )
#                 else:  # No path to factory (would still be 1 if on factory)
#                     return False
#
#                 self._drop_off_cargo_if_necessary()
#
#         # Then route from factory (if successful up to this point)
#         if success:
#             if len(self.unit.action_queue) < self.target_queue_length:
#                 logger.debug(f'adding from factory actions')
#                 # Then loop from factory
#                 success = self.from_factory_route()
#         return success
#
#     def _drop_off_cargo_if_necessary(self):
#         if self.unit.cargo.ice > 0:
#             logger.info(f"dropping off {self.unit.cargo.ice} ice before from_factory")
#             self.unit.action_queue.append(
#                 self.unit.transfer(CENTER, ICE, self.unit.unit_config.CARGO_SPACE)
#             )
#         if self.unit.cargo.ore > 0:
#             logger.info(f"dropping off {self.unit.cargo.ore} ore before from_factory")
#             self.unit.action_queue.append(
#                 self.unit.transfer(CENTER, ORE, self.unit.unit_config.CARGO_SPACE)
#             )
#
#     def resource_then_factory_route(
#         self, path_to_resource, power_remaining_after_moves
#     ) -> bool:
#         success = True
#         logger.info("pathing to resource first")
#         # Move to resource
#         self.pathfinder.append_path_to_actions(self.unit, path_to_resource)
#
#         # Dig as many times as possible
#         n_digs = int(
#             np.floor(power_remaining_after_moves / self.unit.unit_config.DIG_COST)
#         )
#         if n_digs >= 1:
#             self.unit.action_queue.append(self.unit.dig(n=n_digs))
#         else:
#             logger.error(f"n_digs = {n_digs}, should always be greater than 1")
#             success = False
#
#         # Move to factory
#         path_from_resource_to_factory = self._path_to_factory(
#             from_pos=self.resource_pos,
#         )
#         if len(path_from_resource_to_factory) > 0:
#             self.pathfinder.append_path_to_actions(
#                 self.unit, path_from_resource_to_factory
#             )
#         else:  # No path to factory
#             success = False
#
#         # Transfer resources to factory (only if successful up to now)
#         if success:
#             self.unit.action_queue.append(
#                 self.unit.transfer(
#                     CENTER, self.resource_type, self.unit.unit_config.CARGO_SPACE
#                 )
#             )
#         return success
#
#     def _unit_starting_on_factory(self) -> bool:
#         if (
#             self.factory.factory_loc[self.unit_start_pos[0], self.unit_start_pos[1]]
#             == 1
#         ):
#             return True
#         return False
#
#     def from_factory_route(
#         self,
#     ) -> bool:
#         """Generate actions for mining route assuming starting on factory
#         Note: No use of repeat any more for ease of updating at any time
#         """
#         # Move to outside edge of factory (if necessary)
#         success = self._move_to_edge_of_factory()
#         if not success:
#             logger.error(f'{self.unit.log_prefix} failed to path to edge of factory')
#             return False
#
#         # If already has resources, drop off first
#         self._drop_off_cargo_if_necessary()
#
#         # Calculate travel costs
#         (
#             path_to_resource,
#             cost_to_resource,
#             cost_from_resource_to_factory,
#         ) = self._path_to_and_from_resource()
#         travel_cost = cost_to_resource + cost_from_resource_to_factory
#
#         # Aim for as many digs as possible (either max battery, or current available if near max)
#         # How much power do we have at start of run
#         available_power = self.unit.power_remaining()
#         if available_power < 0:
#             logger.warning(
#                 f"{self.unit.log_prefix}: available_power ({available_power}) negative, setting zero instead"
#             )
#             available_power = 0
#         logger.info(f"available_power = {available_power}")
#
#         # near enough to max power not to waste a turn picking up power (otherwise aim for max capacity)
#         if available_power > 0.85 * self.unit.unit_config.BATTERY_CAPACITY:
#             target_power_at_start = available_power
#         else:
#             target_power_at_start = self.unit.unit_config.BATTERY_CAPACITY
#
#         # How many digs can we do
#         target_digs = int(
#             np.floor(
#                 (target_power_at_start - travel_cost) / self.unit.unit_config.DIG_COST
#             )
#         )
#         # Don't dig more than cargo capacity allows
#         target_digs = int(
#             min(
#                 self.unit.unit_config.CARGO_SPACE
#                 / self.unit.unit_config.DIG_RESOURCE_GAIN,
#                 target_digs,
#             )
#         )
#
#         # Total cost of planned route
#         target_power = travel_cost + target_digs * self.unit.unit_config.DIG_COST
#         logger.info(f"target_power = {target_power}")
#
#         # Assume nothing else picking up power from factory
#         # TODO: Make a factory.power_at_turn() method to keep track of upcoming power requests from factory
#         factory_power = self.factory.power + util.num_turns_of_actions(
#             self.unit.action_queue
#         )
#         logger.info(f"factory_power = {target_power}")
#
#         # Pickup power and update n_digs
#         if factory_power + available_power > target_power:
#             power_to_pickup = target_power - available_power
#             if power_to_pickup > 0:
#                 logger.info(
#                     f"picking up {power_to_pickup} power to achieve target of {target_power}"
#                 )
#                 self.unit.action_queue.append(
#                     self.unit.pickup(
#                         POWER,
#                         min(self.unit.unit_config.BATTERY_CAPACITY, power_to_pickup),
#                     )
#                 )
#             else:
#                 logger.info(f"Enough power already, not picking up")
#             n_digs = target_digs
#         elif (
#             factory_power + available_power - travel_cost
#             > self.unit.unit_config.DIG_COST * 3
#         ):
#             logger.info(f"picking up available power {factory_power}")
#             self.unit.action_queue.append(self.unit.pickup(POWER, factory_power))
#             n_digs = int(
#                 np.floor(
#                     (factory_power + available_power - travel_cost)
#                     / self.unit.unit_config.DIG_COST
#                 )
#             )
#         else:
#             # Not enough energy to do a mining run, return now leave unit on factory
#             logger.warning(
#                 f"{self.factory.unit_id} doesn't have enough energy for {self.unit.unit_id} to do a "
#                 f"mining run to {self.resource_pos} from {self.unit.pos}"
#             )
#             return False
#
#         # Add journey out
#         if len(path_to_resource) > 0:
#             self.pathfinder.append_path_to_actions(self.unit, path_to_resource)
#         else:
#             logger.warning(
#                 f"{self.unit.unit_id} has no path to {self.resource_pos} from {self.unit.pos}"
#             )
#             return False
#
#         # Add digs
#         if n_digs >= 1:
#             self.unit.action_queue.append(self.unit.dig(n=n_digs))
#         else:
#             logger.error(
#                 f"{self.unit.unit_id} n_digs = {n_digs}, unit heading off to not mine anything. should always be greater than 1"
#             )
#             return False
#
#         # Add return journey
#         return_path = self._path_to_factory(
#             self.unit.pos,
#         )
#         if len(return_path) > 0:
#             self.pathfinder.append_path_to_actions(self.unit, return_path)
#         else:
#             logger.warning(
#                 f"{self.unit.unit_id} has no path to {self.factory.unit_id} at {self.factory.factory.pos} "
#                 f"from {self.unit.pos}"
#             )
#             return False
#
#         # Add transfer
#         self.unit.action_queue.append(
#             self.unit.transfer(
#                 CENTER, self.resource_type, self.unit.unit_config.CARGO_SPACE
#             )
#         )
#         return True
#
#     def _move_to_edge_of_factory(self) -> bool:
#         cm = self.pathfinder.generate_costmap(self.unit)
#         path = path_to_factory_edge_nearest_pos(
#             pathfinder=self.pathfinder,
#             factory_loc=self.factory.factory_loc,
#             pos=self.unit.pos,
#             pos_to_be_near=self.resource_pos,
#             costmap=cm,
#             margin=2,
#         )
#         if len(path) == 0:
#             logger.error(
#                 f"Apparently no way to get to the edge of the factory without colliding from {self.unit.pos}",
#             )
#             return False
#         elif len(path) > 0:
#             self.pathfinder.append_path_to_actions(self.unit, path)
#             return True
#
#     def _path_to_factory(self, from_pos: Tuple[int, int]) -> np.ndarray:
#         cm = self.pathfinder.generate_costmap(self.unit)
#         return calc_path_to_factory(
#             self.pathfinder,
#             costmap=cm,
#             pos=from_pos,
#             factory_loc=self.factory.factory_loc,
#             margin=2,
#         )
#
#     def _path_to_resource(
#         self,
#         from_pos: Optional[Tuple[int, int]] = None,
#     ) -> np.ndarray:
#         if from_pos is None:
#             from_pos = self.unit.pos
#         cm = self.pathfinder.generate_costmap(self.unit)
#         return self.pathfinder.fast_path(
#             start_pos=from_pos,
#             end_pos=self.resource_pos,
#             costmap=cm,
#             margin=2,
#         )
#
#     @deprecated(deprecated_in='2.2.3')
#     def _cost_of_actions(self, actions: List[np.ndarray], rubble=None):
#         # TODO: could this use future_rubble? Problem is that rubble may not yet be cleared
#         if rubble is None:
#             rubble = self.rubble
#         return power_cost_of_actions(
#             self.unit.start_of_turn_pos, rubble, self.unit, actions
#         )
#


class BaseRoute:
    def __init__(
        self,
        pathfinder: Pather,
        rubble: np.ndarray,
        resource_pos: Tuple[int, int],
        resource_type: int,
        factory: FriendlyFactoryManager,
        unit: FriendlyUnitManager,
    ):
        self.pathfinder = pathfinder
        self.rubble = rubble
        self.resource_pos = resource_pos
        self.resource_type = resource_type
        self.factory = factory
        self.unit = unit
        self.unit_start_pos = unit.pos
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

    def _drop_off_cargo_if_necessary(self):
        if self.unit.cargo.ice > 0:
            logger.info(f"dropping off {self.unit.cargo.ice} ice before from_factory")
            self.unit.action_queue.append(
                self.unit.transfer(CENTER, ICE, self.unit.unit_config.CARGO_SPACE)
            )
        if self.unit.cargo.ore > 0:
            logger.info(f"dropping off {self.unit.cargo.ore} ore before from_factory")
            self.unit.action_queue.append(
                self.unit.transfer(CENTER, ORE, self.unit.unit_config.CARGO_SPACE)
            )

    def _path_to_factory(self, from_pos: Tuple[int, int]) -> np.ndarray:
        cm = self.pathfinder.generate_costmap(self.unit)
        return calc_path_to_factory(
            self.pathfinder,
            costmap=cm,
            pos=from_pos,
            factory_loc=self.factory.factory_loc,
            margin=2,
        )

    def _path_to_resource(
        self,
        from_pos: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        if from_pos is None:
            from_pos = self.unit.pos
        cm = self.pathfinder.generate_costmap(self.unit)
        return self.pathfinder.fast_path(
            start_pos=from_pos,
            end_pos=self.resource_pos,
            costmap=cm,
            margin=2,
        )

    def _move_to_edge_of_factory(self) -> bool:
        cm = self.pathfinder.generate_costmap(self.unit)
        path = path_to_factory_edge_nearest_pos(
            pathfinder=self.pathfinder,
            factory_loc=self.factory.factory_loc,
            pos=self.unit.pos,
            pos_to_be_near=self.resource_pos,
            costmap=cm,
            margin=2,
        )
        if len(path) == 0:
            logger.error(
                f"Apparently no way to get to the edge of the factory without colliding from {self.unit.pos}",
            )
            return False
        elif len(path) > 0:
            self.pathfinder.append_path_to_actions(self.unit, path)
            return True


class ResourceFirstRoute(BaseRoute):
    def create_route(self, path_to_resource, power_remaining_after_moves) -> bool:
        success = True
        logger.info("pathing to resource first")

        success = self._move_to_resource(path_to_resource)
        if not success:
            return False

        n_digs = self._calculate_digs(power_remaining_after_moves)
        if n_digs is None:
            return False

        success = self._move_to_factory()
        if not success:
            return False

        return self._transfer_resources()

    def _move_to_resource(self, path_to_resource):
        self.pathfinder.append_path_to_actions(self.unit, path_to_resource)
        return True

    def _calculate_digs(self, power_remaining_after_moves):
        n_digs = int(
            np.floor(power_remaining_after_moves / self.unit.unit_config.DIG_COST)
        )
        if n_digs >= 1:
            self.unit.action_queue.append(self.unit.dig(n=n_digs))
        else:
            logger.error(f"n_digs = {n_digs}, should always be greater than 1")
            return None
        return n_digs

    def _move_to_factory(self):
        path_from_resource_to_factory = self._path_to_factory(
            from_pos=self.resource_pos,
        )
        if len(path_from_resource_to_factory) > 0:
            self.pathfinder.append_path_to_actions(
                self.unit, path_from_resource_to_factory
            )
            return True
        else:  # No path to factory
            return False

    def _transfer_resources(self):
        self.unit.action_queue.append(
            self.unit.transfer(
                CENTER, self.resource_type, self.unit.unit_config.CARGO_SPACE
            )
        )
        return True


class FactoryFirstRoute(BaseRoute):
    def create_route(self) -> bool:
        if not self._move_to_edge_of_factory():
            logger.error(f"{self.unit.log_prefix} failed to path to edge of factory")
            return False

        self._drop_off_cargo_if_necessary()

        travel_cost = self._calculate_travel_costs()

        (
            available_power,
            target_power_at_start,
        ) = self._determine_available_and_target_power()

        target_digs = self._calculate_target_digs(target_power_at_start, travel_cost)

        target_power = self._calculate_target_power(travel_cost, target_digs)

        factory_power = self._calculate_factory_power()

        n_digs = self._pickup_power_and_determine_digs(
            available_power, target_power, factory_power, travel_cost, target_digs
        )

        if n_digs is False:
            return False

        if not self._add_journey_out():
            return False

        if not self._add_digs(n_digs):
            return False

        if not self._add_return_journey():
            return False

        self._add_transfer()
        return True

    def _calculate_travel_costs(self):
        (
            path_to_resource,
            cost_to_resource,
            cost_from_resource_to_factory,
        ) = self._path_to_and_from_resource()
        travel_cost = cost_to_resource + cost_from_resource_to_factory
        return travel_cost

    def _determine_available_and_target_power(self):
        available_power = self.unit.power_remaining()
        if available_power < 0:
            logger.warning(
                f"{self.unit.log_prefix}: available_power ({available_power}) negative, setting zero instead"
            )
            available_power = 0
        logger.info(f"available_power = {available_power}")

        if available_power > 0.85 * self.unit.unit_config.BATTERY_CAPACITY:
            target_power_at_start = available_power
        else:
            target_power_at_start = self.unit.unit_config.BATTERY_CAPACITY

        return available_power, target_power_at_start

    def _calculate_target_digs(self, target_power_at_start, travel_cost):
        target_digs = int(
            np.floor(
                (target_power_at_start - travel_cost) / self.unit.unit_config.DIG_COST
            )
        )
        target_digs = int(
            min(
                self.unit.unit_config.CARGO_SPACE
                / self.unit.unit_config.DIG_RESOURCE_GAIN,
                target_digs,
            )
        )
        return target_digs

    def _calculate_target_power(self, travel_cost, target_digs):
        target_power = travel_cost + target_digs * self.unit.unit_config.DIG_COST
        logger.info(f"target_power = {target_power}")
        return target_power

    def _calculate_factory_power(self):
        factory_power = self.factory.power + util.num_turns_of_actions(
            self.unit.action_queue
        )
        logger.info(f"factory_power = {factory_power}")
        return factory_power

    def _pickup_power_and_determine_digs(
        self, available_power, target_power, factory_power, travel_cost, target_digs
    ):
        if factory_power + available_power > target_power:
            power_to_pickup = target_power - available_power
            if power_to_pickup > 0:
                logger.info(
                    f"picking up {power_to_pickup} power to achieve target of {target_power}"
                )
                self.unit.action_queue.append(
                    self.unit.pickup(
                        POWER,
                        min(self.unit.unit_config.BATTERY_CAPACITY, power_to_pickup),
                    )
                )
            else:
                logger.info(f"Enough power already, not picking up")
            n_digs = target_digs
        elif (
            factory_power + available_power - travel_cost
            > self.unit.unit_config.DIG_COST * 3
        ):
            logger.info(f"picking up available power {factory_power}")
            self.unit.action_queue.append(self.unit.pickup(POWER, factory_power))
            n_digs = int(
                np.floor(
                    (factory_power + available_power - travel_cost)
                    / self.unit.unit_config.DIG_COST
                )
            )
        else:
            logger.warning(
                f"{self.factory.unit_id} doesn't have enough energy for {self.unit.unit_id} to do a "
                f"mining run to {self.resource_pos} from {self.unit.pos}"
            )
            return False

        return n_digs

    def _add_journey_out(self, path_to_resource):
        if len(path_to_resource) > 0:
            self.pathfinder.append_path_to_actions(self.unit, path_to_resource)
        else:
            logger.warning(
                f"{self.unit.unit_id} has no path to {self.resource_pos} from {self.unit.pos}"
            )
            return False
        return True

    def _add_digs(self, n_digs):
        if n_digs >= 1:
            self.unit.action_queue.append(self.unit.dig(n=n_digs))
        else:
            logger.error(
                f"{self.unit.unit_id} n_digs = {n_digs}, unit heading off to not mine anything. should always be greater than 1"
            )
            return False
        return True

    def _add_return_journey(self, return_path):
        if len(return_path) > 0:
            self.pathfinder.append_path_to_actions(self.unit, return_path)
        else:
            logger.warning(
                f"{self.unit.unit_id} has no path to {self.factory.unit_id} at {self.factory.factory.pos} "
                f"from {self.unit.pos}"
            )
            return False
        return True

    def _add_transfer(self):
        self.unit.action_queue.append(
            self.unit.transfer(
                CENTER, self.resource_type, self.unit.unit_config.CARGO_SPACE
            )
        )
        return True


class MiningRoutePlanner(BaseRoute):
    move_lookahead = 10
    target_queue_length = 20

    def __init__(
        self,
        pathfinder: Pather,
        rubble: np.ndarray,
        resource_pos: Tuple[int, int],
        resource_type: int,
        factory: FriendlyFactoryManager,
        unit: FriendlyUnitManager,
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
        super().__init__(pathfinder, rubble, resource_pos, resource_type, factory, unit)
        # self.pathfinder = pathfinder
        #
        # # Map related
        # self.rubble = rubble
        # self.resource_pos = resource_pos
        # self.resource_type = resource_type

        # Unit related
        self.unit_start_pos = unit.pos
        self.factory = factory

        # This will be changed during route planning
        self.unit = unit
        self.unit.action_queue = []

    def make_route(self, unit_must_move: bool) -> bool:
        success = True
        if self.unit.on_own_factory() and unit_must_move:
            success = self._move_to_edge_of_factory()
            if not success:
                return False

        if not self.unit.on_own_factory():
            success = self._decide_resource_or_factory_first()
            if not success:
                return False

        if success and len(self.unit.action_queue) < self.target_queue_length:
            success = self._add_factory_first_route()

        return success

    def _decide_resource_or_factory_first(self):
        logger.debug(f"not on factory, deciding resource or factory first")
        (
            path_to_resource,
            cost_to_resource,
            cost_from_resource_to_factory,
        ) = self._path_to_and_from_resource()

        power_remaining = (
            self.unit.power_remaining()
            - cost_to_resource
            - cost_from_resource_to_factory
        )

        if len(path_to_resource) == 0:
            return False

        if power_remaining > max(
            2 * self.unit.unit_config.DIG_COST,
            len(path_to_resource) // 2 * self.unit.unit_config.DIG_COST,
        ):
            resource_first_route = ResourceFirstRoute(
                self.pathfinder,
                self.rubble,
                self.resource_pos,
                self.resource_type,
                self.factory,
                self.unit,
            )
            return resource_first_route.create_route(path_to_resource, power_remaining)
        else:
            logger.info(
                f"pathing to {self.factory.factory.unit_id} at {self.factory.pos} first"
            )
            direct_path_to_factory = self._path_to_factory(from_pos=self.unit.pos)
            if len(direct_path_to_factory) > 0:
                self.pathfinder.append_path_to_actions(
                    self.unit, direct_path_to_factory
                )
                self._drop_off_cargo_if_necessary()
                return True
            else:
                return False

    def _add_factory_first_route(self):
        logger.debug(f"adding from factory actions")
        factory_first_route = FactoryFirstRoute(
            self.pathfinder,
            self.rubble,
            self.resource_pos,
            self.resource_type,
            self.factory,
            self.unit,
        )
        return factory_first_route.create_route()

    def _unit_starting_on_factory(self) -> bool:
        if (
            self.factory.factory_loc[self.unit_start_pos[0], self.unit_start_pos[1]]
            == 1
        ):
            return True
        return False


class MiningRecommender:
    def __init__(self, master, ice, ore):
        self.master = master
        self.ice = ice
        self.ore = ore

    def get_resource_map(self, resource_type: int) -> np.ndarray:
        if resource_type == ICE:
            return self.ice.copy()
        elif resource_type == ORE:
            return self.ore.copy()
        else:
            raise NotImplementedError(
                f"{resource_type} not recognized, should be {ICE} or {ORE}"
            )

    def find_nearest_resource(self, resource_map, unit_factory):
        for attempt in range(5):
            nearest_resource = nearest_non_zero(resource_map, unit_factory.pos)
            if nearest_resource is None:
                logger.warning(
                    f"No nearest resource to {unit_factory.unit_id} after {attempt} attempts"
                )
                return None
            if self.is_resource_accessible(unit_factory, nearest_resource):
                return nearest_resource

            # Blank out that resource and try again
            resource_map[nearest_resource[0], nearest_resource[1]] = 0
        return None

    def is_resource_accessible(self, unit_factory, resource_pos):
        unit_pos = unit_factory.pos
        cm = self.master.pathfinder.generate_costmap(unit_factory)
        path_to_resource = self.master.pathfinder.fast_path(
            unit_pos,
            resource_pos,
            costmap=cm,
        )
        return len(path_to_resource) > 0

    def recommend(
        self,
        unit: FriendlyUnitManager,
        resource_type: int = ICE,
        unit_must_move: bool = False,
    ) -> [None, MiningRecommendation]:
        resource_map = self.get_resource_map(resource_type)

        # If unit must move, make sure not to recommend resource under unit
        if unit_must_move:
            resource_map[unit.pos_slice] = 0

        unit_factory = self.master.factories.friendly.get(unit.factory_id, None)
        if unit_factory is None:
            logger.warning(
                f"Factory doesn't exist for {unit.unit_id} with factory_id {unit.factory_id}"
            )
            return None

        nearest_resource = self.find_nearest_resource(resource_map, unit_factory)
        if nearest_resource is None:
            logger.warning(
                f"No free resources ({resource_type}) for {unit_factory.unit_id} after a few attempts"
            )
            return None

        return MiningRecommendation(
            distance_from_factory=util.manhattan(unit_factory.pos, nearest_resource),
            resource_pos=nearest_resource,
            factory_id=unit_factory.unit_id,
            resource_type=resource_type,
        )


class MiningPlanner(Planner):
    def __init__(self, master_state: MasterState):
        self.master: MasterState = master_state

    def __repr__(self):
        return f"MiningPlanner[step={self.master.step}]"

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

    def recommend(
        self,
        unit: FriendlyUnitManager,
        resource_type: int = ICE,
        unit_must_move: bool = False,
    ) -> [None, MiningRecommendation]:
        recommender = MiningRecommender(self.master, self.ice, self.ore)
        return recommender.recommend(
            unit, resource_type=resource_type, unit_must_move=unit_must_move
        )

    def carry_out(
        self,
        unit_manager: FriendlyUnitManager,
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
