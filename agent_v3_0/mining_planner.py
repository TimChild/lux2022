from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
from deprecation import deprecated

from config import get_logger
from new_path_finder import Pather
from master_state import MasterState, Planner
from actions_util import Recommendation
from base_planners import BaseGeneralPlanner, BaseUnitPlanner
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

from unit_status import MineValues, MineActSubCategory, ActStatus

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

    def __init__(self, distance_from_factory: float, resource_pos, factory_id, resource_type: int):
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
        heavy_ignore_light_at_resource: bool,
    ):
        self.pathfinder = pathfinder
        self.rubble = rubble
        self.resource_pos = resource_pos
        self.resource_type = resource_type
        self.factory = factory
        self.unit = unit
        self.heavy_ignore_light_at_resource = heavy_ignore_light_at_resource

    def _path_to_and_from_resource(self):
        path_to_resource = self._path_to_resource()
        cost_to_resource = power_cost_of_path(path_to_resource, self.rubble, self.unit.unit_type)

        path_from_resource_to_factory = self._path_to_factory(from_pos=self.resource_pos)
        cost_from_resource_to_factory = power_cost_of_path(
            path_from_resource_to_factory, self.rubble, self.unit.unit_type
        )
        return path_to_resource, cost_to_resource, cost_from_resource_to_factory

    def _drop_off_cargo_if_necessary(self):
        if self.unit.cargo.ice > 0:
            logger.debug(f"dropping off {self.unit.cargo.ice} ice before from_factory")
            self.unit.action_queue.append(self.unit.transfer(CENTER, ICE, self.unit.unit_config.CARGO_SPACE))
        if self.unit.cargo.ore > 0:
            logger.debug(f"dropping off {self.unit.cargo.ore} ore before from_factory")
            self.unit.action_queue.append(self.unit.transfer(CENTER, ORE, self.unit.unit_config.CARGO_SPACE))

    def _path_to_factory(self, from_pos: Tuple[int, int]) -> np.ndarray:
        cm = self.pathfinder.generate_costmap(self.unit, friendly_light=True)
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
        # Allow heavy to path over lights when heading to resource (they will probably get out of the way unless they are unlucky)
        include_light = False if self.heavy_ignore_light_at_resource and self.unit.unit_type == "HEAVY" else True
        cm = self.pathfinder.generate_costmap(self.unit, friendly_light=include_light)
        return self.pathfinder.fast_path(
            start_pos=from_pos,
            end_pos=self.resource_pos,
            costmap=cm,
            margin=2,
        )

    def _move_to_edge_of_factory(self, must_move=False) -> bool:
        cm = self.pathfinder.generate_costmap(self.unit, friendly_light=True)
        factory_loc = self.factory.factory_loc.copy()
        if must_move:
            factory_loc[self.unit.pos_slice] = 0
        path = path_to_factory_edge_nearest_pos(
            pathfinder=self.pathfinder,
            factory_loc=factory_loc,
            pos=self.unit.pos,
            pos_to_be_near=self.resource_pos,
            costmap=cm,
            margin=2,
        )
        if len(path) == 0:
            logger.warning(
                f"{self.unit.log_prefix} failed to find path to edge of factory with normal pathing, now trying avoiding direct collisions only"
            )
            cm = self.pathfinder.generate_costmap(self.unit, friendly_light=True, collision_only=True)
            # if self.unit.unit_id == 'unit_39':
            #     util.show_map_array(cm).show()
            path = path_to_factory_edge_nearest_pos(
                pathfinder=self.pathfinder,
                factory_loc=factory_loc,
                pos=self.unit.pos,
                pos_to_be_near=self.resource_pos,
                costmap=cm,
                margin=2,
            )
            if len(path) == 0:
                logger.error(
                    f"{self.unit.log_prefix} Apparently no way to get to the edge of {self.factory.unit_id}  at {self.factory.pos} without colliding from {self.unit.pos}",
                )
                return False
            else:
                logger.warning(f"succeeded in finding path avoiding collisions only")
                self.pathfinder.append_path_to_actions(self.unit, path)
                return True
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
        n_digs = int(np.floor(power_remaining_after_moves / self.unit.unit_config.DIG_COST))
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
            self.pathfinder.append_path_to_actions(self.unit, path_from_resource_to_factory)
            return True
        else:  # No path to factory
            return False

    def _transfer_resources(self):
        self.unit.action_queue.append(self.unit.transfer(CENTER, self.resource_type, self.unit.unit_config.CARGO_SPACE))
        return True


class FactoryFirstRoute(BaseRoute):
    def create_route(self) -> bool:
        # if not self._move_to_edge_of_factory():
        #     logger.error(f"{self.unit.log_prefix} failed to path to edge of factory")
        #     return False
        self._drop_off_cargo_if_necessary()

        travel_cost = self._calculate_travel_costs()

        (
            available_power,
            target_power_at_start,
        ) = self._determine_available_and_target_power()

        target_digs = self._calculate_target_digs(target_power_at_start, travel_cost)

        target_power = self._calculate_target_power(travel_cost, target_digs)

        # factory_power = self._calculate_factory_power()
        factory_power = (
            self.factory.power
            if util.num_turns_of_actions(self.unit.action_queue) < 3
            else self.factory.short_term_power
        )
        # factory_power = self.factory.power

        n_digs = self._pickup_power_and_determine_digs(
            available_power, target_power, factory_power, travel_cost, target_digs
        )
        if n_digs <= 0 or n_digs is False:
            logger.warning(
                f"{self.unit.log_prefix} n_digs = {n_digs}, likely not enough power (factory_power={factory_power}, target_power={target_power}, available_power={available_power})"
            )
            return False

        if not self._add_journey_out(self._path_to_resource()):
            logger.warning(
                f"{self.unit.log_prefix} failed to path from {self.factory.unit_id} to resource at {self.resource_pos}"
            )
            return False

        if not self._add_digs(n_digs):
            logger.warning(f"{self.unit.log_prefix} failed to add digs at {self.resource_pos}")
            return False

        if not self._add_return_journey(self._path_to_factory(self.resource_pos)):
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
        target_digs = int(np.floor((target_power_at_start - travel_cost) / self.unit.unit_config.DIG_COST))
        target_digs = int(
            min(
                self.unit.unit_config.CARGO_SPACE / self.unit.unit_config.DIG_RESOURCE_GAIN,
                target_digs,
            )
        )
        return target_digs

    def _calculate_target_power(self, travel_cost, target_digs):
        target_power = travel_cost + target_digs * self.unit.unit_config.DIG_COST
        logger.info(f"target_power = {target_power}")
        return target_power

    # def _calculate_factory_power(self):
    #     factory_power = self.factory.power + util.num_turns_of_actions(
    #         self.unit.action_queue
    #     )
    #     logger.info(f"factory_power = {factory_power}")
    #     return factory_power

    def _pickup_power_and_determine_digs(self, available_power, target_power, factory_power, travel_cost, target_digs):
        if factory_power + available_power > target_power:
            power_to_pickup = target_power - available_power
            if power_to_pickup > 0:
                logger.info(f"picking up {power_to_pickup} power to achieve target of {target_power}")

                self.unit.action_queue.append(
                    self.unit.pickup(
                        POWER,
                        min(self.unit.unit_config.BATTERY_CAPACITY, power_to_pickup),
                    )
                )
            else:
                logger.debug(f"Enough power already, not picking up")
            n_digs = target_digs
        elif factory_power + available_power - travel_cost > self.unit.unit_config.DIG_COST * 3:
            n_digs = int(np.floor((factory_power + available_power - travel_cost) / self.unit.unit_config.DIG_COST))
            if n_digs > 0:
                logger.info(f"picking up available power {factory_power}")
                self.unit.action_queue.append(self.unit.pickup(POWER, factory_power))
            else:
                logger.warning(
                    f"{self.unit.log_prefix} not enough available power at {self.factory.unit_id} ({factory_power}) for mining run to {self.resource_pos}"
                )
                return False

        else:
            logger.warning(
                f"{self.factory.unit_id} doesn't have enough power ({factory_power}) for {self.unit.unit_id} to do a "
                f"mining run to {self.resource_pos} from {self.unit.pos}"
            )
            return False

        return n_digs

    def _add_journey_out(self, path_to_resource):
        if len(path_to_resource) > 0:
            self.pathfinder.append_path_to_actions(self.unit, path_to_resource)
        else:
            logger.warning(f"{self.unit.unit_id} has no path to {self.resource_pos} from {self.unit.pos}")
            return False
        return True

    def _add_digs(self, n_digs):
        if n_digs >= 1:
            self.unit.action_queue.append(self.unit.dig(n=n_digs))
        else:
            logger.error(f"{self.unit.log_prefix} n_digs = {n_digs} should always be greater than 1")
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
        self.unit.action_queue.append(self.unit.transfer(CENTER, self.resource_type, self.unit.unit_config.CARGO_SPACE))
        return True


class MiningRoutePlanner(BaseRoute):
    move_lookahead = 50
    target_queue_length = 20

    def __init__(
        self,
        pathfinder: Pather,
        rubble: np.ndarray,
        resource_pos: Tuple[int, int],
        resource_type: int,
        factory: FriendlyFactoryManager,
        unit: FriendlyUnitManager,
        heavy_ignore_light_at_resource: bool = True,
    ):
        """
        Args:
            pathfinder: Agents pathfinding instance
            rubble: full rubble map
            resource_pos: position of resource to mine
            resource_type: i.e. ICE, ORE, etc
            factory: factory being mined for
            unit: unit to make route for
            heavy_ignore_light_at_resource: If True, heavy units will ignore light units at resources
        """
        super().__init__(
            pathfinder,
            rubble,
            resource_pos,
            resource_type,
            factory,
            unit,
            heavy_ignore_light_at_resource,
        )

        # Unit related
        self.unit_start_pos = unit.pos
        self.factory = factory

        # This will be changed during route planning
        self.unit = unit

    def make_route(self, unit_must_move: bool) -> bool:
        # # If more than a step away, ignore lights (they can get out of the way)
        # if util.manhattan(self.resource_pos, self.unit.pos) > 1:
        #     heavy_ignore_light = True
        # else:
        #     heavy_ignore_light = False

        success = True
        if self.unit.on_own_factory() and unit_must_move:
            logger.info(f"on factory and must move first")
            success = self._move_to_edge_of_factory(must_move=True)
            if not success:
                return False

        if not self.unit.on_own_factory():
            logger.info(f"not on factory, deciding resource or factory first")
            success = self._decide_resource_or_factory_first()
            if not success:
                return False
        else:
            logger.info(f"pathing from factory")
            success = self._add_factory_first_route()
        return success

    def _decide_resource_or_factory_first(self):
        (
            path_to_resource,
            cost_to_resource,
            cost_from_resource_to_factory,
        ) = self._path_to_and_from_resource()

        power_remaining = self.unit.power_remaining() - cost_to_resource - cost_from_resource_to_factory

        if len(path_to_resource) == 0:
            logger.warning(f"{self.unit.log_prefix} failed to path to resource at {self.resource_pos}")
            return False

        if power_remaining > max(
            2 * self.unit.unit_config.DIG_COST,
            len(path_to_resource) // 2 * self.unit.unit_config.DIG_COST,
        ):
            logger.info(f"pathing to resource at {self.resource_pos} first")
            resource_first_route = ResourceFirstRoute(
                self.pathfinder,
                self.rubble,
                self.resource_pos,
                self.resource_type,
                self.factory,
                self.unit,
                heavy_ignore_light_at_resource=self.heavy_ignore_light_at_resource,
            )
            return resource_first_route.create_route(path_to_resource, power_remaining)
        else:
            logger.info(f"pathing to {self.factory.factory.unit_id} at {self.factory.pos} first")
            direct_path_to_factory = self._path_to_factory(from_pos=self.unit.pos)
            if len(direct_path_to_factory) > 0:
                self.pathfinder.append_path_to_actions(self.unit, direct_path_to_factory)
                self._drop_off_cargo_if_necessary()
                return True
            else:
                logger.warning(f"failed to path to {self.factory.unit_id} from {self.unit.pos} to {self.factory.pos}")
                return False

    def _add_factory_first_route(self):
        factory_first_route = FactoryFirstRoute(
            self.pathfinder,
            self.rubble,
            self.resource_pos,
            self.resource_type,
            self.factory,
            self.unit,
            self.heavy_ignore_light_at_resource,
        )
        return factory_first_route.create_route()

    def _unit_starting_on_factory(self) -> bool:
        if self.factory.factory_loc[self.unit_start_pos[0], self.unit_start_pos[1]] == 1:
            return True
        return False


class MiningRecommender:
    def __init__(self, master, ice, ore):
        self.master = master
        self.ice = ice
        self.ore = ore
        # self.assigned_resources   # TODO: Do this tomrrow (and figure out why units are still colliding)

    def get_resource_map(self, resource_type: int) -> np.ndarray:
        if resource_type == ICE:
            return self.ice.copy()
        elif resource_type == ORE:
            return self.ore.copy()
        else:
            raise NotImplementedError(f"{resource_type} not recognized, should be {ICE} or {ORE}")

    def find_nearest_resource(self, unit: FriendlyUnitManager, resource_map: np.ndarray):
        # Generate costmap that allows heavy to ignore light (i.e. should force a light off a resource)
        include_light = False if unit.unit_type == "HEAVY" else True
        cm = self.master.pathfinder.generate_costmap(unit, friendly_light=include_light, enemy_light=include_light)

        # Try 10 nearest resources (nearest to unit, will stop at factory for new actions)
        for attempt in range(10):
            nearest_resource = nearest_non_zero(resource_map, unit.pos)
            if nearest_resource is None:
                logger.warning(f"No nearest resource to {unit.unit_id} at {unit.pos} after {attempt} attempts")
                return None
            if self.is_resource_accessible(unit.pos, nearest_resource, cm):
                return nearest_resource

            # Blank out that resource and try again
            resource_map[nearest_resource[0], nearest_resource[1]] = 0
        return None

    def is_resource_accessible(self, unit_pos: util.POS_TYPE, resource_pos: util.POS_TYPE, costmap: np.ndarray) -> bool:
        # Just check if there exists a path to it according to costmap
        # print(
        #     f"checking {resource_pos},  cm there is {costmap[resource_pos[0], resource_pos[1]]}"
        # )
        path_to_resource = self.master.pathfinder.fast_path(
            unit_pos,
            resource_pos,
            costmap=costmap,
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
            resource_map[unit.start_of_turn_pos[0], unit.start_of_turn_pos[1]] = 0

        unit_factory = self.master.factories.friendly.get(unit.factory_id, None)
        if unit_factory is None:
            logger.error(f"Factory doesn't exist for {unit.unit_id} with factory_id {unit.factory_id}")
            return None

        nearest_resource = self.find_nearest_resource(unit, resource_map)
        if nearest_resource is None:
            logger.warning(f"No free resources ({resource_type}) for {unit.unit_id} after a many attempts")
            return None

        return MiningRecommendation(
            distance_from_factory=util.manhattan(unit_factory.pos, nearest_resource),
            resource_pos=nearest_resource,
            factory_id=unit_factory.unit_id,
            resource_type=resource_type,
        )


class MiningUnitPlanner(BaseUnitPlanner):
    def  __init__(self, master: MasterState, general_planner: MiningPlanner, unit: FriendlyUnitManager):
        super().__init__(master, general_planner, unit)
        self.resource_type = None
        self._action_flag = None

    def _get_best_resource(self):
        """
        - Get list of resources from self.planner
        - Check each in list to determine best value (nearest, unoccupied etc)
        - Return best or None
        """
        rec = self.recommend(self.unit, resource_type, unit_must_move=self.unit.status.turn_status.must_move)
        # success = self.carry_out(self.unit, rec, self.unit.status.turn_status.must_move)
        # self.unit.status.planned_action_queue = self.unit.action_queue.copy()
        # return success
        # pass

    def _check_and_handle_action_flags(self,  status=None):
        """
        If current_action changed from MINE and not ATTACK_TEMPORARY, set plan_step = 1
        elif status stop_here_for_now do nothing
        elif status attack for a few turns, do nothing
        ...
        return
        """
        pass

    def update_planned_actions(self):
        # print(self.unit.status.current_action.sub_category)
        if self.unit.status.current_action.sub_category == MineActSubCategory.ICE:
            self.resource_type = util.ICE
        elif self.unit.status.current_action.sub_category == MineActSubCategory.ORE:
            self.resource_type = util.ORE
        else:
            raise ValueError(f"{self.unit.status.current_action} not correct for Mining")


        # 1. Find the nearest available resource
        if self.unit.status.mine_values.plan_step == 1:
            resource = self._get_best_resource()
            if resource is None:
                logger.warning(f'{self.unit.log_prefix} failed to find available resource}')
                return
            self.unit.status.current_action.step = 2

        # 2. Get at least a min amount of power from factory
        if self.unit.status.current_action.step == 2:
            if self.unit.power < min_power:
                status = self.unit.action_handler.add_pickup(allow_partial=True)
                if self._check_and_handle_action_flags(status):
                    return
            if self.unit.power < min_power:
                # remove pickup
                return do nothing for now
            self.unit.status.current_action.step = 3

        # 3. Path to resource
        if self.unit.status.current_action.step == 3:
            status = self.action_handler.add_path(self.unit, resource)
            if self._check_and_handle_action_flags(status):
                return
            self.unit.status.current_action.step = 4

        # 4. Add dig actions
        if self.unit.status.current_action.step == 4:
            available_power = self.unit.power_remaining()
            power_to_facory = self._calculate_power_to_factory()
            n_digs = (available_power - power_to_facory)//self.unit.unit_config.DIG_COST
            if n_digs > 0:
                status = self.action_handler.add_dig(self.unit, n_digs=n_digs)
            else:
                self.unit.status.update_action_status(ActStatus(ActCategory.WAITING))
                self.unit.status.current_action.step = 1
                return
            if self._check_and_handle_action_flags():
                return
            self.unit.status.current_action.step = 5

        # 5. Dropoff at factory
        if self.unit.status.current_action.step == 5:
            self.unit.status.update_action_status(ActStatus(ActCategory.DROPOFF))
            self.unit.status.current_action.step = 1
            self.unit.status.turn_status.replan_required = True
            return

        logger.error(f'{self.unit.log_prefix} somehow plan_step not valid {self.unit.status.current_action.step}')
        self.unit.status.current_action.step = 1
        self.unit.status.update_action_status(ActStatus(category=ActCategory.DROPOFF))
        return


    def recommend(
        self,
        unit: FriendlyUnitManager,
        resource_type: int = ICE,
        unit_must_move: bool = False,
    ) -> [None, MiningRecommendation]:
        recommender = MiningRecommender(self.master, self.planner.ice, self.planner.ore)
        return recommender.recommend(unit, resource_type=resource_type, unit_must_move=unit_must_move)

    def carry_out(
        self,
        unit: FriendlyUnitManager,
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
            unit=unit,
        )
        self.planner.store_planners[unit.unit_id] = route_planner
        success = route_planner.make_route(unit_must_move=unit_must_move)
        return success


class MiningPlanner(BaseGeneralPlanner):
    def __init__(self, master: MasterState):
        super().__init__(master)

        # Make it easier to see what was going on for debugging
        self.store_planners = {}

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
        self.store_planners = {}

    # New
    def get_unit_planner(self, unit: FriendlyUnitManager) -> MiningUnitPlanner:
        """Return a subclass of BaseUnitPlanner to actually update or create new actions for a single Unit"""
        if unit.unit_id not in self.unit_planners:
            unit_planner = MiningUnitPlanner(self.master, self, unit)
            self.unit_planners[unit.unit_id] = unit_planner
        return self.unit_planners[unit.unit_id]
