from __future__ import annotations
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
import logging
from typing import TYPE_CHECKING, Tuple, Optional, List
from dataclasses import dataclass

import numpy as np
from luxai_s2.unit import UnitType

from master_state import MasterState, Planner
from actions import Recommendation
from path_finder import CollisionParams, PathFinder
from util import (
    ICE,
    ORE,
    nearest_non_zero,
    power_cost_of_actions,
    path_to_actions,
    HEAVY_UNIT,
    LIGHT_UNIT,
    ACT_REPEAT,
    ACT_N,
    POWER,
    CENTER,
    power_cost_of_path,
    calc_path_to_factory,
    path_to_factory_edge_nearest_pos,
    num_turns_of_actions,
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

    def __init__(self, value: float, resource_pos, factory_id, resource_type: int):
        self.value = value
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
        pathfinder: PathFinder,
        rubble: np.ndarray,
        resource_pos: Tuple[int, int],
        resource_type: int,
        factory: FriendlyFactoryManager,
        unit_pos: Tuple[int, int],
        unit_id: str,
        unit_power: int,
        unit_type: str,
    ):
        """
        Args:
            pathfinder: Agents pathfinding instance
            rubble: full rubble map
            resource_pos: position of resource to mine
            resource_type: i.e. ICE, ORE, etc
            factory: factory being mined for
            unit_pos:
            unit_id:
            unit_power:
            unit_type:
        """
        assert unit_type in ['LIGHT', 'HEAVY']
        self.pathfinder = pathfinder

        # Map related
        self.rubble = rubble
        self.resource_pos = resource_pos
        self.resource_type = resource_type
        # Unit related
        self.unit_start_pos = unit_pos
        self.factory = factory

        # These will be changed during route planning
        self.unit = LIGHT_UNIT if unit_type == 'LIGHT' else HEAVY_UNIT
        self.unit.pos = unit_pos
        self.unit.power = unit_power
        self.unit.unit_id = unit_id
        self.unit.action_queue = []

    def log(self, message, level=logging.INFO):
        return logging.log(level=level, msg=f'MiningRoutePlanner: {message}')

    def _path_to_and_from_resource(self):
        path_to_resource = self._path_to_resource(
            collision_params=CollisionParams(
                look_ahead_turns=self.move_lookahead,
                ignore_ids=(self.unit.unit_id,),
                enemy_light=True if self.unit.unit_type == 'LIGHT' else False,
                starting_step=num_turns_of_actions(self.unit.action_queue),
            )
        )
        cost_to_resource = power_cost_of_path(
            path_to_resource, self.rubble, self.unit.unit_type
        )

        path_from_resource_to_factory = self._path_to_factory(
            self.resource_pos, collision_params=None
        )
        cost_from_resource_to_factory = power_cost_of_path(
            path_from_resource_to_factory, self.rubble, self.unit.unit_type
        )
        return path_to_resource, cost_to_resource, cost_from_resource_to_factory

    def make_route(self):
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

            # Decide which to do
            if power_remaining > 3 * self.unit.unit_cfg.DIG_COST:
                # Go to resource first
                self._resource_then_factory(path_to_resource, power_remaining)
            else:
                # Go to factory first
                self.log('pathing to factory first')
                direct_path_to_factory = self._path_to_factory(
                    self.resource_pos,
                    collision_params=CollisionParams(
                        look_ahead_turns=self.move_lookahead,
                        ignore_ids=(self.unit.unit_id,),
                        enemy_light=True if self.unit.unit_type == 'LIGHT' else False,
                        starting_step=0,
                    ),
                )
                self.unit.action_queue.extend(path_to_actions(direct_path_to_factory))
                self.unit.pos = direct_path_to_factory[-1]

        if len(self.unit.action_queue) < self.target_queue_length:
            # Then loop from factory
            self._from_factory_actions()
        return self.unit.action_queue[: self.target_queue_length]

    def _resource_then_factory(self, path_to_resource, power_remaining_after_moves):
        self.log('pathing to resource first')
        # Move to resource
        self.unit.action_queue.extend(path_to_actions(path_to_resource))
        self.unit.pos = path_to_resource[-1]

        # Dig as many times as possible
        n_digs = int(
            np.floor(power_remaining_after_moves / self.unit.unit_cfg.DIG_COST)
        )
        if n_digs >= 1:
            self.unit.action_queue.append(self.unit.dig(n=n_digs))
        else:
            self.log(f'n_digs = {n_digs}, should always be greater than 1')

        # Move to factory
        path_from_resource_to_factory = self._path_to_factory(
            self.resource_pos,
            collision_params=CollisionParams(
                look_ahead_turns=3,
                ignore_ids=(self.unit.unit_id,),
                enemy_light=True if self.unit.unit_type == 'LIGHT' else False,
                starting_step=num_turns_of_actions(self.unit.action_queue),
            ),
        )
        self.unit.action_queue.extend(path_to_actions(path_from_resource_to_factory))
        self.unit.pos = path_from_resource_to_factory[-1]

        # Transfer resources to factory
        self.unit.action_queue.append(
            self.unit.transfer(CENTER, self.resource_type, self.unit.unit_cfg.CARGO_SPACE)
        )

        # Set unit power zero (should be used up by here)
        # self.unit.power = 0

    def _unit_starting_on_factory(self) -> bool:
        if (
            self.factory.factory_loc[self.unit_start_pos[0], self.unit_start_pos[1]]
            == 1
        ):
            return True
        return False

    def _from_factory_actions(
        self,
    ) -> None:
        """Generate actions for mining route assuming starting on factory
        Note: No use of repeat any more for ease of updating at any time
        """
        # Move to outside edge of factory (if necessary)
        self._move_to_edge_of_factory()

        # Calculate travel costs
        (
            path_to_resource,
            cost_to_resource,
            cost_from_resource_to_factory,
        ) = self._path_to_and_from_resource()
        travel_cost = cost_to_resource + cost_from_resource_to_factory

        # Aim for either 10 digs, or 5x travel cost whichever is larger
        available_power = self.unit.power - power_cost_of_actions(
            self.rubble, self.unit, self.unit.action_queue
        )
        target_digs = int(max(10, 5 * travel_cost / self.unit.unit_cfg.DIG_COST))
        # Make sure not exceeding batter capacity
        target_digs = int(min(target_digs, (self.unit.unit_cfg.BATTERY_CAPACITY-travel_cost)/self.unit.unit_cfg.DIG_COST))
        target_power = travel_cost + target_digs * self.unit.unit_cfg.DIG_COST

        # If action queue is short, assume factory won't have more energy, if long, assume it will
        if len(self.unit.action_queue) < 10:
            factory_power = self.factory.factory.power
        else:
            # Assume enough to do anything (will be checking if this is true before unit tries to do it)
            factory_power = 3000

        # Pickup power and update n_digs
        if factory_power > target_power - available_power:
            self.log(f'picking up desired power to achieve target of {target_power}')
            power_to_pickup = target_power - available_power
            if power_to_pickup > 0:
                self.unit.action_queue.append(
                    self.unit.pickup(POWER, power_to_pickup)
                )
            n_digs = target_digs
        elif factory_power + available_power > self.unit.unit_cfg.DIG_COST * 3:
            self.log(f'picking up available power {factory_power}')
            self.unit.action_queue.append(self.unit.pickup(POWER, factory_power))
            n_digs = int(
                np.floor(
                    (factory_power + available_power - travel_cost)
                    / self.unit.unit_cfg.DIG_COST
                )
            )
        else:
            # Not enough energy to do a mining run, return now (everything up to this point is still useful to do)
            return None

        # Add journey out
        self.unit.action_queue.extend(path_to_actions(path_to_resource))
        self.unit.pos = path_to_resource[-1]

        # Add digs
        if n_digs >= 1:
            self.unit.action_queue.append(self.unit.dig(n=n_digs))
        else:
            self.log(f'n_digs = {n_digs}, should always be greater than 1')

        # Add return journey
        return_path = self._path_to_factory(
            self.unit.pos,
            collision_params=CollisionParams(
                look_ahead_turns=self.move_lookahead,
                ignore_ids=(self.unit.unit_id,),
                enemy_light=True if self.unit.unit_type == 'LIGHT' else False,
                starting_step=num_turns_of_actions(self.unit.action_queue),
            ),
        )
        self.unit.action_queue.extend(path_to_actions(return_path))
        self.unit.pos = return_path[-1]

        # Add transfer
        self.unit.action_queue.append(
            self.unit.transfer(CENTER, self.resource_type, self.unit.unit_cfg.CARGO_SPACE)
        )

        # Assume power been used up by now for any further calculations
        # self.unit.power = 0

        # If action queue is long, return, otherwise repeat
        if len(self.unit.action_queue) >= self.target_queue_length:
            return None
        else:
            return self._from_factory_actions()

    def _move_to_edge_of_factory(self):
        path = path_to_factory_edge_nearest_pos(
            self.pathfinder,
            self.factory.factory_loc,
            CollisionParams(
                look_ahead_turns=self.move_lookahead,
                ignore_ids=(self.unit.unit_id,),
                starting_step=num_turns_of_actions(self.unit.action_queue),
            ),
            self.unit.pos,
            self.resource_pos,
            self.rubble,
            margin=2,
            max_delay_by_move_center=5,
        )
        if path is None:
            self.log(
                'Apparently no way to get to the edge of the factory without colliding',
                level=logging.ERROR,
            )
            raise RuntimeError(
                'Apparently no way to get to the edge of the factory without colliding'
            )
        elif len(path) > 0:
            self.unit.action_queue.extend(path_to_actions(path))
            self.unit.pos = path[-1]

    def _path_to_factory(
        self, from_pos: Tuple[int, int], collision_params: CollisionParams = None
    ) -> np.ndarray:
        return calc_path_to_factory(
            self.pathfinder,
            from_pos,
            self.factory.factory_loc,
            rubble=self.rubble,
            margin=2,
            collision_params=collision_params,
        )

    def _path_to_resource(
        self,
        from_pos: Optional[Tuple[int, int]] = None,
        collision_params: CollisionParams = None,
    ) -> np.ndarray:
        if from_pos is None:
            from_pos = self.unit.pos
        return self.pathfinder.path_fast(
            start=from_pos,
            end=self.resource_pos,
            rubble=self.rubble,
            margin=2,
            collision_params=collision_params,
        )

    def _cost_of_actions(self, actions: List[np.ndarray], rubble=None):
        # TODO: could this use future_rubble? Problem is that rubble may not yet be cleared
        if rubble is None:
            rubble = self.rubble
        return power_cost_of_actions(rubble, self.unit, actions)


class MiningPlanner(Planner):
    def __init__(self, master_state: MasterState):
        self.master: MasterState = master_state
        # self.mining_routes: dict = {}
        self._mining_routes = None

    @property
    def mining_routes(self) -> dict:
        if self._mining_routes is None:
            self._mining_routes = self._generate_routes()
        return self._mining_routes

    @property
    def ice(self):
        return self.master.maps.ice

    @property
    def ore(self):
        return self.master.maps.ore

    @property
    def friendly_factories(self):
        """Map of friendly factories (where -1 is non-factory, otherwise factory id)"""
        return self.master.maps.factory_maps.by_player[self.master.player]

    def update(self):
        # For now, only calculate once on first call of .mining_routes
        # TODO: clear _mining_routes periodically to get updated paths (i.e. for rubble cleared or added)
        # self.mining_routes = self._generate_routes()
        pass

    def recommend(
        self, unit_manager: FriendlyUnitManger
    ) -> [None, MiningRecommendation]:
        """
        Make recommendations for this unit to mine resources (i.e. get values of nearby routes)

        Information required to carry out this recommendation should be stored in the recommendation
        Then to carry out the recommendation, the .carry_out(...) method will be called in the same turn

        Args:
            unit_manager: Unit to make recommendation for
            # resource_type: which resource to look for ('ice', 'ore')
        """
        pos = unit_manager.unit.pos
        # TODO: For ICE and ORE
        best_rec = MiningRecommendation(
            value=-999, resource_pos=(-1, -1), factory_id='', resource_type=ICE
        )
        for resource in [ICE]:  # , ORE]:
            # for factory in self.master_plan.factory_managers.values():
            #     routes = self.mining_routes[unit_manager.unit.unit_type][resource][
            #         factory.unit_id
            #     ]

            friendly_factory_positions = self.friendly_factories.copy()
            # make non-factories == 0, (and factory_id = 0 becomes 1)
            friendly_factory_positions += 1
            nearest_factory = nearest_non_zero(friendly_factory_positions, pos)
            if nearest_factory is None:
                unit_manager.log(f'No nearest factory', level=logging.WARNING)
                continue
            nearest_factory_id_num = self.friendly_factories[
                nearest_factory[0], nearest_factory[1]
            ]
            nearest_factory_id = f'factory_{nearest_factory_id_num}'
            routes = self.mining_routes[unit_manager.unit.unit_type][resource][
                nearest_factory_id
            ]

            # Only recommend unoccupied resource tiles
            for i, path in enumerate(routes.paths):
                next_unoccupied_path = None
                # resource_pos = tuple(path[-1])
                if (
                    # resource_pos not in self.master.allocations.resource_allocation
                    # or resource_pos in self.master.allocations.resource_allocation
                    # and not self.master.allocations.resource_allocation[resource_pos].used_by
                    True
                ):
                    next_unoccupied_path = path
                    next_unoccupied_value = routes.values[i]
                    break
            else:
                self.log(f'No unoccupied routes left for {nearest_factory_id}')
                return MiningRecommendation(
                    value=-100, resource_pos=(-1, -1), factory_id='', resource_type=ICE
                )

            # TODO: Might be able to path straight to resource if enough energy
            # path_to_factory = self.master_plan.pathfinder.path(pos, nearest_factory)
            # cost_to_factory = power_cost_of_actions(
            #     self.master_plan.game_state,
            #     unit_manager.unit,
            #     path_to_actions(unit_manager, path_to_factory),
            # )
            # TODO: Take into account occupied routes, also cost of reaching route?

            rec = MiningRecommendation(
                value=next_unoccupied_value,
                resource_pos=next_unoccupied_path[-1]
                if next_unoccupied_path is not None and len(next_unoccupied_path) > 0
                else None,
                factory_id=nearest_factory_id,
                resource_type=resource,
            )
            if rec.value > best_rec.value:
                best_rec = rec
        return best_rec

    def carry_out(
        self, unit_manager: FriendlyUnitManger, recommendation: MiningRecommendation
    ) -> List[np.ndarray]:
        factory = self.master.factories.friendly[recommendation.factory_id]
        route_planner = MiningRoutePlanner(
            self.master.pathfinder,
            self.master.maps.rubble,
            recommendation.resource_pos,
            recommendation.resource_type,
            factory=factory,
            unit_pos=unit_manager.pos,
            unit_id=unit_manager.unit_id,
            unit_power=unit_manager.unit.power,
            unit_type=unit_manager.unit.unit_type,
        )
        actions = route_planner.make_route()
        return actions[:20]

        # # TODO: Check going to resource first (for now always going to factory first
        # dig_repeats = 10
        #
        # unit_pos = unit_manager.unit.pos
        # unit_type = unit_manager.unit.unit_type
        # unit_id = unit_manager.unit_id
        #
        # pathfinder = self.master.pathfinder
        # resource_pos = recommendation.resource_pos
        #
        # # Figure out which factory tile to use
        # factory_id = recommendation.factory_id
        # if not factory_id:
        #     self.log(
        #         f'No factory_id in recommendation for unit {unit_manager.unit_id}',
        #         level=logging.ERROR,
        #     )
        #     return None
        # factory_num = int(factory_id[-1])
        # factory_map = np.array(
        #     self.master.maps.factory_maps.by_player[self.master.player] == factory_num,
        #     dtype=int,
        # )
        # center_coord = self.master.game_state.factories[self.master.player][
        #     factory_id
        # ].pos
        # factory_map[
        #     center_coord[0], center_coord[1]
        # ] = 0  # Don't deliver to center of factory
        # while True:
        #     factory_pos = nearest_non_zero(factory_map, resource_pos)
        #     if factory_pos is None:
        #         self.log(
        #             f'No unallocated spaces at {factory_id}', level=logging.WARNING
        #         )
        #         factory_pos = center_coord
        #         break
        #     elif (
        #         # factory_id in self.master.allocations.factory_tiles
        #         # and factory_pos in self.master.allocations.factory_tiles[factory_id]
        #         # and self.master.allocations.factory_tiles[factory_id][factory_pos].used_by
        #         False
        #     ):
        #         factory_map[factory_pos[0], factory_pos[1]] = 0  # Already occupied
        #     else:
        #         break
        #
        # # Calculate paths avoiding short term collisions
        # paths = []
        # for start, end in zip(
        #     [unit_pos, factory_pos, resource_pos],
        #     [factory_pos, resource_pos, factory_pos],
        # ):
        #     self.log(f'start: {start}, end: {end}', level=logging.DEBUG)
        #     path = pathfinder.path_fast(
        #         start,
        #         end,
        #         rubble=True,
        #         margin=2,
        #         collision_params=CollisionParams(
        #             look_ahead_turns=10,
        #             ignore_ids=(unit_id,),
        #             enemy_light=False if unit_type == UnitType.HEAVY else True,
        #         ),
        #     )
        #     if not path:
        #         self.log(
        #             f'No path found for {unit_id} to get from {start} to {end}',
        #             level=logging.WARNING,
        #         )
        #
        #     paths.append(path)
        #
        # # Actions of regular route (excluding first starting)
        # route_actions = (
        #     path_to_actions(paths[1])
        #     + [unit_manager.unit.dig(n=dig_repeats)]
        #     + path_to_actions(paths[2])
        #     + [
        #         unit_manager.unit.transfer(
        #             CENTER,
        #             recommendation.resource_type,
        #             unit_manager.unit_config.DIG_RESOURCE_GAIN * dig_repeats,
        #         )
        #     ]
        # )
        #
        # # Energy cost of mining route
        # route_cost = power_cost_of_actions(
        #     self.master.maps.rubble, unit_manager.unit, route_actions
        # )
        # route_actions.insert(0, unit_manager.unit.pickup(POWER, int(route_cost * 1.1)))
        # for action in route_actions:
        #     action[ACT_REPEAT] = action[ACT_N]  # Repeat forever
        #
        # # Route to start of mining loop (for now just move to factory, TODO: option to move to resource tile and start cycle form there)
        # actions = path_to_actions(paths[0]) + route_actions
        #
        # # Update pathfinders record of unit path for future calculations this turn
        # pathfinder.update_unit_path(unit_manager, unit_manager.actions_to_path(actions))
        # return actions

    @property
    def resource_maps(self):
        return {ICE: self.ice, ORE: self.ore}

    def nearest_resource(
        self, pos, resource: str, resource_map=None
    ) -> Tuple[int, int]:
        pos = tuple(pos)
        map_hash = hash(resource_map.data.tobytes())
        return self._nearest_resource(pos, resource, resource_map, map_hash=map_hash)

    @cached(
        cache=LRUCache(maxsize=128),
        key=lambda self, pos, resource, resource_map, map_hash: hashkey(
            pos, resource, map_hash
        ),
    )
    def _nearest_resource(
        self, pos, resource: str, resource_map=None, map_hash=None
    ) -> Tuple[int, int]:
        if resource_map is None:
            if resource == ICE:
                resource_map = self.master.game_state.board.ice
            elif resource == ORE:
                resource_map = self.master.game_state.board.ore
            else:
                raise NotImplementedError(f"Dont know {resource}")
        return nearest_non_zero(resource_map, pos)

    def _generate_routes(self):
        """Calculate the X shortest paths from factory to each resource

        Returns:
            routes: dict[unit_type, [dict[resource], [dict[factory_id], MiningRoutes]]]
        """
        base_costs = {
            UnitType.LIGHT: 25,
            UnitType.HEAVY: 250,
        }
        units = {
            UnitType.LIGHT: LIGHT_UNIT,
            UnitType.HEAVY: HEAVY_UNIT,
        }
        routes = {}
        for unit_type in UnitType:
            routes[unit_type.name] = {}

            for resource in [ICE, ORE]:
                routes[unit_type.name][resource] = {}

                for factory in self.master.factories.friendly.values():
                    position = factory.factory.pos
                    resource_map = self.resource_maps[resource].copy()

                    # Paths
                    paths = []
                    for i in range(3):  # X best routes
                        nearest_pos = self.nearest_resource(
                            position, resource, resource_map=resource_map
                        )
                        path = self.master.pathfinder.path_fast(
                            position,
                            nearest_pos,
                            rubble=True,
                        )
                        paths.append(path)
                        resource_map[
                            nearest_pos[0], nearest_pos[1]
                        ] = 0  # So next route goes to next nearest resource

                    # Costs
                    costs = []
                    for path in paths:
                        if len(path) < 1:
                            costs.append(0)
                            self.log(
                                f'Path len 0 for {resource}, {unit_type}, {factory.factory.unit_id}'
                            )
                            continue
                        unit = units[unit_type]
                        unit.pos = path[0]
                        cost = power_cost_of_actions(
                            self.master.maps.rubble,
                            unit,
                            path_to_actions(path),
                        )
                        costs.append(cost)

                    # Values
                    values = []

                    for cost in costs:
                        values.append(base_costs[unit_type] - cost)

                    routes[unit_type.name][resource][factory.unit_id] = MiningRoutes(
                        paths=paths,
                        costs=costs,
                        values=values,
                    )
        return routes
