from __future__ import annotations
from cachetools import LRUCache, cached
from cachetools.keys import hashkey
import logging
from typing import TYPE_CHECKING, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from luxai_s2.unit import UnitType

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
)

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManger


@dataclass
class MiningRoutes:
    paths: list
    costs: list
    values: list


class MiningRecommendation(Recommendation):
    """Recommended mining action between a resource and factory"""

    role = 'miner'
    value = 0

    def __init__(self, value: float, resource_pos, factory_id, resource_type: str):
        self.value = value
        self.resource_pos: Tuple[int, int] = resource_pos
        self.factory_id: str = factory_id
        self.resource_type = resource_type

    def to_action_queue(self, plan: MasterState) -> list[np.ndarray]:
        logging.error('Not Implemented')
        print('error, MiningRecommendation.to_action_queue not implemented')
        return []


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
        # For now, only calculate once
        # self.mining_routes = self._generate_routes()
        pass

    def recommend(self, unit_manager: FriendlyUnitManger):
        """Make recommendations for this unit to mine resources (i.e. get values of nearby routes)

        TODO: Think about whether this should take into account the needs of nearby factories? That is maybe better suited to RL?
        """
        # TODO: Improve this
        if unit_manager.status.role == MiningRecommendation.role:
            return MiningRecommendation(
                value=0, resource_pos=(-1, -1), factory_id='', resource_type=ICE
            )

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
                resource_pos=next_unoccupied_path[-1] if next_unoccupied_path is not None and len(next_unoccupied_path) > 0 else None,
                factory_id=nearest_factory_id,
                resource_type=resource,
            )
            if rec.value > best_rec.value:
                best_rec = rec
        return best_rec

    def carry_out(
        self, unit_manager: FriendlyUnitManger, recommendation: MiningRecommendation
    ):
        # TODO: Check going to resource first (for now always going to factory first
        dig_repeats = 10

        unit_pos = unit_manager.unit.pos
        unit_type = unit_manager.unit.unit_type
        unit_id = unit_manager.unit_id

        pathfinder = self.master.pathfinder
        resource_pos = recommendation.resource_pos

        # Figure out which factory tile to use
        factory_id = recommendation.factory_id
        if not factory_id:
            self.log(
                f'No factory_id in recommendation for unit {unit_manager.unit_id}',
                level=logging.ERROR,
            )
            return None
        factory_num = int(factory_id[-1])
        factory_map = np.array(
            self.master.maps.factory_maps.by_player[self.master.player] == factory_num,
            dtype=int,
        )
        center_coord = self.master.game_state.factories[self.master.player][
            factory_id
        ].pos
        factory_map[
            center_coord[0], center_coord[1]
        ] = 0  # Don't deliver to center of factory
        while True:
            factory_pos = nearest_non_zero(factory_map, resource_pos)
            if factory_pos is None:
                self.log(
                    f'No unallocated spaces at {factory_id}', level=logging.WARNING
                )
                factory_pos = center_coord
                break
            elif (
                # factory_id in self.master.allocations.factory_tiles
                # and factory_pos in self.master.allocations.factory_tiles[factory_id]
                # and self.master.allocations.factory_tiles[factory_id][factory_pos].used_by
                False
            ):
                factory_map[factory_pos[0], factory_pos[1]] = 0  # Already occupied
            else:
                break

        # Calculate paths avoiding short term collisions
        paths = []
        for start, end in zip(
            [unit_pos, factory_pos, resource_pos],
            [factory_pos, resource_pos, factory_pos],
        ):
            self.log(f'start: {start}, end: {end}', level=logging.DEBUG)
            path = pathfinder.path_fast(
                start,
                end,
                step=unit_manager.master.step,
                rubble=True,
                margin=2,
                collision_params=CollisionParams(
                    turns=10,
                    enemy_light=False if unit_type == UnitType.HEAVY else True,
                    ignore_ids=[unit_id],
                ),
            )
            paths.append(path)

        # Actions of regular route (excluding first starting)
        route_actions = (
            path_to_actions(paths[1])
            + [unit_manager.unit.dig(n=dig_repeats)]
            + path_to_actions(paths[2])
            + [
                unit_manager.unit.transfer(
                    CENTER,
                    recommendation.resource_type,
                    unit_manager.unit_config.DIG_RESOURCE_GAIN * dig_repeats,
                )
            ]
        )

        # Energy cost of mining route
        route_cost = power_cost_of_actions(
            self.master.game_state, unit_manager.unit, route_actions
        )
        route_actions.insert(0, unit_manager.unit.pickup(POWER, int(route_cost * 1.1)))
        for action in route_actions:
            action[ACT_REPEAT] = action[ACT_START_N]  # Repeat forever

        # Route to start of mining loop (for now just move to factory, TODO: option to move to resource tile and start cycle form there)
        actions = path_to_actions(paths[0]) + route_actions

        # Update pathfinders record of unit path for future calculations this turn
        pathfinder.update_unit_path(unit_manager, unit_manager.actions_to_path(actions))
        return actions

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
                            step=self.master.step,
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
                            self.master.game_state,
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
