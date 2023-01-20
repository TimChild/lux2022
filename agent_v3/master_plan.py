from __future__ import annotations
import abc
import copy
import logging
from typing import Dict, TYPE_CHECKING, Tuple, List, Optional
from dataclasses import dataclass, field
from functools import lru_cache
import numpy as np

from lux.kit import GameState
from util import ORE, ICE, METAL, WATER

if TYPE_CHECKING:
    from unit_manager import UnitManager
    from factory_manager import FactoryManager
    from path_finder import PathFinder


@dataclass
class ResourceTile(dict):
    pos: Tuple[int, int]
    resource_type: str
    used_by: List[str] = field(default_factory=list)


@dataclass
class FactoryTile(dict):
    pos: Tuple[int, int]
    factory_id: str
    used_by: List[str] = field(default_factory=list)


@dataclass
class FactoryMaps:
    """-1 where no factory, factory_id_num where factory"""

    all: np.ndarray
    friendly: np.ndarray
    enemy: np.ndarray


class Planner(abc.ABC):
    @abc.abstractmethod
    def recommend(self, unit: UnitManager):
        pass

    @abc.abstractmethod
    def carry_out(self, unit: UnitManager, recommendation: Recommendation):
        pass

    @abc.abstractmethod
    def update(self):
        pass

    def log(self, message, level=logging.INFO):
        logging.log(level, message)


class Planners:
    def __init__(self, master_plan: MasterState):
        from mining_planner import MiningPlanner

        self.mining_planner: MiningPlanner = MiningPlanner(master_plan)


class MasterState:
    def __init__(
        self,
        player,
        opp_player,
        unit_managers,
        enemy_unit_managers,
        factory_managers,
        pathfinder: PathFinder,
    ):
        self.player = player
        self.opp_player = opp_player
        self.game_state: GameState = None
        self.step: int = None
        self.unit_managers: Dict[str, UnitManager] = unit_managers
        self.enemy_unit_managers: Dict[str, UnitManager] = enemy_unit_managers
        self.factory_managers: Dict[str, FactoryManager] = factory_managers

        self.resource_allocation: Dict[Tuple[int, int], ResourceTile] = {}
        self.factory_allocation: Dict[str, Dict[Tuple[int, int], FactoryTile]] = {}

        self.unit_allocations: Dict[str, ResourceTile] = {}
        self.pathfinder: PathFinder = pathfinder

        self._factory_maps: FactoryMaps = None

    def update(self, game_state: GameState):
        if game_state.real_env_steps < 0:
            self._early_update(game_state)
        else:
            self._update_units(game_state)
            self._update_factories(game_state)

        self.step = game_state.real_env_steps
        self.game_state = game_state

    def _early_update(self, game_state: GameState):
        """
            Update MasterState for the first few turns where factories are being placed

            Track new factories
        """
        self._factory_maps = None  # Factories being placed, so needs updating on next call
        pass

    def _update_units(self, game_state: GameState):
        """
            Update each unit with new position/action queue/power etc
            If there are new units, add them to tracked units
            If units have died, remove them from tracked units and remove any allocations for them
        """
        pass

    def _update_factories(self, game_state: GameState):
        """
            Update each factory with new power/resources
            If factory has died, update accordingly
        """

    @property
    def factory_maps(self):
        if self._factory_maps is None:
            self._factory_maps = self._generate_factory_maps()
        return self._factory_maps

    def _generate_factory_maps(self):
        factory_map = self.game_state.board.factory_occupancy_map
        factories = self.game_state.factories

        other_maps = {}
        for team in self.game_state.teams:
            fs = factories[team]
            arr = np.ones(factory_map.shape, dtype=int) * -1
            for f in fs.values():
                f_num = int(f.unit_id[-1])
                arr[factory_map == f_num] = f_num
            other_maps[team] = arr

        factory_maps = FactoryMaps(
            all=factory_map,
            friendly=other_maps[self.player],
            enemy=other_maps[self.opp_player],
        )
        return factory_maps

    def _update_resource_allocation(self, remove_units):
        raise
        for unit in remove_units:
            rts = self.unit_allocations.get(unit.unit_id, [])
            for rt in rts:
                rt.used_by.remove(unit.unit_id)

    def assign_unit_resource(self, unit_id: str, position: Tuple[int, int]):
        raise
        position = tuple(position)
        rt = self.resource_allocation.get(
            position,
            ResourceTile(pos=position, resource_type=self.resource_at_tile(position)),
        )
        rt.used_by.append(unit_id)
        self.resource_allocation[position] = rt

        current_allocations = self.unit_allocations.get(unit_id, [])
        current_allocations.append(rt)
        self.unit_allocations[unit_id] = current_allocations

    def assign_unit_factory(
        self, unit_id: str, position: Optional[Tuple[int, int]] = None
    ):
        raise
        position = tuple(position)
        factory_id = f'factory_{self.factory_maps.all[position[0], position[1]]}'
        if factory_id not in self.factory_allocation:
            self.factory_allocation[factory_id] = {}
        ft = self.factory_allocation[factory_id].get(
            position, FactoryTile(pos=position, factory_id=factory_id)
        )
        ft.used_by.append(unit_id)
        self.factory_allocation[factory_id][position] = ft

    def deassign_unit_resource(self, unit_id: str, position: Tuple[int, int] = None):
        raise
        if position is not None:
            position = tuple(position)
            rts = [self.resource_allocation.get(position, [])]
        else:
            rts = self.unit_allocations.get(unit_id, [])
        for rt in rts:
            if unit_id in rt.used_by:
                rt.used_by.remove(unit_id)

    def deassign_unit_factory(self, unit_id: str, factory_id: Optional[str] = None):
        raise
        def _remove_from(factory_id):
            for pos, ft in self.factory_allocation[factory_id].items():
                if unit_id in ft.used_by:
                    ft.used_by.remove(unit_id)

        if factory_id:  # Remove from specified
            _remove_from(factory_id)
        else:  # Remove from all
            for f_id in self.factory_allocation.keys():
                _remove_from(f_id)

    def resource_at_tile(self, pos) -> str:
        pos = tuple(pos)
        if self.game_state.board.ice[pos[0], pos[1]]:
            return ICE
        if self.game_state.board.ore[pos[0], pos[1]]:
            return ORE
        if self.game_state.board.lichen[pos[0], pos[1]]:
            return 'lichen'
        if self.game_state.board.rubble[pos[0], pos[1]]:
            return 'rubble'
        return None


class Recommendation:
    role: str = 'not set'
    value: float = 0
