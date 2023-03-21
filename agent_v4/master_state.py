from __future__ import annotations
import abc
import copy
import logging
from typing import Dict, TYPE_CHECKING, Tuple, List, Optional, Any
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
from lux.kit import GameState

from util import ORE, ICE, METAL, WATER

if TYPE_CHECKING:
    from unit_manager import UnitManager
    from factory_manager import FactoryManager
    from path_finder import PathFinder


class Planner(abc.ABC):
    """E.g. MiningPlanner makes Mining Recommendations
    Other Planners might be:
        - Attack
        - Defend
        - Solar Farm
        - Dig Rubble
        - Dig Lichen
    """

    @abc.abstractmethod
    def recommend(self, unit: UnitManager):
        """Recommend an action for the unit (effectively make a high level obs)"""
        pass

    @abc.abstractmethod
    def carry_out(self, unit: UnitManager, recommendation: Recommendation):
        """TODO: Should this be here?
        Idea would be to make the actions necessary to carry out recommendation
        The Planner instance has probably already calculated what would need to be done, so might be more efficient to ask it again?
        Then clear it's cache at the beginning of each turn?
        """
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """Clear cached recommendations for example"""
        pass

    def log(self, message, level=logging.INFO):
        """Record a logging message in a way that I can easily change the behaviour later"""
        logging.log(level, message)


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

    def _generate_factory_maps(self, game_state: GameState, player: str):
        # TODO: This came from MasterState, needs to be implemented here instead I think
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


@dataclass
class Units:
    light: dict[str, UnitManager]
    heavy: dict[str, UnitManager]
    enemy: Units = None

    @property
    def friendly_units(self) -> dict[str, UnitManager]:
        """Return full list of friendly units"""
        return dict(**self.light, **self.heavy)


@dataclass
class Factories:
    friendly: dict[str, FactoryManager]
    enemy: dict[str, FactoryManager]


class UnitTasks(list):
    pass


@dataclass
class Allocations:
    ice: dict[str, ResourceTile] = field(
        default_factory=dict
    )  # TODO: What to store here
    ore: dict[str, ResourceTile] = field(
        default_factory=dict
    )  # TODO: What to store here
    factory_tiles: dict[str, FactoryTile] = field(
        default_factory=dict
    )  # TODO: What to store here
    units: dict[str, UnitTasks] = field(default_factory=dict)  # TODO: Think about this

    def update(self):
        # TODO: Implement this
        # Remove dead units
        # Remove completed tasks?
        pass

    def assign_unit_resource(self, unit, resource, exclusive: bool = False):
        """Mark a resource as being used by a given unit
        Args:
            exclusive: Whether this resource should be marked for exclusive use by unit
        """
        pass

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


class Maps:
    def __init__(self):
        self.ice: np.ndarray = None
        self.ore: np.ndarray = None
        self.rubble: np.ndarray = None
        self.lichen: np.ndarray = None
        self.factory_maps: FactoryMaps = None

        self.first_update = False

    def update(self, game_state: GameState):
        board = game_state.board
        if not self.first_update:
            self.ice = board.ice
            self.ore = board.ore
            self.first_update = True
        self.rubble = board.rubble
        self.lichen = board.lichen
        self.factory_maps = FactoryMaps(
            all=board.factory_occupancy_map,
            friendly=NotImplemented,
            enemy=NotImplemented,
        )

    def resource_at_tile(self, pos) -> int:
        pos = tuple(pos)
        if self.ice[pos[0], pos[1]]:
            return ICE
        if self.ore[pos[0], pos[1]]:
            return ORE
        return -1  # Invalid


class MasterState:
    def __init__(
        self,
        player,
        env_cfg,
    ):
        self.player = player
        self.opp_player = ("player_1" if self.player == "player_0" else "player_0",)
        self.env_cfg = env_cfg

        self.game_state: GameState = None
        self.step: int = None
        self.pathfinder: PathFinder = PathFinder()

        # TODO: Implement this
        self.units = Units(light={}, heavy={}, enemy=Units(light={}, heavy={}))
        self.factories = Factories(friendly={}, enemy={})
        self.allocations = Allocations()
        self.maps = Maps()

    def update(self, game_state: GameState):
        if game_state.real_env_steps < 0:
            self._early_update(game_state)
        else:
            self._update_units(game_state)
            self._update_factories(game_state)
            self._update_allocations(game_state)

        self.step = game_state.real_env_steps
        self.game_state = game_state

    def _early_update(self, game_state: GameState):
        """
        Update MasterState for the first few turns where factories are being placed

        Track new factories
        """
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
        pass

    def _update_allocations(self, game_state: GameState):
        raise
