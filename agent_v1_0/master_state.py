from __future__ import annotations
import abc
import copy
import logging
from typing import Dict, TYPE_CHECKING, Tuple, List, Optional, Any
from dataclasses import dataclass, field
import collections
from functools import lru_cache

import numpy as np

from lux.kit import GameState

from util import ORE, ICE, METAL, WATER, manhattan
from path_finder import PathFinder

if TYPE_CHECKING:
    from unit_manager import UnitManager
    from factory_manager import FactoryManager
    from actions import Recommendation


def map_nested_dicts(ob, func):
    if isinstance(ob, collections.Mapping):
        return {k: map_nested_dicts(v, func) for k, v in ob.items()}
    else:
        return func(ob)


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
    by_player: dict[str, np.ndarray]

    @classmethod
    def from_game_state(cls, game_state: GameState):
        # TODO: This came from MasterState, needs to be implemented here instead I think
        factory_map = game_state.board.factory_occupancy_map
        factories = game_state.factories

        by_player_maps = {}
        for team in game_state.teams:
            fs = factories[team]
            arr = np.ones(factory_map.shape, dtype=int) * -1
            for f in fs.values():
                f_num = int(f.unit_id[-1])
                arr[factory_map == f_num] = f_num
            by_player_maps[team] = arr

        factory_maps = cls(all=factory_map, by_player=by_player_maps)
        return factory_maps


@dataclass
class Units:
    light: dict[str, UnitManager]
    heavy: dict[str, UnitManager]
    #                  {LIGHT/HEAVY: {unit_id: pos}}
    unit_positions: dict[str, dict[str, tuple[int, int]]] = None
    enemy: Units = None

    @property
    def friendly_units(self) -> dict[str, UnitManager]:
        """Return full list of friendly units"""
        return dict(**self.light, **self.heavy)

    @property
    def enemy_units(self) -> dict[str, UnitManager]:
        """Return full list of enemy units"""
        return self.enemy.friendly_units

    def unit_at_position(self, pos: tuple[int, int]):
        """Get the unit_id of unit at given position"""
        for type, units in self.unit_positions.items():
            for unit_id, unit_pos in units.items():
                if tuple(pos) == tuple(unit_pos):
                    return unit_id
        return None

    def log(self, message, level=logging.INFO):
        logging.log(level=level, msg=message)

    def get_unit(self, unit_id: str):
        if unit_id in self.light:
            return self.light[unit_id]
        elif unit_id in self.heavy:
            return self.heavy[unit_id]
        elif self.enemy is not None:
            return self.enemy.get_unit(unit_id)
        else:
            raise KeyError(f'Unit {unit_id} does not exist')

    def nearest_unit(
        self, pos: tuple[int, int], friendly=True, enemy=True, light=True, heavy=True
    ) -> tuple[str, int]:
        """Return the nearest unit and distance from the given position"""
        unit_positions = {}
        if friendly:
            if light:
                for unit_id, unit in self.light.items():
                    unit_positions[unit_id] = unit.pos
            if heavy:
                for unit_id, unit in self.heavy.items():
                    unit_positions[unit_id] = unit.pos
        if enemy:
            if light:
                for unit_id, unit in self.enemy.light.items():
                    unit_positions[unit_id] = unit.pos
            if heavy:
                for unit_id, unit in self.enemy.heavy.items():
                    unit_positions[unit_id] = unit.pos

        distances = {
            unit_id: manhattan(pos, other_pos)
            for unit_id, other_pos in unit_positions.items()
        }
        if not distances:
            return '', 999
        nearest_id = min(distances, key=distances.get)
        return nearest_id, distances[nearest_id]

    def update(self, game_state: GameState, team: str, master_state: MasterState):
        """
        Update at the beginning of turn
        TODO: May want to do something before calling this to check what has actually changed
        """
        from unit_manager import UnitManager

        units = {'LIGHT': {}, 'HEAVY': {}}
        for id, unit in game_state.units[team].items():
            units[unit.unit_type][id] = unit

        for attr_key, unit_type in zip(['light', 'heavy'], ['LIGHT', 'HEAVY']):
            for unit_id, unit in units[unit_type].items():
                # Update existing Units
                if unit_id in getattr(self, attr_key):
                    getattr(self, attr_key)[unit_id].update(unit)
                # Add new units
                else:
                    factory_id_num = master_state.maps.factory_maps.all[
                        unit.pos[0], unit.pos[1]
                    ]
                    factory_id = f'factory_{factory_id_num}'
                    unit_manager = UnitManager(
                        unit=unit,
                        master_state=master_state,
                        factory_id=factory_id,
                    )
                    getattr(self, attr_key)[unit_id] = unit_manager
                    if team == master_state.player:
                        getattr(master_state.factories.friendly[factory_id], f'{attr_key}_units')[unit_id] = unit_manager

                # Remove dead units
                for k in set(getattr(self, attr_key).keys()) - set(units[unit_type].keys()):
                    self.log(f'Removing unit {k}, assumed dead')
                    dead_unit = getattr(self, attr_key).pop(k)
                    logging.info(f'{team}, {master_state.player}')
                    if team == master_state.player:
                        self.log(f'Removing unit {k} from factory also')
                        getattr(master_state.factories.friendly[dead_unit.factory_id], f'{attr_key}_units').pop(k)

        # Store positions of all units
        self.unit_positions = map_nested_dicts(units, lambda unit: unit.pos)

        # Only update self.enemy if it isn't none (prevent infinite recursion)
        if self.enemy is not None:
            opp_team = 'player_1' if team == 'player_0' else 'player_1'
            self.enemy.update(game_state, opp_team, master_state)


@dataclass
class Factories:
    player: str
    friendly: dict[str, FactoryManager]
    enemy: dict[str, FactoryManager]

    def update(self, game_state: GameState):
        from factory_manager import FactoryManager

        for player_id, factories in game_state.factories.items():
            if player_id == self.player:
                for factory_id, factory in factories.items():
                    if factory_id not in self.friendly:
                        self.friendly[factory_id] = FactoryManager(factory)
                    self.friendly[factory_id].update(factory)
            else:
                for factory_id, factory in factories.items():
                    if factory_id not in self.enemy:
                        self.enemy[factory_id] = FactoryManager(factory)
                    self.enemy[factory_id].update(factory)


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
        self.factory_maps = FactoryMaps.from_game_state(game_state=game_state)
        # self.unit_map = game_state.units.

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
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg = env_cfg

        self.game_state: GameState = None
        self.step: int = None
        self.pathfinder: PathFinder = PathFinder()

        self.units = Units(light={}, heavy={}, enemy=Units(light={}, heavy={}))
        self.factories = Factories(player=self.player, friendly={}, enemy={})
        self.allocations = Allocations()
        self.maps = Maps()

    def update(self, game_state: GameState):
        self.maps.update(game_state)
        if game_state.real_env_steps < 0:
            self._early_update(game_state)
        else:
            self._update_units(game_state)
            self._update_factories(game_state)
            self._update_allocations(game_state)
            self.pathfinder.update(
                rubble=self.maps.rubble,
                friendly_units=self.units.friendly_units,
                enemy_units=self.units.enemy_units,
                enemy_factories=self.factories.enemy,
            )

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
        self.units.update(game_state, self.player, master_state=self)
        pass

    def _update_factories(self, game_state: GameState):
        """
        Update each factory with new power/resources
        If factory has died, update accordingly
        """
        self.factories.update(game_state)
        pass

    def _update_allocations(self, game_state: GameState):
        pass
