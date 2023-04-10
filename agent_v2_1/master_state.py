from __future__ import annotations
import abc
import logging
from typing import TYPE_CHECKING, List
from dataclasses import dataclass, field
import collections

import numpy as np

from agent_v2_1.new_path_finder import Pather
from lux.kit import GameState

from util import ORE, ICE, METAL, WATER, manhattan
from path_finder import PathFinder


if TYPE_CHECKING:
    from lux.unit import Unit
    from unit_manager import UnitManager, FriendlyUnitManger
    from factory_manager import (
        FriendlyFactoryManager,
        EnemyFactoryManager,
    )
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
    def recommend(self, unit: FriendlyUnitManger):
        """Recommend an action for the unit (effectively make a high level obs)"""
        pass

    @abc.abstractmethod
    def carry_out(
        self, unit: FriendlyUnitManger, recommendation: Recommendation
    ) -> List[np.ndarray]:
        """
        Idea would be to make the actions necessary to carry out recommendation
        The Planner instance has probably already calculated a lot of what would need to be done, so efficient to ask it again
        """
        pass

    @abc.abstractmethod
    def update(self, *args, **kwargs):
        """Called at beginning of turn. May want to clear some cached calculations for example"""
        pass


@dataclass
class ResourceTile(dict):
    pos: tuple[int, int]
    resource_type: str
    used_by: list[str] = field(default_factory=list)


@dataclass
class FactoryTile(dict):
    pos: tuple[int, int]
    factory_id: str
    used_by: list[str] = field(default_factory=list)


@dataclass
class FactoryMaps:
    """-1 where no factory, factory_id_num where factory"""

    all: np.ndarray
    friendly: np.ndarray
    enemy: np.ndarray

    @classmethod
    def from_game_state(cls, game_state: GameState, player: str):
        factory_map = game_state.board.factory_occupancy_map
        factories = game_state.factories

        by_team = {}
        for team in game_state.teams:
            fs = factories[team]
            arr = np.ones(factory_map.shape, dtype=int) * -1
            for f in fs.values():
                f_num = int(f.unit_id[-1])
                arr[factory_map == f_num] = f_num
            by_team[team] = arr

        friendly = by_team[player]
        enemy = by_team['player_0' if player == 'player_1' else 'player_1']
        factory_maps = cls(all=factory_map, friendly=friendly, enemy=enemy)
        return factory_maps


@dataclass
class UnitPositions:
    # Unit_id, pos
    light: dict[str, tuple[int, int]]
    heavy: dict[str, tuple[int, int]]

    def unit_at_position(self, pos: tuple[int, int]) -> [None, str]:
        """Get the unit_id of unit at given position"""
        pos = tuple(pos)
        for unit_id, unit_pos in self.light.items():
            if pos == tuple(unit_pos):
                return unit_id
        for unit_id, unit_pos in self.heavy.items():
            if pos == tuple(unit_pos):
                return unit_id
        return None

    def update(self, light: dict[str, UnitManager], heavy: dict[str, UnitManager]):
        """Update at beginning of turn"""
        self.light = {unit_id: unit.pos for unit_id, unit in light.items()}
        self.heavy = {unit_id: unit.pos for unit_id, unit in heavy.items()}


@dataclass
class AllUnits:
    friendly: FriendlyUnits
    enemy: EnemyUnits
    master: MasterState

    def unit_at_position(self, pos: tuple[int, int]) -> [None, str]:
        """Get the unit_id of unit at given position"""
        unit_id = self.friendly.unit_at_position(pos)
        if unit_id is not None:
            return unit_id
        else:
            return self.enemy.unit_at_position(pos)  # Note: None if not there either

    def get_unit(self, unit_id: str) -> UnitManager:
        unit = self.friendly.get_unit(unit_id)
        if unit is not None:
            return unit
        else:
            unit = self.enemy.get_unit(unit_id)
        if unit is None:
            raise KeyError(f'{unit_id} does not exist in friendly or enemy units')

    def nearest_unit(
        self, pos: tuple[int, int], friendly=True, enemy=True, light=True, heavy=True
    ) -> tuple[str, int]:
        """Get unit_id, dist of nearest unit to pos
        Note: Returns '', 999 if no unit found
        """
        nearest_id = ''
        nearest_dist = 999
        if friendly:
            new_id, new_dist = self.friendly.nearest_unit(pos, light=light, heavy=heavy)
            if new_dist < nearest_dist:
                nearest_id, nearest_dist = new_id, new_dist
        if enemy:
            new_id, new_dist = self.enemy.nearest_unit(pos, light=light, heavy=heavy)
            if new_dist < nearest_dist:
                nearest_id, nearest_dist = new_id, new_dist
        return nearest_id, nearest_dist

    def update(self, game_state: GameState):
        self.friendly.update(game_state.units[self.master.player])
        self.enemy.update(game_state.units[self.master.opp_player])


class Units(abc.ABC):
    def __init__(self):
        self.light: dict[str, UnitManager] = dict()
        self.heavy: dict[str, UnitManager] = dict()
        #                  {LIGHT/HEAVY: {unit_id: pos}}
        self.unit_positions: UnitPositions = UnitPositions(light={}, heavy={})

    @property
    def all(self) -> dict[str, UnitManager]:
        """Return full list of units"""
        return dict(**self.light, **self.heavy)

    def unit_at_position(self, pos: tuple[int, int]) -> [None, str]:
        """Get the unit_id of unit at given position"""
        if self.unit_positions is None:
            logging.error(
                f'Requesting unit at {pos} before initializing unit_positions',
            )
            raise RuntimeError(
                f'self.unit_positions is None. Must be initialized first'
            )
        return self.unit_positions.unit_at_position(pos)

    def nearest_unit(
        self, pos: tuple[int, int], light=True, heavy=True
    ) -> tuple[str, int]:
        """Return the nearest unit and distance from the given position
        Note: Returns '', 999 if no unit found
        """
        unit_positions = {}
        if light:
            for unit_id, unit in self.light.items():
                unit_positions[unit_id] = unit.pos
        if heavy:
            for unit_id, unit in self.heavy.items():
                unit_positions[unit_id] = unit.pos

        distances = {
            unit_id: manhattan(pos, other_pos)
            for unit_id, other_pos in unit_positions.items()
        }
        if not distances:
            return '', 999
        nearest_id = min(distances, key=distances.get)
        return nearest_id, distances[nearest_id]

    def get_unit(self, unit_id: str) -> [None, UnitManager]:
        if unit_id in self.light:
            return self.light[unit_id]
        elif unit_id in self.heavy:
            return self.heavy[unit_id]
        else:
            return None

    def _remove_dead(self, units):
        # Remove dead units
        for u_dict in (self.light, self.heavy):
            # If anything in light/heavy list that isn't in full list, must be dead
            for k in set(u_dict.keys()) - set(units.keys()):
                logging.info(f'Removing unit {k}, assumed dead')
                dead_unit = u_dict.pop(k)
                dead_unit.dead()

    @abc.abstractmethod
    def update(self, units: dict[str, Unit]):
        """
        Update at the beginning of turn
        Args:
            units: All units for friendly/enemy team (i.e. when called it will already be decided if friendly or enemy)
        """
        pass


class EnemyUnits(Units):
    def update(self, units: dict[str, Unit]):
        """
        Update at the beginning of turn
        Args:
            units: All units for this team (i.e. already decided if friendly or enemy)
        """
        from unit_manager import EnemyUnitManager  # Avoiding circular import

        unit_dicts = {'LIGHT': self.light, 'HEAVY': self.heavy}
        for unit_id, unit in units.items():
            u_dict = unit_dicts[unit.unit_type]

            # Update existing
            if unit_id in u_dict:
                u_dict[unit_id].update(unit)

            # Add new unit
            else:
                u_dict[unit_id] = EnemyUnitManager(unit)

        # Remove dead units
        self._remove_dead(units)

        # Store positions of all units
        self.unit_positions.update(self.light, self.heavy)


class FriendlyUnits(Units):
    def __init__(self, master: MasterState):
        super().__init__()
        self.master: MasterState = master

    def update(self, units: dict[str, Unit]):
        """
        Update at the beginning of turn
        Args:
            units: All units for this team (i.e. already decided if friendly or enemy)
        """
        from unit_manager import FriendlyUnitManger  # Avoiding circular import

        # For all units on team
        unit_dicts = {'LIGHT': self.light, 'HEAVY': self.heavy}
        for unit_id, unit in units.items():
            u_dict = unit_dicts[unit.unit_type]

            # Update existing
            if unit_id in u_dict:
                u_dict[unit_id].update(unit)

            # Add new unit
            else:
                # Which factory produced this unit
                factory_id_num = self.master.maps.factory_maps.all[
                    unit.pos[0], unit.pos[1]
                ]

                # Convert to factory_id
                if factory_id_num >= 0:
                    factory_id = f'factory_{factory_id_num}'

                else:
                    logging.error(
                        f"No factory under new {unit_id}, assigning `None` for factory_id. Pretty sure this shouldn't "
                        f"happen though",
                    )
                    factory_id = None

                # Create and save new UnitManager
                new_unit = FriendlyUnitManger(
                    unit, master_state=self.master, factory_id=factory_id
                )
                u_dict[unit_id] = new_unit

                # Add this unit to the factory list of units (light_units or heavy_units depending on unit_type
                getattr(
                    self.master.factories.friendly[factory_id],
                    f'{unit.unit_type.lower()}_units',
                )[unit_id] = new_unit

        # Remove dead units
        self._remove_dead(units)

        # Store positions of all units
        self.unit_positions.update(self.light, self.heavy)


@dataclass
class Factories:
    friendly: dict[str, FriendlyFactoryManager]
    enemy: dict[str, EnemyFactoryManager]
    master: MasterState = None

    def update(self, game_state: GameState):
        from factory_manager import FriendlyFactoryManager, EnemyFactoryManager

        # Update Friendly Factories
        factories = game_state.factories[self.master.player]
        f_dict = self.friendly
        for factory_id, factory in factories.items():
            # Update existing
            if factory_id in f_dict:
                f_dict[factory_id].update(factory)
            # Add new
            else:
                f_dict[factory_id] = FriendlyFactoryManager(
                    factory, master_state=self.master
                )
        # Remove dead
        for k in set(f_dict.keys()) - set(factories.keys()):
            dead_factory = f_dict.pop(k)
            logging.info(f'Friendly {k} died, being removed')
            dead_factory.dead()

        # Update Enemy Factories
        factories = game_state.factories[self.master.opp_player]
        f_dict = self.enemy
        for factory_id, factory in factories.items():
            # Update existing
            if factory_id in f_dict:
                f_dict[factory_id].update(factory)
            # Add new
            else:
                f_dict[factory_id] = EnemyFactoryManager(factory)
        # Remove dead
        for k in set(f_dict.keys()) - set(factories.keys()):
            dead_factory = f_dict.pop(k)
            logging.info(f'Friendly {k} died, being removed')
            dead_factory.dead()


# class UnitTasks(list):
#     pass
#
#
# @dataclass
# class Allocations:
#     ice: dict[str, ResourceTile] = field(
#         default_factory=dict
#     )  # TODO: What to store here
#     ore: dict[str, ResourceTile] = field(
#         default_factory=dict
#     )  # TODO: What to store here
#     factory_tiles: dict[str, FactoryTile] = field(
#         default_factory=dict
#     )  # TODO: What to store here
#     units: dict[str, UnitTasks] = field(default_factory=dict)  # TODO: Think about this
#
#     def update(self):
#         # TODO: Implement this
#         # Remove dead units
#         # Remove completed tasks?
#         pass
#
#     def assign_unit_resource(self, unit, resource, exclusive: bool = False):
#         """Mark a resource as being used by a given unit
#         Args:
#             exclusive: Whether this resource should be marked for exclusive use by unit
#         """
#         pass
#
#     def assign_unit_factory(
#         self, unit_id: str, position: Optional[Tuple[int, int]] = None
#     ):
#         raise
#         position = tuple(position)
#         factory_id = f'factory_{self.factory_maps.all[position[0], position[1]]}'
#         if factory_id not in self.factory_allocation:
#             self.factory_allocation[factory_id] = {}
#         ft = self.factory_allocation[factory_id].get(
#             position, FactoryTile(pos=position, factory_id=factory_id)
#         )
#         ft.used_by.append(unit_id)
#         self.factory_allocation[factory_id][position] = ft
#
#     def deassign_unit_resource(self, unit_id: str, position: Tuple[int, int] = None):
#         raise
#         if position is not None:
#             position = tuple(position)
#             rts = [self.resource_allocation.get(position, [])]
#         else:
#             rts = self.unit_allocations.get(unit_id, [])
#         for rt in rts:
#             if unit_id in rt.used_by:
#                 rt.used_by.remove(unit_id)
#
#     def deassign_unit_factory(self, unit_id: str, factory_id: Optional[str] = None):
#         raise
#
#         def _remove_from(factory_id):
#             for pos, ft in self.factory_allocation[factory_id].items():
#                 if unit_id in ft.used_by:
#                     ft.used_by.remove(unit_id)
#
#         if factory_id:  # Remove from specified
#             _remove_from(factory_id)
#         else:  # Remove from all
#             for f_id in self.factory_allocation.keys():
#                 _remove_from(f_id)


class Maps:
    def __init__(self):
        self.ice: np.ndarray = None
        self.ore: np.ndarray = None
        self.rubble: np.ndarray = None
        self.lichen: np.ndarray = None
        self.factory_maps: FactoryMaps = None

        self.first_update = False

    def update(self, game_state: GameState, player: str):
        board = game_state.board
        if not self.first_update:
            self.ice = board.ice
            self.ore = board.ore
            self.first_update = True
        self.rubble = board.rubble
        self.lichen = board.lichen
        self.factory_maps = FactoryMaps.from_game_state(
            game_state=game_state, player=player
        )
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
        self.pathfinder: Pather = (
            None  # This gets set at the beginning of each unit action
        )

        self.units = AllUnits(
            friendly=FriendlyUnits(master=self), enemy=EnemyUnits(), master=self
        )
        self.factories = Factories(friendly={}, enemy={}, master=self)
        # self.allocations = Allocations()
        self.maps = Maps()

    def update(self, game_state: GameState):
        self.maps.update(game_state)
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
        self.factories.update(game_state)

    def _update_units(self, game_state: GameState):
        """
        Update each unit with new position/action queue/power etc
        If there are new units, add them to tracked units
        If units have died, remove them from tracked units and remove any allocations for them
        """
        self.units.update(game_state)
        pass

    def _update_factories(self, game_state: GameState):
        """
        Update each factory with new power/resources
        If factory has died, update accordingly
        """
        self.factories.update(game_state)
        pass

    # def _update_allocations(self, game_state: GameState):
    #     pass
