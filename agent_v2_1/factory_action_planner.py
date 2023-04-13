from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List
import copy

from agent_v2_1.actions import Recommendation
from agent_v2_1.factory_manager import FriendlyFactoryManager
from agent_v2_1.unit_manager import FriendlyUnitManger

from lux.factory import Factory

from master_state import MasterState, Planner
import util
from config import get_logger

logger = get_logger(__name__)

BUILD_LIGHT = 0
BUILD_HEAVY = 1
WATER =  2

@dataclass
class FactoryInfo:
    factory: Factory
    factory_id: str
    power: int
    water: int
    ice: int
    ore: int
    metal: int
    pos: Tuple[int, int]
    num_heavy: int
    num_light: int
    water_cost: int
    connected_zeros: int
    lichen_tiles: int
    total_lichen: int

    def update(self, new: FactoryInfo):
        # never changes
        # self.factory_id
        # self.pos

        # Replace some things
        self.factory = new.factory
        self.connected_zeros = new.connected_zeros
        self.lichen_tiles = new.lichen_tiles
        self.total_lichen = new.total_lichen
        self.num_light = new.num_light
        self.num_heavy = new.num_heavy

        # Average others
        for attr in ["power", "water", "ice", "ore", "metal", "water_cost"]:
            current = getattr(self, attr)
            # Roughly like average of last 10 values
            val = 0.9*current + 0.1*getattr(new, attr)
            setattr(self, attr, val)

@dataclass
class FactoryDesires:
    heavy_mining_ice: int = 1
    light_mining_ore: int = 0
    light_clearing_rubble: int = 0
    light_attacking:  int = 0
    heavy_attacking: int = 0

    def copy(self) -> FactoryDesires:
        return copy.copy(self)


class FactoryActionPlanner:

    def __init__(self, master: MasterState):
        self.master = master
        self.factories = self.master.factories.friendly

        # Caching variables
        self._factory_desires: Dict[str, FactoryDesires] = {}
        self._factory_infos = Dict[str, FactoryInfo] = {}

    def update(self, *args, **kwargs):
        """Calculate the general desires of the factories
        - How many heavy units mining ice
        - How many light units mining ore
        - How many light units clearing rubble
        - How many light units attacking
        - How many heavy units attacking
        - etc
        """
        logger.function_call(f'Updating FactoryActionPlanner')
        # Remove any dead factories
        for k in set(self._factory_desires.keys()) - set(self.master.factories.friendly.keys()):
            logger.info(f'Removing factory {k}, assumed dead')
            self._factory_desires.pop(k)

        # Update their infos
        self._update_factory_info()

        # Calculate new desires for first X turns and then every 50 after that
        if self.master.step < 10 or self.master.step % 50 == 0:
            self._update_factory_desires()

    def _update_factory_info(self):
        """Update info about the factory (uses a sort of rolling average in updating)"""
        for f_id, factory in self.master.factories.friendly.items():
            logger.debug(f'Updating {f_id} info')
            prev_info = self._factory_infos.pop(f_id, None)
            new_info = self._collect_factory_info(factory)
            if prev_info:
                prev_info: FactoryInfo
                prev_info.update(new_info)
                new_info = prev_info
            self._factory_infos[f_id] = new_info

    def _update_factory_desires(self):
        """Update the desires for each factory"""
        # Remove dead factories
        for f_id, factory in self.master.factories.friendly.items():
            logger.debug(f'Updating {f_id} desires')
            # Pop because updating
            previous_desire = self._factory_desires.pop(f_id, None)
            # Get because only using
            info = self._factory_infos.get(f_id, None)
            # Update taking into account previous preferences and info
            if previous_desire and info:
                desires = self._update_single_factory_desire(previous_desire, info)
            # Update without knowing previous preferences (first turn for this factory)
            else:
                desires = self._create_single_factory_desire(info)

            self._factory_desires[f_id] = desires

    def _update_single_factory_desire(self, prev_desire: FactoryDesires, info: FactoryInfo) -> FactoryDesires:
        """Update the desires of the factory based on it's current state and it's previous desires"""
        step = self.master.step
        desires = prev_desire.copy()
        logger.debug(f'Desires before HeavyIce={desires.heavy_mining_ice}, LightRubble={desires.light_clearing_rubble}, LightOre={desires.light_mining_ore}, LightAttack={desires.light_attacking}, HeavyAttack={desires.heavy_attacking}')
        # Early game
        if step < 200:
            # Add more light units
            if info.power > 1000 and info.metal > 10:
                if desires.light_mining_ore < 3:
                    desires.light_mining_ore += 1
                elif desires.light_clearing_rubble < 3:
                    desires.light_clearing_rubble += 1
                elif desires.light_attacking < 1:
                    desires.light_attacking += 1
                else:
                    # Add heavy
                    if info.power > 2000 and info.metal > 100:
                        desires.heavy_mining_ice += 1
        # Early mid game
        elif step < 500:
            pass
        # mid late game
        elif step < 800:
            pass
        else:
            pass
        logger.debug(f'Desires after HeavyIce={desires.heavy_mining_ice}, LightRubble={desires.light_clearing_rubble}, LightOre={desires.light_mining_ore}, LightAttack={desires.light_attacking}, HeavyAttack={desires.heavy_attacking}')
        return desires

    def _create_single_factory_desire(self, info: FactoryInfo) -> Tuple[FactoryDesires, FactoryInfo]:
        """Create the initial desires for factory, assumes this is the beginning of the game. After this the
        desires should be updated based on previous desires"""
        # Defaults
        desires = FactoryDesires(
            heavy_mining_ice=1,
            light_mining_ore=1,
            light_clearing_rubble=0,
            light_attacking=0,
            heavy_attacking=0,
        )

        # Update based on info
        if info.connected_zeros < 15:
            desires.light_clearing_rubble = 1

        return desires, info

    def _collect_factory_info(self, factory: FriendlyFactoryManager) -> FactoryInfo:
        lichen_id = factory.factory.strain_id
        lichen_locations = self.master.game_state.board.lichen_strains == lichen_id
        lichen = self.master.game_state.board.lichen
        lichen[lichen_locations != 1] = 0

        factory_info = FactoryInfo(
            factory=factory.factory,
            factory_id=factory.unit_id,
            power=factory.power,
            water=factory.factory.cargo.water,
            ice=factory.factory.cargo.ice,
            ore=factory.factory.cargo.ore,
            metal=factory.factory.cargo.metal,
            pos=factory.pos,
            num_heavy=len(factory.heavy_units),
            num_light=len(factory.light_units),
            water_cost=factory.factory.water_cost(self.master.game_state),
            connected_zeros=int(np.sum(util.connected_array_values_from_pos(self.master.maps.rubble, factory.pos, connected_value=0))),
            lichen_tiles=int(np.sum(lichen_locations)),
            total_lichen=int(np.sum(lichen)),
        )
        return factory_info

    def decide_factory_actions(self) -> Dict[str, int]:
        actions = {}
        for f_info, f_desire in zip(self._factory_infos.values(), self._factory_desires.values()):
            logger.debug(f'Deciding factory actions for {f_info.factory_id}')
            f_info: FactoryInfo
            f_desire: FactoryDesires
            center_occupied = True if self.master.units.friendly.unit_at_position(f_info.factory.pos) is not None else False
            action = None

            # Only consider building if center not occupied
            if not center_occupied:
                action = self._build_unit_action(f_info, f_desire)

            # If not building unit, consider watering
            if action is None:
                action = self._water(f_info, f_desire)

            if action is not None:
                logger.debug(f'Factory action = {action}')
                actions[f_info.factory_id] = action
        return actions

    def _build_unit_action(self, info: FactoryInfo, desire: FactoryDesires) -> [None, int]:
        total_light_desired = desire.light_attacking + desire.light_clearing_rubble + desire.light_mining_ore
        total_heavy_desired = desire.heavy_attacking + desire.heavy_mining_ice
        if info.factory.power > 2000 and info.factory.can_build_heavy(self.master.game_state) and info.num_heavy < total_heavy_desired:
            return info.factory.build_heavy()
        elif info.factory.power > 1000 and info.factory.can_build_light(self.master.game_state) and info.num_light < total_light_desired:
            return info.factory.build_light()
        else:
            return None

    def _water(self, info: FactoryInfo, desire: FactoryDesires) -> [None, int]:
        if info.water - info.water_cost > 100:
            return info.factory.water()
        return None

