from __future__ import annotations
from typing import Dict, TYPE_CHECKING
import numpy as np
import pandas as pd
import logging

from lux.kit import obs_to_game_state, GameState
from lux.factory import Factory

from master_state import MasterState
from util import (
    manhattan,
    nearest_non_zero,
    convolve_array_kernel,
    factory_map_kernel,
    count_connected_values,
)

from actions import Recommendation

if TYPE_CHECKING:
    from unit_manager import UnitManager


class BuildHeavyRecommendation(Recommendation):
    role = 'heavy'
    value = 0

    def to_action_queue(self, plan: MasterState) -> int:
        return 1


# class FactoryRecommendation(Recommendation):
#     role = 'factory ice'
#
#     def __init__(self, value: int):
#         self.value = value
#
#
# class FactoryPlanner(Planner):
#     def recommend(self, unit: FactoryManager):
#         if unit.factory.power > 1000:
#             pass
#
#
#         return FactoryRecommendation
#
#     def carry_out(self, unit: FactoryManager, recommendation: Recommendation):
#         pass
#
#     def update(self):
#         pass


class FactoryManager:
    def __init__(self, factory: Factory, master_state: MasterState):
        self.unit_id = factory.unit_id
        self.factory = factory

        self.light_units = {}
        self.heavy_units = {}

        self.master = master_state

    @staticmethod
    def place_factory(game_state: GameState, player):
        """Place factory in early_setup"""
        # TODO: Add some other considerations (i.e. near low rubble, far from enemy, near ore)

        # how many factories you have left to place
        factories_to_place = game_state.teams[player].factories_to_place

        # how much water and metal you have in your starting pool to give to new factories
        water_left = game_state.teams[player].water
        metal_left = game_state.teams[player].metal

        # All possible spawns
        potential_spawns = list(zip(*np.where(game_state.board.valid_spawns_mask == 1)))
        df = pd.DataFrame(potential_spawns, columns=['x', 'y'])
        df['pos'] = df.apply(lambda row: (row.x, row.y), axis=1)

        # Find distance to ice
        ice = game_state.board.ice
        df["ice_dist"] = df.apply(
            lambda row: manhattan(row.pos, nearest_non_zero(ice, row.pos)), axis=1
        )
        df = df.sort_values("ice_dist")

        # Keep only top X distance to ice
        df = df.iloc[:20]

        # Value based nearby zero-rubble
        rubble = game_state.board.rubble
        count_arr = count_connected_values(rubble, value=0)
        factory_map_kernel(2, dist_multiplier=0.5)
        kernel = factory_map_kernel(3, dist_multiplier=0.5)
        conv_count_arr = convolve_array_kernel(count_arr, kernel)
        df['zero_rubble_value'] = df.apply(
            lambda row: (conv_count_arr[row.x, row.y]), axis=1
        )
        df = df.sort_values(['ice_dist', 'zero_rubble_value'], ascending=[True, False])
        # return df

        # TODO: Calculate how close to nearest enemy factory for top X
        # TODO: Calculate how close to nearest ore (not so important?)

        return dict(spawn=df.iloc[0].pos, metal=150, water=150)

    def update(self, factory: Factory):
        self.factory = factory

    def log(self, message, level=logging.INFO):
        logging.log(
            level,
            f"Step {self.master.game_state.real_env_steps}, Factory {self.unit_id}: {message}",
        )

    def dead(self):
        """Called when factory is detected as dead"""
        factory_player = f'player_{self.factory.team_id}'
        if factory_player == self.master.player:
            self.log(f'Friendly factory {self.unit_id} dead, looking for units assigned')
            for unit_id, unit in self.master.units.friendly_units.items():
                if unit.factory_id == self.unit_id:
                    self.log(f'Removing {self.unit_id} assignment for unit {unit_id}')
                    unit.factory_id = None
        else:
            self.log(f'Enemy factory {self.unit_id} dead, nothing else to do')


