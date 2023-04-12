from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
import pandas as pd

from lux.kit import GameState
from lux.factory import Factory

from config import get_logger
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
    pass

logger = get_logger(__name__)


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
    def __init__(self, factory: Factory):
        self.unit_id = factory.unit_id
        self.factory = factory

    def update(self, factory: Factory):
        self.factory = factory

    @property
    def pos(self):
        return self.factory.pos


class EnemyFactoryManager(FactoryManager):
    def dead(self):
        logger.info(f'dead, nothing else to do')


class FriendlyFactoryManager(FactoryManager):
    def __init__(self, factory: Factory, master_state: MasterState):
        super().__init__(factory)
        self.master = master_state

        self.light_units = {}
        self.heavy_units = {}

    @property
    def factory_loc(self) -> np.ndarray:
        """Return an array with shape of map with 1s where factory is"""
        arr = np.zeros_like(self.master.maps.rubble, dtype=int)
        arr[self.factory.pos_slice] = 1
        return arr

    @property
    def power(self) -> int:
        return self.factory.power

    @staticmethod
    def place_factory(game_state: GameState, player):
        """Place factory in early_setup"""
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

    def dead(self):
        """Called when factory is detected as dead"""
        logger.info(f'dead, looking for assigned units')

        for unit_id, unit in self.master.units.friendly.all.items():
            if unit.factory_id == self.unit_id:
                logger.info(f'Removing {self.unit_id} assignment for unit {unit_id}')
                unit.factory_id = None
