from __future__ import annotations
from typing import Dict, TYPE_CHECKING

import numpy as np

from master_plan import MasterPlan, Planner, Recommendation
from lux.kit import obs_to_game_state, GameState
from lux.factory import Factory

if TYPE_CHECKING:
    from unit_manager import UnitManager


class FactoryManager:
    def __init__(self, factory: Factory, master_plan: MasterPlan):
        self.unit_id = factory.unit_id
        self.factory = factory
        self.master_plan = master_plan

    @staticmethod
    def place_factory(game_state: GameState, player):
        """Place factory in early_setup"""

        # how many factories you have left to place
        factories_to_place = game_state.teams[player].factories_to_place

        # how much water and metal you have in your starting pool to give to new factories
        water_left = game_state.teams[player].water
        metal_left = game_state.teams[player].metal

        # spawn in random location with default resources
        potential_spawns = np.array(
            list(zip(*np.where(game_state.board.valid_spawns_mask == 1)))
        )
        spawn_loc = potential_spawns[np.random.randint(0, len(potential_spawns))]
        return dict(spawn=spawn_loc, metal=150, water=150)

    def update(self, factory: Factory):
        self.factory = factory

    # def assign_unit(self, unit: UnitManager):
    #     self.master_plan.assign_unit_factory(unit.unit_id, factory_id=self.unit_id)
    #
    # def deassign_unit(self, unit: UnitManager):
    #     self.master_plan.deassign_unit_factory(unit.unit_id, factory_id=self.unit_id)


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
