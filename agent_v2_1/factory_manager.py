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
    from unit_manager import FriendlyUnitManger
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

    def assign_unit(self, unit: FriendlyUnitManger):
        logger.debug(f'Assigning {unit.log_prefix} to {self.factory.unit_id}')
        if unit.unit_type == 'LIGHT':
            self.light_units[unit.unit_id] = unit
        elif unit.unit_id == 'HEAVY':
            self.heavy_units[unit.unit_id] = unit

    @property
    def factory_loc(self) -> np.ndarray:
        """Return an array with shape of map with 1s where factory is"""
        arr = np.zeros_like(self.master.maps.rubble, dtype=int)
        arr[self.factory.pos_slice] = 1
        return arr

    @property
    def power(self) -> int:
        return self.factory.power


    def dead(self):
        """Called when factory is detected as dead"""
        logger.info(f'dead, looking for assigned units')

        for unit_id, unit in self.master.units.friendly.all.items():
            if unit.factory_id == self.unit_id:
                logger.info(f'Removing {self.unit_id} assignment for unit {unit_id}')
                unit.factory_id = None
