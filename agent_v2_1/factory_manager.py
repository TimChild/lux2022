from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict
import numpy as np

from lux.factory import Factory

from config import get_logger
from master_state import MasterState
import util

import actions

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManger

    pass

logger = get_logger(__name__)


class BuildHeavyRecommendation(actions.Recommendation):
    role = "heavy"
    value = 0

    def to_action_queue(self, plan: MasterState) -> int:
        return 1


@dataclass
class UnitActions:
    mining_ice: Dict[str, FriendlyUnitManger]
    mining_ore: Dict[str, FriendlyUnitManger]
    clearing_rubble: Dict[str, FriendlyUnitManger]
    attacking: Dict[str, FriendlyUnitManger]
    nothing: Dict[str, FriendlyUnitManger]


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
        logger.info(f"dead, nothing else to do")


class FriendlyFactoryManager(FactoryManager):
    def __init__(self, factory: Factory, master_state: MasterState):
        super().__init__(factory)
        self.master = master_state

        self.light_units: Dict[str, FriendlyUnitManger] = {}
        self.heavy_units: Dict[str, FriendlyUnitManger] = {}

        # caching
        self._light_actions = None
        self._heavy_actions = None

    def update(self, factory: Factory):
        super().update(factory)
        self._light_actions = None
        self._heavy_actions = None

    def assign_unit(self, unit: FriendlyUnitManger):
        logger.debug(f"Assigning {unit.log_prefix} to {self.factory.unit_id}")
        if unit.unit_type == "LIGHT":
            self.light_units[unit.unit_id] = unit
        elif unit.unit_id == "HEAVY":
            self.heavy_units[unit.unit_id] = unit

    def get_light_actions(self) -> UnitActions:
        if self._light_actions is None:
            self._light_actions = self._get_actions(self.light_units)
        return self._light_actions

    def get_heavy_actions(self) -> UnitActions:
        if self._heavy_actions is None:
            self._heavy_actions = self._get_actions(self.heavy_units)
        return self._heavy_actions

    def _get_actions(self, units: Dict[str, FriendlyUnitManger]) -> UnitActions:
        return UnitActions(
            mining_ice={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action == actions.MINE_ICE
            },
            mining_ore={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action == actions.MINE_ORE
            },
            clearing_rubble={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action == actions.CLEAR_RUBBLE
            },
            attacking={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action == actions.ATTACK
            },
            nothing={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action == actions.NOTHING
            },
        )

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
        logger.info(f"dead, looking for assigned units")

        for unit_id, unit in self.master.units.friendly.all.items():
            if unit.factory_id == self.unit_id:
                logger.info(f"Removing {self.unit_id} assignment for unit {unit_id}")
                unit.factory_id = None
