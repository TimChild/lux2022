from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict
import numpy as np

from lux.factory import Factory

from config import get_logger
from master_state import MasterState
import util

from actions_util import Actions

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManager

    pass

logger = get_logger(__name__)


@dataclass
class UnitActions:
    mining_ice: Dict[str, FriendlyUnitManager]
    mining_ore: Dict[str, FriendlyUnitManager]
    clearing_rubble: Dict[str, FriendlyUnitManager]
    attacking: Dict[str, FriendlyUnitManager]
    nothing: Dict[str, FriendlyUnitManager]


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

        self.light_units: Dict[str, FriendlyUnitManager] = {}
        self.heavy_units: Dict[str, FriendlyUnitManager] = {}

        # Keep track of some values that will change during planning
        self._power = 0
        self.short_term_power = 0

        # caching
        self._light_actions = None
        self._heavy_actions = None

    @property
    def power(self):
        """This is just the start of turn power for now
        Use factory.short_term_power to get expected power in the short term
        """
        return self._power

    @property
    def lichen_id(self) -> int:
        """id num of lichen strain"""
        return self.factory.strain_id

    @property
    def own_lichen(self) -> np.ndarray:
        """map array of amount of own lichen"""
        lichen_locations = self.master.maps.lichen_strains == self.lichen_id
        lichen = self.master.maps.lichen.copy()
        lichen[lichen_locations != 1] = 0
        return lichen

    def update(self, factory: Factory):
        super().update(factory)
        self._light_actions = None
        self._heavy_actions = None
        self._power = factory.power
        self.short_term_power = self._calculate_start_of_turn_short_term_power()

    def _calculate_start_of_turn_short_term_power(self):
        """
        For now, assuming all assigned units only pickup power at own factory, and any power pickup is valid
        """
        short_power = self.power
        for unit_id, unit in dict(**self.light_units, **self.heavy_units).items():
            unit: FriendlyUnitManager
            actions = unit.status.planned_actions
            for action in actions[:2]:
                if action[util.ACT_TYPE] == util.PICKUP and action[util.ACT_RESOURCE] == util.POWER:
                    short_power -= action[util.ACT_AMOUNT]
        return short_power


    def assign_unit(self, unit: FriendlyUnitManager):
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

    @property
    def factory_loc(self) -> np.ndarray:
        """Return an array with shape of map with 1s where factory is"""
        arr = np.zeros_like(self.master.maps.rubble, dtype=int)
        arr[self.factory.pos_slice] = 1
        return arr

    def dead(self):
        """Called when factory is detected as dead"""
        logger.info(f"dead, looking for assigned units")

        for unit_id, unit in self.master.units.friendly.all.items():
            if unit.factory_id == self.unit_id:
                logger.info(f"Removing {self.unit_id} assignment for unit {unit_id}")
                unit.factory_id = None

    def _get_actions(self, units: Dict[str, FriendlyUnitManager]) -> UnitActions:
        return UnitActions(
            mining_ice={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action == Actions.MINE_ICE
            },
            mining_ore={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action == Actions.MINE_ORE
            },
            clearing_rubble={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action == Actions.CLEAR_RUBBLE
            },
            attacking={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action == Actions.ATTACK
            },
            nothing={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action == Actions.NOTHING
            },
        )
