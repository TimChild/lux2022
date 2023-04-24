from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Tuple
import numpy as np

import actions_util

from lux.cargo import UnitCargo
from unit_status import ActCategory, MineActSubCategory, ClearActSubCategory, CombatActSubCategory
from lux.factory import Factory

from config import get_logger
from master_state import MasterState
import util


if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManager

    pass

logger = get_logger(__name__)


@dataclass
class UnitActions:
    mining_ice: Dict[str, FriendlyUnitManager]
    mining_ore: Dict[str, FriendlyUnitManager]
    clearing_rubble: Dict[str, FriendlyUnitManager]
    clearing_lichen: Dict[str, FriendlyUnitManager]
    attacking: Dict[str, FriendlyUnitManager]
    waiting: Dict[str, FriendlyUnitManager]
    nothing: Dict[str, FriendlyUnitManager]


class FactoryManager:
    def __init__(self, factory: Factory, map_shape: Tuple):
        self.unit_id = factory.unit_id
        self.factory = factory

        # Distance from factory
        self.dist_array: np.ndarray = util.pad_and_crop(
            util.stretch_middle_of_factory_array(util.manhattan_kernel(47)), map_shape, self.pos[0], self.pos[1]
        )


    def update(self, factory: Factory):
        self.factory = factory

    @property
    def pos(self):
        return self.factory.pos

    @property
    def cargo(self) -> UnitCargo:
        return self.factory.cargo


class EnemyFactoryManager(FactoryManager):
    def dead(self):
        logger.info(f"dead, nothing else to do")


@dataclass
class FactoryInfo:
    factory_id: str
    num_heavy: int
    num_light: int
    water_cost: int
    connected_growable_space: int
    num_lichen_tiles: int
    total_lichen: int

    # TODO: what about these?
    light_mining_ore: int
    light_mining_ice: int
    light_clearing_rubble: int
    light_clearing_lichen: int
    light_attacking: int
    heavy_mining_ice: int
    heavy_mining_ore: int
    heavy_clearing_rubble: int
    heavy_clearing_lichen: int
    heavy_attacking: int

    @classmethod
    def init(
        cls,
        master: MasterState,
        factory: FriendlyFactoryManager,
    ) -> FactoryInfo:
        current_light_actions = factory.get_light_actions()
        current_heavy_actions = factory.get_heavy_actions()

        lichen = factory.own_lichen
        connected_zeros = util.connected_array_values_from_pos(master.maps.rubble, factory.pos, connected_value=0)
        # ID array of other lichen
        other_lichen = master.maps.lichen_strains.copy()
        # Set own lichen to -1 (like no lichen)
        other_lichen[other_lichen == factory.lichen_id] = -1
        # Zero rubble, and no other lichen, no ice, no ore, not under factory
        connected_growable_space = int(
            np.sum(
                (connected_zeros > 0)
                & (other_lichen < 0)
                & (master.maps.ice == 0)
                & (master.maps.ore == 0)
                & (master.maps.factory_maps.all < 0)
            )
        )

        factory_info = cls(
            factory_id=factory.unit_id,
            num_heavy=len(factory.heavy_units),
            num_light=len(factory.light_units),
            water_cost=factory.factory.water_cost(master.game_state),
            connected_growable_space=connected_growable_space,
            num_lichen_tiles=int(np.sum(lichen > 0)),
            total_lichen=int(np.sum(lichen)),
            light_mining_ore=len(current_light_actions.mining_ore),
            light_mining_ice=len(current_light_actions.mining_ice),
            light_clearing_rubble=len(current_light_actions.clearing_rubble),
            light_clearing_lichen=len(current_light_actions.clearing_lichen),
            light_attacking=len(current_light_actions.attacking),
            heavy_mining_ice=len(current_heavy_actions.mining_ice),
            heavy_mining_ore=len(current_heavy_actions.mining_ore),
            heavy_clearing_rubble=len(current_heavy_actions.clearing_rubble),
            heavy_clearing_lichen=len(current_heavy_actions.clearing_lichen),
            heavy_attacking=len(current_heavy_actions.attacking),
        )
        return factory_info

    def remove_or_add_unit_action(self, unit: FriendlyUnitManager, remove_or_add="remove"):
        if not unit.factory_id == self.factory_id:
            logger.error(
                f"Trying to update factory_info ({self.factory_id}) with unit that has factory id ({unit.factory_id})"
            )
            return None
        logger.info(
            f"Removing {unit.unit_id} assignment of {unit.status.current_action} from factory_info count ({self.factory_id})"
        )
        if remove_or_add == "remove":
            amount = -1
        elif remove_or_add == "add":
            amount = 1
        else:
            raise ValueError(f'got {remove_or_add}, must be "remove" or "add"')

        category = unit.status.current_action.category
        sub_category = unit.status.current_action.sub_category

        if unit.unit_type == "HEAVY":
            if category == ActCategory.MINE and sub_category == MineActSubCategory.ICE:
                self.heavy_mining_ice += amount
            elif category == ActCategory.MINE and sub_category == MineActSubCategory.ORE:
                self.heavy_mining_ore += amount
            elif category == ActCategory.COMBAT:
                self.heavy_attacking += amount
            elif category == ActCategory.CLEAR and sub_category == ClearActSubCategory.RUBBLE:
                self.heavy_clearing_rubble += amount
            elif category == ActCategory.CLEAR and sub_category == ClearActSubCategory.LICHEN:
                self.heavy_clearing_lichen += amount
            else:
                raise NotImplementedError(f"{unit.status.current_action} not implemented")
        else:
            if category == ActCategory.MINE and sub_category == MineActSubCategory.ICE:
                self.light_mining_ice += amount
            elif category == ActCategory.MINE and sub_category == MineActSubCategory.ORE:
                self.light_mining_ore += amount
            elif category == ActCategory.COMBAT:
                self.light_attacking += amount
            elif category == ActCategory.CLEAR and sub_category == ClearActSubCategory.RUBBLE:
                self.light_clearing_rubble += amount
            elif category == ActCategory.CLEAR and sub_category == ClearActSubCategory.LICHEN:
                self.light_clearing_lichen += amount
            else:
                raise NotImplementedError(f"{unit.status.current_action} not implemented")

    @staticmethod
    def _get_connected_zeros(rubble: np.ndarray, factory_pos: util.POS_TYPE):
        return


class FriendlyFactoryManager(FactoryManager):
    def __init__(self, factory: Factory, master: MasterState):
        super().__init__(factory, master.maps.rubble.shape)
        self.master = master
        self.info: FactoryInfo = None

        self.light_units: Dict[str, FriendlyUnitManager] = {}
        self.heavy_units: Dict[str, FriendlyUnitManager] = {}

        self.queue_array: np.ndarray = self._generate_waiting_area()

        # Keep track of some values that will change during planning
        self._power = 0
        self.short_term_power = 0

        # caching
        self._light_actions = None
        self._heavy_actions = None

    def update(self, factory: Factory):
        super().update(factory)
        self._light_actions = None
        self._heavy_actions = None
        self._power = factory.power
        self.short_term_power = self.calculate_power_at_step()
        self.info = FactoryInfo.init(self.master, self)

    @property
    def water_cost(self):
        owned_lichen_tiles = (self.master.maps.lichen_strains == self.factory.strain_id).sum()
        return np.ceil(owned_lichen_tiles / self.master.env_cfg.LICHEN_WATERING_COST_FACTOR)

    def generate_circle_array(self, center: util.POS_TYPE, radius: int, num: int):
        return util.generate_circle_coordinates_array(center, num, radius, self.master.maps.rubble.shape[0])

    def _generate_waiting_area(self) -> np.ndarray:
        arr = functools.reduce(
            np.logical_or,
            [
                self.generate_circle_array(self.pos, radius=r, num=n)
                for r, n in zip([3, 4, 5, 6, 7], [8, 8, 14, 16, 14])
            ],
        ).astype(int)
        # Block out resources from waiting areas
        resources = self.master.maps.ice & self.master.maps.ore
        arr[resources > 0] = 0
        return arr

    @property
    def power(self):
        """This is just the start of turn power for now
        Use factory.calculate_power_at_step(step) to get expected power
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

    def calculate_power_at_step(self, step: int = 1):
        """
        For now, assuming all assigned units only pickup power at own factory, and any power pickup is valid
        """
        power_at_step = self.power
        for unit_id, unit in dict(**self.light_units, **self.heavy_units).items():
            unit: FriendlyUnitManager
            actions = unit.status.planned_action_queue
            actions = actions_util.split_actions_at_step(actions, step)[0]
            for action in actions:
                if action[util.ACT_TYPE] == util.PICKUP and action[util.ACT_RESOURCE] == util.POWER:
                    power_at_step -= action[util.ACT_AMOUNT]
        return power_at_step

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
                if unit.status.current_action.category == ActCategory.MINE
                and unit.status.current_action.sub_category == MineActSubCategory.ICE
            },
            mining_ore={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action.category == ActCategory.MINE
                and unit.status.current_action.sub_category == MineActSubCategory.ORE
            },
            clearing_rubble={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action.category == ActCategory.CLEAR
                and unit.status.current_action.sub_category == ClearActSubCategory.RUBBLE
            },
            clearing_lichen={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action.category == ActCategory.CLEAR
                and unit.status.current_action.sub_category == ClearActSubCategory.LICHEN
            },
            attacking={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action.category == ActCategory.COMBAT
            },
            waiting={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action.category == ActCategory.WAITING
            },
            nothing={
                unit.unit_id: unit
                for unit_id, unit in units.items()
                if unit.status.current_action.category == ActCategory.NOTHING
            },
        )
