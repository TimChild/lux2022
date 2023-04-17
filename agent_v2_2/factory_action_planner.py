from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, TYPE_CHECKING
import copy

from lux.kit import GameState
from lux.factory import Factory

from factory_manager import FriendlyFactoryManager
import actions

from master_state import MasterState
import util
from config import get_logger

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManager

logger = get_logger(__name__)

BUILD_LIGHT = 0
BUILD_HEAVY = 1
WATER = 2


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
    connected_growable_space: int
    num_lichen_tiles: int
    total_lichen: int

    # Below should be equivalent to factory desires
    light_mining_ore: int
    light_mining_ice: int
    light_clearing_rubble: int
    light_attacking: int
    heavy_mining_ice: int
    heavy_mining_ore: int
    heavy_clearing_rubble: int
    heavy_attacking: int

    @classmethod
    def init(
        cls,
        master: MasterState,
        factory: FriendlyFactoryManager,
        previous: [None, FactoryInfo] = None,
    ) -> FactoryInfo:
        current_light_actions = factory.get_light_actions()
        current_heavy_actions = factory.get_heavy_actions()

        lichen = factory.own_lichen
        connected_zeros = util.connected_array_values_from_pos(
            master.maps.rubble, factory.pos, connected_value=0
        )
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
            water_cost=factory.factory.water_cost(master.game_state),
            connected_growable_space=connected_growable_space,
            num_lichen_tiles=int(np.sum(lichen > 0)),
            total_lichen=int(np.sum(lichen)),
            light_mining_ore=len(current_light_actions.mining_ore),
            light_mining_ice=len(current_light_actions.mining_ice),
            light_clearing_rubble=len(current_light_actions.clearing_rubble),
            light_attacking=len(current_light_actions.attacking),
            heavy_mining_ice=len(current_heavy_actions.mining_ice),
            heavy_mining_ore=len(current_heavy_actions.mining_ore),
            heavy_clearing_rubble=len(current_heavy_actions.clearing_rubble),
            heavy_attacking=len(current_heavy_actions.attacking),
        )

        if previous:
            cls.update_averages(factory_info, previous)
        return factory_info

    @staticmethod
    def update_averages(new_info: FactoryInfo, previous_info: FactoryInfo):
        # Average some things
        # Roughly like average of last 10 values
        # new_info.power = int(previous_info.power * 0.9 + 0.1 * new_info.power)
        # new_info.water = int(previous_info.water * 0.9 + 0.1 * new_info.water)
        # new_info.ice = int(previous_info.ice * 0.9 + 0.1 * new_info.ice)
        # new_info.ore = int(previous_info.ore * 0.9 + 0.1 * new_info.ore)
        # new_info.metal = int(previous_info.metal * 0.9 + 0.1 * new_info.metal)
        # new_info.water_cost = int(previous_info.water_cost * 0.9 + 0.1 * new_info.water_cost)
        pass

    def remove_unit_from_current_count(self, unit: FriendlyUnitManager):
        if not unit.factory_id == self.factory_id:
            logger.error(
                f"Trying to update factory_info ({self.factory_id}) with unit that has factory id ({unit.factory_id})"
            )
            return None
        logger.info(
            f"Removing {unit.unit_id} assignment of {unit.status.current_action} from factory_info count ({self.factory_id})"
        )
        if unit.unit_type == "HEAVY":
            if unit.status.current_action == actions.MINE_ICE:
                self.heavy_mining_ice -= 1
            elif unit.status.current_action == actions.MINE_ORE:
                self.heavy_mining_ore -= 1
            elif unit.status.current_action == actions.ATTACK:
                self.heavy_attacking -= 1
        else:
            if unit.status.current_action == actions.MINE_ORE:
                self.light_mining_ore -= 1
            elif unit.status.current_action == actions.CLEAR_RUBBLE:
                self.light_clearing_rubble -= 1
            elif unit.status.current_action == actions.ATTACK:
                self.light_attacking -= 1

    @staticmethod
    def _get_connected_zeros(rubble: np.ndarray, factory_pos: util.POS_TYPE):
        return


@dataclass
class FactoryDesires:
    heavy_mining_ice: int = 1
    heavy_mining_ore: int = 0
    heavy_clearing_rubble: int = 0
    heavy_attacking: int = 0
    light_mining_ore: int = 1
    light_mining_ice: int = 0
    light_clearing_rubble: int = 0
    light_attacking: int = 0

    def total_light(self):
        return (
            self.light_mining_ice
            + self.light_mining_ore
            + self.light_clearing_rubble
            + self.light_attacking
        )

    def total_heavy(self):
        return (
            self.heavy_mining_ice
            + self.heavy_mining_ore
            + self.heavy_clearing_rubble
            + self.heavy_attacking
        )

    def copy(self) -> FactoryDesires:
        return copy.copy(self)

    def update_desires(
        self,
        info: FactoryInfo,
        light_energy_consideration=1000,
        light_rubble_min_tiles=10,
        light_rubble_max_tiles=50,
        light_rubble_max_num=2,
        light_metal_min=50,
        light_metal_max=200,
        light_metal_max_num=5,
        light_water_min=50,
        light_water_max=500,
        light_water_max_num=3,
        light_attack_max_num=2,
        heavy_energy_consideration=1000,
        heavy_rubble_min_tiles=30,
        heavy_rubble_max_tiles=50,
        heavy_rubble_max_num=2,
        heavy_metal_min=100,
        heavy_metal_max=300,
        heavy_metal_max_num=5,
        heavy_water_min=200,
        heavy_water_max=2500,
        heavy_water_max_num=3,
        heavy_attack_max_num=2,
    ):
        # Consider more LIGHT units
        power_req_met = info.power > light_energy_consideration
        # Rubble
        expansion_tiles = info.connected_growable_space - info.num_lichen_tiles
        if expansion_tiles < light_rubble_min_tiles and power_req_met:
            self.light_clearing_rubble = min(
                light_rubble_max_num, info.light_clearing_rubble + 1
            )
        elif expansion_tiles > light_rubble_max_tiles:
            self.light_clearing_rubble = max(0, info.light_clearing_rubble - 1)

        # Ore
        if info.metal < light_metal_min and power_req_met:
            self.light_mining_ore = min(light_metal_max_num, info.light_mining_ore + 1)
        elif info.metal > light_metal_max:
            self.light_mining_ore = max(0, info.light_mining_ore - 1)

        # Ice
        if info.water < light_water_min and power_req_met:
            self.light_mining_ice = min(light_water_max_num, info.light_mining_ice + 1)
        elif info.water > light_water_max:
            self.light_mining_ice = max(0, info.light_mining_ice - 1)

        # Attack
        if info.num_light == self.total_light():
            self.light_attacking = max(light_attack_max_num, info.light_attacking+1)
        elif info.num_light < self.total_heavy():
            self.light_attacking = max(0, info.light_attacking-1)


        power_req_met = info.power > heavy_energy_consideration

        # Consider more HEAVY units
        # Ice
        if info.water < heavy_water_min and power_req_met:
            self.heavy_mining_ice = min(heavy_water_max_num, info.heavy_mining_ice + 1)
        elif info.water > heavy_water_max:
            self.heavy_mining_ice = max(0, info.heavy_mining_ice - 1)

        # Ore
        if info.metal < heavy_metal_min and power_req_met:
            self.heavy_mining_ore = min(heavy_metal_max_num, info.heavy_mining_ore + 1)
        elif info.metal > heavy_metal_max:
            self.heavy_mining_ore = max(0, info.heavy_mining_ore - 1)

        # Rubble
        expansion_tiles = info.connected_growable_space - info.num_lichen_tiles
        if expansion_tiles < heavy_rubble_min_tiles and power_req_met:
            self.heavy_clearing_rubble = min(
                heavy_rubble_max_num, info.heavy_clearing_rubble + 1
            )
        elif expansion_tiles > heavy_rubble_max_tiles:
            self.heavy_clearing_rubble = max(0, info.heavy_clearing_rubble - 1)

        # Attack
        if info.num_heavy == self.total_heavy():
            self.heavy_attacking = max(heavy_attack_max_num, info.heavy_attacking+1)
        elif info.num_heavy < self.total_heavy():
            self.heavy_attacking = max(0, info.heavy_attacking-1)


class FactoryActionPlanner:
    def __init__(self, master: MasterState):
        self.master = master
        self.factories = self.master.factories.friendly

        # Caching variables
        self._factory_desires: Dict[str, FactoryDesires] = {}
        self._factory_infos: Dict[str, FactoryInfo] = {}

    def update(self, *args, **kwargs):
        """Calculate the general desires of the factories
        - How many heavy units mining ice
        - How many light units mining ore
        - How many light units clearing rubble
        - How many light units attacking
        - How many heavy units attacking
        - etc
        """
        logger.info(f"Updating FactoryActionPlanner")
        # Remove any dead factories from lists
        for k in set(self._factory_desires.keys()) - set(
            self.master.factories.friendly.keys()
        ):
            logger.info(f"Removing factory {k}, assumed dead")
            self._factory_desires.pop(k)
            self._factory_infos.pop(k)

        # Update their infos
        self._update_factory_info()

        # Calculate new desires for turn and then every X after that
        if self.master.step == 0 or self.master.step % 10 == 0:
            self._update_factory_desires()

    def get_factory_desires(self) -> Dict[str, FactoryDesires]:
        return self._factory_desires

    def get_factory_infos(self) -> Dict[str, FactoryInfo]:
        return self._factory_infos

    def _update_factory_info(self):
        """Update info about the factory (uses a sort of rolling average in updating)"""
        for f_id, factory in self.master.factories.friendly.items():
            logger.debug(f"Updating {f_id} info")
            self._factory_infos[f_id] = FactoryInfo.init(
                master=self.master,
                factory=factory,
                previous=self._factory_infos.pop(f_id, None),
            )

    def _update_factory_desires(self):
        """Update the desires for each factory"""
        for f_id, factory in self.master.factories.friendly.items():
            logger.debug(f"Updating {f_id} desires")
            # Pop because updating
            previous_desire = self._factory_desires.pop(f_id, None)
            # Get because only using
            info = self._factory_infos.get(f_id, None)
            # Update taking into account previous preferences and info
            if previous_desire and info:
                desires = self._update_single_factory_desire(previous_desire, info)
            # Update without knowing previous preferences (first turn for this factory)
            else:
                desires = FactoryDesires()

            self._factory_desires[f_id] = desires

    def _update_single_factory_desire(
        self, prev_desire: FactoryDesires, info: FactoryInfo
    ) -> FactoryDesires:
        """Update the desires of the factory based on it's current state and it's previous desires"""
        step = self.master.step
        desires = prev_desire.copy()
        logger.debug(
            f"Desires before HeavyIce={desires.heavy_mining_ice}, LightRubble={desires.light_clearing_rubble}, LightOre={desires.light_mining_ore}, LightAttack={desires.light_attacking}, HeavyAttack={desires.heavy_attacking}"
        )
        # Early game
        if step < 200:
            desires.update_desires(
                info=info,
                light_energy_consideration=300,
                light_rubble_min_tiles=10,
                light_rubble_max_tiles=50,
                light_rubble_max_num=4,
                light_metal_min=50,
                light_metal_max=200,
                light_metal_max_num=2,
                light_water_min=20,
                light_water_max=100,
                light_water_max_num=3,
                light_attack_max_num=2,
                heavy_energy_consideration=1000,
                heavy_rubble_min_tiles=30,
                heavy_rubble_max_tiles=50,
                heavy_rubble_max_num=1,
                heavy_metal_min=100,
                heavy_metal_max=300,
                heavy_metal_max_num=2,
                heavy_water_min=200,
                heavy_water_max=500,
                heavy_water_max_num=2,
                heavy_attack_max_num=1,
            )
        # Early mid game
        elif step < 500:
            desires.update_desires(
                info=info,
                light_energy_consideration=600,
                light_rubble_min_tiles=10,
                light_rubble_max_tiles=50,
                light_rubble_max_num=6,
                light_metal_min=50,
                light_metal_max=200,
                light_metal_max_num=1,
                light_water_min=100,
                light_water_max=300,
                light_water_max_num=3,
                light_attack_max_num=5,
                heavy_energy_consideration=500,
                heavy_rubble_min_tiles=20,
                heavy_rubble_max_tiles=40,
                heavy_rubble_max_num=2,
                heavy_metal_min=100,
                heavy_metal_max=300,
                heavy_metal_max_num=2,
                heavy_water_min=500,
                heavy_water_max=1500,
                heavy_water_max_num=3,
                heavy_attack_max_num=2,
            )
        # mid late game
        elif step < 800:
            desires.update_desires(
                info=info,
                light_energy_consideration=1000,
                light_rubble_min_tiles=10,
                light_rubble_max_tiles=30,
                light_rubble_max_num=5,
                light_metal_min=100,
                light_metal_max=200,
                light_metal_max_num=0,
                light_water_min=300,
                light_water_max=1000,
                light_water_max_num=3,
                light_attack_max_num=10,
                heavy_energy_consideration=800,
                heavy_rubble_min_tiles=5,
                heavy_rubble_max_tiles=30,
                heavy_rubble_max_num=1,
                heavy_metal_min=100,
                heavy_metal_max=300,
                heavy_metal_max_num=1,
                heavy_water_min=1000,
                heavy_water_max=2500,
                heavy_water_max_num=4,
                heavy_attack_max_num=3,
            )
        else:
            desires.update_desires(
                info=info,
                light_energy_consideration=500,
                light_rubble_min_tiles=10,
                light_rubble_max_tiles=20,
                light_rubble_max_num=10,
                light_metal_min=100,
                light_metal_max=200,
                light_metal_max_num=0,
                light_water_min=500,
                light_water_max=1000,
                light_water_max_num=5,
                light_attack_max_num=15,
                heavy_energy_consideration=1500,
                heavy_rubble_min_tiles=5,
                heavy_rubble_max_tiles=15,
                heavy_rubble_max_num=2,
                heavy_metal_min=0,
                heavy_metal_max=300,
                heavy_metal_max_num=0,
                heavy_water_min=500,
                heavy_water_max=2500,
                heavy_water_max_num=5,
                heavy_attack_max_num=5,
            )
        logger.debug(
            f"Desires after HeavyIce={desires.heavy_mining_ice}, LightRubble={desires.light_clearing_rubble}, LightOre={desires.light_mining_ore}, LightAttack={desires.light_attacking}, HeavyAttack={desires.heavy_attacking}"
        )
        return desires

    def decide_factory_actions(self) -> Dict[str, int]:
        actions = {}
        for f_info, f_desire in zip(
            self._factory_infos.values(), self._factory_desires.values()
        ):
            logger.debug(f"Deciding factory actions for {f_info.factory_id}")
            f_info: FactoryInfo
            f_desire: FactoryDesires
            center_occupied = (
                True
                if self.master.units.friendly.unit_at_position(f_info.factory.pos)
                is not None
                else False
            )
            action = None

            # Only consider building if center not occupied
            if not center_occupied:
                logger.debug(f"Center not occupied, considering building unit")
                action = self._build_unit_action(f_info, f_desire)

            # If not building unit, consider watering
            if action is None:
                logger.debug(f"Not building unit, considering watering")
                action = self._water(f_info, f_desire)

            if action is not None:
                logger.debug(f"Factory action = {action}")
                actions[f_info.factory_id] = action
        return actions

    def _build_unit_action(
        self, info: FactoryInfo, desire: FactoryDesires
    ) -> [None, int]:
        total_light_desired = desire.total_light()
        total_heavy_desired = desire.total_heavy()

        can_build_light = info.factory.can_build_light(self.master.game_state)
        can_build_heavy = info.factory.can_build_heavy(self.master.game_state)

        logger.debug(
            f"can build heavy={can_build_heavy}, can_build_light={can_build_light}"
        )
        logger.debug(
            f"desired_heavy={total_heavy_desired}, desired_light={total_light_desired}"
        )
        logger.debug(f"current_heavy={info.num_heavy}, current_light={info.num_light}")

        if can_build_heavy:
            if total_heavy_desired > info.num_heavy:
                return info.factory.build_heavy()
        elif can_build_light:
            if total_light_desired > info.num_light:
                return info.factory.build_light()

    def _water(self, info: FactoryInfo, desire: FactoryDesires) -> [None, int]:
        min_water = 100
        if self.master.step <= 300:
            min_water = 200
        elif self.master.step <= 500:
            min_water = 800
        elif self.master.step <= 850:
            min_water = 1500
        elif self.master.step <= 1000:
            min_water = 150
        logger.debug(
            f"Current water = {info.water}, water cost = {info.water_cost}, min_water = {min_water}"
        )
        if info.water - info.water_cost > min_water:
            return info.factory.water()
        return None

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
        df = pd.DataFrame(potential_spawns, columns=["x", "y"])
        df["pos"] = df.apply(lambda row: (row.x, row.y), axis=1)

        # Find distance to ice
        ice = game_state.board.ice
        df["ice_dist"] = df.apply(
            lambda row: util.manhattan(row.pos, util.nearest_non_zero(ice, row.pos)),
            axis=1,
        )
        df = df.sort_values("ice_dist")

        # Keep only top X distance to ice
        df = df.iloc[:20]

        # Value based nearby zero-rubble
        rubble = game_state.board.rubble
        count_arr = util.count_connected_values(rubble, value=0)
        kernel = util.factory_map_kernel(5, dist_multiplier=0.6)
        conv_count_arr = util.convolve_array_kernel(count_arr, kernel)
        df["zero_rubble_value"] = df.apply(
            lambda row: (conv_count_arr[row.x, row.y]), axis=1
        )
        df = df.sort_values(["ice_dist", "zero_rubble_value"], ascending=[True, False])

        # TODO: Calculate how close to nearest enemy factory for top X
        # TODO: Calculate how close to nearest ore (not so important?)

        return dict(spawn=df.iloc[0].pos, metal=150, water=150)
