from __future__ import annotations

import random
import numpy as np
import pandas as pd
from dataclasses import dataclass, is_dataclass
from typing import Dict, TYPE_CHECKING, Optional
import copy

from factory_manager import FactoryInfo, FriendlyFactoryManager
from unit_status import ActCategory, MineActSubCategory, ClearActSubCategory, ActStatus
from lux.kit import GameState

from master_state import MasterState
import util
from config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

BUILD_LIGHT = 0
BUILD_HEAVY = 1
WATER = 2


def _to_df(data_dict):
    df = pd.DataFrame(data_dict, dtype=float).set_index("step")
    _first_row = df.iloc[[0]].copy()
    _first_row.index = [0]
    df = df.div(df.sum(axis=1), axis=0)
    df = pd.concat([df, _first_row]).sort_index()
    return df


#
# _desires_work_ratios_dict = {
#     "step": [20, 200, 500, 850, 1000],
#     "mine ore": [2, 1, 0.5, 0.2, 0],
#     "mine ice": [0.3, 1, 2, 4, 1],
#     "clear rubble": [0.5, 2, 2, 4, 2],
#     "clear lichen": [0, 0, 1, 2, 5],
#     "attack": [0, 1, 2, 3, 0.5],
#     "defend": [1, 1, 1, 1, 2],
#     # "transfer": [0, 0.5, 1, 2, 0],
#     "transfer": [0, 0, 0, 0, 0],
#     "waiting": [0, 0, 1, 2, 0],
# }
_desires_work_ratios_dict = {
    "step": [1000],
    "mine ore": [0],
    "mine ice": [1],
    "clear rubble": [0],
    "clear lichen": [0],
    "attack": [0],
    "defend": [0],
    # "transfer": [0, 0.5, 1, 2, 0],
    "transfer": [0],
    "waiting": [0],
}
DESIRED_WORK_RATIOS_DF = _to_df(_desires_work_ratios_dict)


@dataclass
class WorkRatios:
    """For holding the ratio of each unit desired at step"""

    mine_ore: float
    mine_ice: float
    clear_rubble: float
    clear_lichen: float
    attack: float
    defend: float
    transfer: float
    waiting: float

    @classmethod
    def at_step(cls, step: int):
        df = util.get_interpolated_values(DESIRED_WORK_RATIOS_DF, step)
        inst = cls(
            mine_ore=df["mine ore"],
            mine_ice=df["mine ice"],
            clear_rubble=df["clear rubble"],
            clear_lichen=df["clear lichen"],
            attack=df["attack"],
            defend=df["defend"],
            transfer=df["transfer"],
            waiting=df["waiting"],
        )
        return inst

    def weighted_random_choice(self):
        # field_names = [print(type(field)) for field in self.__dataclass_fields__]
        # field_names = [field.name for field in self.__dataclass_fields__]
        field_names = [field for field in self.__dataclass_fields__]
        field_values = [getattr(self, field_name) for field_name in field_names]
        selected_field = random.choices(field_names, weights=field_values, k=1)[0]

        category, sub_category = None, None

        if selected_field == "mine_ore":
            category = ActCategory.MINE
            sub_category = MineActSubCategory.ORE
        elif selected_field == "mine_ice":
            category = ActCategory.MINE
            sub_category = MineActSubCategory.ICE
        elif selected_field == "clear_rubble":
            category = ActCategory.CLEAR
            sub_category = ClearActSubCategory.RUBBLE
        elif selected_field == "clear_lichen":
            category = ActCategory.CLEAR
            sub_category = ClearActSubCategory.LICHEN
        elif selected_field == "attack":
            category = ActCategory.COMBAT
        elif selected_field == "defend":
            category = ActCategory.COMBAT
        elif selected_field == "transfer":
            category = ActCategory.DROPOFF
        elif selected_field == "waiting":
            category = ActCategory.WAITING

        return ActStatus(category=category, sub_category=sub_category)


_desires_unit_ratios_dict = {
    "step": [1, 100, 500, 850, 1000],
    "heavy": [1, 1, 1, 1, 1],
    "light": [0.01, 10, 5, 3, 2],
}
DESIRED_UNIT_RATIOS_DF = _to_df(_desires_unit_ratios_dict)


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
        return self.light_mining_ice + self.light_mining_ore + self.light_clearing_rubble + self.light_attacking

    def total_heavy(self):
        return self.heavy_mining_ice + self.heavy_mining_ore + self.heavy_clearing_rubble + self.heavy_attacking

    def copy(self) -> FactoryDesires:
        return copy.copy(self)

    # def update_desires(
    #     self,
    #     info: FactoryInfo,
    #     light_energy_consideration=1000,
    #     light_rubble_min_tiles=10,
    #     light_rubble_max_tiles=50,
    #     light_rubble_max_num=2,
    #     light_metal_min=50,
    #     light_metal_max=200,
    #     light_metal_max_num=5,
    #     light_water_min=50,
    #     light_water_max=500,
    #     light_water_max_num=3,
    #     light_attack_max_num=2,
    #     heavy_energy_consideration=1000,
    #     heavy_rubble_min_tiles=30,
    #     heavy_rubble_max_tiles=50,
    #     heavy_rubble_max_num=2,
    #     heavy_metal_min=100,
    #     heavy_metal_max=300,
    #     heavy_metal_max_num=5,
    #     heavy_water_min=200,
    #     heavy_water_max=2500,
    #     heavy_water_max_num=3,
    #     heavy_attack_max_num=2,
    # ):
    #     # Consider more LIGHT units
    #     power_req_met = info.short_term_power > light_energy_consideration
    #     # Rubble
    #     expansion_tiles = info.connected_growable_space - info.num_lichen_tiles
    #     if expansion_tiles < light_rubble_min_tiles and power_req_met:
    #         self.light_clearing_rubble = min(light_rubble_max_num, info.light_clearing_rubble + 1)
    #     elif expansion_tiles > light_rubble_max_tiles:
    #         self.light_clearing_rubble = max(0, info.light_clearing_rubble - 1)
    #
    #     # Ore
    #     if info.metal < light_metal_min and power_req_met:
    #         self.light_mining_ore = min(light_metal_max_num, info.light_mining_ore + 1)
    #     elif info.metal > light_metal_max:
    #         self.light_mining_ore = max(0, info.light_mining_ore - 1)
    #
    #     # Ice
    #     if info.water < light_water_min and power_req_met:
    #         self.light_mining_ice = min(light_water_max_num, info.light_mining_ice + 1)
    #     elif info.water > light_water_max:
    #         self.light_mining_ice = max(0, info.light_mining_ice - 1)
    #
    #     # Attack
    #     if info.num_light == self.total_light():
    #         self.light_attacking = min(light_attack_max_num, info.light_attacking + 1)
    #     elif info.num_light < self.total_heavy():
    #         self.light_attacking = max(0, info.light_attacking - 1)
    #
    #     power_req_met = info.short_term_power > heavy_energy_consideration
    #
    #     # Consider more HEAVY units
    #     # Ice
    #     if info.water < heavy_water_min and power_req_met:
    #         self.heavy_mining_ice = min(heavy_water_max_num, info.heavy_mining_ice + 1)
    #     elif info.water > heavy_water_max:
    #         self.heavy_mining_ice = max(0, info.heavy_mining_ice - 1)
    #
    #     # Ore
    #     if info.metal < heavy_metal_min and power_req_met:
    #         self.heavy_mining_ore = min(heavy_metal_max_num, info.heavy_mining_ore + 1)
    #     elif info.metal > heavy_metal_max:
    #         self.heavy_mining_ore = max(0, info.heavy_mining_ore - 1)
    #
    #     # Rubble
    #     expansion_tiles = info.connected_growable_space - info.num_lichen_tiles
    #     if expansion_tiles < heavy_rubble_min_tiles and power_req_met:
    #         self.heavy_clearing_rubble = min(heavy_rubble_max_num, info.heavy_clearing_rubble + 1)
    #     elif expansion_tiles > heavy_rubble_max_tiles:
    #         self.heavy_clearing_rubble = max(0, info.heavy_clearing_rubble - 1)
    #
    #     # Attack
    #     if info.num_heavy == self.total_heavy():
    #         self.heavy_attacking = max(heavy_attack_max_num, info.heavy_attacking + 1)
    #     elif info.num_heavy < self.total_heavy():
    #         self.heavy_attacking = max(0, info.heavy_attacking - 1)


class FactoryActionPlanner:
    max_ice = 1500
    max_ore = 1100
    max_metal = 500
    max_water = 5000

    mid_water = 2000
    mid_metal = 200

    def __init__(self, master: MasterState):
        self.master = master
        self.factories = self.master.factories.friendly

        # Caching variables
        self._factory_desires: Dict[str, FactoryDesires] = {}
        self._factory_work_ratios: Dict[str, WorkRatios] = {}
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
        # for k in set(self._factory_desires.keys()) - set(self.master.factories.friendly.keys()):
        #     logger.info(f"Removing factory {k}, assumed dead")
        #     self._factory_work_ratios.pop(k)
        #     self._factory_desires.pop(k)
        #     self._factory_infos.pop(k)

        # Update their infos
        # infos = self._update_factory_info()

        # update the desired ratios
        self._update_factory_work_ratios()

        # # Calculate new desires for turn and then every X after that
        # if self.master.step == 0 or self.master.step % 5 == 0:
        #     self._update_factory_desires()

    def get_factory_work_ratios(self) -> Dict[str, WorkRatios]:
        return self._factory_work_ratios

    # def get_factory_desires(self) -> Dict[str, FactoryDesires]:
    #     return self._factory_desires

    # def get_factory_infos(self) -> Dict[str, FactoryInfo]:
    #     return self._factory_infos

    def _update_factory_work_ratios(self):
        for f_id, factory in self.factories.items():
            info = factory.info
            logger.debug(f"Updating {f_id} work_ratios")
            ratios = WorkRatios.at_step(self.master.step)
            if factory.cargo.metal > self.max_metal or factory.cargo.ore > self.max_ore:
                ratios.mine_ore = 0
            if factory.cargo.water > self.max_water or factory.cargo.ice > self.max_ice:
                ratios.mine_ice = 0
            self._factory_work_ratios[f_id] = ratios

    # def _update_factory_info(self) -> Dict[str, FactoryInfo]:
    #     """Update info about the factory (uses a sort of rolling average in updating)"""
    #     for f_id, factory in self.master.factories.friendly.items():
    #         logger.debug(f"Updating {f_id} info")
    #         self._factory_infos[f_id] = FactoryInfo.init(
    #             master=self.master,
    #             factory=factory,
    #             previous=self._factory_infos.pop(f_id, None),
    #         )
    #     return self._factory_infos

    # def _update_factory_desires(self):
    #     """Update the desires for each factory"""
    #     for f_id, factory in self.master.factories.friendly.items():
    #         logger.debug(f"Updating {f_id} desires")
    #         # Pop because updating
    #         previous_desire = self._factory_desires.pop(f_id, None)
    #         # Get because only using
    #         info = self._factory_infos.get(f_id, None)
    #         # Update taking into account previous preferences and info
    #         if previous_desire and info:
    #             desires = self._update_single_factory_desire(previous_desire, info)
    #         # Update without knowing previous preferences (first turn for this factory)
    #         else:
    #             desires = FactoryDesires()
    #
    #         self._factory_desires[f_id] = desires

    # def _update_single_factory_desire(self, prev_desire: FactoryDesires, info: FactoryInfo) -> FactoryDesires:
    #     """Update the desires of the factory based on step of game and other info"""
    #     df = DESIRES_DF

    # def _update_single_factory_desire(self, prev_desire: FactoryDesires, info: FactoryInfo) -> FactoryDesires:
    #     """Update the desires of the factory based on it's current state and it's previous desires"""
    #     step = self.master.step
    #     desires = prev_desire.copy()
    #     logger.debug(
    #         f"Desires before HeavyIce={desires.heavy_mining_ice}, LightRubble={desires.light_clearing_rubble}, LightOre={desires.light_mining_ore}, LightAttack={desires.light_attacking}, HeavyAttack={desires.heavy_attacking}"
    #     )
    #     #  TODO: Debug testing
    #     if step < 90:
    #         desires.update_desires(
    #             info=info,
    #             light_energy_consideration=300,
    #             light_rubble_min_tiles=20,
    #             light_rubble_max_tiles=40,
    #             light_rubble_max_num=0,
    #             light_metal_min=50,
    #             light_metal_max=200,
    #             light_metal_max_num=0,
    #             light_water_min=20,
    #             light_water_max=100,
    #             light_water_max_num=0,
    #             light_attack_max_num=4,
    #             heavy_energy_consideration=1000,
    #             heavy_rubble_min_tiles=30,
    #             heavy_rubble_max_tiles=50,
    #             heavy_rubble_max_num=0,
    #             heavy_metal_min=100,
    #             heavy_metal_max=300,
    #             heavy_metal_max_num=1,
    #             heavy_water_min=200,
    #             heavy_water_max=500,
    #             heavy_water_max_num=1,
    #             heavy_attack_max_num=0,
    #         )
    #     elif step < 200:
    #         desires.update_desires(
    #             info=info,
    #             light_energy_consideration=300,
    #             light_rubble_min_tiles=20,
    #             light_rubble_max_tiles=40,
    #             light_rubble_max_num=5,
    #             light_metal_min=50,
    #             light_metal_max=200,
    #             light_metal_max_num=2,
    #             light_water_min=20,
    #             light_water_max=100,
    #             light_water_max_num=0,
    #             light_attack_max_num=1,
    #             heavy_energy_consideration=1000,
    #             heavy_rubble_min_tiles=30,
    #             heavy_rubble_max_tiles=50,
    #             heavy_rubble_max_num=1,
    #             heavy_metal_min=100,
    #             heavy_metal_max=300,
    #             heavy_metal_max_num=2,
    #             heavy_water_min=200,
    #             heavy_water_max=500,
    #             heavy_water_max_num=2,
    #             heavy_attack_max_num=1,
    #         )
    #     # Early mid game
    #     elif step < 500:
    #         desires.update_desires(
    #             info=info,
    #             light_energy_consideration=1100,
    #             light_rubble_min_tiles=50,
    #             light_rubble_max_tiles=200,
    #             light_rubble_max_num=6,
    #             light_metal_min=50,
    #             light_metal_max=200,
    #             light_metal_max_num=1,
    #             light_water_min=100,
    #             light_water_max=300,
    #             light_water_max_num=0,
    #             light_attack_max_num=2,
    #             heavy_energy_consideration=1000,
    #             heavy_rubble_min_tiles=20,
    #             heavy_rubble_max_tiles=40,
    #             heavy_rubble_max_num=2,
    #             heavy_metal_min=100,
    #             heavy_metal_max=300,
    #             heavy_metal_max_num=2,
    #             heavy_water_min=500,
    #             heavy_water_max=1500,
    #             heavy_water_max_num=3,
    #             heavy_attack_max_num=3,
    #         )
    #     # mid late game
    #     elif step < 800:
    #         desires.update_desires(
    #             info=info,
    #             light_energy_consideration=1100,
    #             light_rubble_min_tiles=70,
    #             light_rubble_max_tiles=100,
    #             light_rubble_max_num=5,
    #             light_metal_min=100,
    #             light_metal_max=200,
    #             light_metal_max_num=0,
    #             light_water_min=300,
    #             light_water_max=1000,
    #             light_water_max_num=0,
    #             light_attack_max_num=5,
    #             heavy_energy_consideration=1000,
    #             heavy_rubble_min_tiles=30,
    #             heavy_rubble_max_tiles=70,
    #             heavy_rubble_max_num=2,
    #             heavy_metal_min=100,
    #             heavy_metal_max=300,
    #             heavy_metal_max_num=2,
    #             heavy_water_min=1000,
    #             heavy_water_max=2500,
    #             heavy_water_max_num=4,
    #             heavy_attack_max_num=3,
    #         )
    #     else:
    #         desires.update_desires(
    #             info=info,
    #             light_energy_consideration=1000,
    #             light_rubble_min_tiles=50,
    #             light_rubble_max_tiles=200,
    #             light_rubble_max_num=10,
    #             light_metal_min=100,
    #             light_metal_max=200,
    #             light_metal_max_num=0,
    #             light_water_min=500,
    #             light_water_max=1000,
    #             light_water_max_num=0,
    #             light_attack_max_num=5,
    #             heavy_energy_consideration=1000,
    #             heavy_rubble_min_tiles=20,
    #             heavy_rubble_max_tiles=50,
    #             heavy_rubble_max_num=4,
    #             heavy_metal_min=0,
    #             heavy_metal_max=300,
    #             heavy_metal_max_num=0,
    #             heavy_water_min=500,
    #             heavy_water_max=2500,
    #             heavy_water_max_num=3,
    #             heavy_attack_max_num=3,
    #         )
    #     logger.debug(
    #         f"Desires after HeavyIce={desires.heavy_mining_ice}, LightRubble={desires.light_clearing_rubble}, LightOre={desires.light_mining_ore}, LightAttack={desires.light_attacking}, HeavyAttack={desires.heavy_attacking}"
    #     )
    #     return desires

    def _possible_build(self, factory: FriendlyFactoryManager) -> Optional[int]:
        current_light = max(0.1, len(factory.light_units))  # avoid div zero
        current_heavy = max(0.1, len(factory.heavy_units))

        light_ratio = current_light / current_heavy

        desired_ratios = util.get_interpolated_values(DESIRED_UNIT_RATIOS_DF, self.master.step)
        desired_light_ratio = desired_ratios.light / desired_ratios.heavy

        power_in_X = factory.calculate_power_at_step(10)

        # If too many lights, consider building heavy
        if light_ratio > desired_light_ratio:
            can_build = factory.factory.can_build_heavy(self.master.game_state)
            if can_build and power_in_X > 0:
                return factory.factory.build_heavy()
        # Otherwise consider building light
        else:
            can_build = factory.factory.can_build_light(self.master.game_state)
            if can_build and power_in_X > 0:
                return factory.factory.build_light()
        return None

    def _possible_water(self, factory: FriendlyFactoryManager) -> Optional[int]:
        """Decide when to water"""
        min_water = 100
        if self.master.step <= 100:
            min_water = 100
        elif self.master.step <= 400:
            min_water = 300
        elif self.master.step <= 500:
            min_water = 600
        elif self.master.step <= 700:
            min_water = 900
        elif self.master.step <= 800:
            min_water = 1200
        elif self.master.step <= 850:
            min_water = 800
        elif self.master.step <= 1000:
            min_water = 50

        steps_remaining = 1000 - self.master.step
        if steps_remaining > 700:
            if factory.cargo.water > min_water:
                logger.debug(f"Current water = {factory.cargo.water}, water cost = {factory.water_cost}, min_water = {min_water}")
                logger.info(f"{factory.unit_id} midgame watering because above min water")
                return factory.factory.water()
        else:
            if factory.cargo.water - min_water > (factory.water_cost * steps_remaining):
                logger.debug(f"Current water = {factory.cargo.water}, water cost = {factory.water_cost}, min_water = {min_water}")
                logger.info(f"{factory.unit_id} endgame watering because won't run out with this water cost")
                return factory.factory.water()
        return None

    def decide_factory_actions(self) -> Dict[str, int]:
        """
        Decide what factory should do based on current step and resources etc
        """
        actions = {}
        for factory in self.factories.values():
            center_occupied = (
                True if self.master.units.friendly.unit_at_position(factory.pos) is not None else False
            )
            action = None

            # Only consider building if center not occupied
            if not center_occupied:
                logger.debug(f"Center not occupied, considering building unit")
                action = self._possible_build(factory)

            # If not building unit, consider watering
            if action is None:
                logger.debug(f"Not building unit, considering watering")
                action = self._possible_water(factory)

            if action is not None:
                logger.debug(f"Factory action = {action}")
                actions[factory.unit_id] = action
        return actions

    # def decide_factory_actions(self) -> Dict[str, int]:
    #     actions = {}
    #     for f_info, f_desire in zip(self._factory_infos.values(), self._factory_desires.values()):
    #         logger.debug(f"Deciding factory actions for {f_info.factory_id}")
    #         f_info: FactoryInfo
    #         f_desire: FactoryDesires
    #         center_occupied = (
    #             True if self.master.units.friendly.unit_at_position(f_info.factory.pos) is not None else False
    #         )
    #         action = None
    #
    #         # Only consider building if center not occupied
    #         if not center_occupied:
    #             logger.debug(f"Center not occupied, considering building unit")
    #             action = self._build_unit_action(f_info, f_desire)
    #
    #         # If not building unit, consider watering
    #         if action is None:
    #             logger.debug(f"Not building unit, considering watering")
    #             action = self._water(f_info, f_desire)
    #
    #         if action is not None:
    #             logger.debug(f"Factory action = {action}")
    #             actions[f_info.factory_id] = action
    #     return actions
    #
    # def _build_unit_action(self, info: FactoryInfo, desire: FactoryDesires) -> [None, int]:
    #     total_light_desired = desire.total_light()
    #     total_heavy_desired = desire.total_heavy()
    #
    #     can_build_light = info.factory.factory.can_build_light(self.master.game_state)
    #     can_build_heavy = info.factory.factory.can_build_heavy(self.master.game_state)
    #
    #     logger.debug(f"can build heavy={can_build_heavy}, can_build_light={can_build_light}")
    #     logger.debug(f"desired_heavy={total_heavy_desired}, desired_light={total_light_desired}")
    #     logger.debug(f"current_heavy={info.num_heavy}, current_light={info.num_light}")
    #
    #     if can_build_heavy:
    #         if total_heavy_desired > info.num_heavy:
    #             return info.factory.factory.build_heavy()
    #     elif can_build_light:
    #         if total_light_desired > info.num_light:
    #             return info.factory.factory.build_light()
    #
    # def _water(self, info: FactoryInfo, desire: FactoryDesires) -> [None, int]:
    #     min_water = 100
    #     if self.master.step <= 300:
    #         min_water = 200
    #     elif self.master.step <= 400:
    #         min_water = 400
    #     elif self.master.step <= 500:
    #         min_water = 600
    #     elif self.master.step <= 700:
    #         min_water = 900
    #     elif self.master.step <= 800:
    #         min_water = 1500
    #     elif self.master.step <= 900:
    #         min_water = 1000
    #     elif self.master.step <= 1000:
    #         min_water = 150
    #     logger.debug(f"Current water = {info.water}, water cost = {info.water_cost}, min_water = {min_water}")
    #     if info.water - info.water_cost > min_water:
    #         return info.factory.factory.water()
    #     return None

    def place_factory(self, game_state: GameState, player):
        """Place factory in early_setup"""
        # how many factories you have left to place
        factories_to_place = game_state.teams[player].factories_to_place

        # how much water and metal you have in your starting pool to give to new factories
        water_left = game_state.teams[player].water
        metal_left = game_state.teams[player].metal

        # how much to give to each factory
        water = water_left // factories_to_place
        metal = metal_left // factories_to_place

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

        # Find distance to ore
        ore = game_state.board.ore
        df["ore_dist"] = df.apply(
            lambda row: util.manhattan(row.pos, util.nearest_non_zero(ore, row.pos)),
            axis=1,
        )

        # Find min distance to friendly other friendly factories
        friendly_factory_map = (self.master.maps.factory_maps.friendly >= 0).astype(int)
        if np.all(friendly_factory_map == 0):
            # No friendly factories placed yet
            df["nearest_friendly_factory"] = [999] * len(df)
        else:
            df["nearest_friendly_factory"] = df.apply(
                lambda row: util.manhattan(row.pos, util.nearest_non_zero(friendly_factory_map, row.pos)),
                axis=1,
            )

        df["ice_less_than_X"] = df["ice_dist"] <= 4
        df["ore_less_than_X"] = df["ore_dist"] <= 8
        df["friendly_further_than_X"] = df["nearest_friendly_factory"] >= 15

        # df = df.sort_values("ice_dist")
        df = df.sort_values(
            [
                "ice_less_than_X",
                "ore_less_than_X",
                "friendly_further_than_X",
                "ice_dist",
                "ore_dist",
            ],
            ascending=[False, False, False, True, True],
        )

        # Keep only top X before doing expensive calculations
        df = df.iloc[:20]

        # Value based nearby zero-rubble
        rubble = game_state.board.rubble
        count_arr = util.count_connected_values(rubble, value=0)
        kernel = util.factory_map_kernel(5, dist_multiplier=0.6)
        conv_count_arr = util.convolve_array_kernel(count_arr, kernel)
        df["zero_rubble_value"] = df.apply(lambda row: (conv_count_arr[row.x, row.y]), axis=1)
        df = df.sort_values(
            [
                "ice_less_than_X",
                "ore_less_than_X",
                "friendly_further_than_X",
                "zero_rubble_value",
            ],
            ascending=[False, False, False, False],
        )
        # df = df.sort_values(["ice_dist", "zero_rubble_value"], ascending=[True, False])

        # print(df.head(10))

        # TODO: Calculate how close to nearest enemy factory for top X

        return dict(spawn=df.iloc[0].pos, metal=metal, water=water)
