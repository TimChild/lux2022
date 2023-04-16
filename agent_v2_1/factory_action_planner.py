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
    from unit_manager import FriendlyUnitManger

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
    connected_zeros: int
    lichen_tiles: int
    total_lichen: int
    # Below should be equivalent to factory desires
    light_mining_ore: int
    light_clearing_rubble: int
    light_attacking: int
    heavy_mining_ice: int
    heavy_mining_ore: int
    heavy_attacking: int

    @classmethod
    def init(cls, master: MasterState, factory: FriendlyFactoryManager) -> FactoryInfo:
        lichen_id, lichen_locations, lichen = cls._get_lichen_info(master, factory)
        current_light_actions = factory.get_light_actions()
        current_heavy_actions = factory.get_heavy_actions()
        connected_zeros = cls._get_connected_zeros(master.maps.rubble, factory.pos)
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
            connected_zeros=connected_zeros,
            lichen_tiles=int(np.sum(lichen_locations)),
            total_lichen=int(np.sum(lichen)),
            light_mining_ore=len(current_light_actions.mining_ore),
            light_clearing_rubble=len(current_light_actions.clearing_rubble),
            light_attacking=len(current_light_actions.attacking),
            heavy_mining_ice=len(current_heavy_actions.mining_ice),
            heavy_mining_ore=len(current_heavy_actions.mining_ore),
            heavy_attacking=len(current_heavy_actions.attacking),
        )
        logger.debug(
            f"light_ore: {factory_info.light_mining_ore}, light_rubble: {factory_info.light_clearing_rubble}, heavy_ice: {factory_info.heavy_mining_ice}"
        )
        return factory_info

    def update(self, master: MasterState, factory: FriendlyFactoryManager):
        lichen_id, lichen_locations, lichen = self._get_lichen_info(master, factory)
        current_light_actions = factory.get_light_actions()
        current_heavy_actions = factory.get_heavy_actions()
        connected_zeros = self._get_connected_zeros(master.maps.rubble, factory.pos)

        # never changes
        # self.factory_id
        # self.pos

        # Average some things
        # Roughly like average of last 10 values
        self.power = int(self.power * 0.9 + 0.1 * factory.power)
        self.water = int(self.water * 0.9 + 0.1 * factory.factory.cargo.water)
        self.ice = int(self.ice * 0.9 + 0.1 * factory.factory.cargo.ice)
        self.ore = int(self.ore * 0.9 + 0.1 * factory.factory.cargo.ore)
        self.metal = int(self.metal * 0.9 + 0.1 * factory.factory.cargo.metal)
        self.water_cost = int(
            self.water_cost * 0.9 + 0.1 * factory.factory.water_cost(master.game_state)
        )

        # Replace some things
        self.factory = factory.factory
        self.connected_zeros = connected_zeros
        self.lichen_tiles = int(np.sum(lichen_locations))
        self.total_lichen = int(np.sum(lichen))
        self.num_light = len(factory.light_units)
        self.num_heavy = len(factory.heavy_units)
        self.light_mining_ore = len(current_light_actions.mining_ore)
        self.light_clearing_rubble = len(current_light_actions.clearing_rubble)
        self.light_attacking = len(current_light_actions.attacking)
        self.heavy_mining_ice = len(current_heavy_actions.mining_ice)
        self.heavy_mining_ore = len(current_heavy_actions.mining_ore)
        self.heavy_attacking = len(current_heavy_actions.attacking)

    def remove_unit_from_current_count(self, unit: FriendlyUnitManger):
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
        return int(
            np.sum(
                util.connected_array_values_from_pos(
                    rubble, factory_pos, connected_value=0
                )
            )
        )

    @staticmethod
    def _get_lichen_info(master: MasterState, factory: FriendlyFactoryManager):
        lichen_id = factory.factory.strain_id
        lichen_locations = master.game_state.board.lichen_strains == lichen_id
        lichen = master.game_state.board.lichen
        lichen[lichen_locations != 1] = 0
        return lichen_id, lichen_locations, lichen


@dataclass
class FactoryDesires:
    heavy_mining_ice: int = 1
    heavy_mining_ore: int = 0
    heavy_attacking: int = 0
    light_mining_ore: int = 0
    light_clearing_rubble: int = 0
    light_attacking: int = 0

    def copy(self) -> FactoryDesires:
        return copy.copy(self)


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
        logger.function_call(f"Updating FactoryActionPlanner")
        # Remove any dead factories
        for k in set(self._factory_desires.keys()) - set(
            self.master.factories.friendly.keys()
        ):
            logger.info(f"Removing factory {k}, assumed dead")
            self._factory_desires.pop(k)
            self._factory_infos.pop(k)

        # Update their infos
        self._update_factory_info()

        # Calculate new desires for first X turns and then every 50 after that
        if self.master.step < 10 or self.master.step % 20 == 0:
            self._update_factory_desires()

    def get_factory_desires(self) -> Dict[str, FactoryDesires]:
        return self._factory_desires

    def get_factory_infos(self) -> Dict[str, FactoryInfo]:
        return self._factory_infos

    def _update_factory_info(self):
        """Update info about the factory (uses a sort of rolling average in updating)"""
        for f_id, factory in self.master.factories.friendly.items():
            logger.debug(f"Updating {f_id} info")
            if not f_id in self._factory_infos:
                self._factory_infos[f_id] = FactoryInfo.init(self.master, factory)
            else:
                self._factory_infos[f_id].update(self.master, factory)

    def _update_factory_desires(self):
        """Update the desires for each factory"""
        # Remove dead factories
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
                desires = self._create_single_factory_desire(info)

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
        if step < 300:
            # Add more light units
            if info.power > 1000 and info.metal > 10:
                if desires.light_mining_ore < 1 and desires.heavy_mining_ore == 0:
                    desires.light_mining_ore += 1
                elif desires.light_clearing_rubble < 2:
                    desires.light_clearing_rubble += 1
                elif desires.light_attacking < 1:
                    desires.light_attacking += 1
                else:
                    # Add heavy
                    if info.power > 2000 and info.metal > 100:
                        if desires.heavy_mining_ice < 2:
                            desires.heavy_mining_ice += 1
                        elif desires.heavy_mining_ore < 1:
                            desires.heavy_mining_ore += 1
        # Early mid game
        elif step < 500:
            if info.power > 1500 and info.metal > 10:
                if desires.light_mining_ore < 1 and desires.heavy_mining_ore == 0:
                    desires.light_mining_ore += 1
                elif desires.light_clearing_rubble < 4:
                    desires.light_clearing_rubble += 1
                elif desires.light_attacking < 1:
                    desires.light_attacking += 1
                else:
                    # Add heavy
                    if info.power > 2500 and info.metal > 100:
                        if desires.heavy_mining_ice < 3:
                            desires.heavy_mining_ice += 1
                        # elif desires.heavy_mining_ore < 1:
                        #     desires.heavy_mining_ore += 1
        # mid late game
        elif step < 800:
            pass
        else:
            pass
        logger.debug(
            f"Desires after HeavyIce={desires.heavy_mining_ice}, LightRubble={desires.light_clearing_rubble}, LightOre={desires.light_mining_ore}, LightAttack={desires.light_attacking}, HeavyAttack={desires.heavy_attacking}"
        )
        return desires

    def _create_single_factory_desire(self, info: FactoryInfo) -> FactoryDesires:
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
            if not center_occupied and self.master.step:
                action = self._build_unit_action(f_info, f_desire)

            # If not building unit, consider watering
            if action is None:
                action = self._water(f_info, f_desire)

            if action is not None:
                logger.debug(f"Factory action = {action}")
                actions[f_info.factory_id] = action
        return actions

    def _build_unit_action(
        self, info: FactoryInfo, desire: FactoryDesires
    ) -> [None, int]:
        total_light_desired = (
            desire.light_attacking
            + desire.light_clearing_rubble
            + desire.light_mining_ore
        )
        total_heavy_desired = desire.heavy_attacking + desire.heavy_mining_ice
        # Really need at least 1 heavy ice miner
        if (
            info.num_heavy == 0
            and desire.heavy_mining_ice > 0
            and info.factory.can_build_heavy(self.master.game_state)
        ):
            return info.factory.build_heavy()
        # The more heavys only if excess power
        elif (
            info.factory.power > 2000
            and info.factory.can_build_heavy(self.master.game_state)
            and info.num_heavy < total_heavy_desired
        ):
            return info.factory.build_heavy()
        # Or lights if less excess power
        elif (
            info.factory.power > 1000
            and info.factory.can_build_light(self.master.game_state)
            and info.num_light < total_light_desired
        ):
            return info.factory.build_light()
        else:
            return None

    def _water(self, info: FactoryInfo, desire: FactoryDesires) -> [None, int]:
        min_water = 100
        if 0 < self.master.step <= 500:
            min_water = 100
        elif 500 < self.master.step <= 850:
            min_water = 1000
        elif 850 < self.master.step < 1000:
            min_water = 50
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
        kernel = util.factory_map_kernel(3, dist_multiplier=0.5)
        conv_count_arr = util.convolve_array_kernel(count_arr, kernel)
        df["zero_rubble_value"] = df.apply(
            lambda row: (conv_count_arr[row.x, row.y]), axis=1
        )
        df = df.sort_values(["ice_dist", "zero_rubble_value"], ascending=[True, False])

        # TODO: Calculate how close to nearest enemy factory for top X
        # TODO: Calculate how close to nearest ore (not so important?)

        return dict(spawn=df.iloc[0].pos, metal=150, water=150)
