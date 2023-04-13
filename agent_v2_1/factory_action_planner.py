import pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict, List

from lux.factory import Factory

from master_state import MasterState


BUILD_LIGHT = 0
BUILD_HEAVY = 1
WATER =  2

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

class FactoryActionPlanner:
    def __init__(self, master: MasterState):
        self.master = master
        self.factories = self.master.factories.friendly

    def calculate_factory_data(self) -> List[FactoryInfo]:
        """Make some observations that will help decide what the factories should do"""
        data = []
        for f_id, factory in self.factories.items():
            factory_info = FactoryInfo(
                factory=factory.factory,
                factory_id=f_id,
                power=factory.power,
                water=factory.factory.cargo.water,
                ice=factory.factory.cargo.ice,
                ore=factory.factory.cargo.ore,
                metal =factory.factory.cargo.metal,
                pos=factory.pos,
                num_heavy=len(factory.heavy_units),
                num_light=len(factory.light_units),
                water_cost=factory.factory.water_cost(self.master.game_state),
            )
            data.append(factory_info)
        df = pd.DataFrame(data)
        return df

    def decide_factory_actions(self) -> Dict[str, int]:
        actions = {}
        factory_data = self.calculate_factory_data()
        for f_data in factory_data:
            action = self.action_for_factory(f_data)
            if action is not None:
                actions[f_data.factory_id] = action
        return actions

    def action_for_factory(self, factory_info: FactoryInfo) -> [None, int]:
        """Actions are: build light, build heavy, water"""
        # Early game
        if self.master.step < 200:
            return self.early_game_action_for_factory(factory_info)
        # Early mid game
        elif self.master.step < 500:
            pass
        # mid late game
        elif self.master.step < 800:
            pass
        else:
            pass

    def early_game_action_for_factory(self, factory_info: FactoryInfo) -> [None, int]:
        factory = factory_info.factory
        if factory.can_build_heavy(self.master.game_state) and factory_info.num_heavy < 1:
            return BUILD_HEAVY
        elif factory.can_build_light(self.master.game_state) and factory_info.metal < 200 and factory_info.num_light < 5:
            return BUILD_LIGHT
        elif factory_info.water > 100:
            return WATER
        return None


