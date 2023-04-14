from __future__ import annotations
from typing import List, TYPE_CHECKING

import util

import numpy as np


if TYPE_CHECKING:
    from actions import Recommendation
    from master_state import MasterState
    from unit_action_planner import CloseUnits
    from unit_manager import FriendlyUnitManger


class CombatPlanner:
    def __init__(self, master: MasterState):
        self.master = master

    def update(self):
        pass

    def attack(self, unit: FriendlyUnitManger, close_units: CloseUnits) -> bool:
        possible_close_enemy: CloseUnits
        index_of_closest = close_units.other_unit_distances.index(
            min(close_units.other_unit_distances)
        )
        enemy_loc = close_units.other_unit_positions[index_of_closest]
        path_to_enemy = self.master.pathfinder.fast_path(unit.pos, enemy_loc)
        if len(path_to_enemy) > 0:
            self.master.pathfinder.append_path_to_actions(unit, path_to_enemy)
            return True
        else:
            return False

    def run_away(self, unit: FriendlyUnitManger):
        path_to_factory = util.calc_path_to_factory(
            self.master.pathfinder,
            unit.pos,
            self.master.factories.friendly[unit.factory_id].factory_loc,
        )
        if len(path_to_factory) > 0:
            self.master.pathfinder.append_path_to_actions(unit, path_to_factory)
            return True
        else:
            return False
