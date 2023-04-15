from __future__ import annotations
from typing import TYPE_CHECKING

from config import get_logger
import util

logger = get_logger(__name__)


if TYPE_CHECKING:
    from master_state import MasterState
    from unit_action_planner import CloseUnits
    from unit_manager import FriendlyUnitManger


class CombatPlanner:
    def __init__(self, master: MasterState):
        self.master = master

    def update(self):
        pass

    def attack(self, unit: FriendlyUnitManger, close_units: CloseUnits) -> bool:
        """Note: Unit MUST move, sitting still will result in automatic death on collision"""
        logger.function_call(f'Attacking enemy')
        # If there is an enemy already detected close, go for that
        if close_units is not None:
            index_of_closest = close_units.other_unit_distances.index(
                min(close_units.other_unit_distances)
            )
            enemy_loc = close_units.other_unit_positions[index_of_closest]
        # Otherwise find the nearest enemy with lower power
        else:
            logger.error(f'Not implemented finding far away enemies yet')
            return False

        # TODO: Try to intercept enemy instead of just aiming for where they are now
        path_to_enemy = self.master.pathfinder.fast_path(unit.pos, enemy_loc)
        if len(path_to_enemy) > 1:
            self.master.pathfinder.append_path_to_actions(unit, path_to_enemy)
            return True
        # Path is saying don't move... that's a bad idea for combat... at least move somewhere
        elif len(path_to_enemy) == 1:
            logger.warning(f'{unit.log_prefix}: Attacking path said to stand still, but that could mean death, moving to cheapest adjacent tile')
            util.move_to_cheapest_adjacent_space(self.master.pathfinder, unit)
        else:
            logger.error(f'{unit.log_prefix}: No path to enemy')
            return False

    def run_away(self, unit: FriendlyUnitManger):
        logger.function_call(f'Running away to factory')
        path_to_factory = util.calc_path_to_factory(
            self.master.pathfinder,
            unit.pos,
            self.master.factories.friendly[unit.factory_id].factory_loc,
        )
        if len(path_to_factory) > 0:
            self.master.pathfinder.append_path_to_actions(unit, path_to_factory)
            return True
        else:
            logger.error(f'{unit.log_prefix}: No path to factory')
            return False
