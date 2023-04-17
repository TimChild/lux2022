from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np

from config import get_logger
from actions import NOTHING
import util

logger = get_logger(__name__)

if TYPE_CHECKING:
    from master_state import MasterState
    from unit_action_planner import CloseUnits
    from unit_manager import FriendlyUnitManager, EnemyUnitManager


class CombatPlanner:
    def __init__(self, master: MasterState):
        self.master = master

    def update(self):
        pass

    def attack(self, unit: FriendlyUnitManager, close_units: CloseUnits) -> bool:
        """Note: Unit MUST move, sitting still will result in automatic death on collision"""
        logger.info(f"Attacking enemy")

        # Find all interceptable enemy locations of interest
        enemy_location_ids = self.master.pathfinder.unit_paths.calculate_likely_unit_collisions(
            unit.pos,
            util.num_turns_of_actions(unit.action_queue),
            exclude_id_nums=[unit.id_num],
            friendly_light=False,
            friendly_heavy=False,
            # Only hunt down same type of unit
            enemy_light=True if unit.unit_type == "LIGHT" else False,
            enemy_heavy=True if unit.unit_type == "HEAVY" else False,
        )
        # Only asking for one anyway, so will always be first in dict
        enemy_location_ids = next(iter(enemy_location_ids.values()))

        # Find nearest that has lower power or lower unit type
        cm = self.master.pathfinder.generate_costmap(unit, collision_only=True)
        power_now = unit.power_remaining(self.master.maps.rubble)
        best_enemy_unit = None
        best_intercept = None
        best_power_at_enemy = None
        best_power_back_to_factory = None
        for i in range(20):
            enemies_map = enemy_location_ids >= 0
            nearest_intercept = util.nearest_non_zero(enemies_map, unit.pos)
            if nearest_intercept is None:
                if best_enemy_unit is None:
                    logger.warning(f"{unit.log_prefix} No intercepts with enemy")
                break

            # Check lower power or type
            enemy_num = enemy_location_ids[nearest_intercept[0], nearest_intercept[1]]
            enemy_id = f"unit_{enemy_num}"
            enemy_unit = self.master.units.enemy.get_unit(enemy_id)
            if enemy_unit is None:
                logger.error(
                    f"{unit.log_prefix}: Found {enemy_id} as nearest enemy, but not in enemy units"
                )
                break
            enemy_unit: EnemyUnitManager

            # May need to check unit type if I change my mind above!

            # Will unit have power to attack and return to factory (Note: Actually power will be decreased by attack, but ignoring that for now)
            path_to_enemy = self.master.pathfinder.fast_path(
                unit.pos, nearest_intercept, costmap=cm
            )
            cost_to_enemy = util.power_cost_of_path(
                path_to_enemy, self.master.maps.rubble, unit.unit_type
            )
            path_to_factory = util.path_to_factory_edge_nearest_pos(
                self.master.pathfinder,
                unit.factory_loc,
                nearest_intercept,
                nearest_intercept,
                costmap=cm,
            )
            cost_to_factory = util.power_cost_of_path(
                path_to_factory, self.master.maps.rubble, unit.unit_type
            )
            power_at_enemy = power_now - cost_to_enemy
            power_back_to_factory = power_at_enemy - cost_to_factory

            if enemy_unit.power-enemy_unit.unit_config.MOVE_COST*len(path_to_enemy) <= power_at_enemy:
                if power_back_to_factory > 0:
                    logger.info(
                        f"Found good enemy unit to intercept {enemy_unit.unit_id}, doing that"
                    )
                    best_enemy_unit = enemy_unit
                    best_intercept = nearest_intercept
                    best_power_at_enemy = power_at_enemy
                    best_power_back_to_factory = power_back_to_factory
                    break
                else:
                    logger.debug(
                        f"Found possible enemy unit to intercept {enemy_unit.unit_id}, looking for better"
                    )
                    best_enemy_unit = enemy_unit
                    best_intercept = nearest_intercept
                    best_power_at_enemy = power_at_enemy
                    best_power_back_to_factory = power_back_to_factory

            # Remove this unit and look for next
            enemy_location_ids[enemy_location_ids == enemy_num] = -1
        else:
            logger.warning(
                f"{unit.log_prefix} Checked 20 enemy units, breaking loop now"
            )

        # Decide whether to attack, do nothing, or return to factory
        if best_enemy_unit is None:
            if power_now < unit.unit_config.BATTERY_CAPACITY * 0.5:
                logger.debug(f"low power ({unit.power}) returning to factory")
                what_do = "factory"
            else:
                logger.debug(
                    "No good intercept, but enough power to stay out, will set success to False so unit can be reassing if necessaary"
                )
                what_do = "nothing"
        else:
            # If really not enough to get there and back
            if best_power_back_to_factory < -unit.unit_config.BATTERY_CAPACITY * 0.2:
                logger.info(
                    f"Found enemy intercept {best_intercept} from {unit.pos}, but really not enough power "
                    f"({best_power_back_to_factory}) to get there and back to factory"
                )
                if power_now < unit.unit_config.BATTERY_CAPACITY * 0.5:
                    logger.debug(f"low power ({unit.power}) returning to factory")
                    what_do = "factory"
                else:
                    logger.debug(
                        f"Enough power to stay out ({power_now}) will set status to doing NOTHING"
                    )
                    what_do = "nothing"
            else:
                what_do = "attack"
                logger.info(
                    f"Attacking {best_enemy_unit.unit_id} at intercept {best_intercept}, "
                    f"power at enemy {best_power_at_enemy}, power at factory {best_power_back_to_factory}"
                )

        if what_do == "attack":
            cm = self.master.pathfinder.generate_costmap(
                unit, ignore_id_nums=[best_enemy_unit.id_num]
            )
            path_to_enemy = self.master.pathfinder.fast_path(
                unit.pos, best_intercept, costmap=cm
            )
            if len(path_to_enemy) > 1:
                self.master.pathfinder.append_path_to_actions(unit, path_to_enemy)
                return True
            elif len(path_to_enemy) == 1:
                logger.warning(
                    f"{unit.log_prefix}: Attacking path said to stand still, but that could mean death, moving to cheapest adjacent tile"
                )
                util.move_to_cheapest_adjacent_space(self.master.pathfinder, unit)
                return True
            else:
                logger.error(
                    f"{unit.log_prefix} error in final pathing to intercept {best_enemy_unit.unit_id}"
                )
                return False
        elif what_do == "factory":
            cm = self.master.pathfinder.generate_costmap(
                unit,
            )
            path_to_factory = util.path_to_factory_edge_nearest_pos(
                self.master.pathfinder,
                unit.factory_loc,
                unit.pos,
                unit.pos,
                costmap=cm,
            )
            if len(path_to_factory) > 0:
                self.master.pathfinder.append_path_to_actions(unit, path_to_factory)
                return True
            else:
                logger.error(f"{unit.log_prefix} error in pathing back to factory")
                return False
        elif what_do == "nothing":
            # If none, if need to move, move, else do nothing
            if cm[unit.pos[0], unit.pos[1]] <= 0:
                logger.info(f"Had to move, so moving to cheapest spot")
                util.move_to_cheapest_adjacent_space(self.master.pathfinder, unit)
            # unit.update_status(NOTHING, success=True)
            return False

    def run_away(self, unit: FriendlyUnitManager):
        logger.info(f"Running away to factory")
        cm = self.master.pathfinder.generate_costmap(unit)
        path_to_factory = util.calc_path_to_factory(
            self.master.pathfinder,
            costmap=cm,
            pos=unit.pos,
            factory_loc=self.master.factories.friendly[unit.factory_id].factory_loc,
        )
        if len(path_to_factory) > 0:
            self.master.pathfinder.append_path_to_actions(unit, path_to_factory)
            return True
        else:
            cm = self.master.pathfinder.generate_costmap(
                unit, enemy_light=False, enemy_heavy=False
            )
            path_to_factory = util.calc_path_to_factory(
                self.master.pathfinder,
                costmap=cm,
                pos=unit.pos,
                factory_loc=self.master.factories.friendly[unit.factory_id].factory_loc,
            )
            if len(path_to_factory) > 0:
                logger.warning(
                    f"{unit.log_prefix}: Path to factory only after allowing collisions with enemy"
                )
                self.master.pathfinder.append_path_to_actions(unit, path_to_factory)
                return True
            logger.error(f"{unit.log_prefix}: No path to factory")
            return False
