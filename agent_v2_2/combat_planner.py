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


class ActionDecider:
    def __init__(
        self,
        unit,
        master,
        best_enemy_unit,
        best_intercept,
        best_power_at_enemy,
        best_power_back_to_factory,
    ):
        self.unit = unit
        self.master = master
        self.best_enemy_unit = best_enemy_unit
        self.best_intercept = best_intercept
        self.best_power_at_enemy = best_power_at_enemy
        self.best_power_back_to_factory = best_power_back_to_factory

    def decide_action(self) -> str:
        if self.best_enemy_unit is None:
            return self.decide_action_no_enemy()
        else:
            return self.decide_action_with_enemy()

    def decide_action_no_enemy(self) -> str:
        power_now = self.unit.power_remaining(self.master.maps.rubble)
        if power_now < self.unit.unit_config.BATTERY_CAPACITY * 0.5:
            logger.debug(f"low power ({self.unit.power}) returning to factory")
            return "factory"
        else:
            logger.debug(
                "No good intercept, but enough power to stay out, will set success to False so unit can be reassigned if necessary"
            )
            return "nothing"

    def decide_action_with_enemy(self) -> str:
        if (
            self.best_power_back_to_factory
            < -self.unit.unit_config.BATTERY_CAPACITY * 0.2
        ):
            return self.decide_action_insufficient_power()
        else:
            return self.decide_action_attack()

    def decide_action_insufficient_power(self) -> str:
        logger.info(
            f"Found enemy intercept {self.best_intercept} from {self.unit.pos}, but really not enough power ({self.best_power_back_to_factory}) to get there and back to factory"
        )
        power_now = self.unit.power_remaining(self.master.maps.rubble)
        if power_now < self.unit.unit_config.BATTERY_CAPACITY * 0.5:
            logger.debug(f"low power ({self.unit.power}) returning to factory")
            return "factory"
        else:
            logger.debug(
                f"Enough power to stay out ({power_now}) will set status to doing NOTHING"
            )
            return "nothing"

    def decide_action_attack(self) -> str:
        logger.info(
            f"Attacking {self.best_enemy_unit.unit_id} at intercept {self.best_intercept}, power at enemy {self.best_power_at_enemy}, power at factory {self.best_power_back_to_factory}"
        )
        return "attack"


class ActionExecutor:
    def __init__(self, unit, master, what_do, best_enemy_unit, best_intercept):
        self.unit = unit
        self.master = master
        self.what_do = what_do
        self.best_enemy_unit = best_enemy_unit
        self.best_intercept = best_intercept

    def execute_action(self) -> bool:
        if self.what_do == "attack":
            return self.execute_attack()
        elif self.what_do == "factory":
            return self.execute_factory()
        elif self.what_do == "nothing":
            return self.execute_nothing()
        return False

    def execute_attack(self) -> bool:
        cm = self.master.pathfinder.generate_costmap(
            self.unit, ignore_id_nums=[self.best_enemy_unit.id_num]
        )
        path_to_enemy = self.master.pathfinder.fast_path(
            self.unit.pos, self.best_intercept, costmap=cm
        )

        if len(path_to_enemy) > 1:
            self.master.pathfinder.append_path_to_actions(self.unit, path_to_enemy)
            return True
        elif len(path_to_enemy) == 1:
            logger.warning(
                f"{self.unit.log_prefix}: Attacking path said to stand still, but that could mean death, moving to cheapest adjacent tile"
            )
            util.move_to_cheapest_adjacent_space(self.master.pathfinder, self.unit)
            return True
        else:
            logger.error(
                f"{self.unit.log_prefix} error in final pathing to intercept {self.best_enemy_unit.unit_id}"
            )
            return False

    def execute_factory(self) -> bool:
        cm = self.master.pathfinder.generate_costmap(self.unit)
        path_to_factory = util.path_to_factory_edge_nearest_pos(
            self.master.pathfinder,
            self.unit.factory_loc,
            self.unit.pos,
            self.unit.pos,
            costmap=cm,
        )

        if len(path_to_factory) > 0:
            self.master.pathfinder.append_path_to_actions(self.unit, path_to_factory)
            return True
        else:
            logger.error(f"{self.unit.log_prefix} error in pathing back to factory")
            return False

    def execute_nothing(self) -> bool:
        cm = self.master.pathfinder.generate_costmap(self.unit)
        if cm[self.unit.pos[0], self.unit.pos[1]] <= 0:
            logger.info(f"Had to move, so moving to cheapest spot")
            util.move_to_cheapest_adjacent_space(self.master.pathfinder, self.unit)
        return False


class BestEnemyUnit:
    def __init__(
        self,
        master: MasterState,
        unit: FriendlyUnitManager,
        enemy_location_ids: np.ndarray,
    ):
        self.master = master
        self.unit = unit
        self.enemy_location_ids = enemy_location_ids
        self.best_enemy_unit = None
        self.best_intercept = None
        self.best_power_at_enemy = None
        self.best_power_back_to_factory = None

    def calculate_power_at_enemy_and_factory(self, path_to_enemy, path_to_factory):
        power_now = self.unit.power_remaining(self.master.maps.rubble)
        cost_to_enemy = util.power_cost_of_path(
            path_to_enemy, self.master.maps.rubble, self.unit.unit_type
        )
        cost_to_factory = util.power_cost_of_path(
            path_to_factory, self.master.maps.rubble, self.unit.unit_type
        )
        power_at_enemy = power_now - cost_to_enemy
        power_back_to_factory = power_at_enemy - cost_to_factory

        return power_at_enemy, power_back_to_factory

    def is_good_enemy(self, enemy_unit, power_at_enemy, path_to_enemy):
        return (
            enemy_unit.power - enemy_unit.unit_config.MOVE_COST * len(path_to_enemy)
            <= power_at_enemy
        )

    def update_best_enemy_unit(
        self, enemy_unit, intercept, power_at_enemy, power_back_to_factory
    ):
        self.best_enemy_unit = enemy_unit
        self.best_intercept = intercept
        self.best_power_at_enemy = power_at_enemy
        self.best_power_back_to_factory = power_back_to_factory

    def remove_current_enemy(self, enemy_num):
        self.enemy_location_ids[self.enemy_location_ids == enemy_num] = -1

    def find_best_enemy_unit(self):
        cm = self.master.pathfinder.generate_costmap(self.unit, collision_only=True)

        for i in range(20):
            enemies_map = self.enemy_location_ids >= 0
            nearest_intercept = util.nearest_non_zero(enemies_map, self.unit.pos)
            if nearest_intercept is None:
                if self.best_enemy_unit is None:
                    logger.warning(f"{self.unit.log_prefix} No intercepts with enemy")
                break

            enemy_num = self.enemy_location_ids[
                nearest_intercept[0], nearest_intercept[1]
            ]
            enemy_id = f"unit_{enemy_num}"
            enemy_unit = self.master.units.enemy.get_unit(enemy_id)
            if enemy_unit is None:
                logger.error(
                    f"{self.unit.log_prefix}: Found {enemy_id} as nearest enemy, but not in enemy units"
                )
                break
            enemy_unit: EnemyUnitManager

            path_to_enemy = self.master.pathfinder.fast_path(
                self.unit.pos, nearest_intercept, costmap=cm
            )
            path_to_factory = util.path_to_factory_edge_nearest_pos(
                self.master.pathfinder,
                self.unit.factory_loc,
                nearest_intercept,
                nearest_intercept,
                costmap=cm,
            )
            (
                power_at_enemy,
                power_back_to_factory,
            ) = self.calculate_power_at_enemy_and_factory(
                path_to_enemy, path_to_factory
            )

            if self.is_good_enemy(enemy_unit, power_at_enemy, path_to_enemy):
                if power_back_to_factory > 0:
                    logger.info(
                        f"Found good enemy unit to intercept {enemy_unit.unit_id}, doing that"
                    )
                    self.update_best_enemy_unit(
                        enemy_unit,
                        nearest_intercept,
                        power_at_enemy,
                        power_back_to_factory,
                    )
                    break
                else:
                    logger.debug(
                        f"Found possible enemy unit to intercept {enemy_unit.unit_id}, looking for better"
                    )
                    self.update_best_enemy_unit(
                        enemy_unit,
                        nearest_intercept,
                        power_at_enemy,
                        power_back_to_factory,
                    )

            self.remove_current_enemy(enemy_num)
        else:
            logger.warning(
                f"{self.unit.log_prefix} Checked 20 enemy units, breaking loop now"
            )


class Attack:
    def __init__(self, unit: FriendlyUnitManager, master: MasterState):
        self.unit = unit
        self.master = master

        self.enemy_location_ids = None
        self.best_enemy_unit = None
        self.best_intercept = None
        self.best_power_at_enemy = None
        self.best_power_back_to_factory = None

    def find_interceptable_enemy_locations(self):
        self.enemy_location_ids = self.master.pathfinder.unit_paths.calculate_likely_unit_collisions(
            self.unit.pos,
            util.num_turns_of_actions(self.unit.action_queue),
            exclude_id_nums=[self.unit.id_num],
            friendly_light=False,
            friendly_heavy=False,
            # Only hunt down same type of unit
            enemy_light=True if self.unit.unit_type == "LIGHT" else False,
            enemy_heavy=True if self.unit.unit_type == "HEAVY" else False,
        )
        # Note: Only asking for one anyway, so will always be first in dict
        self.enemy_location_ids = next(iter(self.enemy_location_ids.values()))

    def find_best_enemy_unit(self):
        # Instantiate BestEnemyUnit class and call find_best_enemy_unit method
        best_enemy_unit_finder = BestEnemyUnit(
            self.master, self.unit, self.enemy_location_ids
        )
        best_enemy_unit_finder.find_best_enemy_unit()

        # Update the Attack class attributes with the results
        self.best_enemy_unit = best_enemy_unit_finder.best_enemy_unit
        self.best_intercept = best_enemy_unit_finder.best_intercept
        self.best_power_at_enemy = best_enemy_unit_finder.best_power_at_enemy
        self.best_power_back_to_factory = (
            best_enemy_unit_finder.best_power_back_to_factory
        )

    def decide_action(self) -> str:
        action_decider = ActionDecider(
            self.unit,
            self.master,
            self.best_enemy_unit,
            self.best_intercept,
            self.best_power_at_enemy,
            self.best_power_back_to_factory,
        )
        return action_decider.decide_action()

    def execute_action(self, what_do: str) -> bool:
        action_executor = ActionExecutor(
            self.unit, self.master, what_do, self.best_enemy_unit, self.best_intercept
        )
        return action_executor.execute_action()

    def perform_attack(self) -> bool:
        self.find_interceptable_enemy_locations()
        self.find_best_enemy_unit()
        what_do = self.decide_action()
        return self.execute_action(what_do)


class CombatPlanner:
    def __init__(self, master: MasterState):
        self.master = master

    def update(self):
        pass

    def attack(self, unit: FriendlyUnitManager) -> bool:
        attack_instance = Attack(unit, self.master)
        return attack_instance.perform_attack()

    # def attack(self, unit: FriendlyUnitManager, close_units: CloseUnits) -> bool:
    #     """Note: Unit MUST move, sitting still will result in automatic death on collision"""
    #     logger.info(f"Attacking enemy")
    #
    #     # Find all interceptable enemy locations of interest
    #     enemy_location_ids = self.master.pathfinder.unit_paths.calculate_likely_unit_collisions(
    #         unit.pos,
    #         util.num_turns_of_actions(unit.action_queue),
    #         exclude_id_nums=[unit.id_num],
    #         friendly_light=False,
    #         friendly_heavy=False,
    #         # Only hunt down same type of unit
    #         enemy_light=True if unit.unit_type == "LIGHT" else False,
    #         enemy_heavy=True if unit.unit_type == "HEAVY" else False,
    #     )
    #     # Note: Only asking for one anyway, so will always be first in dict
    #     enemy_location_ids = next(iter(enemy_location_ids.values()))
    #
    #     # Find nearest enemy that has lower power or lower unit type
    #     cm = self.master.pathfinder.generate_costmap(unit, collision_only=True)
    #     power_now = unit.power_remaining(self.master.maps.rubble)
    #     best_enemy_unit = None
    #     best_intercept = None
    #     best_power_at_enemy = None
    #     best_power_back_to_factory = None
    #     for i in range(20):
    #         # Convert ids to a 1/0 map
    #         enemies_map = enemy_location_ids >= 0
    #         nearest_intercept = util.nearest_non_zero(enemies_map, unit.pos)
    #         if nearest_intercept is None:
    #             if best_enemy_unit is None:
    #                 logger.warning(f"{unit.log_prefix} No intercepts with enemy")
    #             break
    #
    #         # Get that enemy unit
    #         enemy_num = enemy_location_ids[nearest_intercept[0], nearest_intercept[1]]
    #         enemy_id = f"unit_{enemy_num}"
    #         enemy_unit = self.master.units.enemy.get_unit(enemy_id)
    #         if enemy_unit is None:
    #             logger.error(
    #                 f"{unit.log_prefix}: Found {enemy_id} as nearest enemy, but not in enemy units"
    #             )
    #             break
    #         enemy_unit: EnemyUnitManager
    #
    #         # Note: May need to check unit type if I change my mind above!
    #
    #         # Calculate costs to attack this enemy
    #         path_to_enemy = self.master.pathfinder.fast_path(
    #             unit.pos, nearest_intercept, costmap=cm
    #         )
    #         cost_to_enemy = util.power_cost_of_path(
    #             path_to_enemy, self.master.maps.rubble, unit.unit_type
    #         )
    #         path_to_factory = util.path_to_factory_edge_nearest_pos(
    #             self.master.pathfinder,
    #             unit.factory_loc,
    #             nearest_intercept,
    #             nearest_intercept,
    #             costmap=cm,
    #         )
    #         cost_to_factory = util.power_cost_of_path(
    #             path_to_factory, self.master.maps.rubble, unit.unit_type
    #         )
    #         power_at_enemy = power_now - cost_to_enemy
    #         power_back_to_factory = power_at_enemy - cost_to_factory
    #
    #         # Decide whether this is a good enemy to attack
    #         if enemy_unit.power-enemy_unit.unit_config.MOVE_COST*len(path_to_enemy) <= power_at_enemy:
    #             # If definitely good, then break
    #             if power_back_to_factory > 0:
    #                 logger.info(
    #                     f"Found good enemy unit to intercept {enemy_unit.unit_id}, doing that"
    #                 )
    #                 best_enemy_unit = enemy_unit
    #                 best_intercept = nearest_intercept
    #                 best_power_at_enemy = power_at_enemy
    #                 best_power_back_to_factory = power_back_to_factory
    #                 break
    #             # Otherwise, save this but see if there is a better option
    #             else:
    #                 logger.debug(
    #                     f"Found possible enemy unit to intercept {enemy_unit.unit_id}, looking for better"
    #                 )
    #                 best_enemy_unit = enemy_unit
    #                 best_intercept = nearest_intercept
    #                 best_power_at_enemy = power_at_enemy
    #                 best_power_back_to_factory = power_back_to_factory
    #
    #         # Remove this unit and look for next
    #         enemy_location_ids[enemy_location_ids == enemy_num] = -1
    #     else:
    #         logger.warning(
    #             f"{unit.log_prefix} Checked 20 enemy units, breaking loop now"
    #         )
    #
    #     # Decide whether to attack, do nothing, or return to factory based on the best choice of enemy unit
    #     if best_enemy_unit is None:
    #         # If no decide based on current power
    #         if power_now < unit.unit_config.BATTERY_CAPACITY * 0.5:
    #             logger.debug(f"low power ({unit.power}) returning to factory")
    #             what_do = "factory"
    #         else:
    #             logger.debug(
    #                 "No good intercept, but enough power to stay out, will set success to False so unit can be reassing if necessaary"
    #             )
    #             what_do = "nothing"
    #     else:
    #         # If enemy decide based on power cost to attack
    #         if best_power_back_to_factory < -unit.unit_config.BATTERY_CAPACITY * 0.2:
    #             logger.info(
    #                 f"Found enemy intercept {best_intercept} from {unit.pos}, but really not enough power "
    #                 f"({best_power_back_to_factory}) to get there and back to factory"
    #             )
    #             if power_now < unit.unit_config.BATTERY_CAPACITY * 0.5:
    #                 logger.debug(f"low power ({unit.power}) returning to factory")
    #                 what_do = "factory"
    #             else:
    #                 logger.debug(
    #                     f"Enough power to stay out ({power_now}) will set status to doing NOTHING"
    #                 )
    #                 what_do = "nothing"
    #         else:
    #             what_do = "attack"
    #             logger.info(
    #                 f"Attacking {best_enemy_unit.unit_id} at intercept {best_intercept}, "
    #                 f"power at enemy {best_power_at_enemy}, power at factory {best_power_back_to_factory}"
    #             )
    #
    #     # Carry out the decided action
    #     if what_do == "attack":
    #         cm = self.master.pathfinder.generate_costmap(
    #             unit, ignore_id_nums=[best_enemy_unit.id_num]
    #         )
    #         path_to_enemy = self.master.pathfinder.fast_path(
    #             unit.pos, best_intercept, costmap=cm
    #         )
    #         if len(path_to_enemy) > 1:
    #             self.master.pathfinder.append_path_to_actions(unit, path_to_enemy)
    #             return True
    #         elif len(path_to_enemy) == 1:
    #             logger.warning(
    #                 f"{unit.log_prefix}: Attacking path said to stand still, but that could mean death, moving to cheapest adjacent tile"
    #             )
    #             util.move_to_cheapest_adjacent_space(self.master.pathfinder, unit)
    #             return True
    #         else:
    #             logger.error(
    #                 f"{unit.log_prefix} error in final pathing to intercept {best_enemy_unit.unit_id}"
    #             )
    #             return False
    #     elif what_do == "factory":
    #         cm = self.master.pathfinder.generate_costmap(
    #             unit,
    #         )
    #         path_to_factory = util.path_to_factory_edge_nearest_pos(
    #             self.master.pathfinder,
    #             unit.factory_loc,
    #             unit.pos,
    #             unit.pos,
    #             costmap=cm,
    #         )
    #         if len(path_to_factory) > 0:
    #             self.master.pathfinder.append_path_to_actions(unit, path_to_factory)
    #             return True
    #         else:
    #             logger.error(f"{unit.log_prefix} error in pathing back to factory")
    #             return False
    #     elif what_do == "nothing":
    #         # If none, if need to move, move, else do nothing
    #         if cm[unit.pos[0], unit.pos[1]] <= 0:
    #             logger.info(f"Had to move, so moving to cheapest spot")
    #             util.move_to_cheapest_adjacent_space(self.master.pathfinder, unit)
    #         # unit.update_status(NOTHING, success=True)
    #         return False

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
