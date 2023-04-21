from __future__ import annotations
from typing import TYPE_CHECKING, Dict

import numpy as np

from base_planners import BaseUnitPlanner, BaseGeneralPlanner
from unit_status import ActCategory, CombatActSubCategory
from config import get_logger
import actions_util
import util

logger = get_logger(__name__)

if TYPE_CHECKING:
    from master_state import MasterState
    from unit_action_planner import CloseUnits
    from unit_manager import FriendlyUnitManager, EnemyUnitManager


class ActionDecider:
    def __init__(
        self,
        unit: FriendlyUnitManager,
        master: MasterState,
        best_enemy_unit: EnemyUnitManager,
        best_intercept: util.POS_TYPE,
        best_power_at_enemy: int,
        best_power_back_to_factory: int,
    ):
        self.unit = unit
        self.master = master
        self.best_enemy_unit = best_enemy_unit
        self.best_intercept = best_intercept
        self.best_power_at_enemy = best_power_at_enemy
        self.best_power_back_to_factory = best_power_back_to_factory

    def decide_action(self) -> str:
        if self.best_enemy_unit is None:
            logger.debug(f"Deciding attack action with no best enemy")
            return self.decide_action_no_enemy()
        else:
            logger.debug(
                f"Deciding attack action for {self.best_enemy_unit.unit_id} at {self.best_enemy_unit.pos}, best intercept = {self.best_intercept}"
            )
            return self.decide_action_with_enemy()

    def decide_action_no_enemy(self) -> str:
        power_now = self.unit.power_remaining()
        if power_now < self.unit.unit_config.BATTERY_CAPACITY * 0.5:
            logger.debug(f"low power ({self.unit.power}) returning to factory")
            return "factory"
        else:
            logger.debug("No good intercept, but enough power to stay out, won't do anything")
            return "nothing"

    def decide_action_with_enemy(self) -> str:
        if self.best_power_back_to_factory < -self.unit.unit_config.BATTERY_CAPACITY * 0.2:
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
            logger.debug(f"Enough power to stay out ({power_now}) will do nothing")
            return "nothing"

    def decide_action_attack(self) -> str:
        logger.info(
            f"Attacking {self.best_enemy_unit.unit_id} at intercept {self.best_intercept}, power at enemy {self.best_power_at_enemy}, power at factory {self.best_power_back_to_factory}"
        )
        return "attack"


class ActionExecutor:
    def __init__(
        self,
        unit: FriendlyUnitManager,
        master: MasterState,
        what_do: str,
        best_enemy_unit: EnemyUnitManager,
        best_intercept: util.POS_TYPE,
    ):
        self.unit = unit
        self.master = master
        self.what_do = what_do
        self.best_enemy_unit = best_enemy_unit
        self.best_intercept = best_intercept

    def execute_action(self) -> bool:
        if self.what_do == "attack":
            logger.debug(f"Executing attack")
            return self.execute_attack()
        elif self.what_do == "factory":
            logger.debug(f"Executing go to factory")
            return self.execute_factory()
        elif self.what_do == "nothing":
            logger.debug(f"Executing do nothing")
            return self.execute_nothing()
        return False

    def execute_attack(self) -> bool:
        if self.best_enemy_unit is None:
            raise ValueError(f"Must have enemy unit to attack")

        # Probably we have more power if near max capacity, then can ignore enemies in pathing
        avoid_other_light = (
            False
            if self.unit.unit_type == "HEAVY" or self.unit.power > 0.8 * self.unit.unit_config.BATTERY_CAPACITY
            else True
        )
        avoid_other_heavy = (
            False
            if self.unit.unit_type == "HEAVY" and self.unit.power > 0.8 * self.unit.unit_config.BATTERY_CAPACITY
            else True
        )
        logger.debug(
            f"avoid_light = {avoid_other_light}, avoid_heavy = {avoid_other_heavy}, ignoring {self.best_enemy_unit.unit_id}"
        )
        cm = self.master.pathfinder.generate_costmap(
            self.unit,
            ignore_id_nums=[self.best_enemy_unit.id_num],
            enemy_light=avoid_other_light,
            enemy_heavy=avoid_other_heavy,
            # collision_only=True,  # don't collide with other units, but don't avoid enemies either
            enemy_nearby_start_cost=0,
        )
        #  Path to enemy  (with larger margin to allow for navigating around things better)
        path_to_enemy = self.master.pathfinder.fast_path(self.unit.pos, self.best_intercept, costmap=cm, margin=5)
        # if self.unit.unit_id == "unit_27":
        #     fig = util.show_map_array(cm)
        #     if len(path_to_enemy) > 0:
        #         util.plotly_plot_path(fig, path_to_enemy)
        #     fig.show()

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
            logger.warning(
                f"{self.unit.log_prefix} failed in final pathing to intercept {self.best_enemy_unit.unit_id} at intercept {self.best_intercept}. Enemy currently at {self.best_enemy_unit.pos}"
            )
            return False

    def execute_factory(self) -> bool:
        if self.best_enemy_unit:
            ignores = [self.best_enemy_unit.id_num]
        else:
            ignores = []
        cm = self.master.pathfinder.generate_costmap(self.unit, ignore_id_nums=ignores)
        path_to_factory = util.path_to_factory_edge_nearest_pos(
            self.master.pathfinder,
            self.unit.factory_loc,
            self.unit.pos,
            self.unit.pos,
            costmap=cm,
        )

        if len(path_to_factory) > 0:
            self.master.pathfinder.append_path_to_actions(self.unit, path_to_factory)
            power_required = self.unit.unit_config.BATTERY_CAPACITY - self.unit.power_remaining()
            if util.num_turns_of_actions(self.unit.action_queue) < 2:
                if self.unit.factory.short_term_power < power_required:
                    logger.info(f"Probably not enough power at factory, skipping adding power pickup")
                    return True
            logger.debug(f"Adding power pickup back at factory. Aiming to pickup {power_required}")
            self.unit.action_queue.append(self.unit.pickup(util.POWER, power_required))
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
        targeted_enemies: Dict[str, FriendlyUnitManager],
    ):
        self.master = master
        self.unit = unit
        self.enemy_location_ids = enemy_location_ids
        self.targeted_enemies = targeted_enemies
        self.best_enemy_unit = None
        self.best_intercept = None
        self.best_power_at_enemy = None
        self.best_power_back_to_factory = None

    def calculate_power_at_enemy_and_factory(self, path_to_enemy, path_to_factory):
        power_now = self.unit.power_remaining(self.master.maps.rubble)
        cost_to_enemy = util.power_cost_of_path(path_to_enemy, self.master.maps.rubble, self.unit.unit_type)
        cost_to_factory = util.power_cost_of_path(path_to_factory, self.master.maps.rubble, self.unit.unit_type)
        power_at_enemy = power_now - cost_to_enemy
        power_back_to_factory = power_at_enemy - cost_to_factory

        return power_at_enemy, power_back_to_factory

    def likely_enemy_power(self, enemy_unit: EnemyUnitManager, path_to_enemy) -> int:
        actions_to_step = actions_util.split_actions_at_step(enemy_unit.action_queue, len(path_to_enemy))[0]
        turns_of_actions = util.num_turns_of_actions(actions_to_step)
        action_cost = util.power_cost_of_actions(
            enemy_unit.start_of_turn_pos,
            self.master.maps.rubble,
            enemy_unit,
            actions_to_step,
        )
        if turns_of_actions < 0.7 * len(path_to_enemy):
            extra_turns = turns_of_actions - len(path_to_enemy)
            action_cost += extra_turns * enemy_unit.unit_config.MOVE_COST
        return enemy_unit.power - action_cost

    def update_best_enemy_unit(self, enemy_unit, intercept, power_at_enemy, power_back_to_factory):
        self.best_enemy_unit = enemy_unit
        self.best_intercept = intercept
        self.best_power_at_enemy = power_at_enemy
        self.best_power_back_to_factory = power_back_to_factory

    def remove_current_enemy(self, enemy_num):
        self.enemy_location_ids[self.enemy_location_ids == enemy_num] = -1

    def find_best_enemy_unit(self):
        logger.debug(f"Finding best enemy unit to attack")
        cm = self.master.pathfinder.generate_costmap(
            self.unit,
            # using collision only can result in pathing to enemy failing (well if I do collision only for both is that ok?)
            # collision_only=True,
            enemy_nearby_start_cost=0,
            enemy_light=False,
            enemy_heavy=True if self.unit.unit_type == "LIGHT" else False,
        )
        # if self.unit.unit_id == "unit_41":
        #     print("here")
        #     util.show_map_array(cm).show()
        for i in range(20):
            enemies_map = self.enemy_location_ids >= 0
            nearest_intercept = util.nearest_non_zero(enemies_map, self.unit.pos)
            # print(f'Nearest intercept = {nearest_intercept}')
            if nearest_intercept is None:
                if self.best_enemy_unit is None:
                    logger.info(f"{self.unit.log_prefix} No intercepts with enemy")
                break

            enemy_num = self.enemy_location_ids[nearest_intercept[0], nearest_intercept[1]]
            enemy_id = f"unit_{enemy_num}"
            enemy_unit = self.master.units.enemy.get_unit(enemy_id)
            if enemy_unit is None:
                logger.error(f"{self.unit.log_prefix}: Found {enemy_id} as nearest enemy, but not in enemy units")
                break
            elif enemy_id in self.targeted_enemies and self.targeted_enemies[enemy_id].unit_id != self.unit.unit_id:
                logger.debug(
                    f"{self.unit.unit_id} not targetting {enemy_id} because its already targeted by {self.targeted_enemies[enemy_id].unit_id}"
                )
                self.remove_current_enemy(enemy_num)
                continue
            enemy_unit: EnemyUnitManager

            path_to_enemy = self.master.pathfinder.fast_path(self.unit.pos, nearest_intercept, costmap=cm)
            if len(path_to_enemy) == 0:
                logger.debug(f"No path to {enemy_id}")
                self.remove_current_enemy(enemy_num)
                continue

            # print(f'len path to enemy {len(path_to_enemy)}')
            path_to_factory = util.path_to_factory_edge_nearest_pos(
                self.master.pathfinder,
                self.unit.factory_loc,
                nearest_intercept,
                nearest_intercept,
                costmap=cm,
            )
            # print(f'len path to factory {len(path_to_factory)}')
            (
                power_at_enemy,
                power_back_to_factory,
            ) = self.calculate_power_at_enemy_and_factory(path_to_enemy, path_to_factory)

            # Is this a good enemy to attack
            likely_enemy_power = self.likely_enemy_power(enemy_unit, path_to_enemy)
            if likely_enemy_power < power_at_enemy:
                if power_back_to_factory > 0:
                    logger.info(f"Found good enemy unit to intercept {enemy_unit.unit_id}, doing that")
                    self.update_best_enemy_unit(
                        enemy_unit,
                        nearest_intercept,
                        power_at_enemy,
                        power_back_to_factory,
                    )
                    break
                else:
                    logger.debug(f"Found possible enemy unit to intercept {enemy_unit.unit_id}, looking for better")
                    self.update_best_enemy_unit(
                        enemy_unit,
                        nearest_intercept,
                        power_at_enemy,
                        power_back_to_factory,
                    )

            # logger.debug(f'{enemy_id} not a good target for {self.unit.unit_id}')
            # print(f'{enemy_id} not a good target for {self.unit.unit_id}')
            # Remove that unit from intercepts and try again
            self.remove_current_enemy(enemy_num)
        else:
            logger.warning(f"{self.unit.log_prefix} Checked 20 enemy units, breaking loop now")
        if self.best_enemy_unit is None:
            logger.info(f"failed to find a good unit to attack for {self.unit.unit_id} from {self.unit.pos}")


class Attack:
    def __init__(
        self,
        unit: FriendlyUnitManager,
        master: MasterState,
        targeted_enemies: Dict[str, FriendlyUnitManager],
    ):
        self.unit = unit
        self.master = master
        self.targeted_enemies = targeted_enemies
        self.targeted_enemies_reversed = {unit.unit_id: enemy_id for enemy_id, unit in targeted_enemies.items()}

        self.enemy_location_ids = None
        self.best_enemy_unit = None
        self.best_intercept = None
        self.best_power_at_enemy = None
        self.best_power_back_to_factory = None

        self.action_executed: str = None

    def find_interceptable_enemy_locations(self, specific_id_num: int = None):
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
        # if self.unit.unit_id == 'unit_30':
        #     util.show_map_array(self.enemy_location_ids).show()
        if specific_id_num is not None:
            self.enemy_location_ids[self.enemy_location_ids != specific_id_num] = -1

    def find_best_enemy_unit(self):
        # Instantiate BestEnemyUnit class and call find_best_enemy_unit method
        best_enemy_unit_finder = BestEnemyUnit(self.master, self.unit, self.enemy_location_ids, self.targeted_enemies)
        best_enemy_unit_finder.find_best_enemy_unit()

        # Update the Attack class attributes with the results
        self.best_enemy_unit = best_enemy_unit_finder.best_enemy_unit
        self.best_intercept = best_enemy_unit_finder.best_intercept
        self.best_power_at_enemy = best_enemy_unit_finder.best_power_at_enemy
        self.best_power_back_to_factory = best_enemy_unit_finder.best_power_back_to_factory

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
        action_executor = ActionExecutor(self.unit, self.master, what_do, self.best_enemy_unit, self.best_intercept)
        return action_executor.execute_action()

    def continue_attack(self) -> bool:
        enemy_id = self.targeted_enemies_reversed[self.unit.unit_id]
        enemy = self.master.units.enemy.get_unit(enemy_id)
        if enemy is None:
            logger.warning(f"{self.unit.log_prefix} enemy {enemy_id} no longer exists")
            self.targeted_enemies.pop(enemy_id)
            return False
        enemy: EnemyUnitManager
        self.best_enemy_unit = enemy
        self.find_interceptable_enemy_locations(specific_id_num=enemy.id_num)
        enemy_map = (self.enemy_location_ids >= 0).astype(int)
        nearest_intercept = util.nearest_non_zero(enemy_map, self.unit.start_of_turn_pos)

        # If new intercept, maybe can get there better
        if nearest_intercept is not None:
            intercept_valid = self.master.maps.valid_friendly_move[nearest_intercept[0], nearest_intercept[1]] > 0
            if not intercept_valid:
                # probably on own factory
                return False
            else:
                logger.info(
                    f"Continuing attack on {enemy_id}, current dist = {util.manhattan(self.unit.start_of_turn_pos, enemy.start_of_turn_pos)}"
                )
                self.best_intercept = nearest_intercept
                return self.execute_action("attack")
        elif len(self.unit.status.planned_action_queue) > 0:
            logger.info(f"didn't find new intercept for enemy, continuing with previous path")
            # Carry on, enemy will probably pop up again
            self.unit.action_queue = self.unit.status.planned_action_queue.copy()
            return True
        else:
            logger.info(f"no intercept with enemy, and no planned actions, lost enemy, returning false")
            # We've lost the enemy
            return False

    def perform_attack(self) -> bool:
        logger.debug(f"Starting attack planning")
        success = False
        # Continue previous attack
        if self.unit.unit_id in self.targeted_enemies_reversed:
            success = self.continue_attack()

        if not success:
            self.find_interceptable_enemy_locations()
            logger.debug(f"finding best enemy")
            self.find_best_enemy_unit()
            logger.debug(f"Decding what to do next")
            what_do = self.decide_action()
            self.action_executed = what_do
            logger.debug(f"Executing attack behavior {what_do}")
            success = self.execute_action(what_do)
        return success


class CombatUnitPlanner(BaseUnitPlanner):
    def update_planned_actions(self):
        if len(self.unit.status.planned_action_queue) == 0:
            return self.create_new_actions()
        else:
            return self.create_new_actions()

    def create_new_actions(self):
        return self.attack(self.unit)

    def attack(self, unit: FriendlyUnitManager) -> bool:
        attack_instance = Attack(unit, self.master, self.planner.targeted_enemies)
        success = attack_instance.perform_attack()
        if success and attack_instance.best_enemy_unit is not None and attack_instance.action_executed == "attack":
            enemy_id = attack_instance.best_enemy_unit.unit_id
            if (
                enemy_id in self.planner.targeted_enemies
                and self.planner.targeted_enemies[enemy_id].unit_id != unit.unit_id
            ):
                logger.warning(
                    f"{enemy_id} already targeted by {self.planner.targeted_enemies[enemy_id].unit_id}, now {unit.unit_id} is targeting too"
                )
            self.planner.targeted_enemies[enemy_id] = unit
        return success

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
            cm = self.master.pathfinder.generate_costmap(unit, enemy_light=False, enemy_heavy=False)
            path_to_factory = util.calc_path_to_factory(
                self.master.pathfinder,
                costmap=cm,
                pos=unit.pos,
                factory_loc=self.master.factories.friendly[unit.factory_id].factory_loc,
            )
            if len(path_to_factory) > 0:
                logger.warning(f"{unit.log_prefix}: Path to factory only after allowing collisions with enemy")
                self.master.pathfinder.append_path_to_actions(unit, path_to_factory)
                return True
            logger.error(f"{unit.log_prefix}: No path to factory")
            return False


class CombatPlanner(BaseGeneralPlanner):
    def __init__(self, master: MasterState):
        super().__init__(master)
        self.master = master

        # Keep dict of enemy unit_id, friendly unit attacking
        self.targeted_enemies: Dict[str, FriendlyUnitManager] = {}
        self.single_unit_planners: Dict[str, CombatUnitPlanner] = {}

        # Update these each turn
        # Map of current enemy locations and values of attacking that enemy (probably want to account for factory dist when using these)
        self.light_enemy_value_map: np.ndarray = np.zeros(self.master.maps.map_shape)
        self.heavy_enemy_value_map: np.ndarray = np.zeros(self.master.maps.map_shape)

    def update(self):
        # Keep track of targeted enemies (if unit no longer attacking, remove from targeted)
        keys_to_pop = []
        for enemy_id, friendly_unit in self.targeted_enemies.items():
            act_status = friendly_unit.status.current_action
            if act_status.category != ActCategory.COMBAT or act_status.sub_category in [
                CombatActSubCategory.RUN_AWAY,
                CombatActSubCategory.RETREAT_HOLD,
            ]:
                keys_to_pop.append(enemy_id)
                logger.info(f"{self.targeted_enemies[enemy_id].unit_id} no longer targeting {enemy_id}")
        for k in keys_to_pop:
            self.targeted_enemies.pop(k)

        # Update enemy values
        self.light_enemy_value_map = np.zeros(self.master.maps.map_shape)
        self.heavy_enemy_value_map = np.zeros(self.master.maps.map_shape)
        for enemy_id, enemy_unit in self.master.units.enemy.all.items():
            cfg = enemy_unit.unit_config
            pos = enemy_unit.pos
            value = 0
            # If some cargo
            if enemy_unit.cargo.ice + enemy_unit.ore > cfg.CARGO_SPACE * 0.3:
                value += 20
            # If more cargo
            if enemy_unit.cargo.ice + enemy_unit.ore > cfg.CARGO_SPACE * 0.6:
                value += 20
            # If currently on a resource
            if self.master.maps.resource_at_tile(enemy_unit.pos) >= 0:
                value += 40
            # On its own factory (untouchable)
            if self.master.maps.factory_maps.enemy[pos[0], pos[1]] >= 0:
                value = 0
            if enemy_unit.unit_type == "LIGHT":
                self.light_enemy_value_map[pos[0], pos[1]] = value
            else:
                self.heavy_enemy_value_map[pos[0], pos[1]] = value

    def get_unit_planner(self, unit: FriendlyUnitManager) -> CombatUnitPlanner:
        """Return a subclass of BaseUnitPlanner to actually update or create new actions for a single Unit"""
        if unit.unit_id not in self.unit_planners:
            unit_planner = CombatUnitPlanner(self.master, self, unit)
            self.unit_planners[unit.unit_id] = unit_planner
        return self.unit_planners[unit.unit_id]
