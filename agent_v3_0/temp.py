import agent_v3_0.unit_status
from typing import Tuple


class CombatPlanner:
    def __init__(self, master: MasterState):
        self.master = master

        # Keep dict of enemy unit_id, friendly unit attacking
        self.targeted_enemies: Dict[str, FriendlyUnitManager] = {}

    def update(self):
        keys_to_pop = []
        for enemy_id, friendly_unit in self.targeted_enemies.items():
            if friendly_unit.status.current_action != Actions.COMBAT:
                keys_to_pop.append(enemy_id)
                logger.info(f"{self.targeted_enemies[enemy_id].unit_id} no longer targeting {enemy_id}")
        for k in keys_to_pop:
            self.targeted_enemies.pop(k)

    def attack(self, unit: FriendlyUnitManager) -> bool:
        attack_instance = Attack(unit, self.master, self.targeted_enemies)
        success = attack_instance.perform_attack()

        if success and attack_instance.best_enemy_unit is not None and attack_instance.action_executed == "attack":
            enemy_id = attack_instance.best_enemy_unit.unit_id
            if enemy_id in self.targeted_enemies and self.targeted_enemies[enemy_id].unit_id != unit.unit_id:
                logger.warning(
                    f"{enemy_id} already targeted by {self.targeted_enemies[enemy_id].unit_id}, now {unit.unit_id} is targeting too"
                )
            self.targeted_enemies[enemy_id] = unit
        return success

    ### Can probably separate below into separate class ###
    def process_attack_strategy(self, unit: FriendlyUnitManager) -> None:
        current_action = unit.status.current_action

        if current_action == Actions.RETREAT_HOLD:
            self.process_retreat_hold(unit)
        elif current_action == Actions.ATTACK_HOLD:
            self.process_attack_hold(unit)
        elif current_action == Actions.ATTACKING_TEMPORARY:
            self.process_attacking_temporary(unit)
        elif current_action == Actions.ATTACK_CONSERVE:
            self.process_attack_conserve(unit)

    def process_retreat_hold(self, unit: FriendlyUnitManager) -> None:
        if not unit.collisions_in_path_to_next_dest():
            if unit.at_location(unit.status.next_dest):
                if unit.enemies_to_attack_within(agent_v3_0.unit_status.AttackValues.hold_radius):
                    unit.status.current_action = Actions.ATTACK_HOLD
                    unit.status.replan_required = True
        else:
            unit.status.action_queue_empty_ok = True

    def process_attack_hold(self, unit: FriendlyUnitManager) -> None:
        if agent_v3_0.unit_status.AttackValues.hold_position is None:
            agent_v3_0.unit_status.AttackValues.hold_position = unit.position

            # Add a warning or error if the unit is on a factory or resource tile
            # (You'll need to implement a function to check this condition)
            if unit.is_on_factory_or_resource_tile():
                logger.warning("Unit is on a factory or resource tile")

        enemies_to_attack = self.enemies_within_radius(
            agent_v3_0.unit_status.AttackValues.hold_position, agent_v3_0.unit_status.AttackValues.attack_radius
        )
        self.attack(enemies_to_attack)

    def process_attacking_temporary(self, unit: FriendlyUnitManager) -> None:
        if agent_v3_0.unit_status.AttackValues.attack_num_remaining > 0:
            self.continue_attacking()
            agent_v3_0.unit_status.AttackValues.attack_num_remaining -= 1
        else:
            unit.status.current_action = unit.status.previous_action
            unit.status.replan_required = True

    def process_attack_conserve(self, unit: FriendlyUnitManager) -> None:
        # You'll need to implement a function to check if the unit's previous steps were repeated
        if unit.repeated_previous_steps():
            target_dist = get_power_proportion(
                self.unit.available_power(), self.unit.status.min_conserve_radius, self.unit.status.max_reserve_radius
            )
            self.path_toward_factory(dist=target_dist)

            # You'll need to implement a function to check if the unit is closer to the factory
            if unit.is_closer_to_factory_on_lower_power():
                self.move_closer_to_factory()

            # If the unit doesn't need to move, it can stand still and conserve power
            self.stand_still_and_conserve_power()
        else:
            self.attack(enemy)

    #### Maybe below here into another ####
    def perform_attack(self, unit: FriendlyUnitManager, enemy: EnemyUnitManager) -> None:
        dist = unit.distance_to(enemy)
        eliminate_threshold = agent_v3_0.unit_status.AttackValues.eliminiate_threshold

        if dist < eliminate_threshold:
            # Can path directly to enemy for most distances
            if dist == 1 or (dist == 2 and unit.power > enemy.power) or dist > 3:
                unit.path_directly_to_enemy_position(enemy)
                # Have to be careful at the last step towards enemy (dist 2)
            else:
                if unit.status.previous_direction_of_travel:
                    unit.path_to_lowest_cost_perpendicular()
                else:
                    unit.path_away_from_enemy(enemy)
        else:
            # if unit.already_within_eliminate_range(enemy):
            next_dest = unit.status.next_dest
            if util.manhattan(next_dest, enemy.pos) < eliminate_threshold:
                # Carry on moving and wait to pounce
                pass
            else:
                # Path to nearest within kill threshold
                # with current path weighting decreased by a small amount
                self.path_to_nearest_one_away_from_enemy(enemy)


class MineUnitPlanner(BaseUnitPlanner):
    def __init__(self, master: MasterState, general_planner: BaseGeneralPlanner, unit: UnitManager):
        super().__init__(master, general_planner, unit)

    def update_planned_actions(self):
        # Implement the update_planned_actions logic
        pass

    def create_new_actions(self):
        # Implement the create_new_actions logic
        pass

    def path_to_best_resource(self):
        resources = (
            self.planner.get_sorted_resources()
        )  # Replace with the method that gets the sorted list of resources

        for resource in resources:
            enemy = self.master.pathfinder.check_enemy_at_resource(
                resource
            )  # Replace with the method that checks for an enemy at the resource
            if enemy:
                if enemy.power < self.unit.power and enemy.type == self.unit.type:
                    self.unit.status.current_action = Actions.ATTACKING_TEMPORARY
                    self.unit.status.attacking_num_remaining = 5
                    self.unit.status.replan_required = True
                    break
                # No 'elif' case here, as we just move on to the next resource
            else:
                self.unit.path_to_resource(resource)  # Replace with the method that paths the unit to the resource
                break
        else:
            # If no suitable resources found or power is low
            self.unit.path_to_factory_waiting_area()  # Replace with the method that paths the unit to the factory waiting area
            self.unit.status.current_action = Actions.NOTHING
            # If the unit is already on the factory and needs to re-run decision-making, you can add logic here

    def calculate_resource_value(self, resource: Tuple[int, int]) -> float:
        value = self.master.get_resource_value(
            resource
        )  # Replace with the method that gets the base value of the resource

        friendly_units_occupying = self.master.get_friendly_units_occupying_resource(
            resource
        )  # Replace with the method that returns the number of friendly units occupying the resource

        if self.master.distance_to_factory(resource) < 4:
            value = 0
        else:
            value /= 2**friendly_units_occupying

        if self.unit.type == "heavy":  # Assuming there are "heavy" and "light" unit types
            value *= 1 - self.master.get_light_units_occupying_resource(
                resource
            )  # Replace with the method that returns the number of light units occupying the resource

        return value

    def choose_resource_to_mine(self):
        resource_candidates = self.unit.status.mine_values.resource_locations

        for resource in resource_candidates:
            resource_value = self.calculate_resource_value(resource)
            if resource_value >= self.unit.status.mine_values.min_resource_value_cutoff:
                self.unit.path_to_resource(resource)  # Replace with the method that paths the unit to the resource
                self.unit.status.current_action = Actions.MINE  # Assuming there's an action called MINE
                break
        else:
            self.unit.status.current_action = Actions.NOTHING
            self.unit.status.replan_required = True


from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class ActionProportions:
    mine_ice: float
    mine_ore: float
    clear_rubble: float
    clear_lichen: float
    attack_mining_units: float
    attack_nearby_units: float
    guard_location: float


def interpolate_proportions(target_proportions: List[Tuple[int, ActionProportions]]) -> Dict[str, np.ndarray]:
    target_proportions.sort(key=lambda x: x[0])  # Sort by step
    actions = list(target_proportions[0][1].__dict__.keys())

    interpolated_proportions = {action: np.zeros(1001) for action in actions}

    for i in range(len(target_proportions) - 1):
        step0, proportions0 = target_proportions[i]
        step1, proportions1 = target_proportions[i + 1]

        for action in actions:
            start_prop = getattr(proportions0, action)
            end_prop = getattr(proportions1, action)
            interpolated_proportions[action][step0 : step1 + 1] = np.linspace(start_prop, end_prop, step1 - step0 + 1)

    return interpolated_proportions


# Example usage:
target_proportions = [
    (
        0,
        ActionProportions(
            mine_ice=0.2,
            mine_ore=0.2,
            clear_rubble=0.2,
            clear_lichen=0.2,
            attack_mining_units=0.1,
            attack_nearby_units=0.1,
            guard_location=0.0,
        ),
    ),
    (
        1000,
        ActionProportions(
            mine_ice=0.2,
            mine_ore=0.2,
            clear_rubble=0.2,
            clear_lichen=0.2,
            attack_mining_units=0.1,
            attack_nearby_units=0.1,
            guard_location=0.0,
        ),
    ),
]

interpolated_proportions = interpolate_proportions(target_proportions)

# To access the proportion of a specific action at a specific step, use:
# interpolated_proportions[action][step]
