import copy
from enum import Enum, auto
from typing import Dict, Optional
from dataclasses import dataclass

import numpy as np

import util
from master_state import MasterState
from config import get_logger

from unit_manager import FriendlyUnitManager, EnemyUnitManager, UnitManager
from unit_status import ActCategory, ActStatus, CombatActSubCategory

logger = get_logger(__name__)


class CombatPlanner:
    def __init__(self, unit):
        self.unit = unit

    def plan(self):
        # Plan the combat actions
        # 1. Determine the attack mode (e.g., hold, attack, retreat, etc.)
        # 2. Find the nearest enemy target based on the attack mode
        # 3. Path to the target or hold position
        # 4. Add attack actions
        # 5. Check and handle any action flags
        pass


class GeneralPlanner:
    def __init__(self, unit):
        self.unit = unit

    def plan(self):
        # Plan the general actions
        # 1. Determine the general task for the unit (e.g., patrolling, guarding, etc.)
        # 2. Find the target or destination for the task
        # 3. Path to the target or destination
        # 4. Add any required actions related to the task
        # 5. Check and handle any action flags
        pass


class ActionHandler:
    class HandleStatus(Enum):
        SUCCESS = auto()
        INVALID_FIRST_STEP = auto()
        MAX_STEPS_REACHED_PAUSE = auto()

        LOW_POWER_RETURNING = auto()
        ENEMY_NEAR_ATTACKING = auto()
        ENEMY_NEAR_WAIT = auto()
        ENEMY_NEAR_FLEEING = auto()
        # When pathing back to factory etc fails
        PATHING_FAILED = auto()
        # When provided path is invalid (len 0)
        PATH_INVALID_RETURNING = auto()
        DIG_INVALID_RETURNING = auto()
        PICKUP_INVALID_RETURNING = auto()
        TRANSFER_INVALID_RETURNING = auto()

    def __init__(self, master: MasterState, unit: FriendlyUnitManager, max_queue_step_length: int):
        self.master = master
        self.unit = unit
        self.max_queue_step_length = max_queue_step_length

    @property
    def rubble(self):
        return self.master.maps.rubble

    @property
    def pathfinder(self):
        return self.master.pathfinder

    def _add_current_act_status(self, num=1, targeting_enemy=None, allow_partial=None):
        """
        Add the current act status for as many steps as there are actions (i.e. ActStatuses should match with actions)
        """
        if targeting_enemy is not None:
            self.unit.status.current_action.targeting_enemy = targeting_enemy
        if allow_partial is not None:
            self.unit.status.current_action.allow_partial = allow_partial

        act_status = self.unit.status.current_action.copy()
        self.unit.act_statuses.extend([act_status] * num)

    def add_actions_to_queue(self, actions, targeting_enemy=None, allow_partial=None) -> HandleStatus:
        """
        Add actions to the action queue after checking queue length
        Note: This should be the ONLY way actions are added to unit action queue

        Checks:
            If max_num_steps will be exceeded, don't add, return MAX_STEPS_REACHED_PAUSE statu-
        """
        num_new_steps = util.num_turns_of_actions(actions)
        current_action_steps = util.num_turns_of_actions(self.unit.action_queue)

        # If num steps will exceed max planning steps pause before this move
        if current_action_steps + num_new_steps > self.max_queue_step_length:
            return self.HandleStatus.MAX_STEPS_REACHED_PAUSE

        # Otherwise update the action queue and add ActStatus along with each action
        self.unit.action_queue.extend(actions)
        self._add_current_act_status(len(actions), targeting_enemy=targeting_enemy, allow_partial=allow_partial)
        return self.HandleStatus.SUCCESS

    def get_costmap(self):
        """Get costmap at current step"""
        cm = self.pathfinder.generate_costmap(
            self.unit,
            ignore_id_nums=[self.unit.id_num],
            friendly_light=True,
            friendly_heavy=True,
            enemy_light=None,
            enemy_heavy=True,
            # TODO: Check whether collision only works OK
            collision_only=True,
        )
        return cm

    def path_to_nearest_non_zero(self, non_zero_array: np.ndarray, from_pos=None) -> np.ndarray:
        """Calculate path to nearest available non-zero in array"""
        pos = from_pos if from_pos is not None else self.unit.pos
        path = util.calculate_path_to_nearest_non_zero(
            self.pathfinder,
            self.get_costmap(),
            from_pos=pos,
            target_array=non_zero_array,
            near_pos=pos,
            max_attempts=20,
            margin=2,
        )
        return path

    def path_to_factory_cost(self, from_pos: util.POS_TYPE = None) -> int:
        """Calculate cost of pathing to factory from_pos (or last pos of unit)"""
        pos = from_pos if from_pos is not None else self.unit.pos
        path = self.path_to_factory(from_pos=pos)
        return util.power_cost_of_path(path, self.rubble, self.unit.unit_type)

    def path_to_factory(self, from_pos: util.POS_TYPE = None) -> np.ndarray:
        """Get path to factory tile"""
        pos = from_pos if from_pos is not None else self.unit.pos
        array = self.unit.factory.factory_loc
        path = self.path_to_nearest_non_zero(array, from_pos=pos)
        return path

    def path_to_factory_queue(self) -> np.ndarray:
        """Get path to factory queue"""
        array = self.unit.factory.queue_array
        path = self.path_to_nearest_non_zero(array)
        return path

    def add_dropoff(self) -> HandleStatus:
        """Assumes already at factory, then drops off any resources necessary"""
        SUCCESS = self.HandleStatus.SUCCESS
        cargo = self.unit.cargo
        status = None
        if cargo.ice > 0:
            status = self.add_transfer(resource_type=util.ICE, direction=util.CENTER, to_unit=False)
            if status != SUCCESS:
                return status
        if cargo.ore > 0:
            status = self.add_transfer(resource_type=util.ORE, direction=util.CENTER, to_unit=False)
            if status != SUCCESS:
                return status
        if cargo.metal > 0:
            status = self.add_transfer(resource_type=util.METAL, direction=util.CENTER, to_unit=False)
            if status != SUCCESS:
                return status
        if cargo.water > 0:
            status = self.add_transfer(resource_type=util.WATER, direction=util.CENTER, to_unit=False)
            if status != SUCCESS:
                return status
        if status is None:
            logger.warning(f'{self.unit.log_prefix} tried to dropoff with no cargo')
            return SUCCESS
        return status

    def return_to_factory(self) -> HandleStatus:
        """
        Use this to return action to factory because of failed action (e.g. low power, enemy near, etc)

        If unit has resources, this will return unit to factory, add transfer, and set DROPOFF status
        If unit has no resources, this will return unit to nearest factory queue spot, and set WAITING status
        """
        # If unit has cargo, path direct to factory and set DROPOFF
        if self.unit.cargo_total > 0:
            path_to_factory = self.path_to_factory()
            status = self.add_path(path_to_factory)
            if status != self.HandleStatus.SUCCESS:
                return status
            status = self.add_dropoff()
            if status != self.HandleStatus.SUCCESS:
                return status

            self.unit.status.update_action_status(ActStatus(category=ActCategory.DROPOFF))
            return self.HandleStatus.SUCCESS
        else:
            path_to_queue = self.path_to_factory_queue()
            status = self.add_path(path_to_queue)
            if status != self.HandleStatus.SUCCESS:
                return status

            self.unit.status.update_action_status(ActStatus(category=ActCategory.WAITING))
            return self.HandleStatus.SUCCESS

    def get_units_near(
        self,
        pos=None,
        radius=5,
        step=None,
        friendly_light=True,
        friendly_heavy=True,
        enemy_light=True,
        enemy_heavy=True,
    ) -> Dict[str, UnitManager]:
        """Get any enemy unit near pos"""
        pos = pos if pos is not None else self.unit.pos
        step = step if step is not None else util.num_turns_of_actions(self.unit.action_queue)
        unit_nums_array = self.pathfinder.unit_paths.get_unit_nums_near(
            pos,
            step=step,
            radius=radius,
            friendly_light=friendly_light,
            friendly_heavy=friendly_heavy,
            enemy_light=enemy_light,
            enemy_heavy=enemy_heavy,
        )
        unit_dict = {}
        for num in np.unique(unit_nums_array):
            unit_id = f"unit_{num}"
            unit = self.master.units.get_unit(unit_id)
            if unit is not None:
                unit_dict[unit_id] = unit
        return unit_dict

    def get_nearest_unit_near(
        self,
        pos=None,
        radius=5,
        step=None,
        friendly_light=True,
        friendly_heavy=True,
        enemy_light=True,
        enemy_heavy=True,
    ) -> Optional[UnitManager]:
        units = self.get_units_near(
            pos=pos,
            step=step,
            radius=radius,
            friendly_light=friendly_light,
            friendly_heavy=friendly_heavy,
            enemy_light=enemy_light,
            enemy_heavy=enemy_heavy,
        )

        nearest = None
        nearest_dist = 999
        for unit in units.values():
            dist = util.manhattan(pos, unit.pos)
            if dist < nearest_dist:
                nearest = unit
                nearest_dist = dist
        return nearest

    def _handle_nearby_enemy(self, pos=None, available_power=None) -> HandleStatus:
        """Handle enemy near to pos (i.e. temporary attack or run back to factory)"""
        pos = pos if pos is not None else self.unit.pos
        available_power = available_power if available_power is not None else self.unit.power_remaining()

        # If enemy near pos:
        nearest_enemy = self.get_nearest_unit_near(pos, friendly_light=False, friendly_heavy=False)
        if nearest_enemy is not None:
            # If we'll have lower power or enemy is heavy and we are light go back to factory
            if available_power < nearest_enemy.power or (
                nearest_enemy.unit_type == "HEAVY" and self.unit.unit_type == "LIGHT"
            ):
                logger.warning(
                    f"{self.unit.log_prefix}, {nearest_enemy.unit_id} near {pos} that is dangerous, returning to factory"
                )
                status = self.return_to_factory()
                if status != self.HandleStatus.SUCCESS:
                    return status
                status = self.HandleStatus.ENEMY_NEAR_FLEEING
                return status
            # We have more power and are of same or higher type so do a temporary attack
            else:
                logger.warning(
                    f"{self.unit.log_prefix}, {nearest_enemy.unit_id} near {pos} that can be targeted, temporary attack"
                )
                self.unit.status.attack_values.target = nearest_enemy  # set that unit as target
                self.unit.status.attack_values.temp.num_remaining = 10  # max 10 attack steps once there
                self.unit.status.attack_values.position = pos
                self.unit.status.update_action_status(
                    ActStatus(ActCategory.COMBAT, sub_category=CombatActSubCategory.TEMPORARY)
                )
                self.unit.status.turn_status.replan_required = True
                return self.HandleStatus.ENEMY_NEAR_ATTACKING
        return self.HandleStatus.SUCCESS

    def add_path(self, path, targeting_enemy=False) -> HandleStatus:
        """
        Add path to unit after some checks:
        1. If desired  path has failed (i.e. len(path) = 0), return to factory
        2. Will unit have enough power to get back to factory after, if not return to factory
        3. Will dest be near an enemy, if so, either temporary attack or return to factory
        """
        # Did target path fail?
        if len(path) == 0:
            logger.warning(f"{self.unit.log_prefix}, path len 0, returning to factory")
            status = self.return_to_factory()
            if status != self.HandleStatus.SUCCESS:
                return status
            return self.HandleStatus.PATH_INVALID_RETURNING

        # Calculate some useful values
        dest_pos = path[-1]
        available_power = self.unit.power_remaining()
        path_cost = util.power_cost_of_path(path, self.rubble, self.unit.unit_type)
        actions = util.path_to_actions(path)

        # If unit will not have enough power to return after this path (if not pathing to factory now) then return to factory now
        any_factory = self.unit.factory.factory_loc + self.unit.factory.queue_array
        if any_factory[dest_pos[0], dest_pos[1]] <= 0:
            path_to_factory_cost = self.path_to_factory_cost(from_pos=dest_pos)
            if available_power - path_cost - path_to_factory_cost < 0:
                logger.warning(f"{self.unit.log_prefix}, not enough power after move, returning to factory")
                status = self.return_to_factory()
                if status != self.HandleStatus.SUCCESS:
                    return status
                status = self.HandleStatus.LOW_POWER_RETURNING
                return status

        if not targeting_enemy:
            status = self._handle_nearby_enemy(pos=dest_pos, available_power=available_power - path_cost)
            if status != self.HandleStatus.SUCCESS:
                return status

        # Checks passed, add actions to queue
        status = self.add_actions_to_queue(actions, targeting_enemy=targeting_enemy)
        if status != self.HandleStatus.SUCCESS:
            return status
        self.unit.pos = path[-1]
        self.unit.power = available_power - path_cost
        return self.HandleStatus.SUCCESS

    def add_dig(self, n_digs) -> HandleStatus:
        """
        Add digs to unit after some checks:
        - Is unit actually on a diggable resource
        - If enemy near, either temporary attack or return to factory
        - Will unit have enough power to get back to factory after, if not, add as many as possible then return to factory
        """
        # Calculate some useful values
        available_power = self.unit.power_remaining()
        cost_to_factory = self.path_to_factory_cost()
        max_digs = (available_power - cost_to_factory) // self.unit.unit_config.DIG_COST
        resource_type = self.master.maps.resource_at_tile(self.unit.pos)

        if resource_type < 0:
            logger.warning(
                f"{self.unit.log_prefix} trying to add dig where no resource, lichen, or rubble. Returning to factory"
            )
            status = self.return_to_factory()
            if status != self.HandleStatus.SUCCESS:
                return status
            return self.HandleStatus.DIG_INVALID_RETURNING

        status = self._handle_nearby_enemy()
        if status != self.HandleStatus.SUCCESS:
            return status

        if n_digs > max_digs:
            logger.warning(
                f"{self.unit.log_prefix} attempted too many digs, doing as many as possible then pathihng back to factory"
            )
            status = self.add_actions_to_queue(self.unit.dig(n=max_digs))
            if status != self.HandleStatus.SUCCESS:
                return status
            status = self.return_to_factory()
            if status != self.HandleStatus.SUCCESS:
                return status
            return self.HandleStatus.LOW_POWER_RETURNING

        # Checks passed, add actions to queue
        status = self.add_actions_to_queue(self.unit.dig(n=n_digs))
        if status != self.HandleStatus.SUCCESS:
            return status

        # Update unit cargo
        if resource_type == util.ICE:
            self.unit.cargo.ice += n_digs * self.unit.unit_config.DIG_RESOURCE_GAIN
        elif resource_type == util.ORE:
            self.unit.cargo.ore += n_digs * self.unit.unit_config.DIG_RESOURCE_GAIN
        # rubble/lichen don't change cargo
        return self.HandleStatus.SUCCESS

    def add_pickup(self, resource_type: int = util.POWER, amount=0, allow_partial=False) -> HandleStatus:
        """
        Add pickup to unit after some checks:
            - if on factory
            - if picking up more than unit can hold
            - if picking up more than factory has
        """
        pos = self.unit.pos

        # Not on factory
        if self.unit.factory.factory_loc[pos[0], pos[1]] <= 0:
            logger.warning(
                f"{self.unit.log_prefix} trying to do pickup at {pos} which is not on own factory, returning to factory"
            )
            status = self.return_to_factory()
            if status != self.HandleStatus.SUCCESS:
                return status
            return self.HandleStatus.PICKUP_INVALID_RETURNING

        # Calculate some useful values
        if resource_type == util.POWER:
            current = self.unit.power_remaining()
            available = self.unit.factory.calculate_power_at_step()
            max_pickup = self.unit.unit_config.BATTERY_CAPACITY - current
        elif resource_type == util.METAL:
            current = self.unit.cargo.metal
            available = self.unit.factory.cargo.metal
            max_pickup = self.unit.unit_config.CARGO_SPACE - current
        elif resource_type == util.WATER:
            current = self.unit.cargo.water
            available = self.unit.factory.cargo.water
            max_pickup = self.unit.unit_config.CARGO_SPACE - current
        elif resource_type == util.ICE:
            current = self.unit.cargo.ice
            available = self.unit.factory.cargo.ice
            max_pickup = self.unit.unit_config.CARGO_SPACE - current
        elif resource_type == util.ORE:
            current = self.unit.cargo.ore
            available = self.unit.factory.cargo.ore
            max_pickup = self.unit.unit_config.CARGO_SPACE - current
        else:
            raise ValueError(f"{resource_type} not a valid resource type")

        # Picking up more than unit can hold
        if amount > max_pickup or amount > available:
            # Automatically resolve
            if allow_partial:
                amount = max_pickup
            else:
                logger.warning(
                    f"{self.unit.log_prefix} trying to pickup more than available or than space (amount={amount},  available={available}, current={current})"
                )
                status = self.return_to_factory()
                if status != self.HandleStatus.SUCCESS:
                    return status
                return self.HandleStatus.PICKUP_INVALID_RETURNING

        # Checks passed, add actions to queue
        status = self.add_actions_to_queue(
            self.unit.pickup(pickup_resource=resource_type, pickup_amount=amount), allow_partial=allow_partial
        )
        if status != self.HandleStatus.SUCCESS:
            return status

        # Update unit and factory
        factory = self.unit.factory
        if resource_type == util.POWER:
            self.unit.power += amount
            # Factory power calculated based on planned_queues
        elif resource_type == util.METAL:
            self.unit.cargo.metal += amount
            factory.cargo.metal -= amount
        elif resource_type == util.WATER:
            self.unit.cargo.water += amount
            factory.cargo.water -= amount
        elif resource_type == util.ICE:
            self.unit.cargo.ice += amount
            factory.cargo.ice -= amount
        elif resource_type == util.ORE:
            self.unit.cargo.ore += amount
            factory.cargo.ore -= amount
        return self.HandleStatus.SUCCESS

    def add_transfer(self, resource_type, direction, amount=None, to_unit=False) -> HandleStatus:
        """
        Add pickup to unit after some checks:
            - if on factory
            - if picking up more than unit can hold
            - if picking up more than factory has

        amount None will default to expected cargo (setting 999 will mess up how much factory has)
        """
        if to_unit:
            raise NotImplementedError

        pos = util.add_direction_to_pos(self.unit.pos, direction)

        # Not on factory
        if self.unit.factory.factory_loc[pos[0], pos[1]] <= 0:
            logger.warning(
                f"{self.unit.log_prefix} trying to do transfer to {pos} which is not on own factory, returning to factory"
            )
            status = self.return_to_factory()
            if status != self.HandleStatus.SUCCESS:
                return status
            return self.HandleStatus.TRANSFER_INVALID_RETURNING

        # Checks passed, add actions to queue
        status = self.add_actions_to_queue(
            self.unit.transfer(transfer_direction=direction, transfer_resource=resource_type, transfer_amount=amount)
        )
        if status != self.HandleStatus.SUCCESS:
            return status

        # Update unit and factory
        factory = self.unit.factory
        if resource_type == util.POWER:
            amount = amount if amount else self.unit.power
            self.unit.power -= amount
            # Factory power calculated based on planned_queues
        elif resource_type == util.METAL:
            amount = amount if amount else self.unit.cargo.metal
            self.unit.cargo.metal -= amount
            factory.cargo.metal += amount
        elif resource_type == util.WATER:
            amount = amount if amount else self.unit.cargo.water
            self.unit.cargo.water -= amount
            factory.cargo.water += amount
        elif resource_type == util.ICE:
            amount = amount if amount else self.unit.cargo.ice
            self.unit.cargo.ice -= amount
            factory.cargo.ice += amount
        elif resource_type == util.ORE:
            amount = amount if amount else self.unit.cargo.ore
            self.unit.cargo.ore -= amount
            factory.cargo.ore += amount
        return self.HandleStatus.SUCCESS

    """Class to handle adding actions to unit, i.e. like pathfinder currently does with path (adds to action queue and updates pos of unit)

    TODO: Need to update pos of unit to be end of planned action queue in unit update (or after collision checks etc)
    TODO: Each break (should act) point can trigger replanning if necessary (i.e. load the ActStatus from that position in deque and the planned actions from that point)
    TODO: Step the action queue deque at the same time as the old actions (maybe hold on to

        Notes for MultiActionPlanner:
            When calculating ShouldActs up to X steps
                - For each of the following, if True, Load ActStatus from that step, split action queue and update unit pos/power/cargo etc
                    (Most splits will be on the next action (i.e. invalid pickup, transfer, low power))

                - If unit doesn't have enough to get back to factory and current queue does not end near factory and not a specific flag set in unit.status
                    - Set status to that step, and continue plan

                - If unit will collide with enemy (only if threat i.e. higher power equal or higher type, and not ATTACK category)
                    - Set status to that step, and continue plan

                - If unit will collide with friendly
                    - Repath to dest if possible (can be new waiting location, or new factory location, or near factory),
                    - If not handled set COLLISION_FRIENDLY ActReason, set status to that step and continue plan

                # - If unit doesn't have enough to get back to factory and current queue does not end near factory and not a specific flag set in unit.status
                #     - Drop rest of actions, set low_power_flag add WAITING/DROPOFF category of status
                # - If unit will collide with enemy (only if threat i.e. higher power equal or higher type, and not ATTACK category)
                #     - repath to next destination, if no next dest add WAITING/DROPOFF, Set COLLISION_ENEMY ActReason if not handled here
                # - If unit will collide with friendly
                #     - Repath to dest if possible (can be new waiting location, or new factory location, or near factory),
                #     - If not handled set COLLISION_FRIENDLY ActReason
                # - If unit close to enemy, (status will already be loaded at that step), then just let planner plan again from that latest step



            Order units by NOTHING, MINING, CLEARING, ATTACKING, WAITING
                Loop:
                - Calculate and try handle ShouldActs (leaving queue after point if not solved)

                Then by shortest action queue in num steps first
                Loop:
                - Build queue until X steps in length or ends on factory with WAITING status

        Notes for SingleActionPlanner:
            - Get updated actions from relevant action planner (General, Mining, Clearing, Attacking)
            - If status is done_building and first action valid (i.e. move if necessary, or valid pickup etc) the break
            - Otherwise loop again (maybe has updated status) up to max N times

        Notes for AttackPlanner:
            - Make list of hold/attack locations at beginning of turn (Each factory, then friendly resources, then enemy resources, then enemy factory)

            - If in some sort of attacking mode (not RETREAT or RUN_AWAY)
                - If target enemy
                    - If within attack radius from hold pos, attack
                    - Elif num_attacks left, attack
                    - else skip attacking
                - If no target enemy, look in search dist and pick target
                    - attack
                else skip attacking
            - If not enough energy to reach hold location and within X dist from factory, pickup max power then move to hold location
                else return to waiting queue


        Notes for:
        GeneralHandler:
            if status is NOTHING. If cargo, DROPOFF, if not enough power to get to waiting, delay first then go to WAITING
            if status is RUN_AWAY and  unit on factory, set status WAITING
            if status is WAITING and power max move further out and continue waiting
            if status DROPOFF if the action queue is empty and has resources, check they are being dropped off, otherwise set status to waiting
            if WAITING and collision, move to next nearest free waiting spot (just check index 0 of full unitpaths array (keep full and partial in master)


    """
