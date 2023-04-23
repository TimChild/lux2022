from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Union, Optional, Dict, TYPE_CHECKING

import util
from action_validation import ValidActionCalculator
from unit_status import ActCategory, ActStatus, MineActSubCategory, ClearActSubCategory
from collisions import AllCollisionsForUnit

from config import get_logger

logger = get_logger(__name__)

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManager
    from factory_action_planner import FactoryDesires, FactoryInfo
    from master_state import MasterState
    from unit_action_planner import UnitInfo, CloseUnits


class ActionDecider:
    def __init__(
        self,
        unit: FriendlyUnitManager,
        unit_info: UnitInfo,
        action_validator: ValidActionCalculator,
        factory_desires: FactoryDesires,
        factory_info: FactoryInfo,
        close_enemy_units: Union[None, CloseUnits],
        master: MasterState,
    ):
        self.unit = unit
        self.unit_info = unit_info
        self.action_validator = action_validator
        self.factory_desires = factory_desires
        self.factory_info = factory_info
        self.close_units = close_enemy_units
        self.master = master

    def _possible_switch_to_attack(self) -> Optional[ActStatus]:
        """Potentially switch to attacking if an enemy wanders by"""
        # Only if something nearby
        # if self.close_units is not None:
        #     # And not doing something important
        #     if not self.unit.status.current_action in [
        #         Actions.MINE_ICE,
        #         Actions.MINE_ORE,
        #         Actions.ATTACK,
        #     ]:
        #         # And has a reasonable amount of energy
        #         if self.unit.start_of_turn_power > self.unit.unit_config.BATTERY_CAPACITY * 0.3:
        #             # And did not change actions in the last few turns
        #             if self.unit.master.step - self.unit.status.last_action_update_step > 5:
        #                 # And nearby enemy is equal type with lower power
        #                 closest = self.close_units.closest()
        #                 if closest.unit_type == self.unit.unit_type and closest.power < self.unit.start_of_turn_power:
        #                     logger.warning(f"{self.unit.log_prefix} switching to attack")
        #                     return Actions.ATTACK
        # TODO: implement switching to attack in individual planner
        return None

    def _decide_noops(self, unit_must_move: bool) -> bool:
        """Decide what sort of things should be considered in route planning
        Note: I probably want to get rid of this once the new planners are working

        returns:
            bool: has the unit.status.turn_status.recommend_plan_update been updated?

        """

        act_reason = self.unit_info.act_info.reason
        # Validation only
        if act_reason in [
            ActReasons.NEXT_ACTION_PICKUP,
            ActReasons.NEXT_ACTION_TRANSFER,
            ActReasons.NEXT_ACTION_DIG,
        ]:
            if self.action_validator.next_action_valid(self.unit):
                # Additionally check that diggers have enough energy to get back to factory
                if act_reason == ActReasons.NEXT_ACTION_DIG:
                    # Not worrying about collisions for this
                    cheap_path_to_factory = self.master.pathfinder.fast_path(
                        self.unit.start_of_turn_pos, self.unit.factory.pos
                    )
                    if (
                        util.power_cost_of_path(
                            cheap_path_to_factory,
                            self.master.maps.rubble,
                            self.unit.unit_type,
                        )
                        > self.unit.start_of_turn_power
                    ):
                        logger.info(
                            f"Next action dig, but not enough power to get back to factory, recommending update"
                        )
                        self.unit.status.turn_status.recommend_plan_update = True
                        return True
                logger.debug("Next pickup, transfer, dig is valid, do not update")
                self.unit.status.turn_status.recommend_plan_update = False
                return True
            else:
                logger.debug("Next pickup, transfer, dig not valid, continue but update plan")
                self.unit.status.turn_status.recommend_plan_update = True
                return True

        # Avoid collision with friendly
        if act_reason in [
            ActReasons.COLLISION_WITH_FRIENDLY,
        ]:
            logger.debug("Need to avoid collision with friendly")
            self.unit.status.turn_status.recommend_plan_update = True
            return True

        # Close enemies don't matter if running away (as long as not colliding)
        if self.unit.status.current_action.category == ActCategory.RUN_AWAY and act_reason == ActReasons.CLOSE_TO_ENEMY:
            logger.debug("Already running away, no change necessary")
            self.unit.status.turn_status.recommend_plan_update = False
            return True

        # Heavy doesn't care about enemy light
        if (
            act_reason == ActReasons.CLOSE_TO_ENEMY
            and self.unit.unit_type == "HEAVY"
            and all([t == "LIGHT" for t in self.close_units.other_unit_types])
        ):
            logger.debug("Heavy doesn't care about light enemies, continuing path")
            self.unit.status.turn_status.recommend_plan_update = False
            return True

        # Previous action invalid can probably just update plan with current action
        if act_reason == ActReasons.PREVIOUS_ACTION_INVALID:
            logger.debug("Previous action was invalid, may need to update plan of current role")
            self.unit.status.turn_status.recommend_plan_update = True
            return True

        # If already attacking, just update in case new path to enemy
        if (
            act_reason in [ActReasons.CLOSE_TO_ENEMY, ActReasons.COLLISION_WITH_ENEMY]
            and self.unit.status.current_action.category == ActCategory.COMBAT
        ):
            self.unit.status.turn_status.recommend_plan_update = True
            return True

        # Just need update from planned
        if act_reason == ActReasons.NEED_ACTIONS_FROM_PLANNED:
            self.unit.status.turn_status.recommend_plan_update = False
            return True

        # Check if action is still invalid (might not be now that other units have had a chance to move etc)
        if act_reason in [
            ActReasons.NEXT_ACTION_INVALID_MOVE,
            ActReasons.NEXT_ACTION_INVALID,
        ]:
            # If must move, is next action a move anyway (no move not checked in validator, other moves are)
            condition = self.unit.next_action_is_move() if unit_must_move else True
            if self.action_validator.next_action_valid(self.unit) and condition:
                logger.debug("Next action passed validation, suggesting no action update")
                self.unit.status.turn_status.recommend_plan_update = False
                return True
            else:
                logger.debug("Next action not valid, suggesting keep role but update")
                self.unit.status.turn_status.recommend_plan_update = True
                return True

        # If low power, continue with same action but update path in case a difference decision should be made now
        if act_reason == ActReasons.LOW_POWER:
            # TODO: Not sure about this one... I don't want units that are nearly back at factory to continuously repath
            logger.debug("Unit has low power, should consider changing plans")
            self.unit.status.turn_status.recommend_plan_update = True
            return True
        return False

    def _decide_light_unit_action(self, unit_must_move: bool) -> Optional[ActStatus]:
        logger.debug(f"Deciding between light unit actions")
        new_act = self._possible_switch_to_attack()
        if new_act is not None:
            return new_act
        status_updated = self._decide_noops(unit_must_move)
        if status_updated is True:
            return None
        new_act = self._decide_unit_action_based_on_factory_needs(
            self.factory_desires.light_mining_ore,
            self.factory_info.light_mining_ore,
            self.factory_desires.light_clearing_rubble,
            self.factory_info.light_clearing_rubble,
            self.factory_desires.light_mining_ice,
            self.factory_info.light_mining_ice,
            self.factory_desires.light_attacking,
            self.factory_info.light_attacking,
        )
        return new_act

    def _decide_heavy_unit_action(self, unit_must_move: bool) -> Optional[ActStatus]:
        logger.debug(f"Deciding between heavy unit actions")
        new_act = self._possible_switch_to_attack()
        if new_act is not None:
            return new_act
        status_updated = self._decide_noops(unit_must_move)
        if status_updated is True:
            return None
        new_act = self._decide_unit_action_based_on_factory_needs(
            self.factory_desires.heavy_mining_ore,
            self.factory_info.heavy_mining_ore,
            self.factory_desires.heavy_clearing_rubble,
            self.factory_info.heavy_clearing_rubble,
            self.factory_desires.heavy_mining_ice,
            self.factory_info.heavy_mining_ice,
            self.factory_desires.heavy_attacking,
            self.factory_info.heavy_attacking,
        )
        return new_act

    def _decide_unit_action_based_on_factory_needs(
        self,
        desired_mining_ore: int,
        current_mining_ore: int,
        desired_clearing_rubble: int,
        current_clearing_rubble: int,
        desired_mining_ice: int,
        current_mining_ice: int,
        desired_attacking: int,
        current_attacking: int,
    ) -> ActStatus:
        if (
            not self.unit_info.unit.on_own_factory()
            and self.unit_info.unit.status.current_action.category != ActCategory.NOTHING
        ):
            action = self.unit.status.current_action
            logger.debug(
                f"Unit NOT on factory and currently assigned, should continue same job ({self.unit.status.current_action.category}: {self.unit.status.current_action.sub_category})"
            )
        else:
            logger.debug(f"Unit on factory, can decide a new type of action depending on factory needs")

            self.factory_info.remove_unit_from_current_count(self.unit_info.unit)
            action = ActStatus()
            if current_mining_ice < desired_mining_ice:
                action.category = ActCategory.MINE
                action.sub_category = MineActSubCategory.ICE
            elif current_mining_ore < desired_mining_ore:
                action.category = ActCategory.MINE
                action.sub_category = MineActSubCategory.ORE
            elif current_clearing_rubble < desired_clearing_rubble:
                action.category = ActCategory.CLEAR
                action.sub_category = ClearActSubCategory.RUBBLE
            elif current_attacking < desired_attacking:
                action.category = ActCategory.COMBAT
            self.unit.status.update_action_status(action)
        return action

    def decide_action(self, unit_must_move: bool) -> Optional[ActStatus]:
        logger.info(f"Deciding action for {self.unit_info.unit_id}")
        if self.unit_info.unit_type == "LIGHT":
            action = self._decide_light_unit_action(unit_must_move)
        else:  # unit_type == "HEAVY"
            action = self._decide_heavy_unit_action(unit_must_move)

        logger.debug(f"action should be {action}")
        return action


class ActReasons(Enum):
    NOT_ENOUGH_POWER = "not enough power"
    LOW_POWER = "low power"
    NO_ACTION_QUEUE = "no action queue"
    NEED_ACTIONS_FROM_PLANNED = "action queue is short, need new from planned"
    CURRENT_STATUS_NOTHING = "currently no action"
    COLLISION_WITH_ENEMY = "collision with enemy"
    COLLISION_WITH_FRIENDLY = "collision with friendly"
    CLOSE_TO_ENEMY = "close to enemy"
    NEXT_ACTION_INVALID = "next action invalid"
    NEXT_ACTION_INVALID_MOVE = "next action invalid move"
    NEXT_ACTION_PICKUP = "next action pickup"
    NEXT_ACTION_DIG = "next action dig"
    NEXT_ACTION_TRANSFER = "next action transfer"
    NO_REASON_TO_ACT = "no reason to act"
    ATTACKING = "attacking"
    PREVIOUS_ACTION_INVALID = "previous action was invalid"


@dataclass
class ConsiderActInfo:
    unit: FriendlyUnitManager
    should_act: bool = False
    reason: ActReasons = ActReasons.NO_REASON_TO_ACT


@dataclass
class ShouldActInfo:
    reason: ActReasons
    requires_action: bool = True
    step: int = -1


def should_unit_consider_acting(
    unit: FriendlyUnitManager,
    upcoming_collisions: Dict[str, AllCollisionsForUnit],
    close_enemies: Dict[str, CloseUnits],
) -> ConsiderActInfo:
    unit_id = unit.unit_id
    unit_act_reasons = unit.status.turn_status.should_act_reasons

    should_act = True

    # Can't be updated
    if unit.power < (unit.unit_config.ACTION_QUEUE_POWER_COST + unit.unit_config.MOVE_COST):
        # This one is more of a warning, doesn't really require new actions (planned actions should be updated to
        # account for not enough power to do next actions though)
        unit_act_reasons.append(ShouldActInfo(ActReasons.NOT_ENOUGH_POWER, requires_action=False))
        # Todo remove once unused
        should_act = False
    # Previous action invalid
    if unit.status.turn_status.planned_actions_valid_from_last_step is False:
        unit_act_reasons.append(ShouldActInfo(ActReasons.PREVIOUS_ACTION_INVALID))
    # If no queue
    if len(unit.action_queue) == 0:
        unit_act_reasons.append(ShouldActInfo(ActReasons.NO_ACTION_QUEUE))
    # If colliding with enemy
    if unit_id in upcoming_collisions and upcoming_collisions[unit_id].num_collisions(friendly=False, enemy=True) > 0:
        unit_act_reasons.append(ShouldActInfo(ActReasons.COLLISION_WITH_ENEMY))
    # If colliding with friendly
    if unit_id in upcoming_collisions and upcoming_collisions[unit_id].num_collisions(friendly=True, enemy=False) > 0:
        unit_act_reasons.append(ShouldActInfo(ActReasons.COLLISION_WITH_FRIENDLY))

    move_valid = unit.valid_moving_actions(unit.master.maps.valid_friendly_move, max_len=1)
    # Next move invalid
    if move_valid.was_valid is False:
        unit_act_reasons.append(ShouldActInfo(ActReasons.NEXT_ACTION_INVALID_MOVE))
        logger.debug(
            f"Move from {unit.start_of_turn_pos} was invalid for reason {move_valid.invalid_reasons[0]}, action={unit.action_queue[0]}"
        )
    # If close to enemy
    if unit_id in close_enemies:
        unit_act_reasons.append(ShouldActInfo(ActReasons.CLOSE_TO_ENEMY))
    # Attacking needs regular updates
    if unit.status.current_action.category == ActCategory.COMBAT:
        unit_act_reasons.append(ShouldActInfo(ActReasons.ATTACKING))
        # Check pickup is valid
        unit_act_reasons.append(ShouldActInfo(ActReasons.NEXT_ACTION_PICKUP))
    # Check transfer is valid
    if len(unit.action_queue) > 0 and unit.action_queue[0][util.ACT_TYPE] == util.TRANSFER:
        unit_act_reasons.append(ShouldActInfo(ActReasons.NEXT_ACTION_TRANSFER))
    # Check dig is valid
    if len(unit.action_queue) > 0 and unit.action_queue[0][util.ACT_TYPE] == util.DIG:
        unit_act_reasons.append(ShouldActInfo(ActReasons.NEXT_ACTION_DIG))
    # If not doing anything maybe need an update
    if unit.status.current_action.category == ActCategory.NOTHING:
        unit_act_reasons.append(ShouldActInfo(ActReasons.CURRENT_STATUS_NOTHING))
    # TODO: Need to think more about how to handle low power (don't want to keep repathing especially at low power...)
    # If low power (might want to change plans)
    if unit.start_of_turn_power < unit.unit_config.BATTERY_CAPACITY * 0.15:
        unit_act_reasons.append(ShouldActInfo(ActReasons.LOW_POWER))
    # If action queue is short but more actions planned
    if len(unit.start_of_turn_actions) < 2 and len(unit.status.planned_action_queue) >= 2:
        unit_act_reasons.append(ShouldActInfo(ActReasons.NEED_ACTIONS_FROM_PLANNED))
    if len(unit_act_reasons) == 0:
        unit_act_reasons.append(ShouldActInfo(ActReasons.NO_REASON_TO_ACT, requires_action=False))
        # todo remove once unused
        should_act = False
    if should_act:
        logger.info(f"{unit_id} should consider acting -- {unit_act_reasons}")
    else:
        logger.info(f"{unit_id} should not consider acting -- {unit_act_reasons}")
    # todo remove once unused
    old_act_info = ConsiderActInfo(unit, should_act=should_act, reason=unit_act_reasons[0].reason)
    return old_act_info
