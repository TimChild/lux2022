from __future__ import annotations

from unit_status import DestType, ActStatus
from factory_action_planner import FactoryActionPlanner, WorkRatios
from base_planners import BaseUnitPlanner, BaseGeneralPlanner, ActCategory
from typing import TYPE_CHECKING, Dict
from config import get_logger
import util

if TYPE_CHECKING:
    from master_state import MasterState
    from unit_manager import FriendlyUnitManager

logger = get_logger(__name__)

class GeneralUnitPlanner(BaseUnitPlanner):
    factory_power_low_thresh = 300
    factory_power_med_thresh = 700
    factory_power_high_thresh = 1000
    unit_power_low_thresh = 0.1
    unit_power_med_thresh = 0.5
    unit_power_high_thresh = 0.9

    def  __init__(self, master: MasterState, general_planner: GeneralPlanner, unit: FriendlyUnitManager):
        super().__init__(master, general_planner, unit)
        self.planner: GeneralPlanner = general_planner

        # just make this easier to get to since it's used a lot
        self.SUCCESS = self.unit.action_handler.HandleStatus.SUCCESS

    def update_planned_actions(self):

        act_cat = self.unit.status.current_action.category
        if act_cat == ActCategory.DROPOFF:
            status = self.update_dropoff()
            if status != self.SUCCESS:
                return status
        if act_cat == ActCategory.TRANSFER:
            status = self.update_transfer()
            if status != self.SUCCESS:
                return status
        if act_cat in [ActCategory.NOTHING, ActCategory.WAITING]:
            status = self.update_idle()
            if status != self.SUCCESS:
                return status
        else:
            raise ValueError(f'{act_cat} not valid for GeneralPlanner')

    def update_idle(self):
        planned = self.unit.status.planned_action_queue
        start_pos = self.unit.start_of_turn_pos

        # If finishing previous plans
        if len(planned) > 0:
            path = self.unit.current_path(max_len=self.unit.max_queue_step_length)
            end_pos = path[-1]
            if self.unit.factory.factory_loc[end_pos[0], end_pos[1]] > 0 or self.unit.factory.queue_array[end_pos[0], end_pos[1]] > 0:
                return self.SUCCESS
            else:
                self.unit.reset_unit_to_start_of_turn_empty_queue()
                status = self.unit.action_handler.return_to_factory()
                return status

        # Can new actions be assigned to this unit
        if len(planned) == 0:
            # Check in correct place
            if self.unit.on_own_factory() or self.unit.factory.queue_array[start_pos[0], start_pos[1]] > 0:
                new_work = self.possible_assign_new_work()
                if new_work:
                    # New job will generate actions
                    return self.SUCCESS
                if self.unit.status.turn_status.must_move:
                    self.unit.reset_unit_to_start_of_turn_empty_queue()
                    status = self.unit.action_handler.return_to_factory()
                    return status
            else:
                self.unit.reset_unit_to_start_of_turn_empty_queue()
                status = self.unit.action_handler.return_to_factory()
                return status

    def possible_assign_new_work(self):
        if len(self.unit.status.planned_action_queue) > 0:
            raise RuntimeError(f'{self.unit.log_prefix} has {len(self.unit.status.planned_action_queue)} actions remaining')

        self.unit.reset_unit_to_start_of_turn_empty_queue()

        ### From here decide possible new action for unit ###
        # Collect some useful values
        factory = self.unit.factory
        # What factory power will likely be in X steps
        factory_power = factory.calculate_power_at_step(step=10)
        unit_power_ratio = self.unit.start_of_turn_power // self.unit.unit_config.BATTERY_CAPACITY
        act_cat = self.unit.status.current_action.category

        # Generate a new work status to potentially assign
        work_ratios = self.planner.work_ratios[self.unit.factory_id]
        next_action = work_ratios.weighted_random_choice()

        if act_cat == ActCategory.NOTHING:
            self.unit.status.update_action_status(new_action=next_action)
            logger.info(f'NOTHING, assigned {next_action}')
            return self.SUCCESS

        # Assign units that have lots of energy already
        if factory_power > self.factory_power_low_thresh:
            if unit_power_ratio > self.unit_power_high_thresh:
                self.unit.status.update_action_status(new_action=next_action)
                logger.info(f'unit has high power, assigned {next_action}')
                return self.SUCCESS
        if factory_power > self.factory_power_med_thresh:
            if unit_power_ratio > self.unit_power_med_thresh:
                self.unit.status.update_action_status(new_action=next_action)
                logger.info(f'unit has med power, assigned {next_action}')
                return self.SUCCESS
        if factory_power > self.factory_power_low_thresh:
            if unit_power_ratio > self.unit_power_low_thresh:
                self.unit.status.update_action_status(new_action=next_action)
                logger.info(f'unit has low power, assigned {next_action}')
                return self.SUCCESS

        logger.info(f'Not enough factory power to assign ne work to {self.unit.unit_id}')
        return False

    def update_dropoff(self):

        planned = self.unit.status.planned_action_queue
        start_pos = self.unit.start_of_turn_pos

        # If finishing previous plans
        if len(planned) > 0:
            path = self.unit.current_path(max_len=self.unit.max_queue_step_length)
            end_pos = path[-1]
            if self.unit.factory.factory_loc[end_pos[0], end_pos[1]] > 0:
                # assume there will be some resources to dropoff for now
                return self.SUCCESS
            else:
                self.unit.reset_unit_to_start_of_turn_empty_queue()
                status = self.unit.action_handler.return_to_factory()
                return status
        # No plans
        else:
            self.unit.cargo = self.unit.start_of_turn_cargo
            # If still cargo, dropoff
            if self.unit.cargo_total > 0:
                logger.warning(f'{self.unit.log_prefix} having to add the final dropoff actions to empty cargo {self.unit.cargo_total}')
                status = self.unit.action_handler.return_to_factory()
                return status
            # Else
            else:
                self.unit.status.update_action_status(ActStatus(category=ActCategory.WAITING))
                return self.SUCCESS
        logger.error(f"dont think this should be reachable")











class GeneralPlanner(BaseGeneralPlanner):
    def __init__(self, master: MasterState, factory_planner: FactoryActionPlanner):
        super().__init__(master)
        self.factory_planner = factory_planner
        # updated at beginning of turn

    @property
    def work_ratios(self) -> Dict[str, WorkRatios]:
        return self.factory_planner.get_factory_work_ratios()
    def update(self):
        pass
    # New
    def get_unit_planner(self, unit: FriendlyUnitManager) -> GeneralUnitPlanner:
        """Return a subclass of BaseUnitPlanner to actually update or create new actions for a single Unit"""
        if unit.unit_id not in self.unit_planners:
            unit_planner = GeneralUnitPlanner(self.master, self, unit)
            self.unit_planners[unit.unit_id] = unit_planner
        return self.unit_planners[unit.unit_id]