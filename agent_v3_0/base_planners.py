from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, TypeVar, Dict, List, Optional
import abc

from unit_status import ActStatus, ActCategory
from config import get_logger

import util

if TYPE_CHECKING:
    from unit_manager import UnitManager, FriendlyUnitManager
    from master_state import MasterState
    from locations import LocationManager


logger = get_logger(__name__)

# LocationManagerType= TypeVar("LocationManagerType", bound=LocationManager)
BasePlannerType = TypeVar("BasePlannerType", bound="BaseGeneralPlanner")
BaseUnitPlannerType = TypeVar("BaseUnitPlannerType", bound="BaseUnitPlanner")


class BaseGeneralPlanner(abc.ABC):
    """For generating plans for multiple units for a given action type"""

    def __init__(self, master: MasterState):
        self.master = master
        self.unit_planners: Dict[str, BaseUnitPlannerType] = {}

    @abc.abstractmethod
    def update(self):
        """Beginning of turn update
        Update value maps etc
        """
        pass

    @abc.abstractmethod
    def get_unit_planner(self, unit: FriendlyUnitManager) -> BaseUnitPlanner:
        """Return a subclass of BaseUnitPlanner to actually update or create new actions for a single Unit"""
        if unit.unit_id not in self.unit_planners:
            unit_planner = BaseUnitPlanner(self.master, self, unit)
            self.unit_planners[unit.unit_id] = unit_planner
        return self.unit_planners[unit.unit_id]


class BaseUnitPlanner(abc.ABC):
    """For updating plans of a single unit for a certain action type"""

    def __init__(self, master: MasterState, general_planner: BasePlannerType, unit: FriendlyUnitManager):
        self.master = master
        self.pathfinder = master.pathfinder
        self.planner: BasePlannerType = general_planner
        self.unit = unit

    @abc.abstractmethod
    def update_planned_actions(self):
        """Update the planned actions of the unit based on current status
        This will be called every turn, so decide whether things actually need updating per turn
        """
        pass

    def add_path_to_action_queue(self, path: np.ndarray) -> bool:
        """Add the path to action queue and update unit pos"""
        if len(path) > 0:
            self.pathfinder.append_path_to_actions(self.unit, path)
            return True
        return False

    def add_path_to_pos(
        self, dest_pos: util.POS_TYPE, ignore: Optional[List[int]] = None, collision_only=False
    ) -> bool:
        path = self.get_path_to_pos(dest_pos, ignore=ignore, collision_only=collision_only)
        return self.add_path_to_action_queue(path)

    def get_path_to_pos(
        self, dest_pos: util.POS_TYPE, ignore: Optional[List[int]] = None, collision_only=False
    ) -> np.ndarray:
        cm = self.master.pathfinder.generate_costmap(
            self.unit,
            ignore_id_nums=ignore,
            collision_only=collision_only,
        )
        path = self.master.pathfinder.fast_path(self.unit.pos, dest_pos, costmap=cm, margin=3)
        return path

    def add_path_to_factory_queue(self, avoid_collision_only=False) -> bool:
        path = self.get_path_to_factory_queue(avoid_collision_only=avoid_collision_only)
        return self.add_path_to_action_queue(path)

    def get_path_to_factory_queue(self, avoid_collision_only=False) -> np.ndarray:
        cm = self.master.pathfinder.generate_costmap(self.unit, collision_only=avoid_collision_only)
        path_to_factory = util.calculate_path_to_nearest_non_zero(
            self.master.pathfinder,
            costmap=cm,
            from_pos=self.unit.pos,
            target_array=self.unit.factory.queue_array,
            near_pos=self.unit.factory.pos,
            max_attempts=50,
        )
        # if self.unit.unit_id == 'unit_17':
        #     # print(self.unit.pos, self.unit.factory.pos, self.unit.start_of_turn_pos, self.unit.current_path()[0], self.unit.current_path()[-1])
        #     fig = util.show_map_array(cm)
        #     util.plotly_plot_path(fig, path_to_factory)
        #     fig.show()
        return path_to_factory

    def abort(self, delay_abort_to_end_of_queue: bool = False) -> bool:
        """Retreat to factory waiting area and clear ActionStatus"""
        logger.error(
            f"{self.unit.log_prefix}: Aborting current action. Retreating to factory queue and setting status Waiting"
        )
        if not delay_abort_to_end_of_queue:
            self.unit.action_queue = []
        self.unit.status.update_action_status(ActStatus(category=ActCategory.WAITING))
        self.add_path_to_factory_queue(avoid_collision_only=True)
        self.unit.status.planned_action_queue = self.unit.action_queue
        return False


class BaseRouter(abc.ABC):
    """Base class for getting from pos to Location, where Location maybe be a few different things"""

    # def get_path_to_location(self, start_pos: util.POS_TYPE, location: LocationManager, start_step: int) -> util.PATH_TYPE:
    #     """Get path to location (including location delay)"""
    #     pass

    pass
