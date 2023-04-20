from __future__ import annotations
from typing import TYPE_CHECKING, TypeVar, Dict
import abc

import util

if TYPE_CHECKING:
    from unit_manager import UnitManager, FriendlyUnitManager
    from master_state import MasterState
    from locations import LocationManager


# LocationManagerType= TypeVar("LocationManagerType", bound=LocationManager)
# BaseUnitPlannerType = TypeVar("BaseUnitPlannerType", bound=BaseUnitPlanner)

class BaseGeneralPlanner(abc.ABC):
    """For generating plans for multiple units for a given action type"""
    def __init__(self, master: MasterState):
        self.master = master
        self.unit_planners: Dict[str, BaseUnitPlanner] = {}

    @property
    @abc.abstractmethod
    def locations(self) -> LocationManager:
        pass

    @abc.abstractmethod
    def _init_location_manager(self):
        """Initialize a new instance of whatever subclass of location manager"""
        return LocationManager()

    def update(self):
        """Beginning of turn update
        Update value maps etc
        """
        self.update_value_maps()
        self.update_location_manager()

    @abc.abstractmethod
    def get_unit_planner(self, unit: FriendlyUnitManager) -> BaseUnitPlanner:
        """Return a subclass of BaseUnitPlanner to actually update or create new actions for a single Unit"""
        if unit.unit_id not in self.unit_planners:
            unit_planner = BaseUnitPlanner(self.master, self, unit)
            self.unit_planners[unit.unit_id] = unit_planner
        return self.unit_planners[unit.unit_id]

    @abc.abstractmethod
    def update_value_maps(self):
        """Update value maps at beginning of turn"""
        pass

    @abc.abstractmethod
    def update_location_manager(self):
        """Update location manager (i.e. any dead units should be removed, or any units no longer assigned to current
        action type)"""
        pass


class BaseUnitPlanner(abc.ABC):
    """For updating plans of a single unit for a certain action type"""
    def __init__(self, master: MasterState, general_planner: BaseGeneralPlanner, unit: UnitManager):
        self.master = master
        self.planner = general_planner
        self.unit = unit

    @abc.abstractmethod
    def update_planned_actions(self):
        """Update the planned actions of the unit based on current status"""
        pass

    @abc.abstractmethod
    def create_new_actions(self):
        """Replan the actions of the unit from scratch (i.e. assuming newly assigned, starting from end of current planned actions)"""
        pass


class BaseRouter(abc.ABC):
    """Base class for getting from pos to Location, where Location maybe be a few different things"""
    # def get_path_to_location(self, start_pos: util.POS_TYPE, location: LocationManager, start_step: int) -> util.PATH_TYPE:
    #     """Get path to location (including location delay)"""
    #     pass
    pass

