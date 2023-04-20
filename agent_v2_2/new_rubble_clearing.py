from typing import TYPE_CHECKING

from base_planners import BaseUnitPlanner, BaseGeneralPlanner, BaseRouter
from locations import LocationManager

if TYPE_CHECKING:
    from master_state import MasterState
    from unit_manager import FriendlyUnitManager


class RubbleLocationManager(LocationManager):
    pass


class RubbleRouter(BaseRouter):
    pass


class GeneralRubblePlanner(BaseGeneralPlanner):
    def __init__(self, master: MasterState):
        super().__init__(master)
        self._locations = RubbleLocationManager()

    @property
    def locations(self) -> RubbleLocationManager:
        return self._locations

    def get_unit_planner(self, unit: FriendlyUnitManager) -> BaseUnitPlanner:
        pass

    def update_value_maps(self):
        pass

    def update_location_manager(self):
        pass


class UnitRubblePlanner(BaseUnitPlanner):
    def update_planned_actions(self):
        pass

    def create_new_actions(self):
        pass



