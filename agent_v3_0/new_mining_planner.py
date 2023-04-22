from typing import TYPE_CHECKING

from base_planners import BaseUnitPlanner, BaseGeneralPlanner, BaseRouter
from locations import LocationManager

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManager
    from master_state import MasterState


class MiningLocationManager(LocationManager):
    pass


class MiningRouter(BaseRouter):
    pass


class GeneralMiningPlanner(BaseGeneralPlanner):
    def __init__(self, master: MasterState):
        super().__init__(master)
        self._locations = MiningLocationManager()

        self.ice = self.master.maps.ice
        self.ore = self.master.maps.ore

    @property
    def locations(self) -> MiningLocationManager:
        return self._locations

    def get_unit_planner(self, unit: FriendlyUnitManager) -> BaseUnitPlanner:
        pass

    def update_value_maps(self):
        pass

    def update_location_manager(self):
        pass


class UnitMiningPlanner(BaseUnitPlanner):
    def update_planned_actions(self):
        pass

    def add_new_actions(self):
        pass
