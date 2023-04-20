from __future__ import annotations
from typing import Dict, List, TypeVar, TYPE_CHECKING

if TYPE_CHECKING:
    from unit_manager import UnitManager, FriendlyUnitManager

Location = TypeVar("Location")
Unit = TypeVar("Unit", bound=UnitManager)


class LocationManager:
    """Keeps track of units associated with location, or locations associated with unit in a way that they stay in
    sync"""
    def __init__(self):
        self.location_to_units: Dict[Location, List[Unit]] = {}
        self.unit_to_location: Dict[Unit, Location] = {}

    def add_unit(self, location: Location, unit: Unit):
        if location not in self.location_to_units:
            self.location_to_units[location] = []
        self.location_to_units[location].append(unit)
        self.unit_to_location[unit] = location

    def remove_unit(self, unit: Unit):
        location = self.unit_to_location.get(unit)
        if location:
            self.location_to_units[location].remove(unit)
            del self.unit_to_location[unit]

    def update_location(self, unit: Unit, new_location: Location):
        self.remove_unit(unit)
        self.add_unit(new_location, unit)

    def get_units_by_location(self, location: Location) -> List[Unit]:
        return self.location_to_units.get(location, [])

    def get_location_by_unit(self, unit: Unit) -> Location:
        return self.unit_to_location.get(unit)
