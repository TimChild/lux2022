from __future__ import annotations
from typing import List, TYPE_CHECKING
import numpy as np
import re
import abc
import copy

from luxai_s2.unit import UnitCargo
from lux.unit import Unit
from lux.config import UnitConfig

from action_handler import ActionHandler
from unit_status import Status, ActStatus
import util

from config import get_logger

if TYPE_CHECKING:
    from master_state import MasterState
    from factory_manager import FriendlyFactoryManager

logger = get_logger(__name__)


def get_index(lst, index, default=None):
    """Get the element at the specified index in the list, or return the default value if the index is out of range."""
    return lst[index] if 0 <= index < len(lst) else default


class UnitManager(abc.ABC):
    def __init__(self, unit: Unit):
        self.unit_id = unit.unit_id
        self.unit = unit
        self.unit_config: UnitConfig = unit.unit_cfg
        self.id_num = int(re.search(r"\d+", unit.unit_id).group())
        # Overridden for Friendly where pos changes
        self.start_of_turn_pos = tuple(unit.pos)
        self.start_of_turn_power = unit.power
        self.power = unit.power

    def update(self, unit: Unit):
        """Beginning of turn update"""
        self.unit = unit

    @property
    def unit_type(self) -> str:
        return self.unit.unit_type

    @property
    def pos(self):
        return self.unit.pos

    @property
    def action_queue(self):
        return self.unit.action_queue

    @action_queue.setter
    def action_queue(self, value):
        self.unit.action_queue = value

    @property
    def cargo(self):
        return self.unit.cargo

    @property
    def pos_slice(self):
        """Can be used for indexing arrays directly
        Examples:
            r = rubble[unit.pos_slice]
        """
        return np.s_[self.pos[0], self.pos[1]]

    @property
    def cargo_total(self) -> int:
        """Total cargo contents of unit"""
        c = self.cargo
        return c.ice + c.ore + c.metal + c.ice

    def power_cost_of_actions(self, rubble: np.ndarray, actions: List[np.ndarray] = None, max_actions=None):
        actions = actions if actions is not None else self.action_queue
        actions = actions[:max_actions]
        return util.power_cost_of_actions(
            start_pos=self.start_of_turn_pos,
            rubble=rubble,
            unit=self,
            actions=actions,
        )

    def pickup(self, pickup_resource, pickup_amount, repeat=0, n=1):
        return self.unit.pickup(pickup_resource, pickup_amount, repeat, n)

    def _valid_moving_actions(
        self,
        costmap: np.ndarray,
        start_of_turn_pos,
        action_queue,
        max_len=20,
        ignore_repeat=False,
    ) -> util.ValidActionsMoving:
        return util.calculate_valid_move_actions(
            start_of_turn_pos,
            action_queue,
            valid_move_map=costmap,
            max_len=max_len,
            ignore_repeat=ignore_repeat,
        )

    def valid_moving_actions(self, costmap, max_len=20, ignore_repeat=False) -> util.ValidActionsMoving:
        return self._valid_moving_actions(costmap, self.start_of_turn_pos, self.action_queue, max_len, ignore_repeat)

    @property
    def log_prefix(self) -> str:
        return f"{self.unit_type} {self.unit_id}(spos{self.start_of_turn_pos})(pos{self.pos}):"

    def current_path(self, max_len: int = 10, actions=None) -> np.ndarray:
        """Return current path from start of turn based on current action queue
        Adds a single no-move if not enough power to do the next move (hopefully avoids collisions better?)
        """
        if actions is None:
            actions = self.action_queue
        path = util.actions_to_path(self.start_of_turn_pos, actions, max_len=max_len)
        return path

    def actions_to_path(self, actions: [None, List[np.ndarray]] = None, max_len=20) -> np.ndarray:
        """
        Return a list of coordinates of the path the actions represent starting from unit.pos
        (which may have been updated since beginning of turn)
        """
        if actions is None:
            actions = self.unit.action_queue
        return util.actions_to_path(self.unit.pos, actions, max_len=max_len)

    @abc.abstractmethod
    def dead(self):
        """Called when unit dies, should tidy up anything related to unit"""
        pass


class EnemyUnitManager(UnitManager):
    def dead(self):
        logger.info(f"Enemy unit {self.unit_id} dead, nothing more to do")


class FriendlyUnitManager(UnitManager):
    # Maximum number of steps to plan ahead for (pause after that)
    max_queue_step_length = 50

    def __init__(self, unit: Unit, master_state: MasterState, factory_id: str):
        super().__init__(unit)
        self.factory_id = factory_id
        self.master: MasterState = master_state
        self.status: Status = Status(
            master=self.master,
        )
        self.action_handler = ActionHandler(self.master, self, self.max_queue_step_length)

        # Add these for convenience
        self.dig = unit.dig
        self.transfer = unit.transfer
        self.pickup = unit.pickup

        # Keep track of start of turn values (these are changed during planning)
        self.start_of_turn_actions = list(copy.copy(unit.action_queue))
        self.start_of_turn_pos = tuple(unit.pos)
        self.start_of_turn_power = unit.power
        self.start_of_turn_cargo = copy.copy(unit.cargo)
        self._action_queue = list(copy.copy(unit.action_queue))
        self._cargo = copy.copy(unit.cargo)
        self._power = unit.power
        self._pos = tuple(unit.pos)

        # Calculated per turn
        # Updated from status after stepped (this should be the one updated during planning)
        self.act_statuses: List[ActStatus] = []

    @property
    def log_prefix(self) -> str:
        log_prefix = super().log_prefix
        log_prefix += f"({self.status.current_action.category}:{self.status.current_action.sub_category}):\n\t\t\t"
        return log_prefix

    def update(self, unit: Unit):
        """Beginning of turn update"""
        super().update(unit)
        # Avoid changing the actual pos of unit.pos (which the env also uses)
        self.start_of_turn_actions = list(copy.copy(unit.action_queue))
        self.start_of_turn_pos = tuple(unit.pos)
        self.start_of_turn_power = unit.power
        self.start_of_turn_cargo = copy.copy(unit.cargo)

        # Init values from real unit
        self._action_queue = list(copy.copy(unit.action_queue))
        self._pos = tuple(unit.pos)
        self._power = unit.power
        self._cargo = copy.copy(unit.cargo)

        # Update after the unit is updated (uses start_of_turn...)
        self.status.update(self, self.master)
        # update from planned actions
        self.act_statuses = copy.copy(self.status.planned_act_statuses)

    def _recalculate_current_values(self):
        """Assuming all actions are valid, just add up the transfers etc"""
        from mining_planner import MineActSubCategory

        self.pos = self.start_of_turn_pos
        self.cargo = copy.copy(self.start_of_turn_cargo)
        self.power = self.start_of_turn_power

        self.pos = self.current_path(max_len=self.max_queue_step_length)[-1]
        self.power -= self.power_cost_of_actions()

        # And now cargo
        for action, act_status in zip(self.action_queue, self.act_statuses):
            act_type = action[util.ACT_TYPE]
            rtype = action[util.ACT_RESOURCE]
            dir = action[util.ACT_DIRECTION]
            amount = action[util.ACT_AMOUNT]
            n = action[util.ACT_N]
            if act_type == util.DIG:
                if act_status.sub_category == MineActSubCategory.ICE:
                    self.cargo.ice += self.unit_config.DIG_RESOURCE_GAIN * n
                elif act_status.sub_category == MineActSubCategory.ORE:
                    self.cargo.ore += self.unit_config.DIG_RESOURCE_GAIN * n
                continue
            if act_type == util.TRANSFER:
                if rtype == util.ICE:
                    self.cargo.ice -= amount
                elif rtype == util.ORE:
                    self.cargo.ore -= amount
                elif rtype == util.WATER:
                    self.cargo.water -= amount
                elif rtype == util.METAL:
                    self.cargo.metal -= amount
                continue

    def reset_to_step(self, step: int):
        self.status._reset_to_step(step)
        self.action_queue = self.status.planned_action_queue.copy()
        self.act_statuses = self.status.planned_act_statuses.copy()
        self._recalculate_current_values()

    @property
    def cargo(self):
        return self._cargo

    @cargo.setter
    def cargo(self, value):
        # if not isinstance(value, UnitCargo):
        if not hasattr(value, "metal"):  # TODO: Can switch back when not autoreloading in jupyter
            raise ValueError(f"got {value}, expected UnitCargo")
        self._cargo = copy.copy(value)

    @property
    def pos(self) -> util.POS_TYPE:
        return self._pos

    @pos.setter
    def pos(self, value):
        if value is None or len(value) != 2:
            raise ValueError(f"got {value} with type {type(value)} for pos")
        self._pos = tuple(value)

    @property
    def power(self) -> int:
        return self._power

    @power.setter
    def power(self, value):
        self._power = value

    @property
    def action_queue(self) -> List[np.ndarray]:
        return self._action_queue

    @action_queue.setter
    def action_queue(self, value):
        if not isinstance(value, list):
            raise TypeError(f"got {value}, expected List[np.ndarray]")
        self._action_queue = value

    def reset_unit_to_start_of_turn_empty_queue(self):
        """Reset unit to start of turn, with empty queue, for planning actions from scratch"""
        self.action_queue = []
        self.act_statuses = []
        self.power = self.start_of_turn_power
        self.pos = self.start_of_turn_pos
        self.cargo = self.start_of_turn_cargo

    def run_actions(
        self, actions_to_run: List[np.ndarray], act_statuses: List[ActStatus]
    ) -> ActionHandler.HandleStatus:
        """Run actions from start of turn to calculate pos, power, cargo and adding to action_queue and act_statuses
        Returns act status, and action_queue and act_statuses will be updated to the latest point before the raised status

        Note: this will break for anything not SUCCESS, but other statuses may not be considered failures
        """

        def _add_moves(move_acts: List[np.ndarray], targeting_enemy) -> ActionHandler.HandleStatus:
            path = util.actions_to_path(self.pos, move_acts, max_len=self.max_queue_step_length)
            status_ = self.action_handler.add_path(path, targeting_enemy=targeting_enemy)
            return status_

        if not len(actions_to_run) == len(act_statuses):
            raise ValueError(f"{len(actions_to_run)} == {len(act_statuses)} did not evaluate True")

        self.reset_unit_to_start_of_turn_empty_queue()
        SUCCESS = self.action_handler.HandleStatus.SUCCESS

        # Should collect move actions together as add_paths
        move_actions = []
        status = self.action_handler.HandleStatus.SUCCESS  # Default to success for no queue
        for action, act_status in zip(actions_to_run, act_statuses):
            self.status.current_action = act_status
            act_type = action[util.ACT_TYPE]
            rtype = action[util.ACT_RESOURCE]
            dir = action[util.ACT_DIRECTION]
            amount = action[util.ACT_AMOUNT]
            n = action[util.ACT_N]

            # If move type, collect it and move on to next
            if act_type == util.MOVE:
                move_actions.append(action)
                continue
            # Now add all the moves in one go
            elif len(move_actions) > 0:
                status = _add_moves(move_actions, act_status.targeting_enemy)
                if status != SUCCESS:
                    return status
                move_actions = []

            if act_type == util.DIG:
                status = self.action_handler.add_dig(n_digs=n)
                if status != SUCCESS:
                    return status
                continue

            if act_type == util.TRANSFER:
                status = self.action_handler.add_transfer(
                    resource_type=rtype, direction=dir, amount=amount, to_unit=False
                )
                if status != SUCCESS:
                    return status
                continue

            if act_type == util.PICKUP:
                status = self.action_handler.add_pickup(
                    resource_type=rtype, amount=amount, allow_partial=act_status.allow_partial
                )
                if status != SUCCESS:
                    return status
                continue

        # In case last action was a move
        if len(move_actions) > 0:
            # Will be using from the last value of the loop
            status = _add_moves(move_actions, act_status.targeting_enemy)
        return status

    @property
    def factory(self) -> FriendlyFactoryManager:
        if self.factory_id:
            factory = self.master.factories.friendly.get(self.factory_id, None)
            if factory is not None:
                return factory
        logger.error(f"{self.log_prefix}: f_id={self.factory_id} not in factories")
        raise ValueError(f"{self.log_prefix} has no factory")

    def dist_array(self, start_of_turn=True) -> np.ndarray:
        if start_of_turn:
            pos = self.start_of_turn_pos
        else:
            pos = self.pos
        return util.pad_and_crop(util.manhattan_kernel(30), self.master.maps.map_shape, pos[0], pos[1])

    def current_path(self, max_len: int = 10, actions=None, planned_actions=True) -> np.ndarray:
        """Return current path from start of turn based on current action queue"""
        if actions is None:
            actions = self.status.planned_action_queue if planned_actions else self.action_queue
        return super().current_path(max_len=max_len, actions=actions)

    def valid_moving_actions(
        self, costmap, max_len=20, ignore_repeat=False, planned_actions=True
    ) -> util.ValidActionsMoving:
        """Calculate the moving actions based on the planned actions queue"""
        if planned_actions:
            actions = self.status.planned_action_queue
        else:
            actions = self.action_queue
        return self._valid_moving_actions(costmap, self.start_of_turn_pos, actions, max_len, ignore_repeat)

    def next_action_is_move(self) -> bool:
        """Bool if next action is a move action"""
        if len(self.start_of_turn_actions) == 0:
            return False
        next_action = self.start_of_turn_actions[0]
        if next_action[util.ACT_TYPE] == util.MOVE and next_action[util.ACT_DIRECTION] != util.CENTER:
            return True
        return False

    def on_own_factory(self) -> bool:
        """Is this unit on its own factory"""
        factory_loc = self.factory_loc
        if factory_loc is not None:
            return self.factory_loc[self.pos[0], self.pos[1]] == 1
        return False
        # return (
        #     self.factory_loc[self.start_of_turn_pos[0], self.start_of_turn_pos[1]] == 1
        # )

    def pickup(self, pickup_resource, pickup_amount, repeat=0, n=1):
        """Keep track of a couple of turns of power pickups"""
        turns_of_actions = util.num_turns_of_actions(self.action_queue)
        # If this pickup is in very near future check the power is available
        if turns_of_actions < 2 and pickup_resource == util.POWER:
            logger.debug(f"Checking power pickup is valid and removing power from factory")
            factory_num = self.master.maps.factory_maps.friendly[self.pos_slice]
            if factory_num >= 0:
                factory_id = f"factory_{factory_num}"
                factory = self.master.factories.friendly.get(factory_id)
                if factory.short_term_power < pickup_amount:
                    logger.warning(
                        f"{self.log_prefix} planning to pickup {pickup_amount} but {factory_id} expects to have {factory.short_term_power}"
                    )
                factory.short_term_power -= pickup_amount
            else:
                logger.warning(f"{self.log_prefix} Did not find factory at {self.pos} for pickup")
        return self.unit.pickup(pickup_resource, pickup_amount, repeat, n)

    @property
    def factory_loc(self) -> [None, np.ndarray]:
        """Shortcut to the factory_loc of this unit or None if not assigned a factory"""
        factory = self.factory
        if factory is not None:
            return factory.factory_loc

    def power_cost_of_actions(self, rubble: np.ndarray = None, actions: List[np.ndarray] = None, max_actions=None):
        rubble = self.master.maps.rubble if rubble is None else rubble
        return super().power_cost_of_actions(rubble, actions, max_actions)

    def power_remaining(self, rubble: np.ndarray = None, actions: List[np.ndarray] = None) -> int:
        """Return power remaining at final step in actions so far"""
        rubble = self.master.maps.rubble if rubble is None else rubble
        return (
            self.start_of_turn_power
            - self.unit_config.ACTION_QUEUE_POWER_COST
            - self.power_cost_of_actions(rubble=rubble, actions=actions)
        )

    def dead(self):
        """Called when unit is detected as dead"""
        logger.warning(f"{self.log_prefix} Friendly unit dead")
        if self.factory_id:
            logger.info(f"removing from {self.factory_id} units also")
            fkey = "light" if self.unit.unit_type == "LIGHT" else "heavy"
            popped = getattr(self.master.factories.friendly[self.factory_id], f"{fkey}_units").pop(self.unit_id, None)
            if popped is None:
                logger.warning(f"{self.log_prefix}  was not in {self.factory_id} units")
