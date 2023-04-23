from __future__ import annotations
from typing import TYPE_CHECKING, Optional, List, Tuple
from enum import Enum
import abc

import util
import numpy as np

if TYPE_CHECKING:
    from agent import GeneralObs, UnitObs, FactoryObs
    from master_state import MasterState
    from unit_manager import FriendlyUnitManager
    from factory_manager import FactoryManager


# class Actions(Enum):
#     MINE_ICE = "mine_ice"
#     MINE_ORE = "mine_ore"
#     CLEAR_RUBBLE = "clear_rubble"
#     ATTACK = "attack"
#     RUN_AWAY = "run_away"
#     NOTHING = "do_nothing"
#     CONTINUE_NO_CHANGE = "continue previous action no change"
#     CONTINUE_UPDATE = "continue previous action but update"
#     CONTINUE_RESOLVED = "continue the should act has been resolved"
#

###################### OLD stuff that is still minimally used


class HighLevelAction(abc.ABC):
    """
    The commands that will be converted to action queues before returning to the lux engine
    Examples:
        # Simple ones that can basically be passed straight through
        - Build Heavy Unit (maybe just need to wait a turn or two)

        # More complex actions
        - Mine at X,Y for Factory Z
        - Dig Lichen in area near X, Y
    """

    pass


class Recommendation(HighLevelAction):
    role: str = None


######################### End of OLD


def split_actions_at_step(actions, split_step) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Split actions into two lists at a given step"""
    before_split = []
    after_split = []
    current_step = 0

    for action in actions:
        action_duration = action[util.ACT_N]

        if current_step + action_duration <= split_step:
            before_split.append(action)
        elif current_step < split_step:
            split_duration = split_step - current_step
            remaining_duration = action_duration - split_duration

            before_action = action.copy()
            before_action[util.ACT_N] = split_duration
            before_split.append(before_action)

            after_action = action.copy()
            after_action[util.ACT_N] = remaining_duration
            after_split.append(after_action)
        else:
            after_split.append(action)

        current_step += action_duration

    return before_split, after_split


def split_actions_at_two_steps(
    actions, split_step1, split_step2
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Split actions into three parts (useful for replacing middle part of actions)"""
    if split_step1 > split_step2:
        raise ValueError(f"{split_step1} > {split_step2}")
    before_split1, remaining = split_actions_at_step(actions, split_step1)
    between_split1_and_split2, after_split2 = split_actions_at_step(remaining, split_step2 - split_step1)
    return before_split1, between_split1_and_split2, after_split2


def replace_actions(existing_actions, start_step, end_step, new_actions) -> List[np.ndarray]:
    """Replace the actions between start step and end step with new actions
    Note: Does not replace the action at start step or end step, only between!
    Note: New actions may have different length"""
    before, middle, after = split_actions_at_two_steps(existing_actions, start_step, end_step)
    new = before + new_actions + after
    return new


def find_first_equal_pair_index(arr: np.ndarray, direction: str = "forward") -> Optional[int]:
    """Find index of first value of consecutive pair, or index of last value of last consecutive pair"""
    shifted_arr = np.roll(arr, shift=-1, axis=0)
    equal_pairs = np.all(arr[:-1] == shifted_arr[:-1], axis=1)

    if len(equal_pairs) == 0:
        return None

    if direction == "backward":
        equal_pairs = equal_pairs[::-1]

    index = np.argmax(equal_pairs)
    # if len(equal_pairs) > 0:
    #     index = np.argmax(equal_pairs)
    # else:
    #     index = None

    if index is None or not equal_pairs[index]:
        return None

    if direction == "backward":
        index = len(arr) - 1 - index

    return int(index)


def find_dest_step_from_step(full_path, step, direction="forward") -> int:
    """Find the index in the path corresponding to the last time a unit was at a fixed position (backward) or the first time it is at a new fixed position (forward)
    If no fixed positions backward, returns 0
    If no fixed positions forward, returns last index len(full_path) - 1
    """
    assert direction in ["forward", "backward"]
    if step < 0:
        raise ValueError(f" {step}(step) < 0 doesn't make sense")
    # if not 0 < step < len(full_path) - 1:
    #     raise ValueError(f" 0 < {step}(step) < {len(full_path)-1}(len path-1) not true")
    if direction == "forward":
        if step > len(full_path) - 1:
            return len(full_path) - 1
        path = full_path[step:]
    else:
        if step < 1:
            return 0
        path = full_path[: step + 1]

    equal_index = find_first_equal_pair_index(path, direction=direction)
    if equal_index is not None:
        if direction == "forward":
            # Count from step (backward is from beginning anyway)
            equal_index += step
        return equal_index
    else:
        if direction == "forward":
            return len(full_path) - 1
        else:
            return 0


def was_unit_moving_at_step(actions, step) -> bool:
    """Did unit move to arrive at given step... I.e. if action[0] is move, True for step 1 (undefined for step 0 returns False)
    I.e. Will unit move on step 1.
    E.g. If there is a collision on step 2, you'd want to know if unit **was** moving at step 2
    """
    if step == 0:
        # logger.warning(f"Can't know if unit was moving to get to current step 0, first action is whether moving at step 1")
        return False
    act = action_at_step(actions, step)
    if act is not None and act[util.ACT_TYPE] == util.MOVE and act[util.ACT_DIRECTION] != util.CENTER:
        return True
    return False


def will_unit_move_at_step(actions, step) -> bool:
    """
    Is units next move at given step going to be a move... I.e. if action[0] is move, True for step 0
    Usually you'd want to know if a unit WAS moving at a given step (i.e. did unit move into the collision)
    May want to use this for, will unit move from current step or to know if unit travelling (i.e. if it WAS moving and WILL move at step then it's travelling)
    """
    return was_unit_moving_at_step(actions, step + 1)


def action_at_step(actions, step) -> Optional[np.ndarray]:
    """Get the action at the given step (returns the whole action with whatever n value it had)"""
    if step == 0:
        # logger.warning(f"Can't know unit action at current step 0, first action is for step 1")
        # Don't know what action was taken to get to step 0 (i.e. where we are now)
        return None
    current_step = 0
    for action in actions:
        action_steps = action[util.ACT_N]
        if current_step + action_steps >= step:
            return action
        current_step += action_steps
    # Don't know what's happening after actions
    return None
