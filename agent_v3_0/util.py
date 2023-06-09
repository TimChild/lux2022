from __future__ import annotations
import time
import os
import threading
from dataclasses import dataclass
from scipy.ndimage import distance_transform_cdt
from collections import deque
import functools
from tqdm.auto import tqdm
from typing import Tuple, List, Union, Optional, TYPE_CHECKING, Dict, Callable
import copy
from scipy import ndimage
from scipy.signal import convolve2d
import numpy as np
import math
import re
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
import plotly.colors as pc
from plotly.subplots import make_subplots
from itertools import product

from luxai_s2 import LuxAI_S2
from luxai_s2.unit import UnitType

from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to
from lux.unit import Unit
from lux.cargo import UnitCargo

from config import get_logger
from new_path_finder import Pather


if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManager, UnitManager
    from factory_manager import FriendlyFactoryManager
    from master_state import MasterState

logger = get_logger(__name__)

POS_TYPE = Union[Tuple[int, int], np.ndarray, Tuple[np.ndarray]]
PATH_TYPE = Union[List[Tuple[int, int]], np.ndarray]


ENV_CFG = EnvConfig()
LIGHT_UNIT = Unit(
    team_id=-1,
    unit_id="none",
    unit_type="LIGHT",
    pos=np.array((0, 0)),
    power=1000,
    cargo=UnitCargo(),
    env_cfg=ENV_CFG,
    unit_cfg=ENV_CFG.ROBOTS["LIGHT"],
    action_queue=[],
)
HEAVY_UNIT = Unit(
    team_id=-1,
    unit_id="none",
    unit_type="HEAVY",
    pos=np.array((0, 0)),
    power=1000,
    cargo=UnitCargo(),
    env_cfg=ENV_CFG,
    unit_cfg=ENV_CFG.ROBOTS["HEAVY"],
    action_queue=[],
)

MOVE_DELTAS = np.array([[0, 0], [0, -1], [1, 0], [0, 1], [-1, 0]])
MOVE_DIRECTIONS = np.array([0, 1, 2, 3, 4])

CENTER = 0
UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

ICE = 0
ORE = 1
WATER = 2
METAL = 3
POWER = 4

# Actions:
# (type, direction, resource, amount, repeat, n)
ACT_TYPE = 0
ACT_DIRECTION = 1
ACT_RESOURCE = 2
ACT_AMOUNT = 3
ACT_REPEAT = 4
ACT_N = 5

# Types:
MOVE = 0
TRANSFER = 1
PICKUP = 2
DIG = 3
DESTRUCT = 4
RECHARGE = 5


################# General #################


def timeout_decorator(timeout):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            result_container = {"result": None}

            def target_fn():
                result_container["result"] = fn(*args, **kwargs)

            thread = threading.Thread(target=target_fn)
            thread.start()
            thread.join(timeout)

            if thread.is_alive():
                raise TimeoutError(f"Function '{fn.__name__}' timed out after {timeout} seconds")
            else:
                return result_container["result"]

        return wrapper

    return decorator


class MyEnv:
    def __init__(self, seed: int, Agent, OtherAgent):
        self.seed = seed

        self.env = LuxAI_S2()
        self.obs = self.env.reset(self.seed)
        self.env_step = 0
        self.real_env_steps = 0

        self.Agent = Agent
        self.OtherAgent = OtherAgent
        self.agent: Agent = None
        self.other_agent: OtherAgent = None
        self.agents: Dict[str, Agent] = {}
        self.init_agents()
        # self.agent = Agent("player_0", self.env.env_cfg)
        # self.other_agent = OtherAgent("player_1", self.env.env_cfg)
        # self.agents = {a.player: a for a in [self.agent, self.other_agent]}

        # For being able to undo
        self._previous_step = deque(maxlen=5)
        self._previous_real_steps = deque(maxlen=5)
        self._previous_obs = deque(maxlen=5)
        self._previous_env = deque(maxlen=5)

    def init_agents(self):
        self.agent = self.Agent("player_0", self.env.env_cfg)
        self.other_agent = self.OtherAgent("player_1", self.env.env_cfg)
        self.agents = {a.player: a for a in [self.agent, self.other_agent]}

    def run_early_setup(self):
        """Run through the placing factories stage"""
        # Reset
        self.init_agents()
        self.obs = self.env.reset(self.seed)
        self.env_step = 0
        self.real_env_steps = 0

        # Run placing factory actions
        while self.env.state.real_env_steps < 0:
            self._record_state()
            actions = self.get_early_actions()
            self.env_step += 1
            self.real_env_steps = self.env.state.real_env_steps
            self.obs, rewards, dones, infos = self.env.step(actions)

    def get_early_actions(self):
        """Get next early setup actions"""
        actions = {}
        for player in self.agents:
            o = self.obs[player]
            acts = self.agents[player].early_setup(self.env_step, o)
            actions[player] = acts
        return actions

    def undo(self):
        """Undoes the last turn (can be done a few times in a row)"""
        self.env_step, self.real_env_steps, self.obs, self.env = (
            self._previous_step.pop(),
            self._previous_real_steps.pop(),
            self._previous_obs.pop(),
            self._previous_env.pop(),
        )

    def _record_state(self):
        """So that I can easily undo actions later"""
        self._previous_step.append(self.env_step)
        self._previous_real_steps.append(self.env_step)
        self._previous_obs.append(self.obs)
        self._previous_env.append(copy.deepcopy(self.env))

    def step(self) -> bool:
        """Progress a single step"""
        logger.info(f"Carrying out real step {self.real_env_steps}, env step {self.env_step}")

        self._record_state()

        try:
            actions = self.get_actions()
            self.env_step += 1
            self.obs, rewards, dones, infos = self.env.step(actions)
            self.real_env_steps = self.env.state.real_env_steps
            if any(dones.values()) and self.env_step < 1000:
                print(
                    f"One of the players dones came back True, previous state restored, use myenv.get_actions() to "
                    f"repeat gathering latest turns actions"
                )
                self.undo()
                return False
        except Exception as e:
            print(
                f"Error caught while running step, previous state restored, use myenv.get_actions() to "
                f"repeat gathering actions"
            )
            self.undo()
            raise e
        return True

    def get_actions(self):
        """Get next normal step actions"""
        actions = {}
        for player, agent in self.agents.items():
            acts = agent.act(self.env_step, self.obs[player])
            actions[player] = acts
        return actions

    def run_to_step(self, real_env_step: int):
        """Keep running until reaching real_env_step"""
        num_steps = real_env_step - 1 - self.real_env_steps
        if num_steps < 1:
            print(f"Already at step {self.real_env_steps}")
            return
        for _ in tqdm(range(num_steps), total=num_steps):
            # while self.real_env_steps < real_env_step:
            success = self.step()
            if not success:
                break

    def show(self) -> go.Figure:
        """Display the current env state"""
        print(f"Displaying Step: {self.real_env_steps}")
        return show_env(self.env)


class MyReplayEnv(MyEnv):
    def __init__(
        self, seed, Agent, replay_json, log_file_path: str, my_player="player_0", collector: CollectInfoFromEnv = None
    ):
        self.replay_json = replay_json
        self.player = my_player
        self.other_player = "player_1" if self.player == "player_0" else "player_0"
        self.log_file_path = log_file_path
        self.collector = collector
        super().__init__(seed, Agent, Agent)

        self._all_actions = self._all_replay_actions()

    def init_agents(self):
        super().init_agents()
        if self.player == "player_0":
            self.agents = {"player_0": self.agent}
        else:
            self.agent = self.other_agent
            self.agents = {"player_1": self.other_agent}
        self.other_agent = None

    def get_early_actions(self):
        actions = super().get_early_actions()
        actions[self.other_player] = self.get_replay_actions()
        return actions

    def get_actions(self):
        actions = super().get_actions()
        actions[self.other_player] = self.get_replay_actions()
        return actions

    def step(self) -> bool:
        success = super().step()
        if success and self.collector is not None:
            self.collector.collect(self)
        return success

    def _all_replay_actions(self):
        if "actions" in self.replay_json:
            return [self.replay_json["actions"][i][self.other_player] for i in range(len(self.replay_json["actions"]))]
        elif "steps" in self.replay_json:
            return [
                self.replay_json["steps"][i + 1][int(self.other_player[-1])]["action"]
                for i in range(len(self.replay_json["steps"]) - 1)
            ]

    def get_replay_actions(self, step=None):
        if step is None:
            step = self.env_step
        all_step_actions = self._all_actions
        if len(all_step_actions) > step:
            actions = all_step_actions[step]
            cleaned_actions = {}
            for k, acts in actions.items():
                # If a unit action
                if isinstance(acts, list) and not k == "spawn":
                    # Only if that unit is still on that team
                    if k in self.env.state.units[self.other_player].keys():
                        cleaned_actions[k] = np.array(acts, dtype=int)
                    else:
                        logger.warning(f"Removed enemy actions for {k} because no longer on that team")
                # Factory action
                else:
                    cleaned_actions[k] = acts
            return cleaned_actions
        logger.info(f"No more replay actions for {self.other_player} stopped at {len(all_step_actions)}")
        return {}

    def _get_num_lines(self):
        with open(self.log_file_path, "r") as file:
            return len(file.readlines())

    def run_until_log_increases(self, num_lines: int, max_step=None):
        initial_num_lines = self._get_num_lines()
        with tqdm(total=num_lines, desc="Watching log file") as pbar:
            while True:
                success = self.step()
                current_num_lines = self._get_num_lines()
                lines_increased = current_num_lines - initial_num_lines

                pbar.n = lines_increased
                pbar.refresh()

                if (
                    success is False
                    or lines_increased >= num_lines
                    or (max_step is not None and self.real_env_steps >= max_step)
                ):
                    break
                time.sleep(0.01)  # Adjust the sleep time if needed
        print(f"Logfile increased by {lines_increased}")


class CollectInfoFromEnv:
    def __init__(self, player: str):
        self.player = player
        # For each step, unit_id, num since last update
        self.unit_last_update_step: Dict[int, Dict[str, int]] = {}

        self.agent = None
        self.master: MasterState = None

    def _collect_friendly_unit_info(self):
        step_info = {}
        for unit_id, unit in self.master.units.friendly.all.items():
            unit: FriendlyUnitManager
            step_info[unit_id] = unit.status.last_action_update_step
        self.unit_last_update_step[self.master.step] = step_info

    def collect(self, env: MyEnv):
        self.agent = env.agents[self.player]
        self.master: MasterState = self.agent.master

        self._collect_friendly_unit_info()

    def plot_unit_last_update_step(self) -> go.Figure:
        # Prepare data for plotting
        unit_data = {}
        for idx, d in self.unit_last_update_step.items():
            for unit, value in d.items():
                if unit not in unit_data:
                    unit_data[unit] = []
                unit_data[unit].append((idx, value))

        # Create a Plotly Scatter plot
        fig = go.Figure()
        fig.update_layout(template="plotly_white")

        for unit, values in unit_data.items():
            x_values = [v[0] for v in values]
            y_values = [v[1] for v in values]
            fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="markers+lines", name=unit))

        # Set the plot's title and axis labels
        fig.update_layout(
            title="Last update step vs Real Env Steps",
            xaxis_title="Step Index",
            yaxis_title="Last update step",
        )
        return fig


##########################################


def generate_circle_coordinates_array(center: Tuple[int, int], N: int, radius: float, max_coord: int) -> np.ndarray:
    x, y = center
    angles = np.linspace(0, 2 * np.pi, N + 1)[:-1]  # Exclude the last angle (2 * pi) to avoid overlap

    # Initialize an empty numpy array
    coord_array = np.zeros((max_coord, max_coord))

    for angle in angles:
        # Check if the angle is in one of the cardinal direction gaps
        angle_deg = np.degrees(angle) % 360
        new_x = round(x + radius * np.cos(angle))
        new_y = round(y + radius * np.sin(angle))

        # Check if the coordinates are within the specified range
        if 0 < new_x < max_coord and 0 < new_y < max_coord:
            coord_array[new_x, new_y] = 1  # Set the value at the coordinate to 1

    return coord_array


def nearest_non_zero(array: np.ndarray, pos: Union[np.ndarray, Tuple[int, int]]) -> Optional[Tuple[int, int]]:
    """Nearest location from pos in array where value is positive"""
    locations = np.argwhere(array > 0)
    if len(locations) == 0:
        return None
    # TODO: For long lists of locations, I could do a manhattan distance kernel thing to find nearest
    # TODO: Could break if 0 found
    distances = np.array([manhattan(loc, pos) for loc in locations])
    closest = locations[np.argmin(distances)]
    return tuple(closest)

def non_zero_coords(arr: np.ndarray) -> List[Tuple[int, int]]:
    non_zero_indices = np.nonzero(arr)
    coordinates = list(zip(non_zero_indices[0], non_zero_indices[1]))
    return coordinates

def num_turns_of_actions(actions: Union[np.ndarray, List[np.ndarray]]) -> int:
    """Calculate how many turns the actions will take including their n values, does not include repeat (and should not)"""
    if len(actions) == 0:
        return 0
    actions = np.array(actions)
    durations = actions[:, ACT_N]
    return int(np.sum(durations))


def power_cost_of_actions(
    start_pos: POS_TYPE,
    rubble: np.ndarray,
    unit: UnitManager,
    actions: List[np.ndarray],
):
    """Power requirements of a list of actions

    Note: Does not check for invalid moves or actions
    """
    unit_cfg = unit.unit_config
    pos = tuple(start_pos)

    cost = 0
    for action in actions:
        for _ in range(action[ACT_N]):
            act_type = action[ACT_TYPE]
            if act_type == MOVE:
                pos = pos + MOVE_DELTAS[ACT_DIRECTION]
                try:
                    rubble_at_target = rubble[pos[0], pos[1]]
                except IndexError:
                    logger.error(f"Index Error invalid position is ({pos}) continuing anyway")
                    rubble_at_target = 100
                cost += math.ceil(unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target)
            elif act_type == TRANSFER or (act_type == PICKUP and action[ACT_RESOURCE] != POWER):
                pass  # No cost
            elif act_type == PICKUP and action[ACT_RESOURCE] == POWER:
                cost -= action[ACT_AMOUNT]
            elif act_type == DESTRUCT:
                cost += unit_cfg.SELF_DESTRUCT_COST
            elif act_type == RECHARGE:
                cost -= unit_cfg.CHARGE
            elif act_type == DIG:
                cost += unit_cfg.DIG_COST
    return cost


def count_connected_values(arr: np.ndarray, value: int = 0) -> np.ndarray:
    """Returns an array where the values are how many connected `value`s there are"""
    # Define structuring element for labeling
    struct = ndimage.generate_binary_structure(2, 1)

    # Label connected regions of zeros
    labeled_arr, num_labels = ndimage.label(arr == value, structure=struct)

    # Count the number of labels in each region
    count_arr = np.zeros_like(arr)
    for i in range(1, num_labels + 1):
        count_arr[labeled_arr == i] = np.sum(labeled_arr == i)
    return count_arr


@functools.lru_cache(maxsize=500)
def manhattan_kernel(max_dist: int) -> np.ndarray:
    """
    Make a kernel with manhattan distance weightings
    Args:
        max_dist: How far out from center kernel should extend
    """
    # Make grid
    x = np.arange(-max_dist, max_dist + 1)
    y = np.arange(-max_dist, max_dist + 1)
    xx, yy = np.meshgrid(x, y)

    # Manhattan Distance from center
    manhattan_dist = np.abs(xx - 0) + np.abs(yy - 0)

    return manhattan_dist


def factory_map_kernel(search_dist, dist_multiplier=0.8):
    """Generate kernel that extends `search_dist` away from factory with 3x3 blanked out (for factory size) in the
    middle and decreasing value for distance from factory"""
    manhattan_dist = manhattan_kernel(search_dist)

    # Value accounting for dist_multiplier
    ones = np.ones(manhattan_dist.shape) / dist_multiplier
    values = ones * dist_multiplier**manhattan_dist

    # Set middle to zero (will become blocked out factory part)
    mid_index = search_dist  # This will always also be the middle index
    values[mid_index, mid_index] = 0

    values = stretch_middle_of_factory_array(values)
    return values


def convolve_array_kernel(arr, kernel, fill=0):
    """Convolve array and kernel returning same dimensions as array (filling edges with `fill` for conv)"""
    convolved = convolve2d(arr, kernel, mode="same", boundary="fill", fillvalue=fill)
    return convolved


def pad_and_crop(small_arr, large_arr_shape, x1, y1, fill_value=0):
    """
    Pads the edges of small_arr with zeros so that the middle of the small_arr ends up at
    the coordinate (x1, y1) of large_arr. The small_arr will be cropped to fit inside the
    large_arr. Returns the padded and cropped version of small_arr.

    Args:
        small_arr: a 2D numpy array
        large_arr_shape: shape of large array to pad small array for
        x1: the x-coordinate of the center of small_arr in large_arr
        y1: the y-coordinate of the center of small_arr in large_arr
        fill_value: Value to fill the padded areas with

    Returns:
        A padded and cropped version of small_arr
    """
    x_size, y_size = small_arr.shape

    x_start = max(0, x1 - x_size // 2)
    y_start = max(0, y1 - y_size // 2)
    x_end = min(large_arr_shape[0], x1 + x_size // 2 + 1 - (x_size % 2 == 0))
    y_end = min(large_arr_shape[1], y1 + y_size // 2 + 1 - (y_size % 2 == 0))

    small_arr_start_x = x_start - x1 + x_size // 2
    small_arr_start_y = y_start - y1 + y_size // 2
    padded_arr = np.full(large_arr_shape, fill_value=fill_value, dtype=small_arr.dtype)
    padded_arr[x_start:x_end, y_start:y_end] = small_arr[
        small_arr_start_x : small_arr_start_x + (x_end - x_start),
        small_arr_start_y : small_arr_start_y + (y_end - y_start),
    ]

    return padded_arr


def connected_array_values_from_pos(arr: np.ndarray, pos: POS_TYPE, connected_value=0) -> np.ndarray:
    """Figure out what area of zeros is connected to factory and return that array"""
    struct = ndimage.generate_binary_structure(rank=2, connectivity=1)
    # Note:
    # rank 2 = image dimensions
    # connectivity 1 = adjacent only (no diagonals)

    labelled_arr, num_labels = ndimage.label(arr == connected_value, structure=struct)
    # Note:
    # labelled_arr = unique values for each connected area of 0s

    id_at_pos = labelled_arr[pos[0], pos[1]]
    connected_array = np.zeros(arr.shape)
    connected_array[labelled_arr == id_at_pos] = 1
    return connected_array


def append_zeros(arr, side):
    """
    Appends zeros to the given side of a 2D numpy array.

    Parameters:
        arr (numpy.ndarray): The original 2D numpy array.
        side (str): The side to append the zeros to. Can be 'top', 'bottom', 'left', or 'right'.

    Returns:
        numpy.ndarray: The resulting 2D numpy array with zeros appended to the given side.
    """
    if side == "top":
        zeros = np.zeros(arr.shape[1])
        return np.vstack((zeros, arr))
    elif side == "bottom":
        zeros = np.zeros(arr.shape[1])
        return np.vstack((arr, zeros))
    elif side == "left":
        zeros = np.zeros(arr.shape[0])
        return np.hstack((zeros[:, np.newaxis], arr))
    elif side == "right":
        zeros = np.zeros(arr.shape[0])
        return np.hstack((arr, zeros[:, np.newaxis]))
    else:
        raise ValueError("Invalid side specified. Must be 'top', 'bottom', 'left', or 'right'.")


def create_boundary_array(arr, boundary_num=0):
    """Creates array of boundaries around boundary_num areas where the boundary value is equal to the size of the boundary_num area"""
    arr = np.array(arr)  # Convert input to numpy array if not already
    boundary_areas = arr == boundary_num  # Create a boolean array where True represents boundary_num

    # Create a binary structure to label connected zero areas not including diagonal connections
    s = ndimage.generate_binary_structure(2, 1)
    labeled_array, num_features = ndimage.label(boundary_areas, structure=s)  # Label connected zero areas

    result = np.zeros_like(arr)  # Create a new array with the same shape as the input array, filled with zeros

    for label_num in range(1, num_features + 1):
        mask = labeled_array == label_num
        size = np.count_nonzero(mask)

        # Create a dilated version of the mask to find the boundary
        dilated_mask = np.pad(mask, 1, mode="constant", constant_values=0)
        for _ in range(2):
            dilated_mask = np.maximum.reduce(
                [
                    dilated_mask[:-1, :-1],
                    dilated_mask[:-1, 1:],
                    dilated_mask[1:, :-1],
                    dilated_mask[1:, 1:],
                ]
            )

        # Set the boundary values to the size
        # boundary_mask = dilated_mask[1:-1, 1:-1] & ~mask
        boundary_mask = dilated_mask & ~mask
        result[boundary_mask] = size

    return result


class SubsetExtractor:
    def __init__(self, array: np.ndarray, pos: Tuple[int, int], radius: int, fill_value=0):
        """
        For getting a subset of a 2D numpy array based on a given coordinate and radius.

        Args:
            array: 2D numpy array to extract a subset from.
            pos: position of center of the subset (x, y) using lux ordering arr[x, y]
            radius: distance from the center of the subset to its edge.
            fill_value: The value to fill the subset with if any of the indices are out of bounds.
        """
        self.array = array
        self.pos = pos
        self.radius = radius
        self.fill_value = fill_value

    def get_subset(self) -> np.ndarray:
        """
        Returns a subset of a 2D numpy array based on a given coordinate and radius.

        Returns:
            np.ndarray: Subset of the input array.
        """
        x, y = self.pos
        y_min = max(0, y - self.radius)
        y_max = min(self.array.shape[0] - 1, y + self.radius)
        x_min = max(0, x - self.radius)
        x_max = min(self.array.shape[1] - 1, x + self.radius)

        subset = np.full((2 * self.radius + 1, 2 * self.radius + 1), self.fill_value)
        subset[
            self.radius - (x - x_min) : self.radius + (x_max - x) + 1,
            self.radius - (y - y_min) : self.radius + (y_max - y) + 1,
        ] = self.array[x_min : x_max + 1, y_min : y_max + 1]
        return subset

    def convert_coordinate(self, original_coord) -> Tuple[int, int]:
        """
        Converts a coordinate from the original array to the corresponding coordinate in the subset array.

        Args:
            original_coord: position in the original array (x, y) using lux ordering arr[x, y]

        Returns:
            Tuple[int, int]: Corresponding position in the subset array.
        """
        ox, oy = original_coord
        x, y = self.pos
        return ox - (x - self.radius), oy - (y - self.radius)


def stretch_middle_of_factory_array(array):
    """Stretch out the middle (i.e. factory is 3x3 not 1x1)"""
    shape = array.shape
    assert shape[0] == shape[1]
    assert shape[0] % 2 == 1

    mid_index = np.floor(array.shape[0] / 2).astype(int)

    # y-direction
    array = np.insert(array, mid_index, array[mid_index, :], axis=0)
    array = np.insert(array, mid_index, array[mid_index, :], axis=0)
    # x-direction
    array = np.insert(array, mid_index, array[:, mid_index], axis=1)
    array = np.insert(array, mid_index, array[:, mid_index], axis=1)
    return array


def manhattan_distance_between_values(input_array, value=0):
    """Calculates manhattan distance between areas of `value`"""
    # Create a mask of the input array where the values are True where there are zeros
    mask = input_array == value

    # Calculate the Manhattan distance using distance_transform_cdt function from SciPy
    output_array = distance_transform_cdt(mask, metric="taxicab")

    return output_array


def direction_to(src, target):
    """Just get basic direction from src to target"""
    # From lux_kit.utils
    # direction (0 = center, 1 = up, 2 = right, 3 = down, 4 = left)
    ds = target - src
    dx = ds[0]
    dy = ds[1]
    if dx == 0 and dy == 0:
        return 0
    if abs(dx) > abs(dy):
        if dx > 0:
            return 2
        else:
            return 4
    else:
        if dy > 0:
            return 3
        else:
            return 1


def add_direction_to_pos(pos: POS_TYPE, direction: int) -> Tuple[int, int]:
    """Add direction to pos"""
    return np.array(pos) + MOVE_DELTAS[direction]


def find_border_coords(arr, value=0):
    """Find the coords just outside an area of `value`
    Examples:
        Assuming array was all 1s except for 2x2 of 0s, the coords of x's would be returned
        1 1 x x 1
        1 x 0 0 x
        1 x 0 0 x
        1 1 x x 1
    """
    # Create a boolean mask for the non-zero values
    mask = arr == value

    # Create a structuring element for dilation
    struc = ndimage.generate_binary_structure(2, 1)  # Not not diagonals

    # Dilate the mask using binary_dilation
    dilated = ndimage.binary_dilation(mask, struc) & ~mask

    # Find the coordinates of the values just outside of the square of non-zeros
    coords = np.argwhere(dilated)
    return coords


################# Moving #################
def manhattan(a, b):
    return sum(abs(val1 - val2) for val1, val2 in zip(a, b))


def count_consecutive(lst: Union[List[int], np.ndarray]) -> Tuple[List[int], List[int]]:
    """
    Compute the unique values and the number of consecutive occurrences of each value in a list.

    Args:
        lst: The input list.

    Returns:
        A tuple (values, counts), where `values` is a list of unique values from the input list,
        and `counts` is a list of the number of consecutive occurrences of each value in the input list.
        The i-th element of `counts` corresponds to the i-th element of `values`.
    """
    values = []
    counts = []
    current_value = None
    current_count = 0
    for x in lst:
        if x != current_value:
            if current_value is not None:
                values.append(current_value)
                counts.append(current_count)
            current_value = x
            current_count = 1
        else:
            current_count += 1
    if current_value is not None:
        values.append(current_value)
        counts.append(current_count)
    return values, counts


def list_of_tuples_to_array(lst: List[List[Tuple[int, int]]]) -> np.ndarray:
    """
    Convert a list of sublists of tuples to a 3D NumPy array.
    For converting a list of paths to a single array of paths

    Args:
        lst: The input list of sublists of tuples.

    Returns:
        A 3D NumPy array with shape (len(lst), max_len, 2), where max_len is the maximum length
        of the sublists, and missing values are filled with NaN.
    """
    max_len = max(len(sublst) for sublst in lst)
    result = np.full((len(lst), max_len, 2), np.nan)
    for i, sublst in enumerate(lst):
        for j, tup in enumerate(sublst):
            result[i, j, :] = tup
    return result


def path_to_actions(path):
    """Converts path to actions (combines same direction moves by setting higher n)"""
    if len(path) == 0:
        logger.warning(f"path_to_actions got empty path {path}")
        return []
    pos = path[0]
    directions = []
    for p in path[1:]:
        d = direction_to(pos, p)
        pos = add_direction_to_pos(pos, d)
        directions.append(d)

    directions, ns = count_consecutive(directions)

    actions = []
    for d, n in zip(directions, ns):
        actions.append(LIGHT_UNIT.move(d, n=n))  # Note: UnitType doesn't matter
    return actions


def actions_to_path(
    start_pos: POS_TYPE,
    actions: List[np.ndarray],
    max_len: int = 20,
    ignore_repeat: bool = False,
) -> np.ndarray:
    """Convert list of actions (from start_pos) into a path
    Note: First value in path is current position, repeated values when not moving
    """
    path = [start_pos]
    pos = copy.copy(start_pos)
    actions = copy.copy(actions)
    actions = list(actions)
    for i, action in enumerate(actions):
        for _ in range(action[ACT_N]):
            if action[0] == MOVE:
                direction = action[1]
            else:
                direction = CENTER
            pos = pos + MOVE_DELTAS[direction]
            path.append(pos)
            # If only calculating to specific max_len, return once reached
            if len(path) >= max_len:
                return np.array(path)

        # Account for repeat actions
        if not ignore_repeat and action[ACT_REPEAT] > 0:
            loop_act = copy.copy(action)
            loop_act[ACT_N] = action[ACT_REPEAT]
            actions.append(loop_act)

    return np.array(path)


def move_to_new_spot_on_factory(pathfinder: Pather, unit: FriendlyUnitManager, factory: FriendlyFactoryManager):
    """Move to a new spot on factory (i.e. if current spot is going to be occupied by another unit next turn)"""
    success = False
    pos = np.array(unit.pos)
    best_direction = None
    best_cost = 999
    costmap = pathfinder.generate_costmap(unit)
    for direction, delta in zip(MOVE_DIRECTIONS[1:], MOVE_DELTAS[1:]):
        new_x, new_y = pos + delta
        try:
            # If on factory
            if factory.factory_loc[new_x, new_y]:
                new_cost = costmap[new_x, new_y]
                # If this is better based on costmap
                if 0 < new_cost < best_cost:
                    best_direction = direction
                    best_cost = new_cost
        except IndexError as e:
            logger.info(f"Index error for ({new_x, new_y}), probably near edge of map, ignoring this position")
    # Move unit to best location if available
    if best_direction is not None:
        pathfinder.append_direction_to_actions(unit, best_direction)
        success = True
    else:
        logger.error(f"No best_direction to move on factory")
    return success


def move_to_cheapest_adjacent_space(pathfinder: Pather, unit: FriendlyUnitManager, collision_only=False) -> bool:
    """Move to cheapest location wherever unit currently is"""
    cm = pathfinder.generate_costmap(unit, collision_only=collision_only)
    pos = np.array(unit.pos)
    best_direction = None
    best_cost = 999
    for direction, delta in zip(MOVE_DIRECTIONS[1:], MOVE_DELTAS[1:]):
        new_x, new_y = pos + delta
        try:
            new_cost = cm[new_x, new_y]
            if 0 < new_cost < best_cost:
                best_direction = direction
                best_cost = new_cost
        except IndexError as e:
            logger.info(f"Index error for ({new_x, new_y}), probably near edge of map, ignoring this position")
            continue
    if best_direction is None:
        logger.error(f"{unit.log_prefix}: No adjacent direction to move, not doing anything")
        success = False
    else:
        logger.info(f"Moved unit to cheapest adjacent space at {pos+best_direction}")
        pathfinder.append_direction_to_actions(unit, best_direction)
        success = True
    return success


################### Plotting #################
# def get_test_env(path=None):
#     if path is None:
#         path = "test_state.pkl"
#     with open(path, "rb") as f:
#         state = pickle.load(f)
#     env = LuxAI_S2()
#     _ = env.reset(state.seed)
#     env.set_state(state)
#     return env


def game_state_from_env(env: LuxAI_S2):
    return obs_to_game_state(env.state.env_steps, env.state.env_cfg, env.state.get_obs())


# def run(
#     agent1,
#     agent2,
#     map_seed=None,
#     save_state_at=None,
#     max_steps=1000,
#     return_type='replay',
# ):
#     make_fig = False
#     if return_type == 'figure':
#         make_fig = True
#     elif return_type == 'replay':
#         pass
#     else:
#         raise ValueError(f'return_type={return_type} not valid')
#
#     env = LuxAI_S2()
#     # This code is partially based on the luxai2022 CLI:
#     # https://github.com/Lux-AI-Challenge/Lux-Design-2022/blob/main/luxai_runner/episode.py
#
#     obs = env.reset(seed=map_seed)
#     state_obs = env.state.get_compressed_obs()
#
#     agents = {
#         "player_0": agent1("player_0", env.state.env_cfg),
#         "player_1": agent2("player_1", env.state.env_cfg),
#     }
#
#     game_done = False
#     rewards, dones, infos = {}, {}, {}
#
#     for agent_id in agents:
#         rewards[agent_id] = 0
#         dones[agent_id] = 0
#         infos[agent_id] = {"env_cfg": dataclasses.asdict(env.state.env_cfg)}
#
#     replay = {"observations": [state_obs], "actions": [{}]}
#
#     if make_fig:
#         fig = initialize_step_fig(env)
#     i = 0
#     while not game_done and i < max_steps:
#         i += 1
#         if save_state_at and env.state.real_env_steps == save_state_at:
#             print(f'Saving State at Real Env Step :{env.state.real_env_steps}')
#             import pickle
#
#             with open('test_state.pkl', 'wb') as f:
#                 pickle.dump(env.get_state(), f)
#
#         actions = {}
#         for agent_id, agent in agents.items():
#             agent_obs = obs[agent_id]
#
#             if env.state.real_env_steps < 0:
#                 agent_actions = agent.early_setup(env.env_steps, agent_obs)
#             else:
#                 agent_actions = agent.act(env.env_steps, agent_obs)
#
#             for key, value in agent_actions.items():
#                 if isinstance(value, list):
#                     agent_actions[key] = np.array(value, dtype=int)
#
#             actions[agent_id] = agent_actions
#
#         new_state_obs, rewards, dones, infos = env.step(actions)
#
#         change_obs = env.state.get_change_obs(state_obs)
#         state_obs = new_state_obs["player_0"]
#         obs = new_state_obs
#
#         replay["observations"].append(change_obs)
#         replay["actions"].append(actions)
#
#         players_left = len(dones)
#         for key in dones:
#             if dones[key]:
#                 players_left -= 1
#
#         if players_left < 2:
#             game_done = True
#
#     if make_fig:
#         add_env_step(fig, env)
#     if make_fig:
#         return fig
#     else:
#         replay = to_json(replay)
#     return replay


def show_map_array(map_array):
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=map_array.T,
        ),
    )
    fig.update_yaxes(
        autorange="reversed",
    )
    fig.update_layout(height=400, width=500)
    return fig


def show_env(env, mode="plotly", fig=None):
    game_state = game_state_from_env(env)
    if mode == "mpl":
        fig, ax = plt.subplots(1)
        _mpl_add_board(ax, game_state)
        _mpl_add_factories(ax, game_state)
        _mpl_add_units(ax, game_state)
    elif mode == "plotly":
        if fig is None:
            fig = go.Figure()
            fig.update_layout(
                width=400,
                height=400,
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=10, b=10),
            )
            fig.update_coloraxes(showscale=False)
        _plotly_add_board(fig, game_state)
        fig.layout.shapes = []  # remove any previously existing shapes
        _plotly_add_factories(fig, game_state)
        _plotly_add_units(fig, game_state)
    return fig


def _plotly_add_board(fig, state: GameState):
    step = state.real_env_steps
    fig.add_trace(
        go.Heatmap(
            z=state.board.rubble.T,
            colorscale="OrRd",
            name="Rubble",
            uid=f"{step}_Rubble",
            showscale=False,
        )
    )
    fig.update_layout(
        yaxis_scaleanchor="x",
        yaxis_autorange="reversed",
    )

    # Add Ice and Ore
    for c, arr, name in zip(
        # cyan, yellow, green
        ["#00ffff", "#ffff14", "#15b01a"],
        [state.board.ice.T, state.board.ore.T, state.board.lichen.T],
        ["Ice", "Ore", "Lichen"],
    ):
        cmap = [
            (0.0, c),
            (1.0, c),
        ]
        arr = arr.copy()
        arr = arr.astype(np.float32)
        arr[arr == 0] = np.nan
        fig.add_trace(
            go.Heatmap(
                z=arr,
                colorscale=cmap,
                name=name,
                uid=f"{step}_{name}",
                hoverinfo="skip",
                showscale=False,
            )
        )
    return fig


def _plotly_add_factories(fig, state: GameState):
    step = state.real_env_steps
    team_color = dict(
        player_0="purple",
        player_1="orange",
    )
    # existing_shapes = {s.name: i for i, s in enumerate(fig.layout.shapes)}
    for agent in state.factories:
        if agent not in state.teams:
            continue
        team = state.teams[agent]
        for factory in state.factories[agent].values():
            name = factory.unit_id
            x, y = factory.pos
            # if name not in existing_shapes:
            fig.add_shape(
                type="rect",
                name=name,
                x0=x - 1.5,
                x1=x + 1.5,
                y0=y - 1.5,
                y1=y + 1.5,
                fillcolor=team_color[agent],
            )
            # else:
            #     pass # Factories don't move
            custom_data = [
                [
                    factory.unit_id,
                    f"player_{factory.team_id}",
                    factory.power,
                    factory.cargo.ice,
                    factory.cargo.ore,
                    factory.cargo.water,
                    factory.cargo.metal,
                ]
            ]
            hovertemplate = "<br>".join(
                [
                    "%{customdata[0]} %{customdata[1]}",
                    "Power: %{customdata[2]}",
                    "Ice: %{customdata[3]}",
                    "Ore: %{customdata[4]}",
                    "Water: %{customdata[5]}",
                    "Metal: %{customdata[6]}",
                    "<extra></extra>",
                ]
            )
            unit_num = int(re.search("(\d+)$", factory.unit_id)[0])
            fig.add_trace(
                go.Heatmap(
                    x=[x],
                    y=[y],
                    z=[unit_num],
                    name=name,
                    uid=f"{step}_{name}",
                    customdata=custom_data,
                    hovertemplate=hovertemplate,
                    showscale=False,
                )
            )
    return fig


def _plotly_add_units(fig, state: GameState, add_path=True):
    step = state.real_env_steps
    team_color = dict(
        player_0="purple",
        player_1="orange",
    )
    # existing_shapes = {s.name: i for i, s in enumerate(fig.layout.shapes)}
    for agent, team in state.teams.items():
        for unit in state.units[agent].values():
            name = unit.unit_id
            x, y = unit.pos
            # if name not in existing_shapes:
            fig.add_shape(
                type="rect",
                name=name,
                x0=x - 0.5,
                x1=x + 0.5,
                y0=y - 0.5,
                y1=y + 0.5,
                line=dict(color=team_color[agent]),
                fillcolor="black" if unit.unit_type == UnitType.HEAVY else "white",
            )
            # else:
            #     fig.layout.shapes[existing_shapes[name]].update(
            #         x0=x - 0.5,
            #         x1=x + 0.5,
            #         y0=y - 0.5,
            #         y1=y + 0.5,
            #     )
            custom_data = [
                [
                    unit.unit_id,
                    unit.agent_id,
                    unit.unit_type,
                    unit.pos,
                    unit.power,
                    unit.cargo.ice,
                    unit.cargo.ore,
                    unit.cargo.water,
                    unit.cargo.metal,
                ]
            ]
            hovertemplate = "<br>".join(
                [
                    "%{customdata[0]} %{customdata[1]}",
                    "Type: %{customdata[2]}",
                    "Pos: %{customdata[3]}",
                    "Power: %{customdata[4]}",
                    "Ice: %{customdata[5]}",
                    "Ore: %{customdata[6]}",
                    "Water: %{customdata[7]}",
                    "Metal: %{customdata[8]}",
                    "<extra></extra>",
                ]
            )
            unit_num = int(re.search("(\d+)$", unit.unit_id)[0])
            fig.add_trace(
                go.Heatmap(
                    x=[x],
                    y=[y],
                    z=[unit_num],
                    name=name,
                    uid=f"{step}_{name}",
                    customdata=custom_data,
                    hovertemplate=hovertemplate,
                    showscale=False,
                )
            )
            if add_path:
                path = actions_to_path(unit.pos, unit.action_queue)
                if len(path) > 1:
                    path = path[1:]
                    plotly_plot_path(fig, path, step=step)
    return fig


def plotly_plot_path(fig, path, step=-1):
    fig.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            name="path",
            uid=f"{step}_path",
            mode="markers",
            marker=dict(symbol="circle-open", size=5, color="black"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    return fig


def _slider_initialized(fig):
    if fig.layout.sliders:
        return True
    return False


def initialize_step_fig(env):
    fig = show_env(env, mode="plotly")
    shapes = fig.layout.shapes
    steps = [
        dict(
            method="update",
            label=f"{env.env_steps - 5}",
            args=[
                {"visible": [True] * len(fig.data)},  # Update Datas
                {"shapes": shapes},
            ],  # Update Layout
        )
    ]
    sliders = [dict(active=0, currentvalue={"prefix": "Step: "}, pad={"t": 50}, steps=steps)]
    fig.update_layout(sliders=sliders, height=600)
    return fig


def _add_env_state(fig, env):
    datas_before = len(fig.data)
    fig = show_env(env, fig=fig)
    shapes = fig.layout.shapes
    datas_after = len(fig.data)
    new_datas = datas_after - datas_before

    steps = list(fig.layout.sliders[0].steps)
    for step in steps:
        step["args"][0]["visible"].extend([False] * new_datas)
    steps.append(
        dict(
            method="update",
            label=f"{env.env_steps - 5}",
            args=[
                {"visible": [False] * datas_before + [True] * new_datas},  # Update Datas
                {"shapes": shapes},
            ],  # Update Layout
        )
    )
    fig.layout.sliders[0].update(steps=steps)
    return fig


def add_env_step(fig, env):
    if not _slider_initialized(fig):
        raise ValueError(f"Fig not initialied for steps")

    fig = _add_env_state(fig, env)
    return fig


def mpl_plot_path(ax, path):
    ax.scatter(path[:, 0], path[:, 1], marker="+", s=10, c="black")


def _mpl_add_board(ax, state):
    ax.imshow(state.board.rubble.T, cmap="OrRd")

    # Add Ice and Ore
    for c, arr in zip(
        ["xkcd:cyan", "xkcd:yellow", "xkcd:green"],
        [state.board.ice.T, state.board.ore.T, state.board.lichen.T],
    ):
        cmap = colors.ListedColormap(["black", c])
        bounds = [0, 0.5, 1]
        norm = colors.BoundaryNorm(bounds, cmap.N)
        ax.imshow(
            arr,
            cmap=cmap,
            norm=norm,
            alpha=1.0 * (arr > 0),
        )
    return ax


def _mpl_add_factories(ax, state: GameState):
    team_color = dict(
        player_0="tab:purple",
        player_1="tab:orange",
    )
    for agent in state.factories:
        if agent not in state.teams:
            continue
        team = state.teams[agent]
        for factory in state.factories[agent].values():
            ax.add_patch(
                Rectangle(
                    [p - 1.5 for p in factory.pos],
                    width=3,
                    height=3,
                    color=team_color[agent],
                    fill=False,
                    hatch="xxxx",
                ),
            )
    return ax


def _mpl_add_units(ax, state: GameState):
    team_color = dict(
        player_0="tab:purple",
        player_1="tab:orange",
    )
    for agent in state.units:
        if agent not in state.teams:
            continue
        team = state.teams[agent]
        for unit in state.units[agent].values():
            ax.add_patch(
                Rectangle(
                    [p - 0.5 for p in unit.pos],
                    width=1,
                    height=1,
                    edgecolor=team_color[agent],
                    fill=True,
                    facecolor="black" if unit.unit_type == UnitType.HEAVY else "white",
                    linewidth=2,
                ),
            )
    return ax


def get_subplot_locations(rows, cols, invert=False):
    """Return a single list of the subplot locations in figure with rows/cols of subplots"""
    if invert:
        rows, cols = cols, rows
    return list(product(range(1, rows + 1), range(1, cols + 1)))


def figures_to_subplots(figs, title=None, rows=None, cols=None, shared_data=False, **kwargs):
    """
    Combine multiple plotly figures into a single figure with subplots where the legend and/or colorbar can be shared between them (only if all 2D or all 1D)
    """

    def _figs_2d(figs):
        fig_is_2d = [False] * len(figs)
        for i, fig in enumerate(figs):
            for data in fig.data:
                if isinstance(data, go.Heatmap):
                    fig_is_2d[i] = True
        return fig_is_2d

    def _figs_contain_2d(figs):
        def are_all_true(l):
            return all(l)

        def are_some_true(l):
            return any(l) and not all(l)

        def are_all_false(l):
            return not all(l) and not any(l)

        fig_is_2d = _figs_2d(figs)
        if are_all_true(fig_is_2d):
            return "all"
        elif are_all_false(fig_is_2d):
            return "none"
        else:
            return "some"

    def _figs_all_2d(figs):
        if _figs_contain_2d(figs) == "all":
            return True
        return False

    def _figs_all_1d(figs):
        if _figs_contain_2d(figs) == "none":
            return True
        return False

    def _move_2d_data(
        dest_fig: go.Figure,
        source_figs: List[go.Figure],
        fig_locations: List[tuple],
        match_colorscale: bool,
        specify_rows=None,  # If only moving a subset of figs
        specify_cols=None,  # If only moving a subset of figs
    ):  # , leave_legend_space=False):
        rows = max([l[0] for l in fig_locations]) if specify_rows is None else specify_rows
        cols = max([l[1] for l in fig_locations]) if specify_cols is None else specify_cols
        locations_axis_dict = {loc: i + 1 for i, loc in enumerate(get_subplot_locations(rows, cols))}

        if not match_colorscale:
            if cols == 1:
                xs = [1.00]
            elif cols == 2:
                xs = [0.43, 1.00]
            elif cols == 3:
                xs = [0.245, 0.625, 1.00]
            else:
                raise NotImplementedError
            # if leave_legend_space:
            #     xs = [x*0.8 for x in xs]
            if rows == 1:
                len_ = 1
                ys = [0.5]
            elif rows == 2:
                len_ = 0.4
                ys = [0.81, 0.19]
            elif rows == 3:
                len_ = 0.25
                ys = [0.89, 0.5, 0.11]
            else:
                raise NotImplementedError
            colorbar_locations = {
                (r, c): loc for (r, c), loc in zip(product(range(1, rows + 1), range(1, cols + 1)), product(ys, xs))
            }

        # move data from each figure to subplots (matching colors)
        for fig, (row, col) in zip(source_figs, fig_locations):
            axis_num = locations_axis_dict[(row, col)]
            for j, data in enumerate(fig.data):
                if isinstance(data, go.Heatmap):
                    if match_colorscale:
                        data.coloraxis = "coloraxis"
                    else:
                        data.coloraxis = f"coloraxis{axis_num}"
                dest_fig.add_trace(data, row=row, col=col)
            if not match_colorscale:
                colorbar_location = colorbar_locations[(row, col)]
                dest_fig.update_layout({f"coloraxis{axis_num}": fig.layout.coloraxis})  # Copy across most info
                y, x = colorbar_location
                dest_fig.update_layout(
                    {f"coloraxis{axis_num}_colorbar": dict(x=x, y=y, len=len_)}
                )  # Position the individual colorbar

            dest_fig.update_layout(
                {
                    f"xaxis{axis_num}_title": fig.layout.xaxis.title,
                    f"yaxis{axis_num}_title": fig.layout.yaxis.title,
                }
            )

    def _move_1d_data(
        dest_fig: go.Figure,
        source_figs: List[go.Figure],
        fig_locations: List[tuple],
        match_colors: bool,
        no_legend=False,
        specify_rows=None,
        specify_cols=None,
    ):
        rows = max([l[0] for l in fig_locations]) if specify_rows is None else specify_rows
        cols = max([l[1] for l in fig_locations]) if specify_cols is None else specify_cols
        locations_axis_dict = {loc: i + 1 for i, loc in enumerate(get_subplot_locations(rows, cols))}

        # match the first figures colors if they were specified
        if hasattr(source_figs[0].data[0], "line") and source_figs[0].data[0].line.color and match_colors:
            colors = [d.line.color for d in source_figs[0].data]
        else:
            colors = pc.DEFAULT_PLOTLY_COLORS

        # move data from each figure to subplots (matching colors)
        for fig, (row, col) in zip(source_figs, fig_locations):
            axis_num = locations_axis_dict[(row, col)]
            showlegend = True if axis_num == 1 and not no_legend else False
            for j, data in enumerate(fig.data):
                color = colors[j % len(colors)]  # % to cycle through colors if more data than colors
                if match_colors:
                    data.update(showlegend=showlegend, legendgroup=j, line_color=color)
                dest_fig.add_trace(data, row=row, col=col)
            dest_fig.update_layout(
                {
                    f"xaxis{axis_num}_title": fig.layout.xaxis.title,
                    f"yaxis{axis_num}_title": fig.layout.yaxis.title,
                }
            )

    def _copy_annotations(dest_fig, source_figs):
        """Copy annotations to dest_fig (updating xref and yref if multiple source figs)
        Note: Does NOT copy annotations that use xref/yref = 'paper'
        """
        for i, fig in enumerate(source_figs):
            annotations = fig.layout.annotations
            for annotation in annotations:
                if annotation.xref != "paper" and annotation.yref != "paper":
                    annotation.update(
                        xref=f"x{i + 1}",
                        yref=f"y{i + 1}",
                    )
                    dest_fig.add_annotation(annotation)

    def _copy_shapes(dest_fig, source_figs):
        """Copy shapes to dest_fig (updating xref and yref if multiple source figs)"""
        for i, fig in enumerate(source_figs):
            num_str = f"{i + 1}" if i > 0 else ""  # Plotly names axes 'x', 'x2', 'x3' etc.
            shapes = fig.layout.shapes
            for shape in shapes:
                if shape.xref == "paper" and shape.yref == "paper":
                    shape.update(
                        xref=f"x{num_str} domain",
                        yref=f"y{num_str} domain",
                    )
                else:
                    shape.update(
                        xref=shape.xref.replace("x", f"x{num_str}"),
                        yref=shape.yref.replace("y", f"y{num_str}"),
                    )
                    dest_fig.add_shape(shape)

    # set defaults
    if rows is None and cols is None:
        if len(figs) <= 9:
            cols = int(np.ceil(np.sqrt(len(figs))))
            rows = int(np.ceil(len(figs) / cols))
        else:
            raise NotImplementedError(f"Only implemented up to 3x3")

    if not rows:
        rows = 1 if cols else len(figs)
    if not cols:
        cols = 1

    # get single list of fig row/cols
    fig_locations = get_subplot_locations(rows, cols)
    figs_2d = _figs_2d(figs)

    if _figs_all_2d(figs):
        horizontal_spacing = 0.15 if not shared_data else None
        full_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                fig.layout.title.text if fig.layout.title.text else f"fig {i}" for i, fig in enumerate(figs)
            ],
            horizontal_spacing=horizontal_spacing,
            **kwargs,
        )
        _move_2d_data(
            dest_fig=full_fig,
            source_figs=figs,
            fig_locations=fig_locations,
            match_colorscale=shared_data,
        )
    elif _figs_all_1d(figs):
        full_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                fig.layout.title.text if fig.layout.title.text else f"fig {i}" for i, fig in enumerate(figs)
            ],
            **kwargs,
        )
        _move_1d_data(
            dest_fig=full_fig,
            source_figs=figs,
            fig_locations=fig_locations,
            match_colors=shared_data,
        )
        full_fig.update_layout(
            legend_title=figs[0].layout.legend.title if figs[0].layout.legend.title else "",
        )
    else:  # Some are 2D some are 1D  (Legends are removed, not easy to deal with...)
        horizontal_spacing = 0.15 if not shared_data else None
        full_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                fig.layout.title.text if fig.layout.title.text else f"fig {i}" for i, fig in enumerate(figs)
            ],
            horizontal_spacing=horizontal_spacing,
            **kwargs,
        )
        _move_2d_data(
            dest_fig=full_fig,
            source_figs=[fig for fig, is_2d in zip(figs, figs_2d) if is_2d is True],
            fig_locations=[location for location, is_2d in zip(fig_locations, figs_2d) if is_2d is True],
            match_colorscale=shared_data,
            specify_rows=rows,
            specify_cols=cols,
        )
        _move_1d_data(
            dest_fig=full_fig,
            source_figs=[fig for fig, is_2d in zip(figs, figs_2d) if is_2d is False],
            fig_locations=[location for location, is_2d in zip(fig_locations, figs_2d) if is_2d is False],
            match_colors=shared_data,
            no_legend=True,
            specify_rows=rows,
            specify_cols=cols,
        )

    _copy_annotations(dest_fig=full_fig, source_figs=figs)
    _copy_shapes(dest_fig=full_fig, source_figs=figs)
    full_fig.update_layout(
        title=title,
        height=300 * rows,
    )
    return full_fig

def calc_path_to_available_nearest_pos(
        pathfinder: Pather,
        costmap: np.ndarray,
        from_pos: Tuple[int, int],
        target_array: np.ndarray,
        near_pos: Tuple[int, int],
        max_attempts = 20,
        margin=2,
) -> np.ndarray:
    """Calculate path from pos to the nearest non-zero of array that will be unoccupied on arrival (closest to near_pos)
    Args:
        pathfinder: Usually agents pathfinding instance
        costmap: np.ndarray
        from_pos: Position to path from
        target_array: non-zero array where aiming for
        near_pos: Pos to aim nearest
        margin: How far outside the bounding box of pos/factory to search for best path (higher => slower)
    """
    array = target_array.copy()
    attempts = 0
    # Try path to the nearest factory tile that will be unoccupied at arrival
    while attempts < max_attempts:
        attempts += 1
        coord = nearest_non_zero(array, near_pos)
        # If already there
        if tuple(coord) == tuple(from_pos):
            return np.array([from_pos])  # Path is list of positions
        if coord is not None:
            path = pathfinder.fast_path(
                from_pos,
                coord,
                costmap=costmap,
                margin=margin,

            )
            if len(path) > 0:
                return path
            array[coord[0], coord[1]] = 0
    return np.array([])

def calc_path_to_factory(
    pathfinder: Pather,
    costmap: np.ndarray,
    pos: Tuple[int, int],
    factory_loc: np.ndarray,
    margin=2,
) -> np.ndarray:
    """Calculate path from pos to the nearest point on factory that will be unoccupied on arrival
    Args:
        pathfinder: Usually agents pathfinding instance
        costmap: np.ndarray
        pos: Position to path from
        factory_loc: array with ones where factory is
        rubble: False = ignore rubble costs, True = beginning of turn rubble costs, Array = use array rubble costs
        margin: How far outside the bounding box of pos/factory to search for best path (higher => slower)
    """
    factory_loc = factory_loc.copy()
    original_nearest_location = nearest_non_zero(factory_loc, pos)

    path = calc_path_to_available_nearest_pos(pathfinder, costmap, pos, factory_loc, pos, max_attempts=9, margin=margin)
    #
    # attempts = 0
    # # Try path to the nearest factory tile that will be unoccupied at arrival
    # while attempts < 9:
    #     attempts += 1
    #     nearest_factory = nearest_non_zero(factory_loc, pos)
    #     if tuple(nearest_factory) == tuple(pos):
    #         return np.array([pos])  # Path is list of positions
    #     if nearest_factory is not None:
    #         path = pathfinder.fast_path(
    #             pos,
    #             nearest_factory,
    #             costmap=costmap,
    #             margin=margin,
    #         )
    #         if len(path) > 0:
    #             return path
    #         factory_loc[nearest_factory[0], nearest_factory[1]] = 0
    # Path to the nearest tile anyway
    if len(path) == 0:
    # else:
        logger.warning(
            f"No path to any factory tile without collisions from {pos}  (best factory loc would be {original_nearest_location}), returning path without considering collisions"
        )
        path = pathfinder.fast_path(
            pos,
            original_nearest_location,
            costmap=pathfinder.base_costmap,
            margin=margin,
        )
    return path


def set_middle_of_factory_loc_zero(factory_loc: np.ndarray):
    """Returns an array where the usual factory_loc that contains a 3x3 block of ones is replaced with a zero in the middle"""
    arr = factory_loc.copy()
    # Find the indices of non-zero elements
    indices = np.argwhere(arr == 1)

    # Calculate the center of the 3x3 block
    center = indices.mean(axis=0).astype(int)

    # Set the middle element of the 3x3 block to zero
    arr[tuple(center)] = 0
    return arr


def path_to_factory_edge_nearest_pos(
    pathfinder: Pather,
    factory_loc: np.ndarray,
    pos: Tuple[int, int],
    pos_to_be_near: Tuple[int, int],
    costmap: np.ndarray,
    margin=1,
) -> Union[np.ndarray, None]:
    """Path to the best location on the factory edge without collision
    Args:
        factory_loc: array with 1s where factory is
        pos: target position to try get nearest edge to
    """
    if np.sum(factory_loc) == 9:
        factory_loc = set_middle_of_factory_loc_zero(factory_loc)
    assert np.sum(factory_loc) == 8
    if pos_to_be_near is None:
        raise ValueError(f"Need to provide pos_to_be_near")

    # if np.all(pos == (22, 32)):
    #     show_map_array(costmap).show()
    #     show_map_array(factory_loc).show()

    factory_loc = factory_loc.copy()

    attempts = 0
    # Try path to the factory tile nearest target
    while attempts < 8:
        attempts += 1
        nearest_factory = nearest_non_zero(factory_loc, pos_to_be_near)
        if nearest_factory is None:
            break
        elif tuple(nearest_factory) == tuple(pos):
            return np.array([pos])  # Already there
        else:
            path = pathfinder.fast_path(
                pos,
                nearest_factory,
                costmap=costmap,
                margin=margin,
            )
            if len(path) > 0:
                return path
        factory_loc[nearest_factory[0], nearest_factory[1]] = 0

    logger.warning(f"No path to edge of factory from {pos} aiming to be near {pos_to_be_near} found without collisions")
    return np.array([])
    # if max_delay_by_move_center > 0:
    #     logger.info(f'Adding delay to finding path to edge of factory')
    #     new_collision_params = CollisionParams(
    #         look_ahead_turns=collision_params.look_ahead_turns,
    #         ignore_ids=collision_params.ignore_ids,
    #         friendly_light=collision_params.friendly_light,
    #         friendly_heavy=collision_params.friendly_heavy,
    #         enemy_light=collision_params.enemy_light,
    #         enemy_heavy=collision_params.enemy_heavy,
    #         starting_step=collision_params.starting_step + 1,
    #     )
    #     next_path = path_to_factory_edge_nearest_pos(
    #         pathfinder,
    #         factory_loc,
    #         new_collision_params,
    #         pos,
    #         pos_to_be_near,
    #         rubble,
    #         margin,
    #         max_delay_by_move_center=max_delay_by_move_center - 1,
    #     )
    #     return np.concatenate((np.array([pos]), next_path))
    # else:
    #     logger.warning(f'No path to edge of factory found without collisions')
    #     return None


def power_cost_of_path(path: PATH_TYPE, rubble: np.ndarray, unit_type) -> int:
    """Cost to move along path including rubble
    Note: Assumes first path is current location (i.e. not part of cost)
    """
    assert unit_type in ["LIGHT", "HEAVY"]
    unit_cfg = LIGHT_UNIT.unit_cfg if unit_type == "LIGHT" else HEAVY_UNIT.unit_cfg
    path = np.array(path)
    if len(path) <= 1:
        return 0
    elif path.ndim != 2:
        raise ValueError(f"Path should be (N,2) in shape, got {path.shape}")

    cost = 0
    for pos in path[1:]:
        rubble_at_pos = rubble[pos[0], pos[1]]
        cost += math.floor(unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_pos)
    return cost


@dataclass
class ValidActionsMoving:
    was_valid: bool
    valid_actions: List[np.ndarray]
    invalid_steps: List[int]
    invalid_reasons: List[str]


def calculate_valid_move_actions(
    start_pos: POS_TYPE,
    action_queue: List[np.ndarray],
    valid_move_map: np.ndarray,
    max_len: int = 20,
    ignore_repeat=False,
) -> ValidActionsMoving:
    """
    Will the actions create a valid path (i.e. in bounds and not through any obstructions)
    Replaces invalid moves with move.CENTER

    Note: Actions are returned without repeats

    Args:
        start_pos: Starting position of action_queue
        action_queue: Actions to check (should include non-move actions, they are counted as center moves)
        valid_move_map: The costmap to check for move validity
        max_len: Checks path up to max_len
        ignore_repeat: Whether to only run to end of actions and not repeat them

    Note: Must check all actions in one go to be able to account for repeat actions (i.e. if short list of actions that loop)

    Note: Can use this to check for collisions if setting <= 0 in costmap for collisions, but mostly intended for
    actual not possible moves (i.e. enemy factory or off edge of map)
    """

    # NECESSARY TO AVOID ENV CRASH BECAUSE OF CHANGING POS
    pos = tuple(start_pos)
    action_queue = copy.copy(action_queue)
    action_queue = list(action_queue)

    max_x, max_y = valid_move_map.shape

    valid_actions = []
    invalid_steps = []
    invalid_reasons = []
    step = 0
    for i, action in enumerate(action_queue):
        # Not easy to build valid including repeat, so just force no repeat for now
        action = action.copy()
        repeat = action[ACT_REPEAT]
        action[ACT_REPEAT] = 0

        # Not moving, must be valid in terms of move
        next_n = action[ACT_N]
        if action[ACT_TYPE] != MOVE or (action[ACT_TYPE] == MOVE and action[ACT_DIRECTION] == CENTER):
            # If won't exceed max len
            if step + next_n < max_len:
                valid_actions.append(action)
                step += next_n
            # Only up to max len then break
            else:
                action[ACT_N] = max_len - step
                valid_actions.append(action)
                break

        # Moving, more complicated (might have to change part of move to move center)
        else:
            direction = action[ACT_DIRECTION]
            valid_n = 0
            invalid_reason = None
            # Check where in move becomes invalid if it does
            for j in range(next_n):
                new_x, new_y = pos + MOVE_DELTAS[direction]

                # Not valid if out of index
                if not 0 <= new_x < max_x or not 0 <= new_y < max_y:
                    invalid_reason = "index_error"
                    break
                # Not valid if blocked
                elif valid_move_map[new_x, new_y] <= 0:
                    invalid_reason = "blocked"
                    break
                else:
                    valid_n += 1
                    pos += MOVE_DELTAS[direction]

                step += 1
                # If at max len then break
                if step == max_len:
                    break

            # Fully valid
            if valid_n == next_n:
                valid_actions.append(action)
            # Max len reached but valid
            elif valid_n < next_n and step == max_len:
                action[ACT_N] = valid_n
                valid_actions.append(action)
            # Something invalid
            else:
                if valid_n > 0:
                    # Append the valid part
                    action[ACT_N] = valid_n
                    valid_actions.append(action)
                num_invalid = min(next_n - valid_n, max_len - step)
                invalid_steps.extend([step + i for i in range(0, num_invalid)])
                invalid_reasons.extend([invalid_reason] * num_invalid)
                # Append move CENTER where invalid
                valid_actions.append(np.array([0, 0, 0, 0, 0, num_invalid]))
                step += num_invalid

        # Break if max_len reached
        if step >= max_len:
            break

        # Add action to end of list with 1 less repeat if necessary
        if not ignore_repeat and repeat > 0:
            action[ACT_N] = repeat
            action[ACT_REPEAT] = repeat
            action_queue.append(action)

    was_valid = True
    if len(invalid_steps) > 0:
        was_valid = False

    return ValidActionsMoving(
        was_valid=was_valid,
        valid_actions=valid_actions,
        invalid_steps=invalid_steps,
        invalid_reasons=invalid_reasons,
    )
