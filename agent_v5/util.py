import sys

from typing import Tuple, List, Union, Optional, TYPE_CHECKING
import logging
from scipy import ndimage
from scipy.signal import convolve2d
import dataclasses
import numpy as np
import pickle
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
from lux.kit import obs_to_game_state, GameState, to_json, from_json
from lux.config import UnitConfig, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from lux.unit import Unit, move_deltas
from lux.cargo import UnitCargo

from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

if TYPE_CHECKING:
    from unit_manager import FriendlyUnitManger

UTIL_VERSION = 'AGENT_V2'

ENV_CFG = EnvConfig()
LIGHT_UNIT = Unit(
    team_id=-1,
    unit_id='none',
    unit_type='LIGHT',
    pos=np.array((0, 0)),
    power=1000,
    cargo=UnitCargo(),
    env_cfg=ENV_CFG,
    unit_cfg=ENV_CFG.ROBOTS['LIGHT'],
    action_queue=[],
)
HEAVY_UNIT = Unit(
    team_id=-1,
    unit_id='none',
    unit_type='HEAVY',
    pos=np.array((0, 0)),
    power=1000,
    cargo=UnitCargo(),
    env_cfg=ENV_CFG,
    unit_cfg=ENV_CFG.ROBOTS['HEAVY'],
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
ACT_START_N = 5

# Types:
MOVE = 0
TRANSFER = 1
PICKUP = 2
DIG = 3
DESTRUCT = 4
RECHARGE = 5


################# General #################


def nearest_non_zero(
    array: np.ndarray, pos: Union[np.ndarray, Tuple[int, int]]
) -> Optional[Tuple[int, int]]:
    """Nearest location from pos in array where value is positive"""

    locations = np.argwhere(array > 0)
    if len(locations) == 0:
        return None
    distances = np.array([manhattan(loc, pos) for loc in locations])
    closest = locations[np.argmin(distances)]
    return tuple(closest)


def power_cost_of_actions(state: GameState, unit: Unit, actions: List[np.ndarray]):
    """Power requirements of a list of actions

    Note: Does not check for invalid moves or actions
    """
    # step = state.real_env_steps

    unit_cfg = unit.unit_cfg

    pos = unit.pos

    cost = 0
    for action in actions:
        print(action)
        for _ in range(action[ACT_START_N]):
            act_type = action[ACT_TYPE]
            if act_type == MOVE:
                pos = pos + MOVE_DELTAS[ACT_DIRECTION]
                rubble_at_target = state.board.rubble[pos[0], pos[1]]
                cost += math.ceil(
                    unit_cfg.MOVE_COST
                    + unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
                )
            elif act_type == TRANSFER or (
                act_type == PICKUP and action[ACT_RESOURCE] != POWER
            ):
                pass  # No cost
            elif act_type == PICKUP and action[ACT_RESOURCE] == POWER:
                cost += action[ACT_AMOUNT]  # pickup has negative value
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


def pad_and_crop(small_arr, large_arr, x1, y1, fill_value=0):
    """
    Pads the edges of small_arr with zeros so that the middle of the small_arr ends up at
    the coordinate (x1, y1) of large_arr. The small_arr will be cropped to fit inside the
    large_arr. Returns the padded and cropped version of small_arr.

    Args:
        small_arr: a 2D numpy array
        large_arr: a 2D numpy array
        x1: the x-coordinate of the center of small_arr in large_arr
        y1: the y-coordinate of the center of small_arr in large_arr
        fill_value: Value to fill the padded areas with

    Returns:
        A padded and cropped version of small_arr
    """
    x_size, y_size = small_arr.shape

    x_start = max(0, x1 - x_size // 2)
    y_start = max(0, y1 - y_size // 2)
    x_end = min(large_arr.shape[0], x1 + x_size // 2 + 1 - (x_size % 2 == 0))
    y_end = min(large_arr.shape[1], y1 + y_size // 2 + 1 - (y_size % 2 == 0))

    small_arr_start_x = x_start - x1 + x_size // 2
    small_arr_start_y = y_start - y1 + y_size // 2
    padded_arr = np.full(large_arr.shape, fill_value=fill_value, dtype=small_arr.dtype)
    padded_arr[x_start:x_end, y_start:y_end] = small_arr[
        small_arr_start_x : small_arr_start_x + (x_end - x_start),
        small_arr_start_y : small_arr_start_y + (y_end - y_start),
    ]

    return padded_arr


def connected_factory_zeros(rubble, factory_pos) -> np.ndarray:
    """Figure out what area of zeros is connected to factory"""
    struct = ndimage.generate_binary_structure(rank=2, connectivity=1)
    # Note:
    # rank 2 = image dimensions
    # connectivity 1 = adjacent only (no diagonals)

    labelled_arr, num_labels = ndimage.label(rubble == 0, structure=struct)
    # Note:
    # labelled_arr = unique values for each connected area of 0s

    factory_zeros_id = labelled_arr[factory_pos[0], factory_pos[1]]
    factory_zeros = np.zeros(rubble.shape)
    factory_zeros[labelled_arr == factory_zeros_id] = 1
    return factory_zeros


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
        raise ValueError(
            "Invalid side specified. Must be 'top', 'bottom', 'left', or 'right'."
        )


def create_boundary_array(arr):
    """Creates array of boundaries around zero areas where the boundary value is equal to the size of the zero area"""
    arr = np.array(arr)  # Convert input to numpy array if not already
    zero_areas = arr == 0  # Create a boolean array where True represents zeros

    # Create a binary structure to label connected zero areas not including diagonal connections
    s = ndimage.generate_binary_structure(2, 1)
    labeled_array, num_features = ndimage.label(
        zero_areas, structure=s
    )  # Label connected zero areas

    result = np.zeros_like(
        arr
    )  # Create a new array with the same shape as the input array, filled with zeros

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
    def __init__(
        self, array: np.ndarray, pos: Tuple[int, int], radius: int, fill_value=0
    ):
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

        Args:
            pos: position of center of the subset (x, y) using lux ordering arr[x, y]
            radius: distance from the center of the subset to its edge.

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
            pos: position in the original array (x, y) using lux ordering arr[x, y]

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

    # Get the indices of the True values in the mask
    mask_indices = np.array(np.where(mask)).T

    # Create a meshgrid for the input array
    x_grid, y_grid = np.meshgrid(
        np.arange(input_array.shape[0]), np.arange(input_array.shape[1]), indexing="ij"
    )

    # Calculate the Manhattan distance between all points in the grid and the mask_indices
    distances = np.abs(x_grid[..., np.newaxis] - mask_indices[:, 0]) + np.abs(
        y_grid[..., np.newaxis] - mask_indices[:, 1]
    )

    # Get the minimum distance for each point in the grid
    output_array = np.min(distances, axis=-1)

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


def add_direction_to_pos(pos: Tuple[int, int], direction: int) -> Tuple[int, int]:
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


class PathFinder:
    def __init__(self, cost_map):
        self.cost_map = cost_map
        self.grid = Grid(matrix=cost_map)

    def path(self, start, end):
        finder = AStarFinder()
        start = self.grid.node(*start)
        end = self.grid.node(*end)
        path, runs = finder.find_path(start, end, self.grid)
        path = np.array(path)
        return path


def path_to_actions(path):
    pos = path[0]
    actions = []
    for p in path[1:]:
        actions.append(
            LIGHT_UNIT.move(direction_to(pos, p))
        )  # Note: UnitType doesn't matter
        pos = p
    return actions


def actions_to_path(unit, actions):
    deltas = {
        0: (0, 0),
        1: (0, -1),
        2: (1, 0),
        3: (0, 1),
        4: (-1, 0),
    }
    pos = np.array(unit.pos)
    path = [pos]
    for action in actions:
        if action[0] == MOVE:
            direction = action[1]
        else:
            direction = CENTER
        pos = pos + deltas[direction]
        path.append(pos)
    return np.array(path)


################### Plotting #################
def get_test_env(path=None):
    if path is None:
        path = "test_state.pkl"
    with open(path, "rb") as f:
        state = pickle.load(f)
    env = LuxAI_S2()
    _ = env.reset(state.seed)
    env.set_state(state)
    return env


def game_state_from_env(env: LuxAI_S2):
    return obs_to_game_state(
        env.state.env_steps, env.state.env_cfg, env.state.get_obs()
    )


def run(
    agent1,
    agent2,
    map_seed=None,
    save_state_at=None,
    max_steps=1000,
    return_type='replay',
):
    make_fig = False
    if return_type == 'figure':
        make_fig = True
    elif return_type == 'replay':
        pass
    else:
        raise ValueError(f'return_type={return_type} not valid')

    env = LuxAI_S2()
    # This code is partially based on the luxai2022 CLI:
    # https://github.com/Lux-AI-Challenge/Lux-Design-2022/blob/main/luxai_runner/episode.py

    obs = env.reset(seed=map_seed)
    state_obs = env.state.get_compressed_obs()

    agents = {
        "player_0": agent1("player_0", env.state.env_cfg),
        "player_1": agent2("player_1", env.state.env_cfg),
    }

    game_done = False
    rewards, dones, infos = {}, {}, {}

    for agent_id in agents:
        rewards[agent_id] = 0
        dones[agent_id] = 0
        infos[agent_id] = {"env_cfg": dataclasses.asdict(env.state.env_cfg)}

    replay = {"observations": [state_obs], "actions": [{}]}

    if make_fig:
        fig = initialize_step_fig(env)
    i = 0
    while not game_done and i < max_steps:
        i += 1
        if save_state_at and env.state.real_env_steps == save_state_at:
            print(f'Saving State at Real Env Step :{env.state.real_env_steps}')
            import pickle

            with open('test_state.pkl', 'wb') as f:
                pickle.dump(env.get_state(), f)

        actions = {}
        for agent_id, agent in agents.items():
            agent_obs = obs[agent_id]

            if env.state.real_env_steps < 0:
                agent_actions = agent.early_setup(env.env_steps, agent_obs)
            else:
                agent_actions = agent.act(env.env_steps, agent_obs)

            for key, value in agent_actions.items():
                if isinstance(value, list):
                    agent_actions[key] = np.array(value, dtype=int)

            actions[agent_id] = agent_actions

        new_state_obs, rewards, dones, infos = env.step(actions)

        change_obs = env.state.get_change_obs(state_obs)
        state_obs = new_state_obs["player_0"]
        obs = new_state_obs

        replay["observations"].append(change_obs)
        replay["actions"].append(actions)

        players_left = len(dones)
        for key in dones:
            if dones[key]:
                players_left -= 1

        if players_left < 2:
            game_done = True

    if make_fig:
        add_env_step(fig, env)
    if make_fig:
        return fig
    else:
        replay = to_json(replay)
    return replay


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
            uid=f'{step}_Rubble',
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
        arr = arr.astype(np.float32)
        arr[arr == 0] = np.nan
        fig.add_trace(
            go.Heatmap(
                z=arr,
                colorscale=cmap,
                name=name,
                uid=f'{step}_{name}',
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
                    f'player_{factory.team_id}',
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
                    uid=f'{step}_{name}',
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
                    uid=f'{step}_{name}',
                    customdata=custom_data,
                    hovertemplate=hovertemplate,
                    showscale=False,
                )
            )
            if add_path:
                path = actions_to_path(unit, unit.action_queue)
                if len(path) > 1:
                    path = path[1:]
                    plotly_plot_path(fig, path, step=step)
    return fig


def plotly_plot_path(fig, path, step=-1):
    fig.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            name='path',
            uid=f'{step}_path',
            mode="markers",
            marker=dict(symbol="circle-open", size=5, color="black"),
            hoverinfo='skip',
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
            label=f"{env.env_steps-5}",
            args=[
                {"visible": [True] * len(fig.data)},  # Update Datas
                {"shapes": shapes},
            ],  # Update Layout
        )
    ]
    sliders = [
        dict(active=0, currentvalue={"prefix": "Step: "}, pad={"t": 50}, steps=steps)
    ]
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
            label=f"{env.env_steps-5}",
            args=[
                {
                    "visible": [False] * datas_before + [True] * new_datas
                },  # Update Datas
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


def figures_to_subplots(
    figs, title=None, rows=None, cols=None, shared_data=False, **kwargs
):
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
        rows = (
            max([l[0] for l in fig_locations]) if specify_rows is None else specify_rows
        )
        cols = (
            max([l[1] for l in fig_locations]) if specify_cols is None else specify_cols
        )
        locations_axis_dict = {
            loc: i + 1 for i, loc in enumerate(get_subplot_locations(rows, cols))
        }

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
                (r, c): loc
                for (r, c), loc in zip(
                    product(range(1, rows + 1), range(1, cols + 1)), product(ys, xs)
                )
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
                dest_fig.update_layout(
                    {f"coloraxis{axis_num}": fig.layout.coloraxis}
                )  # Copy across most info
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
        rows = (
            max([l[0] for l in fig_locations]) if specify_rows is None else specify_rows
        )
        cols = (
            max([l[1] for l in fig_locations]) if specify_cols is None else specify_cols
        )
        locations_axis_dict = {
            loc: i + 1 for i, loc in enumerate(get_subplot_locations(rows, cols))
        }

        # match the first figures colors if they were specified
        if (
            hasattr(source_figs[0].data[0], "line")
            and source_figs[0].data[0].line.color
            and match_colors
        ):
            colors = [d.line.color for d in source_figs[0].data]
        else:
            colors = pc.DEFAULT_PLOTLY_COLORS

        # move data from each figure to subplots (matching colors)
        for fig, (row, col) in zip(source_figs, fig_locations):
            axis_num = locations_axis_dict[(row, col)]
            showlegend = True if axis_num == 1 and not no_legend else False
            for j, data in enumerate(fig.data):
                color = colors[
                    j % len(colors)
                ]  # % to cycle through colors if more data than colors
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
            num_str = (
                f"{i + 1}" if i > 0 else ""
            )  # Plotly names axes 'x', 'x2', 'x3' etc.
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
                fig.layout.title.text if fig.layout.title.text else f"fig {i}"
                for i, fig in enumerate(figs)
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
                fig.layout.title.text if fig.layout.title.text else f"fig {i}"
                for i, fig in enumerate(figs)
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
            legend_title=figs[0].layout.legend.title
            if figs[0].layout.legend.title
            else "",
        )
    else:  # Some are 2D some are 1D  (Legends are removed, not easy to deal with...)
        horizontal_spacing = 0.15 if not shared_data else None
        full_fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[
                fig.layout.title.text if fig.layout.title.text else f"fig {i}"
                for i, fig in enumerate(figs)
            ],
            horizontal_spacing=horizontal_spacing,
            **kwargs,
        )
        _move_2d_data(
            dest_fig=full_fig,
            source_figs=[fig for fig, is_2d in zip(figs, figs_2d) if is_2d is True],
            fig_locations=[
                location
                for location, is_2d in zip(fig_locations, figs_2d)
                if is_2d is True
            ],
            match_colorscale=shared_data,
            specify_rows=rows,
            specify_cols=cols,
        )
        _move_1d_data(
            dest_fig=full_fig,
            source_figs=[fig for fig, is_2d in zip(figs, figs_2d) if is_2d is False],
            fig_locations=[
                location
                for location, is_2d in zip(fig_locations, figs_2d)
                if is_2d is False
            ],
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
