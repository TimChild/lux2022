import sys

sys.path.append(
    '../lux_kit'
)  #  lux_kit is a copy of https://github.com/Lux-AI-Challenge/Lux-Design-2022/tree/main/kits/python
from typing import Tuple, List, Union, Optional
from lux.kit import obs_to_game_state, GameState, EnvConfig, to_json, from_json
from lux.config import UnitConfig, EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
from lux.unit import Unit, move_deltas, UnitCargo
import dataclasses
from luxai2022.state import state
from luxai2022 import LuxAI2022
from luxai2022.unit import UnitType
from luxai2022.team import Team
import numpy as np
import sys
import json
import pickle
import math
import re
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

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
# (type, direction, resource, amount, repeat)
ACT_TYPE = 0
ACT_DIRECTION = 1
ACT_RESOURCE = 2
ACT_AMOUNT = 3
ACT_REPEAT = 4

# Types:
MOVE = 0
TRANSFER = 1
PICKUP = 2
DIG = 3
DESTRUCT = 4
RECHARGE = 5


################# General #################


def get_test_env(path=None):
    if path is None:
        path = "test_state.pkl"
    with open(path, "rb") as f:
        state = pickle.load(f)
    env = LuxAI2022()
    _ = env.reset(state.seed)
    env.set_state(state)
    return env


def game_state_from_env(env):
    return obs_to_game_state(env.state.env_steps, env.env_cfg, env.state.get_obs())


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

    env = LuxAI2022()
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

    Note: Excluding weather effects
    Note: Does not check for invalid moves or actions
    """
    # TODO: Include weather??
    # step = state.real_env_steps

    unit_cfg = unit.unit_cfg

    pos = unit.pos

    cost = 0
    for action in actions:
        act_type = action[ACT_TYPE]
        if act_type == MOVE:
            pos = pos + move_deltas[ACT_DIRECTION]
            rubble_at_target = state.board.rubble[pos[0], pos[1]]
            cost += math.ceil(
                unit_cfg.MOVE_COST + unit_cfg.RUBBLE_MOVEMENT_COST * rubble_at_target
            )
        elif act_type in [TRANSFER, PICKUP]:
            pass  # No cost
        elif act_type == DESTRUCT:
            cost += unit_cfg.SELF_DESTRUCT_COST
        elif act_type == RECHARGE:
            cost -= unit_cfg.CHARGE
    return cost


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
