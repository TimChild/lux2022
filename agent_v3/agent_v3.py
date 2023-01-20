from __future__ import annotations

import abc

with open('agent_v3_log.log', 'w') as f:
    pass
import logging

logging.basicConfig(filename='agent_v3_log.log', level=logging.INFO)
logging.info('Starting Log')

from dataclasses import dataclass, asdict
from unit_manager import UnitManager
from master_plan import MasterState, Planners
from factory_manager import FactoryManager
from path_finder import PathFinder, CollisionParams

from typing import Dict, TYPE_CHECKING, List, Any
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import my_turn_to_place_factory

from util import ACT_TYPE, MOVE
import numpy as np
import logging

if TYPE_CHECKING:
    from master_plan import Recommendation


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig):
        logging.info(f'initializing agent for player {player}')
        self.player = player
        self.env_cfg: EnvConfig = env_cfg
        np.random.seed(0)

        # Additional initialization
        self.last_obs = None
        self.state: MasterState = MasterState(
            player=self.player,
            opp_player="player_1" if self.player == "player_0" else "player_0",
            unit_managers={},
            enemy_unit_managers={},
            factory_managers={},
            pathfinder=PathFinder(),
        )

    def log(self, message, level=logging.INFO, **kwargs):
        logging.log(level, f'{self.player} {message}', **kwargs)

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        """Required API for Agent. This is called until all factories are placed"""
        self._beginning_of_step_update(step, obs, remainingOverageTime)

        game_state = self.state.game_state
        action = dict()
        if step == 0:
            action = self.bid(obs)
        else:
            # factory placement period
            my_turn_to_place = my_turn_to_place_factory(
                game_state.teams[self.player].place_first, step
            )
            factories_to_place = game_state.teams[self.player].factories_to_place
            if factories_to_place > 0 and my_turn_to_place:
                # TODO: Improve factory placement
                action = FactoryManager.place_factory(game_state, self.player)
        self.log(f'Early setup action {action}')
        return action

    def bid(self, obs):
        """Bid for starting factory (default to 0)"""
        return dict(faction="TheBuilders", bid=0)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """Required API for Agent. This is called every turn after early_setup is complete"""
        self._beginning_of_step_update(step, obs, remainingOverageTime)

        # Get processed observations (i.e. the obs that I will use to train a PPO agent)
        # First - general observations of the full game state (i.e. not factory/unit specific)
        general_processed_obs = self._get_general_processed_obs()

        # Then observations for units/factories that might want to act
        unit_obs = {}
        for unit_id, unit in self.state.unit_managers.items():
            if unit_should_consider_acting(unit, self.state):
                unit_obs[unit_id] = calculate_unit_obs(unit, self.state)

        factory_obs = {}
        for factory_id, factory in self.state.factory_managers.items():
            if factory_should_consider_acting(factory, self.state):
                factory_obs[factory_id] = calculate_factory_obs(
                    factory, self.state
                )

        # ML agent can use obs to generate action here (note: not the basic actions of the game)
        # For now, I'll just make some algorithms to return the high level actions
        unit_high_level_actions = {}
        for unit_id, obs in unit_obs.items():
            full_obs = combine_obs(general_processed_obs, unit_obs)
            unit_high_level_actions[unit_id] = calculate_high_level_unit_actions(
                full_obs
            )

        factory_high_level_actions = {}
        for factory_id, obs in factory_obs.items():
            full_obs = combine_obs(general_processed_obs, factory_obs)
            factory_high_level_actions[
                factory_id
            ] = calculate_high_level_factory_actions(full_obs)

        # Convert back to actions that the game supports
        unit_actions = generate_unit_low_level_actions(unit_high_level_actions)
        factory_actions = generate_factory_low_level_actions(factory_high_level_actions)
        return dict(**unit_actions, **factory_actions)

    def _beginning_of_step_update(
        self, step: int, obs: dict, remainingOverageTime: int
    ):
        """Use the step and obs to update any turn based info (e.g. map changes)"""
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        # TODO: Use last obs to see what has changed to optimize update?
        self.state.update(game_state)
        self.last_obs = obs

    def _get_general_processed_obs(self) -> GeneralObs:
        """Get a fixed length DF/array of obs that can be passed into an ML agent

        Thoughts:
            - Mostly generated from MasterPlan
            - Some additional info added based on metadata?
        """
        pass


class MyObs(abc.ABC):
    """General form of my observations s.t."""

    @abc.abstractmethod
    def to_array(self):
        """Return observations as a numpy array"""
        pass


@dataclass
class GeneralObs(MyObs):

    def to_array(self):
        return np.array([])


def combine_obs(general_obs, unit_obs):
    """Combine the general and unit obs into a single array that can be passed into ML"""
    pass


def calculate_high_level_unit_actions(processed_obs):
    """Take the processed obs, and return high level actions per unit/factory

    Examples:
        - Mine X Ore for factory X
        - Mine X Ice for factory X
        - Attack area X
        - Defend area X
        - Solar Farm at X
    """
    pass


def calculate_high_level_factory_actions(processed_obs):
    """Take the processed obs, and return high level actions per factory

    Examples:
        - Make Light Unit
        - Make Heavy Unit
    """
    pass


def unit_should_consider_acting(unit: UnitManager, plan: MasterState) -> bool:
    """Whether unit should consider acting this turn
    If no, can save the cost of calculating obs/options for that unit
    """
    pass


def calculate_unit_obs(unit: UnitManager, plan: MasterState) -> UnitObs:
    """Calculate observations that are specific to a particular unit"""
    pass


@dataclass
class UnitObs(MyObs):
    def to_array(self):
        return np.array([])


def factory_should_consider_acting(factory: FactoryManager, plan: MasterState) -> bool:
    """Whether factory should consider acting this turn
    If no, can save the cost of calculating obs/options for that factory
    """
    return True


def calculate_factory_obs(factory: FactoryManager, plan: MasterState) -> FactoryObs:
    """Calculate observations that are specific to a particular factory"""
    pass


@dataclass
class FactoryObs(MyObs):
    def to_array(self):
        return np.array([])


def generate_unit_low_level_actions(
    high_level_actions: dict[str, Any]
) -> dict[str, list[np.ndarray]]:
    """Take the high level action (e.g. Mine X Ice for factory X) and turn it into
    the low level action the game uses (e.g. Move up, dig, etc)"""
    return {}


def generate_factory_low_level_actions(
    high_level_actions: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Take the high level action (e.g. ???) and turn it into
    the low level action the game uses (e.g. ???)

    Not sure what these will look like yet
    """
    return {}


"""
PPO implementation ideas:
Actions:
- Makes decision per unit (units/factories treated equal)
    - Does action space include both unit and factory actions, then mask invalid?
    - Or can I somehow use say 6 outputs and just change the meaning for Factory actions?
        - Probably there is something wrong with this
    - How to make fixed length of actions?
        - Mine Ore, Mine Ice, Attack, Defend
            - But how to specify Defend Factory 1 or Factory 2 etc?
            - Then actions aren't fixed length
- Per Factory?
    - I.e. factory decides if it wants more ice, ore, defence, attack etc.
    - Then units are completely algorithmic
- Other variables
    - There are also many other variables (e.g. thresholds) for when algorithmic things should happen
    - How can these be fine tuned? Maybe separate to the PPO?
        - I.e. train PPO, then tune params, then train PPO, then tune params?


Observations:
- Some sort of gauss peaks for beginning, middle, end game (i.e. some flag that might change strategy for different
parts of game)
- Some calculated states of the game (i.e. total resources, resources per factory, how many light, how many heavy)
- How to give information on what units/factories are doing? 
    - Not fixed length...
    - Could give positions at least as an array of values size of map (but that is a lot!)
"""


# class Agent:
#     def __init__(self, player: str, env_cfg: EnvConfig) -> None:
#         logging.info('initializing agent')
#         self.player = player
#         self.opp_player = "player_1" if self.player == "player_0" else "player_0"
#         np.random.seed(0)
#         self.env_cfg: EnvConfig = env_cfg
#
#         # Things the Agent keeps track of
#         self.pathfinder: PathFinder = PathFinder()
#         self.unit_managers: Dict[str, UnitManager] = {}
#         self.enemy_unit_managers: Dict[str, UnitManager] = {}
#         self.factory_managers: Dict[str, FactoryManager] = {}
#         self.master_plan: MasterPlan = MasterPlan(
#             player=self.player,
#             opp_player=self.opp_player,
#             unit_managers=self.unit_managers,
#             enemy_unit_managers=self.enemy_unit_managers,
#             factory_managers=self.factory_managers,
#             pathfinder=self.pathfinder,
#         )
#         self.planners = Planners(self.master_plan)
#
#     @property
#     def UnitManager(self):
#         return UnitManager
#
#     @property
#     def FactoryManager(self):
#         return FactoryManager
#
#     def log(self, message, level=logging.INFO):
#         step = (
#             self.master_plan.game_state.real_env_steps
#             if self.master_plan and self.master_plan.game_state
#             else 'not initialized'
#         )
#         logging.log(
#             level,
#             f"Player {self.player}, Step: {step} - {message}",
#         )
#
#     def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
#         """Required API for Agent. This is called until all factories are placed"""
#         game_state = obs_to_game_state(step, self.env_cfg, obs)
#         action = dict()
#         if step == 0:
#             action = self.bid(obs)
#         else:
#             # factory placement period
#             my_turn_to_place = my_turn_to_place_factory(
#                 game_state.teams[self.player].place_first, step
#             )
#             factories_to_place = game_state.teams[self.player].factories_to_place
#             if factories_to_place > 0 and my_turn_to_place:
#                 action = self.FactoryManager.place_factory(game_state, self.player)
#         self.log(f'Early setup action {action}')
#         return action
#
#     def bid(self, obs):
#         """Bid for starting factory (default to 0)"""
#         return dict(faction="TheBuilders", bid=0)
#
#     def act(self, step: int, obs, remainingOverageTime: int = 60):
#         """Required API for Agent. This is called every turn after early_setup is complete"""
#         # Set some useful variables for the step that can be used in factory_actions or unit_actions etc
#         self.log('entering act')
#         self.update_step_info(step, obs, remainingOverageTime)
#         factory_actions = self.factory_actions()
#         unit_actions = self.unit_actions()
#         actions = dict(**factory_actions, **unit_actions)
#
#         actions = self._sanitize_actions(actions)
#
#         self.log(f'Returning Actions: {actions}')
#         return actions
#
#     def update_step_info(self, step, obs, remainingOverageTime):
#         """Update for new step in game"""
#         game_state = obs_to_game_state(step, self.env_cfg, obs)
#
#         # Update FactoryManagers
#         new_factories = set(game_state.factories[self.player].keys()) - set(
#             self.factory_managers.keys()
#         )
#         for factory_id in new_factories:
#             self.factory_managers[factory_id] = FactoryManager(
#                 game_state.factories[self.player][factory_id], self.master_plan
#             )
#
#         dead_factories = set(self.factory_managers.keys()) - set(
#             game_state.factories[self.player].keys()
#         )
#         for factory_id in dead_factories:
#             dead_factory = self.factory_managers.pop(factory_id)
#             self.log(f'Factory {dead_factory} died')
#
#         for factory_id, factory in game_state.factories[self.player].items():
#             self.factory_managers[factory_id].update(factory)
#
#         # Update UnitManagers
#         managers = {
#             self.player: self.unit_managers,
#             self.opp_player: self.enemy_unit_managers,
#         }
#         all_dead_units = []
#         for player in game_state.teams:
#             unit_managers = managers[player]
#             new_units = set(game_state.units[player].keys()) - set(unit_managers.keys())
#             for unit_id in new_units:
#                 unit_managers[unit_id] = UnitManager(
#                     game_state.units[player][unit_id],
#                     self.master_plan,
#                 )
#
#             dead_units = set(unit_managers.keys()) - set(
#                 game_state.units[player].keys()
#             )
#             for unit_id in dead_units:
#                 dead_unit = unit_managers.pop(unit_id)
#                 all_dead_units.append(dead_unit)
#                 self.log(f'Unit {dead_unit} died')
#
#             for unit_id, unit in game_state.units[player].items():
#                 unit_managers[unit_id].update(unit)
#
#         # Update MasterPlan
#         self.master_plan.update(game_state=game_state, dead_units=all_dead_units)
#
#         # Update PathFinder
#         self.pathfinder.update(
#             game_state.board.rubble,
#             self.unit_managers.values(),
#             self.enemy_unit_managers.values(),
#             enemy_factories=game_state.factories[self.opp_player],
#         )
#
#         # Update Planners
#         self.planners.mining_planner.update()
#
#     def factory_actions(self):
#         actions = dict()
#         game_state = self.master_plan.game_state
#         factories = game_state.factories[self.player]
#         for unit_id, factory in factories.items():
#             if self.env_cfg.max_episode_length - game_state.real_env_steps < 100:
#                 if factory.water_cost(game_state) <= factory.cargo.water:
#                     actions[unit_id] = factory.water()
#                     continue
#             pos = factory.pos
#             collision = self.pathfinder.check_collisions(
#                 np.array([pos], dtype=int), collision_params=CollisionParams(turns=1)
#             )
#             if collision is None:
#                 # if factory.power >= factory.build_light_power_cost(
#                 #     self.master_plan.game_state
#                 # ) and factory.cargo.metal >= factory.build_light_metal_cost(
#                 #     self.master_plan.game_state
#                 # ):
#                 #     actions[unit_id] = factory.build_light()
#                 #     continue
#                 if factory.power >= factory.build_heavy_power_cost(
#                     self.master_plan.game_state
#                 ) and factory.cargo.metal >= factory.build_heavy_metal_cost(
#                     self.master_plan.game_state
#                 ):
#                     actions[unit_id] = factory.build_heavy()
#                     continue
#         return actions
#
#     def unit_actions(self):
#         actions = {}
#
#         unit_action_recommendations = (
#             self._get_unit_role_recommendations()
#         )  # with unit_id, role, value, actions
#
#         for unit_id, recommendations in unit_action_recommendations.items():
#             manager = self.unit_managers[unit_id]
#             best = sorted(recommendations, key=lambda x: x.value)[-1]
#             if best.role != self.unit_managers[unit_id].status.role:
#                 manager.log(f'Switching role from {manager.status.role} to {best.role}')
#             if best.value > 0:
#                 actions[manager.unit_id] = self.planners.mining_planner.carry_out(
#                     unit_manager=self.unit_managers[unit_id], recommendation=best
#                 )
#
#         actions = self._sanitize_unit_actions(actions)
#         return actions
#
#     def _get_unit_role_recommendations(self) -> Dict[str, List[Recommendation]]:
#         recommendations = {}
#
#         for unit_manager in self.unit_managers.values():
#             mining_recommendation = self.planners.mining_planner.recommend(unit_manager)
#
#             recommendations[unit_manager.unit_id] = [mining_recommendation]
#         return recommendations
#
#     def _sanitize_unit_actions(self, actions):
#         clean_actions = actions.copy()
#         for k, v in actions.items():
#             # if v[ACT_TYPE] == MOVE:
#             #     pass
#             if len(v) > 20:
#                 self.log(f'Actions for {k} longer than 20', level=logging.WARNING)
#                 # clean_actions.pop(k)
#                 # v = v[:20]
#                 v = [np.array([0, 0, 0, 0, 0], dtype=int)]  # no op (temporary)
#             clean_actions[k] = np.array(
#                 v, dtype=int
#             )  # At least required for move_direction
#         return clean_actions
#
#     def _sanitize_actions(self, actions):
#         """Sanitize actions before sending to env.step(...)"""
#         for k, v in actions.items():
#             actions[k] = np.asanyarray(v, dtype=int)
#         return actions
#

if __name__ == '__main__':
    obs = GeneralObs(3)
    obs.test()


if __name__ == '_main__':
    run_type = 'start'
    import time

    start = time.time()
    ########## PyCharm ############
    from luxai2022.env import LuxAI2022
    from util import initialize_step_fig, add_env_step
    from util import get_test_env, show_env, run
    from luxai_runner.utils import to_json
    import dataclasses
    import json

    if run_type == 'test':
        pass

    elif run_type == 'start':
        ### From start running X steps
        # fig = run(Agent, Agent, map_seed=1, max_steps=40, return_type='figure')
        # fig.show()

        replay = run(
            Agent,
            Agent,
            map_seed=1,
            max_steps=1000,
            save_state_at=None,
            return_type='replay',
        )
        with open('replay.json', 'w') as f:
            json.dump(replay, f)

    elif run_type == 'checkpoint':
        ### From checkpoint
        env = get_test_env('test_state.pkl')
        fig = show_env(env)
        fig.show()

    print(f'Finished in {time.time()-start:.3g} s')
    ####################################

    ####### JUPYTER ONLY  ########
    # from lux_eye import run_agents
    # run_agents(Agent, Agent, map_seed=1, save_state_at=None)
    #######################

    # env = get_test_env()
    # show_env(env)
    #
    # game_state = game_state_from_env(env)
    # unit = game_state.units["player_0"]["unit_4"]
    #
    # # First move unit to start on Factory tile -- down, down, left
    # empty_actions = {
    #     "player_0": {},
    #     "player_1": {},
    # }
    #
    # actions = {
    #     "player_0": {unit.unit_id: [unit.move(DOWN, repeat=1), unit.move(LEFT)]},
    #     "player_1": {},
    # }
    #
    # obs, rews, dones, infos = env.step(actions)
    # fig = initialize_step_fig(env)
    # for i in range(2):
    #     obs, rews, dones, infos = env.step(empty_actions)
    #     add_env_step(fig, env)
    #
    # # Set work flow -- pickup, right, up, up, up, dig, down, down, down, left, transfer
    # actions = {
    #     "player_0": {
    #         unit.unit_id: [
    #             unit.pickup(POWER, 200, repeat=-1),
    #             unit.move(RIGHT, repeat=-1),
    #             unit.move(UP, repeat=-1),
    #             unit.move(UP, repeat=-1),
    #             unit.move(UP, repeat=-1),
    #             unit.dig(repeat=-1),
    #             unit.dig(repeat=-1),
    #             unit.move(DOWN, repeat=-1),
    #             unit.move(DOWN, repeat=-1),
    #             unit.move(DOWN, repeat=-1),
    #             unit.move(LEFT, repeat=-1),
    #             unit.transfer(CENTER, ICE, unit.cargo.ice, repeat=-1),
    #         ]
    #     },
    #     "player_1": {},
    # }
    #
    # obs, rews, dones, infos = env.step(actions)
    # add_env_step(fig, env)
    # for i in range(30):
    #     obs, rews, dones, infos = env.step(empty_actions)
    #     add_env_step(fig, env)
    # fig.show()
