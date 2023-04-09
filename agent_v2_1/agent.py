from __future__ import annotations
import abc
from typing import TYPE_CHECKING

from lux.kit import obs_to_game_state
from lux.config import EnvConfig
from lux.utils import my_turn_to_place_factory

from dataclasses import dataclass, field
import numpy as np
import logging

# logging.basicConfig(filename='agent_log.log', level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logging.info('Starting Log')

from unit_manager import UnitManager, FriendlyUnitManger
from master_state import MasterState
from factory_manager import FriendlyFactoryManager
from actions import (
    unit_should_consider_acting,
    factory_should_consider_acting,
    calculate_high_level_unit_action,
    calculate_high_level_factory_actions,
)

from mining_planner import MiningPlanner
from rubble_clearing_planner import RubbleClearingPlanner
from factory_manager import BuildHeavyRecommendation


if TYPE_CHECKING:
    from .master_state import Recommendation


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig):
        logging.info(f'initializing agent for player {player}')
        self.player = player
        self.env_cfg: EnvConfig = env_cfg
        np.random.seed(0)

        # Additional initialization
        self.last_obs = None
        self.master: MasterState = MasterState(
            player=self.player,
            env_cfg=env_cfg,
        )

        self.mining_planner = MiningPlanner(self.master)
        self.rubble_clearing_planner = RubbleClearingPlanner(self.master)

    # def log(self, message, level=logging.INFO, **kwargs):
    #     logging.log(level, f'{self.player} {message}', **kwargs)

    def bid(self, obs):
        """Bid for starting factory (default to 0)"""
        return dict(faction="TheBuilders", bid=0)

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        """Required API for Agent. This is called until all factories are placed"""
        self._beginning_of_step_update(step, obs, remainingOverageTime)

        action = dict()
        if step == 0:
            action = self.bid(obs)
        else:
            # factory placement period
            my_turn_to_place = my_turn_to_place_factory(
                self.master.game_state.teams[self.player].place_first, step
            )
            factories_to_place = self.master.game_state.teams[
                self.player
            ].factories_to_place
            if factories_to_place > 0 and my_turn_to_place:
                action = FriendlyFactoryManager.place_factory(
                    self.master.game_state, self.player
                )
        logging.info(f'Early setup action {action}')
        return action

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """Required API for Agent. This is called every turn after early_setup is complete"""
        logging.info(f'======== Start of turn for {self.player} ============')
        self._beginning_of_step_update(step, obs, remainingOverageTime)

        # Get processed observations (i.e. the obs that I will use to train a PPO agent)
        # First - general observations of the full game state (i.e. not factory/unit specific)
        general_obs = self._get_general_processed_obs()

        # Then additional observations for units (not things they already store)
        unit_obs: dict[str, UnitObs] = {}
        for unit_id, unit in self.master.units.friendly.all.items():
            if unit_should_consider_acting(unit, self.master):
                uobs = calculate_unit_obs(unit, self.master)
                unit_obs[unit_id] = uobs

        # Then additional observations for factories (not things they already store)
        factory_obs = {}
        for factory_id, factory in self.master.factories.friendly.items():
            if factory_should_consider_acting(factory, self.master):
                fobs = calculate_factory_obs(factory, self.master)
                factory_obs[factory_id] = fobs

        # Unit Recommendations
        unit_recommendations = {}
        for unit_id in unit_obs.keys():
            unit = self.master.units.friendly.get_unit(unit_id)
            # TODO: maybe add some logic here as to whether to get recommendations??
            mining_rec = self.mining_planner.recommend(unit_manager=unit)
            rubble_clearing_rec = self.rubble_clearing_planner.recommend(unit=unit)
            unit_recommendations[unit_id] = {
                'mine_ice': mining_rec,
                'clear_rubble': rubble_clearing_rec,
            }

        # Unit Actions
        unit_actions = {}
        for unit_id in unit_obs.keys():
            unit = self.master.units.friendly.get_unit(unit_id)
            uobs = unit_obs[unit_id]
            if unit.factory_id:
                fobs = factory_obs[unit.factory_id]
                factory = self.master.factories.friendly[unit.factory_id]
            else:
                fobs = None
                factory = None
            u_action = self.calculate_unit_actions(
                unit=unit,
                uobs=uobs,
                factory=factory,
                fobs=fobs,
                unit_recommendations=unit_recommendations[unit_id],
            )
            if u_action is not None:
                unit_actions[unit_id] = u_action

        # Factory Actions
        factory_actions = {}
        for factory_id in factory_obs.keys():
            factory = self.master.factories.friendly[factory_id]
            fobs = factory_obs[factory_id]
            f_action = calculate_factory_action(
                factory=factory, fobs=fobs, master=self.master
            )
            if f_action is not None:
                factory_actions[factory_id] = f_action

        logging.debug(f'{self.player} Unit actions: {unit_actions}')
        logging.info(f'{self.player} Factory actions: {factory_actions}')
        return dict(**unit_actions, **factory_actions)

    def calculate_unit_actions(
        self,
        unit: FriendlyUnitManger,
        uobs: UnitObs,
        factory: [None, FriendlyFactoryManager],
        fobs: [None, FactoryObs],
        unit_recommendations: dict[str, Recommendation],
    ) -> [list[np.ndarray], None]:
        def factory_has_heavy_ice_miner(factory: FriendlyFactoryManager):
            units = factory.heavy_units
            for unit_id, unit in units.items():
                if unit.status.role == 'mine_ice':
                    return True
            return False

        if factory is None:
            logging.info(f'{unit.unit_id} has no factory. Doing nothing')
            return None

        # Make at least 1 heavy mine ice
        if (
            not factory_has_heavy_ice_miner(factory)
            and unit.unit.unit_type == 'HEAVY'
            and unit.status.role != 'mine_ice'
        ) or (unit.status.role == 'mine_ice' and len(unit.unit.action_queue) == 0):
            logging.info(f'{unit.unit_id} assigned to mine_ice (for {unit.factory_id})')
            mining_rec = unit_recommendations.pop('mine_ice', None)
            if mining_rec is not None:
                unit.status.role = 'mine_ice'
                return self.mining_planner.carry_out(unit, recommendation=mining_rec)
            else:
                raise RuntimeError(f'no `mine_ice` recommendation for {unit.unit_id}')

        # Make at least one light mine rubble
        if (unit.unit.unit_type == 'LIGHT' and not unit.status.role) or (
            unit.status.role == 'clear_rubble' and len(unit.unit.action_queue) == 0
        ):
            logging.info(
                f'{unit.unit_id} assigned to clear_rubble (for {unit.factory_id})'
            )

            rubble_clearing_rec = unit_recommendations.pop('clear_rubble', None)
            if rubble_clearing_rec is not None:
                unit.status.role = 'clear_rubble'
                return self.rubble_clearing_planner.carry_out(
                    unit, recommendation=rubble_clearing_rec
                )
            pass
        return None

    def _beginning_of_step_update(
        self, step: int, obs: dict, remainingOverageTime: int
    ):
        """Use the step and obs to update any turn based info (e.g. map changes)"""
        logging.info(f'Beginning of step update for step {step}')
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        # TODO: Use last obs to see what has changed to optimize update? Or master does this?
        self.master.update(game_state)
        self.mining_planner.update()
        self.rubble_clearing_planner.update()
        self.last_obs = obs

    def _get_general_processed_obs(self) -> GeneralObs:
        """Get a fixed length DF/array of obs that can be passed into an ML agent

        Thoughts:
            - Mostly generated from MasterPlan
            - Some additional info added based on metadata?
        """
        obs = GeneralObs(
            num_friendly_heavy=len(self.master.units.friendly.heavy.keys()),
        )
        return obs


class MyObs(abc.ABC):
    """General form of my observations (i.e. some general requirements)"""

    pass


@dataclass
class GeneralObs(MyObs):
    num_friendly_heavy: int


@dataclass
class UnitObs(MyObs):
    """Object for holding recommendations and other observations relevant to unit on a per-turn basis"""

    id: str
    nearest_enemy_light_distance: int
    nearest_enemy_heavy_distance: int
    current_role: str
    current_action: str


@dataclass
class FactoryObs(MyObs):
    id: str
    center_occupied: bool  # Center tile (i.e. where units are built)


def calculate_unit_obs(unit: FriendlyUnitManger, plan: MasterState) -> UnitObs:
    """Calculate observations that are specific to a particular unit

    Include all the basic stuff like id etc

    Something like, get some recommended actions for the unit given the current game state?
    Those recommendations can include some standard information (values etc.) that can be used to make an ML interpretable observation along with some extra information that identifies what action to take if this action is recommended

    """
    id = unit.unit_id
    nearest_enemy_light_distance = plan.units.nearest_unit(
        pos=unit.unit.pos, friendly=False, enemy=True, light=True, heavy=False
    )
    nearest_enemy_heavy_distance = plan.units.nearest_unit(
        pos=unit.unit.pos, friendly=False, enemy=True, light=False, heavy=True
    )
    uobs = UnitObs(
        id=id,
        nearest_enemy_light_distance=nearest_enemy_light_distance,
        nearest_enemy_heavy_distance=nearest_enemy_heavy_distance,
        current_role=unit.status.role,
        current_action=unit.status.current_action,
    )

    return uobs


def calculate_factory_obs(
    factory: FriendlyFactoryManager, master: MasterState
) -> FactoryObs:
    """Calculate observations that are specific to a particular factory"""

    center_tile_occupied = (
        # True if plan.maps.unit_at_tile(factory.factory.pos) is not None else False
        True
        if master.units.unit_at_position(factory.factory.pos) is not None
        else False
    )

    return FactoryObs(id=factory.unit_id, center_occupied=center_tile_occupied)


def calculate_factory_action(
    factory: FriendlyFactoryManager, fobs: FactoryObs, master: MasterState
) -> [np.ndarray, None]:
    # Building Units
    if (
        fobs.center_occupied is False and master.step < 800
    ):  # Only consider if middle is free and not near end of game
        # Want at least one heavy mining ice
        if (
            len(factory.heavy_units) < 1
            and factory.factory.cargo.metal > master.env_cfg.ROBOTS['HEAVY'].METAL_COST
            and factory.factory.power > master.env_cfg.ROBOTS['HEAVY'].POWER_COST
        ):
            logging.info(f'{factory.factory.unit_id} building Heavy')
            return factory.factory.build_heavy()

        # Want at least one light to do other things
        if (
            len(factory.light_units) < 1
            and factory.factory.cargo.metal > master.env_cfg.ROBOTS['LIGHT'].METAL_COST
            and factory.factory.power > master.env_cfg.ROBOTS['LIGHT'].POWER_COST
        ):
            logging.info(f'{factory.factory.unit_id} building Light')
            return factory.factory.build_light()

    # Watering Lichen
    water_cost = factory.factory.water_cost(master.game_state)
    if (
        factory.factory.cargo.water > 1000 or master.step > 800
    ):  # Either excess water or near end game
        water_cost = factory.factory.water_cost(master.game_state)
        if factory.factory.cargo.water - water_cost > min(
            100, 1000 - master.game_state.real_env_steps
        ):
            logging.info(
                f'{factory.factory.unit_id} watering with water={factory.factory.cargo.water} and water_cost={water_cost}'
            )
            return factory.factory.water()

    return None


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


# if __name__ == '__main__':
#     obs = GeneralObs()

if __name__ == '__main__':
    run_type = 'start'
    import time

    start = time.time()
    ########## PyCharm ############
    from util import get_test_env, show_env, run
    import json

    if run_type == 'test':
        pass

    elif run_type == 'start':
        ### From start running X steps
        # fig = run(Agent, Agent, map_seed=1, max_steps=40, return_type='figure')
        # fig.show()

        replay = run(
            Agent,
            # BasicAgent,
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

    print(f'Finished in {time.time() - start:.3g} s')
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
