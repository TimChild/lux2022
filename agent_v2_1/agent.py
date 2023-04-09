from __future__ import annotations
import abc
from typing import TYPE_CHECKING, List, Dict, Tuple
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
import logging

from lux.kit import obs_to_game_state
from lux.config import EnvConfig
from lux.utils import my_turn_to_place_factory

from unit_manager import UnitManager, FriendlyUnitManger, EnemyUnitManager
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

import util

# logging.basicConfig(filename='agent_log.log', level=logging.INFO)
logging.basicConfig(level=logging.INFO)
logging.info('Starting Log')

if TYPE_CHECKING:
    from .master_state import Recommendation

    # from .path_finder import PathFinder
    from .new_path_finder import NewPathFinder


@dataclass
class UnitsToAct:
    needs_to_act: List[FriendlyUnitManger]
    should_not_act: List[FriendlyUnitManger]
    has_updated_actions: List[FriendlyUnitManger] = list


@dataclass
class Collision:
    """First collision only"""
    unit_id: str
    other_unit_id: str
    pos: Tuple[int, int]
    step: int


@dataclass
class Collisions:
    friendly: Dict[str, Collision]  # Collisions with friendly unit
    enemy: Dict[str, Collision]  # Collisions with enemy units


@dataclass
class CloseEnemies:
    """All nearby enemies"""
    unit_id: str
    unit_pos: Tuple[int, int]
    enemy_id: List[str]
    enemy_pos: List[Tuple[int, int]]
    distance: List[int]


@dataclass
class UnitPaths(abc.ABC):
    light: Dict[str, np.ndarray] = dict
    heavy: Dict[str, np.ndarray] = dict


@dataclass
class FriendlyUnitPaths(UnitPaths):
    pass


@dataclass
class EnemyUnitPaths(UnitPaths):
    pass


@dataclass
class AllUnitPaths:
    friendly: FriendlyUnitPaths = FriendlyUnitPaths
    enemy: EnemyUnitPaths = EnemyUnitPaths


class TurnPlanner:
    search_dist = 10

    def __init__(self, master: MasterState, pathfinder: NewPathFinder):
        """Assuming this is called after beginning of turn update"""
        self.master = master
        self.pathfinder = pathfinder

        # Changed during turn planning
        self.unit_paths = AllUnitPaths()  # Begins empty

        # Caching
        self._costmap: np.ndarray = None
        self._upcoming_collisions: Collisions = None
        self._close_enemies: Dict[str, CloseEnemies] = {}

    def units_should_consider_acting(
        self, units: List[FriendlyUnitManger]
    ) -> UnitsToAct:
        """
        Determines which units should potentially act this turn, and which should continue with current actions
        Does this based on:
            - collisions in next couple of turns
            - enemies nearby
            - empty action queue

        Args:
            units: list of friendly units

        Returns:
            Instance of UnitsToAct
        """
        upcoming_collisions = self.calculate_collisions()
        close_to_enemy = self.calculate_close_enemies()
        needs_to_act = []
        should_not_act = []
        for unit in units:
            should_act = False
            # If no queue
            if len(unit.unit.action_queue) == 0:
                logging.info(f'no actions -- {unit.unit_id} should consider acting')
                should_act = True
            elif unit.unit_id in upcoming_collisions.friendly:
                logging.info(
                    f'collision with friendly -- {unit.unit_id} should consider acting'
                )
                should_act = True
            elif unit.unit_id in upcoming_collisions.enemy:
                logging.info(
                    f'collision with friendly -- {unit.unit_id} should consider acting'
                )
                should_act = True
            elif unit.unit_id in close_to_enemy:
                logging.info(f'close to enemy -- {unit.unit_id} should consider acting')
                should_act = True

            if should_act:
                needs_to_act.append(unit)
            else:
                should_not_act.append(unit)
        return UnitsToAct(needs_to_act=needs_to_act, should_not_act=should_not_act)

    def calculate_collisions(self, check_steps=2) -> Collisions:
        """Calculates the upcoming collisions based on action queues of all units"""
        if self._upcoming_collisions is None:
            pass
            self._upcoming_collisions = None
        return self._upcoming_collisions

    def calculate_close_enemies(self) -> Dict[str, CloseEnemies]:
        """Calculates which units are close to enemies"""
        if self._close_enemies is None:
            pass
            self._close_enemies = None
        return self._close_enemies

    def collect_unit_data(self, units: List[FriendlyUnitManger]) -> pd.DataFrame:
        """
        Collects data from units and stores it in a pandas dataframe.

        Args:
            units: List of FriendlyUnitManger objects.

        Returns:
            A pandas dataframe containing the unit data.
        """
        data = []
        for unit in units:
            unit_distance_map = util.pad_and_crop(
                util.manhattan_kernel(self.search_dist),
                large_arr=self.master.maps.rubble,
                x1=unit.pos[0],
                y1=unit.pos[1],
                fill_value=self.search_dist,
            )
            unit_factory = self.master.factories.friendly.get(unit.factory_id, None)

            data.append(
                {
                    'distance_to_factory': unit_distance_map[
                        unit_factory.factory.pos[0], unit_factory.factory.pos[1]
                    ]
                    if unit_factory
                    else np.nan,
                    'is_heavy': unit.unit.unit_type == 'HEAVY',
                    'enough_power_to_move': unit.unit.power
                    > unit.unit_config.MOVE_COST
                    + unit.unit_config.ACTION_QUEUE_POWER_COST,
                    'power': unit.unit.power,
                    'ice': unit.unit.cargo.ice,
                    'ore': unit.unit.cargo.ore,
                }
            )

        df = pd.DataFrame(data)
        return df

    def sort_units_by_priority(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Sorts units by priority based on the provided dataframe.

        Args:
            df: A pandas dataframe containing the unit data.

        Returns:
            A sorted pandas dataframe with units ordered by priority.
        """
        sorted_df = df.sort_values(
            by=['is_heavy', 'enough_power_to_move', 'power', 'ice', 'ore'],
            ascending=[True, False, True, False, True],
        )

        return sorted_df

    def base_costmap(self) -> np.ndarray:
        """
        Calculates the base costmap based on:
            - rubble array
            - most travelled?

        Returns:
            A numpy array representing the costmap.
        """
        if self._costmap is None:
            costmap = self.pathfinder.get_costmap(self.master.maps.rubble)
            self._costmap = costmap
        return self._costmap

    def get_costmap_with_paths(
        self, base_costmap: np.ndarray, unit: FriendlyUnitManger
    ) -> np.ndarray:
        """
        Updates the costmap with the paths of the units that have determined paths this turn (not acting, done acting, or enemy)

        Args:
            base_costmap: A numpy array representing the costmap.
            unit: Unit to get the costmap for (i.e. distances calculated relative to this unit)
        """
        new_cost = base_costmap.copy()
        # Add enemy paths
        for unit_id, path in self.unit_paths.enemy.heavy.items():
            # if current unit is heavy and enemy has signficantly lower energy
            # make lower cost
            # update_costmap_with_gaussian_paths(base_costmap, unit)
            # else:
            # make higher cost
            pass
        for unit_id, path in self.unit_paths.enemy.light.items():
            # if current unit is heavy or enemy has signficantly lower energy
            # make lower cost
            # else:
            # make higher cost
            pass

        # Add friendly paths
        for unit_id, path in self.unit_paths.friendly.heavy.items():
            # high cost (really want to avoid collision with own units that have set paths already)
            pass
        for unit_id, path in self.unit_paths.friendly.light.items():
            # high cost (really want to avoid collision with own units that have set paths already)
            pass

        return new_cost

    def update_pathfinder(self, unit: FriendlyUnitManger) -> NewPathFinder:
        """Update the shared pathfinder with the specific costmap for the current unit"""
        base_costmap = self.base_costmap()
        # TODO: Could do more to the base costmap here (make areas higher or lower)
        new_costmap = self.get_costmap_with_paths(
            base_costmap=base_costmap, unit=unit
        )
        # TODO: or here
        self.pathfinder.costmap = new_costmap
        return self.pathfinder

    def process_units(
        self,
        friendly_units: List[FriendlyUnitManger],
        enemy_units: List[EnemyUnitManager],
    ) -> Dict[str, List[np.ndarray]]:
        """
        Processes the units by choosing the paths for units that need to act this turn.

        Args:
            friendly_units: All friendly units after beginning of turn update
            enemy_units: All enemy units after beginning of turn update

        Returns:
            Actions to update units with
        """
        units_to_act = self.units_should_consider_acting(self.master.units.friendly.all)
        unit_data = self.collect_unit_data(units_to_act)
        sorted_units = self.sort_units_by_priority(unit_data)

        base_costmap = self.base_costmap()

        for unit in sorted_units.itertuples():
            unit: FriendlyUnitManger
            pathfinder = self.update_pathfinder(unit)

            actions_before = unit.unit.action_queue
            # Figure out new actions for unit  (i.e. RoutePlanners)
            pass

            if unit.unit.action_queue == actions_before:
                units_to_act.should_not_act.append(unit)
            else:
                units_to_act.has_updated_actions.append(unit)

        actions = {}
        for unit in units_to_act.has_updated_actions:
            if len(unit.unit.action_queue) > 0:
                actions[unit.unit_id] = unit.unit.action_queue
        return actions


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
