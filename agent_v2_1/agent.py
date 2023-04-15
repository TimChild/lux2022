from __future__ import annotations

import time
from typing import TYPE_CHECKING
import numpy as np

from unit_action_planner import UnitActionPlanner
from lux.kit import obs_to_game_state
from lux.config import EnvConfig
from lux.utils import my_turn_to_place_factory

from master_state import MasterState
from mining_planner import MiningPlanner
from rubble_clearing_planner import RubbleClearingPlanner
from combat_planner import CombatPlanner
from factory_action_planner import FactoryActionPlanner

from config import get_logger


logger = get_logger(__name__)


if TYPE_CHECKING:
    pass


class FactoryTurnPlanner:
    def __init__(self, master: MasterState):
        self.master = master

    def get_observation_df(self):
        """
        - Gather info about factories and put into DF
        - Sort so that top is highest priority factory (not sure this will make much difference anyway)
        - Some of this is probably useful for units to decide on actions too
        """
        # factory_obs = {}
        # for factory_id, factory in self.master.factories.friendly.items():
        #     if factory_should_consider_acting(factory, self.master):
        #         fobs = calculate_factory_obs(factory, self.master)
        #         factory_obs[factory_id] = fobs

    def get_actions(self):
        """
        - Determine actions based on info in DF

        """
        # factory_obs = self.get_observation_df()
        #
        # for index, row in factory_obs.iterrows():
        #
        #
        # # Factory Actions
        # factory_actions = {}
        # for factory_id in factory_obs.keys():
        #     factory = self.master.factories.friendly[factory_id]
        #     fobs = factory_obs[factory_id]
        #     f_action = calculate_factory_action(
        #         factory=factory, fobs=fobs, master=self.master
        #     )
        #     if f_action is not None:
        #         factory_actions[factory_id] = f_action


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig):
        logger.info(f"Initializing agent for player {player}")
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
        self.combat_planner = CombatPlanner(self.master)
        self.factory_action_planner = FactoryActionPlanner(self.master)
        self.unit_action_planner = UnitActionPlanner(self.master)

    def _beginning_of_step_update(
            self, step: int, obs: dict, remainingOverageTime: int
    ):
        """Use the step and obs to update any turn based info (e.g. map changes)"""
        logger.info(f"Beginning of step update for step {step}")
        game_state = obs_to_game_state(step, self.env_cfg, obs)
        # TODO: Use last obs to see what has changed to optimize update? Or master does this?
        self.master.update(game_state)
        self.last_obs = obs

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
                action = self.factory_action_planner.place_factory(
                    self.master.game_state, self.player
                )
        logger.info(f"Early setup action {action}")
        return action

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """Required API for Agent. This is called every turn after early_setup is complete"""
        tstart = time.time()
        logger.warning(
            f"====================== Start of turn {self.master.game_state.real_env_steps+1} for {self.player} =============================="
        )
        self._beginning_of_step_update(step, obs, remainingOverageTime)

        self.mining_planner.update()
        self.rubble_clearing_planner.update()
        self.factory_action_planner.update()

        factory_desires = self.factory_action_planner.get_factory_desires()
        factory_infos = self.factory_action_planner.get_factory_infos()

        self.unit_action_planner.update(factory_desires=factory_desires, factory_infos=factory_infos)

        factory_actions = self.factory_action_planner.decide_factory_actions()

        unit_actions = self.unit_action_planner.decide_unit_actions(
            mining_planner=self.mining_planner,
            rubble_clearing_planner=self.rubble_clearing_planner,
            combat_planner=self.combat_planner,
            factory_desires=factory_desires,
            factory_infos=factory_infos,
        )

        logger.verbose(f"{self.player} Unit actions: {unit_actions}")
        logger.debug(f"{self.player} Factory actions: {factory_actions}")
        logger.warning(
            f"========================= End of turn {self.master.game_state.real_env_steps+1} for {self.player}: Took {time.time()-tstart:.1f}s ==========================="
        )
        return dict(**unit_actions, **factory_actions)



if __name__ == "__main__":
    pass
