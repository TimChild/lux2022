import numpy as np
from lux.kit  import obs_to_game_state

def my_turn_to_place_factory(place_first: bool, step: int):
    if place_first:
        if step % 2 == 1:
            return True
    else:
        if step % 2 == 0:
            return True
    return False

# def obs_to_game_state(step, env_cfg, obs):
    
#     units = dict()
#     for agent in obs["units"]:
#         units[agent] = dict()
#         for unit_id in obs["units"][agent]:
#             unit_data = obs["units"][agent][unit_id]
#             cargo = UnitCargo(**unit_data["cargo"])
#             unit = Unit(
#                 **unit_data,
#                 unit_cfg=env_cfg.ROBOTS[unit_data["unit_type"]],
#                 env_cfg=env_cfg
#             )
#             unit.cargo = cargo
#             units[agent][unit_id] = unit
            

#     factory_occupancy_map = np.ones_like(obs["board"]["rubble"], dtype=int) * -1
#     factories = dict()
#     for agent in obs["factories"]:
#         factories[agent] = dict()
#         for unit_id in obs["factories"][agent]:
#             f_data = obs["factories"][agent][unit_id]
#             cargo = UnitCargo(**f_data["cargo"])
#             factory = Factory(
#                 **f_data,
#                 env_cfg=env_cfg
#             )
#             factory.cargo = cargo
#             factories[agent][unit_id] = factory
#             factory_occupancy_map[factory.pos_slice] = factory.strain_id
#     teams = dict()
#     for agent in obs["teams"]:
#         team_data = obs["teams"][agent]
#         faction = FactionTypes[team_data["faction"]]
#         teams[agent] = Team(**team_data, agent=agent)

#     return GameState(
#         env_cfg=env_cfg,
#         env_steps=step,
#         board=Board(
#             rubble=obs["board"]["rubble"],
#             ice=obs["board"]["ice"],
#             ore=obs["board"]["ore"],
#             lichen=obs["board"]["lichen"],
#             lichen_strains=obs["board"]["lichen_strains"],
#             factory_occupancy_map=factory_occupancy_map,
#             factories_per_team=obs["board"]["factories_per_team"],
#             valid_spawns_mask=obs["board"]["valid_spawns_mask"]
#         ),
#         units=units,
#         factories=factories,
#         teams=teams

#     )




class Agent:
    def __init__(self, player: str, env_cfg):
        self.player = player
        self.env_cfg = env_cfg

        self._factory_placed = False

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        """Required API for Agent. This is called until all factories are placed"""
        if step == 0:
            return dict(faction="TheBuilders", bid=26)
        else:
            # factory placement period
            game_state = obs_to_game_state(step, self.env_cfg, obs)
            my_turn_to_place = my_turn_to_place_factory(
                game_state.teams[self.player].place_first, step
            )
            if my_turn_to_place and self._factory_placed is False:
                potential_spawns = list(
                    zip(*np.where(game_state.board.valid_spawns_mask == 1))
                )
                water_left = game_state.teams[self.player].water
                metal_left = game_state.teams[self.player].metal
                self._factory_placed = True
                return dict(
                    spawn=potential_spawns[0], metal=metal_left, water=water_left
                )
            return {}

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        return {}