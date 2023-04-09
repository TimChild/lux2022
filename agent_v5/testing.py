from luxai_s2.env import LuxAI_S2
import matplotlib.pyplot as plt
import numpy as np
from util import figures_to_subplots, show_env
from agent_v5.agent import Agent

if __name__ == '__main__':

    env = LuxAI_S2()
    # Run the early_setup phase
    agents = {player: Agent(player, env.state.env_cfg) for player in env.possible_agents}
    agent = agents["player_0"]
    # obs = env.reset(seed=42)  # resets an environment with a seed
    obs = env.reset(seed=178220973)  # resets an environment with a seed
    step = 0

    while env.state.real_env_steps < 0:
        actions = {}
        for player in env.agents:
            o = obs[player]
            acts = agents[player].early_setup(step, o)
            actions[player] = acts
        step += 1
        obs, rewards, dones, infos = env.step(actions)

    while env.state.real_env_steps < 54:
        print(f"Carrying out real step {step}, env step {env.state.real_env_steps}")
        actions = {player: agent.act(step, obs[player]) for player, agent in agents.items()}
        step += 1
        obs, rewards, dones, infos = env.step(actions)

    show_env(env).show(renderer='browser')

    while env.state.real_env_steps < 70:
        print(f"Carrying out real step {step}, env step {env.state.real_env_steps}")
        actions = {player: agent.act(step, obs[player]) for player, agent in agents.items()}
        step += 1
        obs, rewards, dones, infos = env.step(actions)
    # print(f"Carrying out real step {step}, env step {env.state.real_env_steps}")
    # actions = {player: agent.act(step, obs[player]) for player, agent in agents.items()}
    # step += 1
    # obs, rewards, dones, infos = env.step(actions)
    #
    # show_env(env).show(renderer='browser')
