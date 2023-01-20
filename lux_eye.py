import dataclasses
import json
import numpy as np
from luxai2022.env import LuxAI2022
from luxai_runner.utils import to_json
from IPython import get_ipython
from IPython.display import display, HTML

def run_agents(agent1, agent2, map_seed=None, save_state_at=None):
    env = LuxAI2022()

    # This code is partially based on the luxai2022 CLI:
    # https://github.com/Lux-AI-Challenge/Lux-Design-2022/blob/main/luxai_runner/episode.py

    obs = env.reset(seed=map_seed)
    state_obs = env.state.get_compressed_obs()

    agents = {
        "player_0": agent1("player_0", env.state.env_cfg),
        "player_1": agent2("player_1", env.state.env_cfg)
    }

    game_done = False
    rewards, dones, infos = {}, {}, {}

    for agent_id in agents:
        rewards[agent_id] = 0
        dones[agent_id] = 0
        infos[agent_id] = {
            "env_cfg": dataclasses.asdict(env.state.env_cfg)
        }

    replay = {
        "observations": [state_obs],
        "actions": [{}]
    }

    i = 0
    while not game_done:
        i += 1

        actions = {}
        for agent_id, agent in agents.items():
            agent_obs = obs[agent_id]
            
            if save_state_at and env.state.real_env_steps == save_state_at:
                import pickle
                with open('test_state.pkl', 'wb') as f:
                    pickle.dump(env.get_state(), f)

            if env.state.real_env_steps < 0:
                agent_actions = agent.early_setup(env.env_steps, agent_obs)
            else:
                agent_actions = agent.act(env.env_steps, agent_obs)

            for key, value in agent_actions.items():
                if isinstance(value, list):
                    agent_actions[key] = np.array(value)

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

    execution_id = get_ipython().execution_count

    html = f"""
<iframe
    src="https://jmerle.github.io/lux-eye-2022/kaggle"
    width="1040"
    height="560"
    id="luxEye2022IFrame{execution_id}"
    frameBorder="0"
></iframe>

<script>
document.querySelector('#luxEye2022IFrame{execution_id}').addEventListener('load', () => {{
    document.querySelector('#luxEye2022IFrame{execution_id}').contentWindow.postMessage({json.dumps(to_json(replay))}, 'https://jmerle.github.io');
}});
</script>
    """

    display(HTML(html))