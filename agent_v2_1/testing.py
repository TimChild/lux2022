from luxai_s2.env import LuxAI_S2
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import logging
import sys
import os
from pathlib import Path

from util import MyEnv, figures_to_subplots, show_env
from config import update_logging_level


parent_dir = Path(os.getcwd()).resolve().parent
if str(parent_dir) not in sys.path:
    print(f"Adding {parent_dir} to path")
    sys.path.insert(0, str(parent_dir))

# from agent_v2_0.agent import Agent as Agent_v2_0
from agent import Agent


if __name__ == '__main__':
    update_logging_level(logging.ERROR)
    seed = 42
    # seed = 178220973

    # Run initial setup (placing factories)
    myenv = MyEnv(seed, Agent, Agent)
    myenv.run_early_setup()
    myenv.run_to_step(100)
    myenv.show()