import gym
from gym import envs
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import gym_jsbsim
import gym_jsbsim.properties as prp
from gym_jsbsim.environment import JsbSimEnv
from gym_jsbsim.tasks import Shaping, MyFlightTask
import pathlib
import random
import math
import toml

cfg = toml.load('gym-jsbsim-cfg.toml')



def simulate(steps, render = False):
    env.reset()
    action=[0.0]   
    for i in range(steps):
        if render: 
            env.render()
            if i%10 ==0:
                time = env.sim[prp.sim_time_s]
                t_roll= env.sim[env.properties['target_roll']]
                t_pitch= env.sim[env.properties['target_pitch']]
                print ('{:.0f}s: t_roll={:.1f} t_pitch={:.1f}'
                    ' position=[{:.2f},{:.2f}]          '
                    .format(time,t_roll, t_pitch, 
                    env.sim[prp.roll_rad], env.sim[prp.pitch_rad]),end='\r')
            action=env.action_space.sample()
            env.step(np.array(action))
    return

env = gym_jsbsim.environment.JsbSimEnv(cfg = cfg, task_type = MyFlightTask, shaping = Shaping.STANDARD)
render = not (cfg.get('visualiser') or {}).get('enable') == False
simulate(1000, render)
