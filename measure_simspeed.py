import jsbgym_flex
from jsbgym_flex.environment import JsbSimEnv
from jsbgym_flex.tasks import Shaping, MyFlightTask
import toml
from datetime import datetime

cfg = toml.load('gym-jsbsim-cfg_example.toml')

def simCycle(name, env, outer, inner):
    print ('Simulation {}: {}x{} steps'.format(name, outer, inner),end='\r') 
    before = datetime.now()
    for _ in range(outer):
        env.reset()  
        for _ in range(inner):
            action=env.action_space.sample()
            env.step(action)
    after = datetime.now()
    dt = (after-before)
    print ('Simulation {}: {}x{} steps: {}.{}s'.format(name, outer, inner, dt.seconds, dt.microseconds) )

env = jsbgym_flex.environment.JsbSimEnv(cfg = cfg, task_type = MyFlightTask, shaping = Shaping.STANDARD)
name = 'JSBSIM'
simCycle(name, env, 1, 100)
simCycle(name, env, 10, 100)
simCycle(name, env, 1, 10000)
simCycle(name, env, 10, 1000)
simCycle(name, env, 100, 100)
env.close()