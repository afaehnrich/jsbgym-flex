import gym
from gym import envs
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import cm
import gym_jsbsim
import gym_jsbsim.properties as prp
from gym_jsbsim.aircraft import *
from gym_jsbsim.environment import *
from gym_jsbsim.tasks import TurnHeadingControlTask, MyFlightTask
from gym_jsbsim.pid import PID_angle
import pathlib
from pid_controller.pid import PID, twiddle
from sklearn.metrics import mean_squared_error
import random
import math
import toml
#from gym_jsbsim.configuration import Configuration

cfg = toml.load('gym-jsbsim-cfg.toml')


class single_pid: 
    def __init__(self, kp, ki, kd, min = -1, max = 1,):
        self.pid = PID(p=kp, i=ki, d=kd)
        self.min = min
        self.max = max

    def get_val(self, target, outer, time):
        self.pid.target = target
        ret = self.pid (feedback=outer, curr_tm = time)
        ret = np.clip(ret, self.min, self.max)
        return ret

    def set_pid (self, kp, ki, kd):
        self.pid.Kp = kp
        self.pid.Ki = ki
        self.pid.Kd = kd


def simulate():
    rolls=[]
    actions=[]
    times=[]
    env.reset()
    action=[0.0,0.0,0.0]   
    target = random.random()
    targets=[]
    x=[]
    y=[]
    z=[]
    nicks=[]
    t_roll=0
    t_nick=0
    for i in range(100000):
        env.render()
        roll = env.sim[prp.roll_rad]
        roll_rate =env.sim[prp.p_radps]
        time = env.sim[prp.sim_time_s]
        rolls.append(roll)
        actions.append(action[0])
        times.append(time)
        targets.append(target)
        x.append(env.sim[prp.altitude_sl_ft])
        y.append(env.sim[prp.dist_travel_lat_m])
        z.append(env.sim[prp.dist_travel_lon_m])
        nicks.append(env.sim[prp.pitch_rad])

        if i==500: target = -random.random()
        if i%10 ==0:
            cfg = toml.load('gym-jsbsim-cfg.toml')
            r=cfg['pid']['roll']
            n=cfg['pid']['pitch']
            pid_roll.tune(r['p'],r['i'],r['d'])
            pid_roll.anti_windup = r['windup']
            pid_pitch.tune(n['p'],n['i'],n['d'])
            pid_pitch.anti_windup = n['windup']
            t_roll=r['target']
            t_pitch=n['target']
            pid_roll.target = t_roll
            pid_pitch.target = t_pitch
            print ('{:.0f}: pid_r=[{:.3f},{:.3f},{:5f} pid_n=[{:.3f},{:.3f},{:.5f}] t_roll={:.1f} t_nick={:.1f}'
                ' action=[{:.2f},{:.2f}] position=[{:.2f},{:.2f}]          '
                .format(time, r['p'], r['i'], r['d'], n['p'], n['i'], n['d'], t_roll, t_pitch, 
                action[0],action[1], roll, env.sim[prp.pitch_rad]),end='\r')
        action[0] = pid_roll(roll)
        action[1] = pid_pitch(env.sim[prp.pitch_rad])
        env.step(np.array(action))
        #    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd)
    return rolls, actions, targets, times,x, y, z, nicks


r=cfg['pid']['roll']
p=cfg['pid']['pitch']
pid_roll = PID_angle(r['p'], r['i'], r['d'], angle_max=2*math.pi, out_min=-1, out_max=1,
     target=r['target'], anti_windup=r['windup'] )
pid_pitch = PID_angle(p['p'], p['i'], p['d'], angle_max=2*math.pi, out_min=-1, out_max=1,
     target=p['target'], anti_windup=p['windup'] )

np.set_printoptions(precision=3, suppress=True)
#jsbsim_path = pathlib.Path.joinpath( pathlib.Path(__file__).parent.absolute(), 'JSBSim') 
#jsbsim_path='/usr/share/games/flightgear'
env = gym_jsbsim.environment.JsbSimEnv(cfg = cfg, task_type = MyFlightTask, shaping = Shaping.STANDARD)

rolls, actions, targets, times, x, y, z, nicks = simulate()
#from mpl_toolkits import mplot3d
#ax = plt.axes(projection='3d')
#ax.scatter3D(x, y, z);
plt.show()
plt.plot(nicks)
plt.show()

'''
State space:

0  (name='position/h-sl-ft', description='altitude above mean sea level [ft]', min=-1400, max=85000)
1  (name='position/lat-geod-deg', decription='geocentric latitude [deg]', min=-90, max=90)
2  (name='position/long-gc-deg', description='geodesic longitude [deg]', min=-180, max=180)
3  (name='attitude/pitch-rad', description='pitch [rad]', min=-1.5707963267948966, max=1.5707963267948966)
4  (name='attitude/roll-rad', description='roll [rad]', min=-3.141592653589793, max=3.141592653589793)
5  (name='velocities/u-fps', description='body frame x-axis velocity [ft/s]', min=-2200, max=2200)
6  (name='velocities/v-fps', description='body frame y-axis velocity [ft/s]', min=-2200, max=2200)
7  (name='velocities/w-fps', description='body frame z-axis velocity [ft/s]', min=-2200, max=2200)
8  (name='velocities/p-rad_sec', description='roll rate [rad/s]', min=-6.283185307179586, max=6.283185307179586)
9  (name='velocities/q-rad_sec', description='pitch rate [rad/s]', min=-6.283185307179586, max=6.283185307179586)
10  (name='velocities/r-rad_sec', description='yaw rate [rad/s]', min=-6.283185307179586, max=6.283185307179586)
11 (name='fcs/left-aileron-pos-norm', description='left aileron position, normalised', min=-1, max=1)
12 (name='fcs/right-aileron-pos-norm', description='right aileron position, normalised', min=-1, max=1)
13 (name='fcs/elevator-pos-norm', description='elevator position, normalised', min=-1, max=1)
14 (name='fcs/rudder-pos-norm', description='rudder position, normalised', min=-1, max=1)
15 (name='error/altitude-error-ft', description='error to desired altitude [ft]', min=-1400, max=85000)
16 (name='aero/beta-deg', description='sideslip [deg]', min=-180, max=180)
17 (name='error/track-error-deg', description='error to desired track [deg]', min=-180, max=180)
18 (name='info/steps_left', description='steps remaining in episode', min=0, max=300)
'''
