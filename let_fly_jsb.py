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
from gym_jsbsim.tasks import TurnHeadingControlTask
import pathlib
from pid_controller.pid import PID, twiddle
from sklearn.metrics import mean_squared_error
import random
import math
import json

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
            with open('pid.json','r') as f:
                try:
                    p=json.load(f)
                except:
                    print('error loading JSON')
                r=p['pid_roll']
                n=p['pid_nick']
                pid_roll.set_pid(r['p'],r['i'],r['d'])
                pid_nick.set_pid(n['p'],n['i'],n['d'])
                t_roll=p['target_roll']
                t_nick=p['target_nick']
                print ('pid_r=[{:.3f},{:.3f},{:5f} pid_n=[{:.3f},{:.3f},{:.5f}] t_roll={:.1f} t_nick={:.1f}'
                    ' action=[{:.2f},{:.2f}] position=[{:.2f},{:.2f}]          '
                    .format(r['p'], r['i'], r['d'], n['p'], n['i'], n['d'], t_roll, t_nick, 
                    action[0],action[1], roll, env.sim[prp.pitch_rad]),end='\r')
        action[0] = pid_roll.get_val(t_roll, roll, time)
        action[1] = pid_nick.get_val(t_nick, env.sim[prp.pitch_rad], time)

        #action[0] = 0
        #action[1] = 0

        #action[1]=0
        env.step(np.array(action))
        #    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd)
    return rolls, actions, targets, times,x, y, z, nicks

#pid_roll = io_pid( 1.362, 1.0, .288, .793, 0.285, .234, -1, 1) # -0.00799882575461252   
#pid_roll = io_pid( 0.9, 0.,  0.4, 0.4, 0.4, 0.1, -1, 1)#    -0.056860760633386735
#pid_roll = io_pid (0.99,  0.2,   0.127, 0.647, 0.2,   0.01, -1, 1 )  #gANZ OK # -0.02146936449263489
#pid_roll = io_pid( 5, 2,  0.001, 0.5, 0.3, 0.001, -1, 1) # anfangs extrem, dann gut
#pid_roll = io_pid(0.685, 1.,    0.443, 0.381, 0.,    0.066,-1,1)#    -0.07142133264873667
#pid_roll = io_pid(0.928, 0.,    0.28,  0.527, 0.,    0., -1, 1 ) #   -0.03351313101882057
#pid_roll = io_pid(0.835, 0.468, 0.,    0.731, 0.,    0., -1, 1  )#    -0.017521506445783038
#pid_roll = io_pid( 2, 1.5,  0.01, 0.5, 0.3, 0.001, -1, 1) # anfangs extrem, dann gut
#pid_roll = io_pid(0.969, 0.856, 0.242, 0.607, 0.737, 0.128, -1, 1)#    -0.08736105006175784
#pid_roll = io_pid(0.77,  0.,    0.285, 0.541, 0.,    0.,  -1, 1)#    -0.07065646914239888
#pid_roll = io_pid(0.77,  0.001,    0.3, 0.4, 0.2,    0.,  -1, 1)#    -0.07065646914239888
#pid_roll = io_pid(0.982, 0.,    0.,    0.533, 0.,    0., -1,1)
#pid_roll = io_pid(0.721, 0.,    0.717, 0.34,  0.31,  0., -1,1)  
#pid_roll = io_pid(0.686, 1.,    0.,    0.358, 0.918, 0.0, -1,1)
#pid_roll = io_pid(1.0, 0.,    0.,    0.5, 0., 0.000, -1,1)

#pid_nick = io_pid(1, .6,    0.0,  0.5, 0.4, 0.0001, -1,1)

#pid_roll = single_pid(.015, 10., 0.00001, -1,1) aus der XML
#pid_nick = single_pid(-0.005, 100.,    0.00001, -1,1) aus der XML
pid_roll = single_pid(.00015, 000., 0.0000, -1,1) 
pid_nick = single_pid(-.00015, 100000., 0.0000, -1,1) 
with open('pid.json','r') as f:
    p=json.load(f)
    r=p['pid_roll']
    n=p['pid_nick']
    pid_roll.set_pid(r['p'],r['i'],r['d'])
    pid_nick.set_pid(n['p'],n['i'],n['d'])

np.set_printoptions(precision=3, suppress=True)
#jsbsim_path = pathlib.Path.joinpath( pathlib.Path(__file__).parent.absolute(), 'JSBSim') 
#jsbsim_path='/usr/share/games/flightgear'
jsbsim_path ='../jsbsim'
env = gym_jsbsim.environment.JsbSimEnv(task_type = TurnHeadingControlTask, aircraft = cessna172P,
                 agent_interaction_freq = 10, shaping = Shaping.STANDARD, jsbsim_dir=str(jsbsim_path))

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
