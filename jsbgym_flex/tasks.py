import gym
import numpy as np
import random
import types
import math
import enum
import warnings
import geopy
from collections import namedtuple
import jsbgym_flex.properties as prp
from jsbgym_flex import assessors, rewards, utils
from jsbgym_flex.simulation import Simulation
from jsbgym_flex.properties import BoundedProperty, Property
from jsbgym_flex.aircraft import Aircraft
from jsbgym_flex.rewards import RewardStub
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Dict, Tuple, NamedTuple, Type
from jsbgym_flex.pid import PID_angle
import random


class Task(ABC):

    @abstractmethod
    def take_action (self, sim: Simulation, action: Sequence[float]):
        ...
    
    @abstractmethod
    def get_observation (self, sim: Simulation, action: Sequence[float], env) \
            -> Tuple[NamedTuple, float, bool, Dict]:
        ...

    @abstractmethod
    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        ...

    '''
    @abstractmethod
    def get_initial_conditions(self) -> Optional[Dict[Property, float]]:
        ...
    '''
    @abstractmethod
    def get_state_space(self) -> gym.Space:
        ...

    @abstractmethod
    def get_action_space(self) -> gym.Space:
        ...

    @abstractmethod
    def init_reward(self, init):
        ...

class FlightTask(Task, ABC):

    last_agent_reward = Property('reward/last_agent_reward', 'agent reward from step; includes'
                                                             'potential-based shaping reward')
    last_assessment_reward = Property('reward/last_assess_reward', 'assessment reward from step;'
                                                                   'excludes shaping')
                                                                   
    #def __init__(self, action_properties, state_properties, #initial_states, init_sequence,  pid_controls, 
    #            #simulation_dt, controller_dt, observation_dt, 
    #            debug: bool = False) -> None:
    def __init__(self, cfg, sim, env, #initial_states, init_sequence,  pid_controls, 
                #simulation_dt, controller_dt, observation_dt, 
                debug: bool = False) -> None:
        random.seed()
        self.cfg = cfg
        self.sim = sim
        self.env = env
        self.action_properties = tuple(env._choose_properties(cfg.get('actions'), env.properties))
        self.state_properties = tuple(env._choose_properties(cfg.get('states'), env.properties))                
        #self.action_properties = tuple(action_properties)
        #self.state_properties = tuple(state_properties)
        self.last_state = None
        self._make_state_class()
        #self.initial_states = initial_states
        #self.init_sequence = init_sequence
        self.debug = debug
        #self.pid_controls = pid_controls
        self.reset()

    def reset(self):
        self.terminal = False
        self.init_reward(self.cfg['init'], self.sim, self.env)
        pass



    def _make_state_class(self) -> None:
        """ Creates a namedtuple for readable State data """
        # get list of state property names, containing legal chars only
        legal_attribute_names = [prop.get_legal_name() for prop in
                                 self.state_properties]
        self.State = namedtuple('State', legal_attribute_names)

    def take_action (self, sim: Simulation, action: Sequence[float]):
        # input actions
        for prop, command in zip(self.action_properties, action):
            sim[prop] = command
        
    def get_observation (self, sim: Simulation, action: Sequence[float], env) \
            -> Tuple[NamedTuple, float, bool, Dict]:
        #state = self.State(*(sim[prop] for prop in self.state_properties))
        state = [sim[prop] for prop in self.state_properties]
        done = self._is_terminal(sim, env)
        reward = self._calculate_reward(state, self.last_state, done, sim, env)
        if done:
            reward = self._reward_terminal_override(reward, sim, env)
        if self.debug:
            self._validate_state(state, done, action, reward)
        self._store_reward(reward, sim)
        self.last_state = state
        info = {'reward': reward}

        #return state, reward.agent_reward(), done, info
        return state, reward, done, info

    def _validate_state(self, state, done, action, reward):
        if any(math.isnan(el) for el in state):  # float('nan') in state doesn't work!
            msg = (f'Invalid state encountered!\n'
                   f'State: {state}\n'
                   f'Prev. State: {self.last_state}\n'
                   f'Action: {action}\n'
                   f'Terminal: {done}\n'
                   f'Reward: {reward}')
            warnings.warn(msg, RuntimeWarning)

    def _store_reward(self, reward: rewards.Reward, sim: Simulation):
        #sim[self.last_agent_reward] = reward.agent_reward()
        #sim[self.last_assessment_reward] = reward.assessment_reward()
        pass

    def update_custom_properties(self, sim: Simulation, env) -> None:
        """ Calculates any custom properties which change every timestep. """
        pass

    def observe_first_state(self, sim: Simulation, env) -> np.ndarray:
        self._new_episode_init(sim, env)
        self.update_custom_properties(sim, env)
        state = [sim[prop] for prop in self.state_properties]
        self.last_state = state
        return state

    def _new_episode_init(self, sim: Simulation, env) -> None:
        pass
    '''
    #@abstractmethod
    def get_initial_conditions(self) -> Dict[Property, float]:
        return self.initial_states
    #    ...
    '''
    def get_state_space(self) -> gym.Space:
        state_lows = np.array([state_var.min for state_var in self.state_properties])
        state_highs = np.array([state_var.max for state_var in self.state_properties])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    def get_action_space(self) -> gym.Space:
        action_lows = np.array([act_var.min for act_var in self.action_properties])
        action_highs = np.array([act_var.max for act_var in self.action_properties])
        return gym.spaces.Box(low=action_lows, high=action_highs, dtype='float')


    @abstractmethod
    def _calculate_reward(self, state, last_state, done, sim, env):

        ...

    @abstractmethod
    def _is_terminal(self, sim: Simulation, env) -> bool:
        """ Determines whether the current episode should terminate.

        :param sim: the current simulation
        :return: True if the episode should terminate else False
        """
        ...

    @abstractmethod
    def _reward_terminal_override(self, reward: rewards.Reward, sim: Simulation, env) -> bool:
        """
        Determines whether a custom reward is needed, e.g. because
        a terminal condition is met.
        """
        ...

    @abstractmethod
    def init_reward(self, init, sim, env):
        ...


class AFHeadingControlTask(FlightTask):

    def init_reward(self, init, sim, env):
        self.head_target = init['head_target']
        env.set_property('pos_dx_m', 0)
        env.set_property('pos_dy_m', 0)
        self.start = (env.get_property('initial_latitude_geod_deg'), env.get_property('initial_longitude_geoc_deg'))


    def _calculate_reward(self, state, last_state, done, sim, env):
        dy = env.get_property('dist_travel_lat_m')
        dx = env.get_property('dist_travel_lon_m')
        if env.get_property('lat_geod_deg') < self.start[0]: dy = -dy
        if env.get_property('lng_geoc_deg') < self.start[1]: dx = -dx
        env.set_property('pos_dx_m', dx)
        env.set_property('pos_dy_m', dy)
        head = env.get_property('heading_rad')
        d1 = abs(head - self.head_target)
        d2 = abs (head- (2*math.pi + self.head_target))
        reward = -abs(min(d1,d2)) #Annäherung aus zwei Richtungen möglich
        if reward > -0.1: reward  = (0.1 + reward)*10
        return reward

    def _is_terminal(self, sim: Simulation, env) -> bool:
        if env.get_property('sim_time_s') > 200: return True
        return False

    def _reward_terminal_override(self, reward: rewards.Reward, sim: Simulation, env) -> bool:
        return reward

class FlyAlongLineTask(FlightTask):

    def init_reward(self, init, sim, env):
        env.set_property('pos_dx_m', 0)
        env.set_property('pos_dy_m', 0)
        self.start = (env.get_property('initial_latitude_geod_deg'), env.get_property('initial_longitude_geoc_deg'))
        pass

    
    def _calculate_reward(self, state, last_state, done, sim, env):
        dy = env.get_property('dist_travel_lat_m')
        dx = env.get_property('dist_travel_lon_m')
        if env.get_property('lat_geod_deg') < self.start[0]: dy = -dy
        if env.get_property('lng_geoc_deg') < self.start[1]: dx = -dx
        env.set_property('pos_dx_m', dx)
        env.set_property('pos_dy_m', dy)
        delta_lat = env.get_property('lat_geod_deg') - env.get_property('initial_latitude_geod_deg')
        delta_lon = env.get_property('lng_geoc_deg') - env.get_property('initial_longitude_geoc_deg')
        #reward = -abs(delta_lon)
        #if reward > -0.01: reward = -reward
        reward = abs(delta_lat)
        return reward

    def _is_terminal(self, sim: Simulation, env) -> bool:
        if env.get_property('sim_time_s') > 200: return True
        return False

    def _reward_terminal_override(self, reward: rewards.Reward, sim: Simulation, env) -> bool:
        return reward


class FindTargetTask_Heading(FlightTask):

    def _new_episode_init(self, sim, env):
        rng_min = self.init.get('target_min_range')
        rng_max = self.init.get('target_max_range')
        if rng_min and rng_max:
            r = random.uniform(rng_min, rng_max)
            deg = random.uniform(0, 2 * math.pi)
            self.target_dx = math.sin(deg)*r
            self.target_dy = math.cos(deg)*r
            self.target_head = random.uniform(0, 2 * math.pi)
        else:
            self.target_dx = self.init['target_dx_m']
            self.target_dy = self.init['target_dy_m']           
            self.target_head = self.init['target_heading']
        env.set_property('target_dx_m', self.target_dx)
        env.set_property('target_dy_m', self.target_dy)
        env.set_property('target_heading', self.target_head)


    def update_custom_properties(self, sim: Simulation, env) -> None:
        dy = env.get_property('dist_travel_lat_m')
        dx = env.get_property('dist_travel_lon_m')
        if env.get_property('lat_geod_deg') < self.start[0]: dy = -dy
        if env.get_property('lng_geoc_deg') < self.start[1]: dx = -dx
        env.set_property('pos_dx_m', dx)
        env.set_property('pos_dy_m', dy)
        self.alpha = xy2heading(dx,dy, self.target_dx,self.target_dy)
        env.set_property('alpha', self.alpha)
        self.dist = math.sqrt( (self.target_dx - dx)**2 + (self.target_dy - dy)**2 )
        env.set_property('target_distance', self.dist)

    def init_reward(self, init, sim, env):
        self.init = init
        self.max_dist_target = init['max_dist_target']
        self.max_steps = init['max_steps']
        self.step = 0
        self.start = (env.get_property('initial_latitude_geod_deg'), env.get_property('initial_longitude_geoc_deg'))


    def _calculate_reward(self, state, last_state, done, sim, env):
        self.step += 1
        if self.dist < self.max_dist_target:
            heading_dif = env.get_property('heading_rad') - env.get_property('target_heading')
            heading_dif = limit_angle(heading_dif, 2* math.pi)
            reward = 10-heading_dif*1
            self.terminal = True
            print('FindTarget: Succes')
        elif self.terminal:
            reward = self.alpha - env.get_property('heading_rad')
            reward = -abs(limit_angle(reward, 2* math.pi))            
            print('Distant am Schluss: {:.0f}m'.format(self.dist))
        else:
            reward = self.alpha - env.get_property('heading_rad')
            reward = -abs(limit_angle(reward, 2* math.pi))
        return reward

    def _is_terminal(self, sim: Simulation, env) -> bool:
        if self.step > self.max_steps: self.terminal = True
        return self.terminal

    def _reward_terminal_override(self, reward: rewards.Reward, sim: Simulation, env) -> bool:
        return reward

class FindTargetTask(FindTargetTask_Heading):

    def _calculate_reward(self, state, last_state, done, sim, env): 
        self.step += 1
        if self.dist < self.max_dist_target:
            heading_dif = env.get_property('heading_rad') - env.get_property('target_heading')
            heading_dif = limit_angle(heading_dif, 2* math.pi)
            reward = 10-heading_dif*1
            self.terminal = True
            print('FindTarget: Succes')
        elif self.terminal:
            reward = - self.dist/10000
            print('Distant am Schluss: {:.0f}m'.format(self.dist))
        else:
            reward = -self.dist/10000
        return reward


class FindTargetMapTask(FlightTask):
    
    def init_reward(self, init, sim, env):
        print('INIT:',init)
        env.set_property('target_lat_geod_deg', init['target_lat'])
        env.set_property('target_lng_geoc_deg', init['target_lng'])
        env.set_property('target_heading', init['target_head'])
        self.max_dist_target = init['max_dist_target']
    
    def _calculate_reward(self, state, last_state, done, sim, env):
        delta_lat = env.get_property('lat_geod_deg') - env.get_property('target_lat_geod_deg')
        delta_lng = env.get_property('lng_geoc_deg') - env.get_property('target_lng_geoc_deg')
        #dist = math.sqrt(delta_lat**2+delta_lng**2)
        dist = delta_lat**2+delta_lng**2
        if math.sqrt(dist) < self.max_dist_target:
            heading_dif = env.get_property('heading_rad') - env.get_property('target_heading')
            heading_dif = limit_angle(heading_dif, 2* math.pi)
            reward = 100-heading_dif*10
            self.terminal = True
            print('FindTargetMap: Succes')
        else:
            reward = - dist
        return reward

    def _is_terminal(self, sim: Simulation, env) -> bool:
        #if env.get_property('sim_time_s') > 200: return True
        return self.terminal

    def _reward_terminal_override(self, reward: rewards.Reward, sim: Simulation, env) -> bool:
        return reward

def limit_angle( angle, max_angle):
        #limitiert den Winkel auf einen +/-halben Kreis, z.B. auf max. -180°..180°
        half = max_angle / 2
        return (angle + half ) % max_angle - half


def xy2heading(x1,y1,x2,y2):
    if ((x1-x2) == 0) and ((y1-y2) > 0):
        return math.pi
    elif ((x1-x2) == 0) and ((y1-y2) < 0):
        return 0
    elif ((x1-x2) > 0) and ((y1-y2) == 0):
        return math.pi*1.5
    elif ((x1-x2) < 0) and ((y1-y2) == 0):
        return math.pi*0.5
    elif ((x1-x2) == 0) and ((y1-y2) == 0):
        return 0 #eigentlich keine Winkelberechnung möglich, wenn beide Punkte identisch
    kw = math.atan((y1-y2)/(x1-x2))
    if x2>x1 and y2>y1:
        return math.pi/2-kw
    elif x2>x1 and y2<y1:
        return math.pi/2-kw
    elif x2<x1 and y2<y1:
        return 1.5*math.pi-kw
    elif x2<x1 and y2>y1:
        return 1.5*math.pi-kw

class Shaping(enum.Enum):
    STANDARD = 'STANDARD'
    EXTRA = 'EXTRA'
    EXTRA_SEQUENTIAL = 'EXTRA_SEQUENTIAL'

task_dict = {
    'HeadingControl': AFHeadingControlTask,
    'FlyAlongLine': FlyAlongLineTask,
    'FindTarget': FindTargetTask,
    'FindTargetHead': FindTargetTask_Heading
}
