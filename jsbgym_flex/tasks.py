import gym
import numpy as np
import random
import types
import math
import enum
import warnings
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
                                                                   
    def __init__(self, action_properties, state_properties, #initial_states, init_sequence,  pid_controls, 
                #simulation_dt, controller_dt, observation_dt, 
                debug: bool = False) -> None:
        self.action_properties = tuple(action_properties)
        self.state_properties = tuple(state_properties)
        self.last_state = None
        self._make_state_class()
        #self.initial_states = initial_states
        #self.init_sequence = init_sequence
        self.debug = debug
        #self.pid_controls = pid_controls
        self.reset()

    def reset(self):
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
        state = self.State(*(sim[prop] for prop in self.state_properties))
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

    def update_custom_properties(self, sim: Simulation) -> None:
        """ Calculates any custom properties which change every timestep. """
        pass

    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        #self._new_episode_init(sim)
        self.update_custom_properties(sim)
        state = self.State(*(sim[prop] for prop in self.state_properties))
        self.last_state = state
        return state

    '''
    def _new_episode_init(self, sim: Simulation) -> None:
        """
        This method is called at the start of every episode. It is used to set
        the value of any controls or environment properties not already defined
        in the task's initial conditions.

        By default it simply starts the aircraft engines.
        """
        for (prop, val) in self.init_sequence.items():
            sim[prop] = val
        #sim.start_engines()
        #sim.raise_landing_gear()
        #sim.set_throttle_mixture_controls(0.8, 0.8)
        self._store_reward(RewardStub(1.0, 1.0), sim)

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
    def init_reward(self, init):
        ...


class AFHeadingControlTask(FlightTask):

    def init_reward(self, init):
        self.head_target = init['head_target']

    def _calculate_reward(self, state, last_state, done, sim, env):
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

    def init_reward(self, init):
        pass

    
    def _calculate_reward(self, state, last_state, done, sim, env):
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

class FindTargetTask(FlightTask):

    def init_reward(self, init):
        pass

    
    def _calculate_reward(self, state, last_state, done, sim, env):
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

class Shaping(enum.Enum):
    STANDARD = 'STANDARD'
    EXTRA = 'EXTRA'
    EXTRA_SEQUENTIAL = 'EXTRA_SEQUENTIAL'

task_dict = {
    'HeadingControl': AFHeadingControlTask,
    'FlyAlongLine': FlyAlongLineTask
}
