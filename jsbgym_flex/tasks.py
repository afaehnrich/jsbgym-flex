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
    def task_step(self, sim: Simulation, action: Sequence[float], sim_steps: int) \
            -> Tuple[np.ndarray, float, bool, Dict]:
        ...

    @abstractmethod
    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        ...

    @abstractmethod
    def get_initial_conditions(self) -> Optional[Dict[Property, float]]:
        ...

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
                                                                   
    def __init__(self, action_properties, state_properties, initial_states, init_sequence,  pid_controls, 
                simulation_dt, controller_dt, observation_dt, debug: bool = False) -> None:
        self.action_variables = tuple(action_properties)
        self.state_variables = tuple(state_properties)
        self.last_state = None
        self._make_state_class()
        self.initial_states = initial_states
        self.init_sequence = init_sequence
        self.debug = debug
        self.pid_controls = pid_controls
        self.simulation_dt = simulation_dt
        self.controller_dt = controller_dt
        self.observation_dt = observation_dt
        if self.simulation_dt is None: raise ValueError('No simulation Frequency given.')
        if self.observation_dt is None: raise ValueError('No observation Frequency given.')
        if len(self.pid_controls) > 0: 
            if self.controller_dt is None:
                raise ValueError('Automatic controllers present, but no'
                                ' controller Frequency given.')
        else: self.controller_dt = self.observation_dt
        if not (self.simulation_dt <= self.controller_dt):
            raise ValueError('Condition not met: step width simulation <= step width controller')
        if not (self.simulation_dt <= self.observation_dt):
            raise ValueError('Condition not met: step width simulation <= step width observation')
        self.reset()

    def reset(self):
        self.controller_timer = self.controller_dt
        self.observation_timer = self.observation_dt



    def _make_state_class(self) -> None:
        """ Creates a namedtuple for readable State data """
        # get list of state property names, containing legal chars only
        legal_attribute_names = [prop.get_legal_name() for prop in
                                 self.state_variables]
        self.State = namedtuple('State', legal_attribute_names)

    def task_step(self, sim: Simulation, action: Sequence[float], env) \
            -> Tuple[NamedTuple, float, bool, Dict]:
        # input actions
        for prop, command in zip(self.action_variables, action):
            sim[prop] = command

        # run simulation
        #for _ in range(sim_steps):
        #    sim.run()
        while self.observation_timer > 0:
            sim.run()
            self.controller_timer -= self.simulation_dt
            self.observation_timer -= self.simulation_dt
            if self.controller_timer <= 0:
                self._run_pid_controls(self.pid_controls, sim)
                self.controller_timer += self.controller_dt
        self.observation_timer += self.observation_dt
        self._update_custom_properties(sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
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

    def _run_pid_controls(self, pid_controls, sim):
        for (pid, input, output, target) in pid_controls.values():
            sim[output] = pid(sim[input], sim[target])

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

    def _update_custom_properties(self, sim: Simulation) -> None:
        """ Calculates any custom properties which change every timestep. """
        pass

    def observe_first_state(self, sim: Simulation) -> np.ndarray:
        self._new_episode_init(sim)
        self._update_custom_properties(sim)
        state = self.State(*(sim[prop] for prop in self.state_variables))
        self.last_state = state
        return state

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

    def get_state_space(self) -> gym.Space:
        state_lows = np.array([state_var.min for state_var in self.state_variables])
        state_highs = np.array([state_var.max for state_var in self.state_variables])
        return gym.spaces.Box(low=state_lows, high=state_highs, dtype='float')

    def get_action_space(self) -> gym.Space:
        action_lows = np.array([act_var.min for act_var in self.action_variables])
        action_highs = np.array([act_var.max for act_var in self.action_variables])
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

class HeadingControlTask(FlightTask):
    """
    A task in which the agent must perform steady, level flight maintaining its
    initial heading.
    """
    THROTTLE_CMD = 0.8
    MIXTURE_CMD = 0.8
    INITIAL_HEADING_DEG = 270
    DEFAULT_EPISODE_TIME_S = 60.
    ALTITUDE_SCALING_FT = 150
    TRACK_ERROR_SCALING_DEG = 8
    ROLL_ERROR_SCALING_RAD = 0.15  # approx. 8 deg
    SIDESLIP_ERROR_SCALING_DEG = 3.
    MIN_STATE_QUALITY = 0.0  # terminate if state 'quality' is less than this
    MAX_ALTITUDE_DEVIATION_FT = 1000  # terminate if altitude error exceeds this
    target_track_deg = BoundedProperty('target/track-deg', 'desired heading [deg]',
                                       prp.heading_deg.min, prp.heading_deg.max)
    track_error_deg = BoundedProperty('error/track-error-deg',
                                      'error to desired track [deg]', -180, 180)
    altitude_error_ft = BoundedProperty('error/altitude-error-ft',
                                        'error to desired altitude [ft]',
                                        prp.altitude_sl_ft.min,
                                        prp.altitude_sl_ft.max)
    action_variables = (prp.aileron_cmd, prp.elevator_cmd, prp.rudder_cmd)

    def __init__(self, shaping_type: Shaping, step_frequency_hz: float, aircraft: Aircraft,
                 episode_time_s: float = DEFAULT_EPISODE_TIME_S, positive_rewards: bool = True):
        """
        Constructor.

        :param step_frequency_hz: the number of agent interaction steps per second
        :param aircraft: the aircraft used in the simulation
        """
        self.max_time_s = episode_time_s
        episode_steps = math.ceil(self.max_time_s * step_frequency_hz)
        self.steps_left = BoundedProperty('info/steps_left', 'steps remaining in episode', 0,
                                          episode_steps)
        self.aircraft = aircraft
        self.extra_state_variables = (self.altitude_error_ft, prp.sideslip_deg,
                                      self.track_error_deg, self.steps_left)
        self.state_variables = FlightTask.base_state_variables + self.extra_state_variables
        self.positive_rewards = positive_rewards
        assessor = self.make_assessor(shaping_type)
        super().__init__(assessor)

    def make_assessor(self, shaping: Shaping) -> assessors.AssessorImpl:
        base_components = self._make_base_reward_components()
        shaping_components = ()
        return self._select_assessor(base_components, shaping_components, shaping)

    def _make_base_reward_components(self) -> Tuple[rewards.RewardComponent, ...]:
        base_components = (
            rewards.AsymptoticErrorComponent(name='altitude_error',
                                             prop=self.altitude_error_ft,
                                             state_variables=self.state_variables,
                                             target=0.0,
                                             is_potential_based=False,
                                             scaling_factor=self.ALTITUDE_SCALING_FT),
            rewards.AsymptoticErrorComponent(name='travel_direction',
                                             prop=self.track_error_deg,
                                             state_variables=self.state_variables,
                                             target=0.0,
                                             is_potential_based=False,
                                             scaling_factor=self.TRACK_ERROR_SCALING_DEG),
            # add an airspeed error relative to cruise speed component?
        )
        return base_components

    def _select_assessor(self, base_components: Tuple[rewards.RewardComponent, ...],
                         shaping_components: Tuple[rewards.RewardComponent, ...],
                         shaping: Shaping) -> assessors.AssessorImpl:
        if shaping is Shaping.STANDARD:
            return assessors.AssessorImpl(base_components, shaping_components,
                                          positive_rewards=self.positive_rewards)
        else:
            wings_level = rewards.AsymptoticErrorComponent(name='wings_level',
                                                           prop=prp.roll_rad,
                                                           state_variables=self.state_variables,
                                                           target=0.0,
                                                           is_potential_based=True,
                                                           scaling_factor=self.ROLL_ERROR_SCALING_RAD)
            no_sideslip = rewards.AsymptoticErrorComponent(name='no_sideslip',
                                                           prop=prp.sideslip_deg,
                                                           state_variables=self.state_variables,
                                                           target=0.0,
                                                           is_potential_based=True,
                                                           scaling_factor=self.SIDESLIP_ERROR_SCALING_DEG)
            potential_based_components = (wings_level, no_sideslip)

        if shaping is Shaping.EXTRA:
            return assessors.AssessorImpl(base_components, potential_based_components,
                                          positive_rewards=self.positive_rewards)
        elif shaping is Shaping.EXTRA_SEQUENTIAL:
            altitude_error, travel_direction = base_components
            # make the wings_level shaping reward dependent on facing the correct direction
            dependency_map = {wings_level: (travel_direction,)}
            return assessors.ContinuousSequentialAssessor(base_components, potential_based_components,
                                                          potential_dependency_map=dependency_map,
                                                          positive_rewards=self.positive_rewards)

    def get_initial_conditions(self) -> Dict[Property, float]:
        extra_conditions = {prp.initial_u_fps: self.aircraft.get_cruise_speed_fps(),
                            prp.initial_v_fps: 0,
                            prp.initial_w_fps: 0,
                            prp.initial_p_radps: 0,
                            prp.initial_q_radps: 0,
                            prp.initial_r_radps: 0,
                            prp.initial_roc_fpm: 0,
                            prp.initial_heading_deg: self.INITIAL_HEADING_DEG,
                            }
        return {**self.base_initial_conditions, **extra_conditions}

    def _update_custom_properties(self, sim: Simulation) -> None:
        self._update_track_error(sim)
        self._update_altitude_error(sim)
        self._decrement_steps_left(sim)

    def _update_track_error(self, sim: Simulation):
        v_north_fps, v_east_fps = sim[prp.v_north_fps], sim[prp.v_east_fps]
        track_deg = prp.Vector2(v_east_fps, v_north_fps).heading_deg()
        target_track_deg = sim[self.target_track_deg]
        error_deg = utils.reduce_reflex_angle_deg(track_deg - target_track_deg)
        sim[self.track_error_deg] = error_deg

    def _update_altitude_error(self, sim: Simulation):
        altitude_ft = sim[prp.altitude_sl_ft]
        target_altitude_ft = self._get_target_altitude()
        error_ft = altitude_ft - target_altitude_ft
        sim[self.altitude_error_ft] = error_ft

    def _decrement_steps_left(self, sim: Simulation):
        sim[self.steps_left] -= 1

    def _is_terminal(self, sim: Simulation) -> bool:
        # terminate when time >= max, but use math.isclose() for float equality test
        terminal_step = sim[self.steps_left] <= 0
        state_quality = sim[self.last_assessment_reward]
        state_out_of_bounds = state_quality < self.MIN_STATE_QUALITY  # TODO: issues if sequential?
        return terminal_step or state_out_of_bounds or self._altitude_out_of_bounds(sim)

    def _altitude_out_of_bounds(self, sim: Simulation) -> bool:
        altitude_error_ft = sim[self.altitude_error_ft]
        return abs(altitude_error_ft) > self.MAX_ALTITUDE_DEVIATION_FT

    def _get_out_of_bounds_reward(self, sim: Simulation) -> rewards.Reward:
        """
        if aircraft is out of bounds, we give the largest possible negative reward:
        as if this timestep, and every remaining timestep in the episode was -1.
        """
        reward_scalar = (1 + sim[self.steps_left]) * -1.
        return RewardStub(reward_scalar, reward_scalar)

    def _reward_terminal_override(self, reward: rewards.Reward, sim: Simulation) -> rewards.Reward:
        if self._altitude_out_of_bounds(sim) and not self.positive_rewards:
            # if using negative rewards, need to give a big negative reward on terminal
            return self._get_out_of_bounds_reward(sim)
        else:
            return reward

    def _new_episode_init(self, sim: Simulation) -> None:
        super()._new_episode_init(sim)
        sim.set_throttle_mixture_controls(self.THROTTLE_CMD, self.MIXTURE_CMD)
        sim[self.steps_left] = self.steps_left.max
        sim[self.target_track_deg] = self._get_target_track()

    def _get_target_track(self) -> float:
        # use the same, initial heading every episode
        return self.INITIAL_HEADING_DEG

    def _get_target_altitude(self) -> float:
        return self.INITIAL_ALTITUDE_FT

    def get_props_to_output(self) -> Tuple:
        return (prp.u_fps, prp.altitude_sl_ft, self.altitude_error_ft, self.target_track_deg,
                self.track_error_deg, prp.roll_rad, prp.sideslip_deg, self.last_agent_reward,
                self.last_assessment_reward, self.steps_left)


class TurnHeadingControlTask(HeadingControlTask):
    """
    A task in which the agent must make a turn from a random initial heading,
    and fly level to a random target heading.
    """

    def get_initial_conditions(self) -> [Dict[Property, float]]:
        initial_conditions = super().get_initial_conditions()
        random_heading = random.uniform(prp.heading_deg.min, prp.heading_deg.max)
        initial_conditions[prp.initial_heading_deg] = random_heading
        return initial_conditions

    def _get_target_track(self) -> float:
        # select a random heading each episode
        return random.uniform(self.target_track_deg.min,
                              self.target_track_deg.max)
