import gym
import numpy as np
from gym_jsbsim_simple.tasks import Shaping, HeadingControlTask
from gym_jsbsim_simple.simulation import Simulation
from gym_jsbsim_simple.visualiser import FigureVisualiser, FlightGearVisualiser
from gym_jsbsim_simple.aircraft import Aircraft, cessna172P, aircrafts
from typing import Type, Tuple, Dict
from gym_jsbsim_simple.properties import BoundedProperty, Property
from gym_jsbsim_simple.pid import PID_angle

#from gym_jsbsim_simple.configuration import Configuration


class JsbSimEnv(gym.Env):
    #metadata = {'render.modes': ['human', 'flightgear']}

    def __init__(self, cfg: dict, task_type: Type[HeadingControlTask],
                 shaping: Shaping=Shaping.STANDARD):
        self.cfg = cfg or {}
        self.cfgenv = cfg.get('environment') or {}
        self.properties = self._load_properties(cfg.get('properties'))
        action_properties = self._choose_properties(self.cfgenv.get('actions'), self.properties)
        observation_properties = self._choose_properties(self.cfgenv.get('observations'), self.properties)
        initial_states = self._get_initial_states(self.cfgenv.get('initial_state'), self.properties)
        init_sequence = self._get_initial_states(self.cfgenv.get('init_sequence'), self.properties)
        self.pid_controls = self._load_pids(cfg.get('pid'), self.properties)
        simulation_dt = self.cfgenv.get('simulation_stepwidth')
        controller_dt = self.cfgenv.get('controller_stepwidth')
        observation_dt = self.cfgenv.get('observation_stepwidth')
        self.jsbsim_dir = self.cfgenv.get('path_jsbsim') or ''
        self.aircraft = aircrafts[self.cfgenv.get('aircraft')]
        #self.task = task_type(shaping, agent_interaction_freq, self.aircraft)
        self.task = task_type( action_properties, observation_properties, initial_states, init_sequence,
                                self.pid_controls, simulation_dt, controller_dt, observation_dt,
                                debug = False)
        init_conditions = self.task.get_initial_conditions()
        self.sim = self._init_new_sim(simulation_dt , self.aircraft, init_conditions)
        #self.sim_steps_per_agent_step: int = self.simulation_freq  // self.observation_freq
        # set Space objects
        self.observation_space: gym.spaces.Box = self.task.get_state_space()
        self.action_space: gym.spaces.Box = self.task.get_action_space()
        # set visualisation objects
        self.visualisers = self._load_visualisers(cfg)
        self.step_delay = None

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        if not (action.shape == self.action_space.shape):
            raise ValueError('mismatch between action and action space size')

        state, reward, done, info = self.task.task_step(self.sim, action)#, self.sim_steps_per_agent_step)
        return np.array(state), reward, done, info

    def reset(self):
        init_conditions = self.task.get_initial_conditions()
        self.task.reset()
        self.sim.reinitialise(init_conditions)
        for v in self.visualisers:
            v.reset()
        state = self.task.observe_first_state(self.sim)
        return np.array(state)

    def _init_new_sim(self, dt, aircraft, initial_conditions):
        vis = self.cfg.get('visualiser') or {}
        allow_fg = vis.get('flightgear') is not None
        return Simulation( jsbsim_dir=self.jsbsim_dir, sim_dt=dt, aircraft=aircraft,
                          init_conditions=initial_conditions, allow_flightgear_output = allow_fg)

    def _load_properties(self, cfgprop):
        cfgprop = cfgprop or {}
        properties = {}
        for prop, attr in cfgprop.items():
            name = attr.get('name')
            desc = attr.get('desc')
            min = attr.get('min')
            max = attr.get('max')
            if min and max:
                p = BoundedProperty(name, desc, min, max)
            else:
                p =  Property(name, desc)
            properties.update({prop: p})
        return properties

    def _choose_properties(self, prop_names, properties):
        prop_names = prop_names or []
        chosen = []
        for name in prop_names:
            p =  properties.get(name)
            if p: chosen.append(p)
        return chosen

    def get_property(self, name):
        p = self.properties.get(name)
        if p is None: return None
        return self.sim[p]
    
    def set_property(self, name, value):
        p = self.properties.get(name)
        if p is None: return
        self.sim[p] = value

    def _get_initial_states(self, cfgstates, properties):
        cfgstates = cfgstates or {}
        states = {}
        for name,value in cfgstates.items():
            p =  properties.get(name)
            if not p: continue
            states.update({p:value})
        return states

    def _load_pids(self,cfgpids, properties):
        cfgpids = cfgpids or {}
        pids = {}
        for name, par in cfgpids.items():
            type = par['type']
            if type == 'pid_angle':
                p = PID_angle(name, par.get('p'), par.get('i'), par.get('d'),  0,
                              par.get('angle_max'), par.get('out_min'), par.get('out_max'), par.get('anti_windup'))
                input = self.properties[par.get('input')]
                output = self.properties[par.get('output')]
                target = self.properties[par.get('target')]
                pids.update({name:(p, input, output, target)})
        return pids

    def _load_visualisers(self, cfg):
        cfgvis =cfg.get('visualiser') or {}
        if cfgvis.get('enable') == False:
            return []
        visualisers = []
        for key, item in cfgvis.items() or {}:
            if key == 'figure':
                visualiser =FigureVisualiser(item, self.task.get_props_to_output())
                visualisers.append(visualiser)
            elif key == 'flightgear':
                visualiser = FlightGearVisualiser(item, self.aircraft)
                visualisers.append(visualiser)
        return visualisers

    def render(self, mode='flightgear', flightgear_blocking=True):
        for v in self.visualisers:
            v.plot(self.sim)

    def close(self):
        if self.sim:
            self.sim.close()
        for v in self.visualisers:
            v.close()
        
    def seed(self, seed=None):
        """
        Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        gym.logger.warn("Could not seed environment %s", self)
        return


class NoFGJsbSimEnv(JsbSimEnv):
    """
    An RL environment for JSBSim with rendering to FlightGear disabled.

    This class exists to be used for training agents where visualisation is not
    required. Otherwise, restrictions in JSBSim output initialisation cause it
    to open a new socket for every single episode, eventually leading to
    failure of the network.
    """
    metadata = {'render.modes': ['human']}

    def _init_new_sim(self, dt: float, aircraft: Aircraft, initial_conditions: Dict):
        return Simulation(sim_frequency_hz=dt, jsbsim_dir=self.jsbsim_dir,
                          aircraft=aircraft,
                          init_conditions=initial_conditions,
                          allow_flightgear_output=False)

    def render(self, mode='human', flightgear_blocking=True):
        if mode == 'flightgear':
            raise ValueError('flightgear rendering is disabled for this class')
        else:
            super().render(mode, flightgear_blocking)
