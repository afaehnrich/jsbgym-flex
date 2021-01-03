import gym
import numpy as np
from jsbgym_flex.tasks import Shaping, task_dict
from jsbgym_flex.simulation import Simulation
from jsbgym_flex.visualiser import FigureVisualiser, FlightGearVisualiser
from jsbgym_flex.aircraft import Aircraft, cessna172P, aircrafts
from typing import Type, Tuple, Dict
from jsbgym_flex.properties import BoundedProperty, Property
from jsbgym_flex.pid import PID_angle
from collections import namedtuple  

#from jsbgym_flex.configuration import Configuration


class JsbSimEnv(gym.Env):
    metadata = {'render.modes': ['human', 'flightgear']}

    Task = namedtuple('Task', ['task', 'state_properties', 'action_properties', \
        'observation_space', 'action_space', 'actor', 'critic'])

    def __init__(self, cfg: dict, #task_type: Type[HeadingControlTask],
                 shaping: Shaping=Shaping.STANDARD):
        self.cfg = cfg or {}
        self.cfgenv = cfg.get('environment')
        #self.cfgtask = next(iter(cfg.get('tasks').values())) Für Multi-Agent (TODO)
        self.properties = self._load_properties(cfg.get('properties'))
        self.initial_states = self._get_initial_states(self.cfgenv.get('initial_state'), self.properties)
        self.init_sequence = self._get_initial_states(self.cfgenv.get('init_sequence'), self.properties)
        active_pids = self.cfgenv.get('pids')
        self.pid_controls = self._load_pids(cfg.get('pid'), active_pids, self.properties)
        print (self.pid_controls)
        self._init_timers()
        self.jsbsim_dir = self.cfgenv.get('path_jsbsim') or ''
        self.aircraft = aircrafts[self.cfgenv.get('aircraft')]
        #self.task = task_type(shaping, agent_interaction_freq, self.aircraft)
        self.tasks = [self._init_task(cfg) for cfg in self.cfg.get('tasks').values()]
        self.action_space = [task.action_space for task in self.tasks]
        #init_conditions = self.task.get_initial_conditions()
        self.sim = self._init_new_sim(self.simulation_dt , self.aircraft, self.initial_states) # init_conditions)
        #self.sim_steps_per_agent_step: int = self.simulation_freq  // self.observation_freq
        # set Space objects
        # set visualisation objects
        self.visualisers = self._load_visualisers(cfg)
        self.step_delay = None
    
    def _init_task(self, cfgtask):
        action_properties = tuple(self._choose_properties(cfgtask.get('actions'), self.properties))
        state_properties = tuple(self._choose_properties(cfgtask.get('states'), self.properties))
        actor = cfgtask.get('actor')
        critic = cfgtask.get('critic')
        task_type = task_dict.get(cfgtask.get('name'))
        task = task_type( action_properties, state_properties, debug = False)
        task.init_reward(self.cfgenv.get('task_init'))
        observation_space:gym.spaces.Box = task.get_state_space()
        action_space: gym.spaces.Box = task.get_action_space()
        ret_task = self.Task(task, state_properties,action_properties, \
            observation_space, action_space , actor , critic)
        return ret_task

    def step(self, action_n: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        for task, action in zip(self.tasks, action_n):
            if not (action.shape == task.action_space.shape):
                raise ValueError('mismatch between action and action space size')
            task.task.take_action(self.sim, action)
        while self.observation_timer > 0:
            self.sim.run()
            self.controller_timer -= self.simulation_dt
            self.observation_timer -= self.simulation_dt
            if self.controller_timer <= 0:
                self._run_pid_controls(self.pid_controls, self.sim)
                self.controller_timer += self.controller_dt
        self.observation_timer += self.observation_dt
        for task in self.tasks:
            task.task.update_custom_properties(self.sim)
        state, reward, done, info = zip(*[task.task.get_observation(self.sim, action, self) for task in self.tasks])
        #state, reward, done, info = self.task.task_step(self.sim, action, self)#, self.sim_steps_per_agent_step)
        return np.array(state), reward, done, info

    def reset(self):
        #init_conditions = self.task.get_initial_conditions()
        for task in self.tasks: task.task.reset()
        self.sim.reinitialise(self.initial_states)
        for v in self.visualisers:
            v.reset()
        self.controller_timer = self.controller_dt
        self.observation_timer = self.observation_dt
        for (prop, val) in self.init_sequence.items():
            self.sim[prop] = val
        #sim.start_engines()
        #sim.raise_landing_gear()
        #sim.set_throttle_mixture_controls(0.8, 0.8)
        state = [task.task.observe_first_state(self.sim) for task in self.tasks]
        return np.array(state)

    def _init_timers(self):
        self.simulation_dt = self.cfgenv.get('simulation_stepwidth')
        self.controller_dt = self.cfgenv.get('controller_stepwidth')
        self.observation_dt = self.cfgenv.get('observation_stepwidth')
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
        self.controller_timer = self.controller_dt
        self.observation_timer = self.observation_dt
   
        

    def _init_new_sim(self, dt, aircraft, initial_conditions):
        vis = self.cfg.get('visualiser')
        allow_fg = vis.get('flightgear') is not None
        return Simulation( jsbsim_dir=self.jsbsim_dir, sim_dt=dt, aircraft=aircraft,
                          init_conditions=initial_conditions, allow_flightgear_output = allow_fg)

    def _load_properties(self, cfgprop):
        cfgprop = cfgprop
        properties = {}
        for prop, attr in cfgprop.items():
            name = attr.get('name')
            desc = attr.get('desc')
            min = attr.get('min')
            max = attr.get('max')
            if min is None or max is None:
                p =  Property(name, desc)
            else:
                p = BoundedProperty(name, desc, min, max)
            properties.update({prop: p})
        return properties

    def _choose_properties(self, prop_names, properties):
        prop_names = prop_names
        chosen = []
        for name in prop_names:
            p =  properties.get(name)
            if p is None: raise Exception('choose_property: Property "{}" not found.'.format(name))
            chosen.append(p)
        return chosen

    def get_property(self, name):
        p = self.properties.get(name)
        if p is None: raise Exception('get_property: Property "{}" not found.'.format(name))
        return self.sim[p]
    
    def set_property(self, name, value):
        p = self.properties.get(name)
        if p is None: raise Exception('set_property: Property "{}" not found.'.format(name))
        self.sim[p] = value

    def _get_initial_states(self, cfgstates, properties):
        cfgstates = cfgstates or {}
        states = {}
        for name,value in cfgstates.items():
            p =  properties.get(name)
            if p is None: raise Exception('_get_initial_state: Property "{}" not found.'.format(name))
            states.update({p:value})
        return states

    def _load_pids(self,cfgpids, active_pids, properties):
        cfgpids = cfgpids or {}
        pids = {}
        for name in active_pids:
            par = cfgpids.get(name)
            if par is None: raise Exception('_load_pids: PID"{}" not found.'.format(name))
        #for name, par in cfgpids.items():
        #    if not name in active_pids: continue
            type = par['type']
            if type == 'pid_angle':
                p = PID_angle(name, par.get('p'), par.get('i'), par.get('d'),  0,
                              par.get('angle_max'), par.get('out_min'), par.get('out_max'), par.get('anti_windup'))
                input = self.properties[par.get('input')]
                output = self.properties[par.get('output')]
                target = self.properties[par.get('target')]
                pids.update({name:(p, input, output, target)})
        return pids

    def _run_pid_controls(self, pid_controls, sim):
        for (pid, input, output, target) in pid_controls.values():
            sim[output] = pid(sim[input], sim[target])


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
