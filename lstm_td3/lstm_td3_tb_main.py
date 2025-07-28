from copy import deepcopy
import numpy as np
# import pybullet_envs    # To register tasks in PyBullet
import gymnasium as gym
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from lstm_td3.utils.logx import EpochLogger, setup_logger_kwargs, colorize
import itertools
from lstm_td3.env_wrapper.pomdp_wrapper import POMDPWrapper
# from lstm_td3.env_wrapper.env import make_bullet_task
import os
import os.path as osp
import json
from collections import namedtuple
from tqdm import tqdm

# Add beam environment imports
import contextlib
import pickle
import matplotlib.pyplot as plt
import imageio
import blond.utils.bmath as bm
from blond.beam.beam import Beam, Proton
from blond.beam.distributions import matched_from_distribution_function
from blond.beam.profile import CutOptions, Profile
from blond.input_parameters.rf_parameters import RFStation
from blond.input_parameters.ring import Ring
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.impedances.impedance import (InducedVoltageFreq, InductiveImpedance,
                                        TotalInducedVoltage)
from blond.impedances.impedance_sources import InputTable
from scipy.constants import c
from gymnasium import spaces


# FIXES IMPLEMENTED:
# 1. Fixed the issue with the gymnasium environment (used an older version of gymnasium)
# 2. Fixed the issue with the logger kwargs
# 3. Fixed the issue with the data directory
# 4. Fixed the issue with the experiment name


# ===== BEAM ENVIRONMENT FUNCTIONS AND CLASSES =====

def load_magnetic_field(b_choice: int):
    """Load and process magnetic field program"""
    if b_choice == 1:
        sync_momentum = np.load('/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/programs/unnamed/Bfield.npy')
        t_arr = sync_momentum[0].flatten()  # Time array [s]
        B_field = sync_momentum[1].flatten() * 1e-4  # Convert to Tesla

    elif b_choice == 2:
        sync_momentum = np.load('/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/programs/TOF/Bfield.npy')
        t_arr = sync_momentum[0].flatten()  # Time array [s]
        B_field = sync_momentum[1].flatten() * 1e-4  # Convert to Tesla

    # Trim to injection/extraction window
    inj_idx = np.where(t_arr <= 275)[0][-1]
    ext_idx = np.where(t_arr >= 805)[0][0]

    return {
        't_arr': (t_arr[inj_idx:ext_idx] - t_arr[inj_idx]) / 1e3,
        'B_field': B_field[inj_idx:ext_idx],
        'sync_momentum': B_field[inj_idx:ext_idx] * 8.239 * c
    }

def initialize_accelerator_parameters():
    """Return fixed machine parameters"""
    return {
        'radius': 25.0,          # Machine radius [m]
        'bend_radius': 8.239,    #Bending radius [m]
        'gamma_transition': 4.4, # Transition gamma
        'n_particles': 0.9e13,   # Number of particles
        'n_macroparticles': 1e6, # Macro-particle count
        'harmonic_numbers': [1, 2], # RF harmonics
        'n_rf_systems': 2        # Number of RF systems
    }

def calculate_bdot_parameters(B_data):
    """Calculate B-dot and key indices"""
    B_dot = np.gradient(B_data['B_field'], B_data['t_arr'])
    ind_max = np.argmax(B_dot)
    return [0, ind_max // 2, (2 * ind_max) // 3, ind_max]

def simulate_beam_profile_PSB(params, phi, random_phase_shift, noise=None, imp=True):
    """Parallelizable beam profile simulation function"""
    # Suppress any stdout/stderr during the simulation
    with open(os.devnull, 'w') as devnull, \
         contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull):

        B_choice, total_voltage, v_ratio, B_ind, filling_factor = params
        bm.use_precision('single')

        # Calculate actual B-dot value from index
        B_data = load_magnetic_field(B_choice)
        v1 = total_voltage / (1 + v_ratio)
        v2 = v1 * v_ratio
        B_dot = np.gradient(B_data['B_field'], B_data['t_arr'])[B_ind]

        # --- MAIN SIMULATION CODE ---
        machine_params = initialize_accelerator_parameters()
        ring = Ring(
            2 * np.pi * machine_params['radius'],
            1 / machine_params['gamma_transition'] ** 2,
            (B_data['t_arr'][B_ind:B_ind+2], B_data['sync_momentum'][B_ind:B_ind+2]),
            Proton(),
            bending_radius=machine_params['bend_radius']
        )
        rf_station = RFStation(
            ring,
            machine_params['harmonic_numbers'],
            [v1, v2],
            [np.pi, np.pi + phi + random_phase_shift],
            machine_params['n_rf_systems']
        )
        beam = Beam(ring, machine_params['n_macroparticles'], machine_params['n_particles'])
        n_slices = 1000
        profile = Profile(
            beam,
            CutOptions(cut_left=0, cut_right=ring.t_rev[0], n_slices=n_slices)
        )

        # Compute synchronous phase and emittance
        phi_s_app = ring.delta_E[0][0] / v1
        if np.abs(phi_s_app) > 1:
            phi_s_app = np.sign(phi_s_app) * 1.1
        else:
            phi_s = np.arcsin(phi_s_app)
        emittance = (
            8 / (2 * np.pi * 1 / ring.t_rev[0])
            * (1 - phi_s_app) / (1 + phi_s_app)
            * np.sqrt(
                (2 * total_voltage * ring.beta[0][0]**2 * ring.energy[0][0])
                / (2 * np.pi * np.abs(1 / machine_params['gamma_transition']**2
                      - 1 / ring.gamma[0][0]**2))
            )
        )

        # Finemet cavity impedance table
        F_C = np.loadtxt(
            '/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/EX_02_Finemet.txt',
            dtype=float, skiprows=1
        )
        F_C[:, 3] *= np.pi / 180
        F_C[:, 5] *= np.pi / 180
        F_C[:, 7] *= np.pi / 180

        option = "closed loop"
        if option == "open loop":
            Re_Z = F_C[:, 4] * np.cos(F_C[:, 3])
            Im_Z = F_C[:, 4] * np.sin(F_C[:, 3])
        elif option == "closed loop":
            Re_Z = F_C[:, 2] * np.cos(F_C[:, 5])
            Im_Z = F_C[:, 2] * np.sin(F_C[:, 5])
        else:
            Re_Z = F_C[:, 6] * np.cos(F_C[:, 7])
            Im_Z = F_C[:, 6] * np.sin(F_C[:, 7])
        F_C_table = InputTable(F_C[:, 0], 13 * Re_Z, 13 * Im_Z)

        # Impedance contributions
        steps = InductiveImpedance(
            beam, profile, 34.6669349520904 / 10e9 * ring.f_rev,
            rf_station, deriv_mode='diff'
        )
        dir_space_charge = InductiveImpedance(
            beam, profile,
            -376.730313462 / (ring.beta[0] * ring.gamma[0]**2),
            rf_station
        )
        imp_list = [F_C_table]
        ind_volt_freq = InducedVoltageFreq(
            beam, profile, imp_list, frequency_resolution=2e5
        )
        total_induced_voltage = TotalInducedVoltage(
            beam, profile, [ind_volt_freq, steps, dir_space_charge]
        )
        total_induced_voltage.track()

        tracker = RingAndRFTracker(
            rf_station, beam,
            Profile=profile, TotalInducedVoltage=total_induced_voltage
        )
        full_rrf = FullRingAndRF([tracker])

        try:
            full_rrf.track()
            matched_from_distribution_function(
                beam, full_rrf,
                distribution_type='parabolic_line',
                emittance=emittance * filling_factor,
                process_pot_well=True,
                TotalInducedVoltage=total_induced_voltage
            )
            profile.track()
        except Exception as e:
            return {
                'status': 'failed',
                'error': str(e),
                'total_voltage': total_voltage,
                'v_ratio': v_ratio,
                'B_dot': B_dot,
                'B_ind': B_ind,
                'emittance': emittance * filling_factor,
                'filling_factor': filling_factor,
                'phi': phi,
                'B': B_data['B_field'][B_ind],
                'phi_s': phi_s,
                'random_phase_shift': random_phase_shift
            }

        # Center profile and re-track
        half_trev = ring.t_rev[0] / 2
        center_of_weight = np.mean(beam.dt)
        profile.cut_options.cut_left = center_of_weight - half_trev
        profile.cut_options.cut_right = center_of_weight + half_trev
        profile.set_slices_parameters()
        profile.track()

        norm_bin_centers = profile.bin_centers / ring.t_rev[0]

        if noise is None:
            return {
                'status': 'created',
                'total_voltage': total_voltage,
                'v_ratio': v_ratio,
                'B_dot': B_dot,
                'B_ind': B_ind,
                'emittance': emittance * filling_factor,
                'filling_factor': filling_factor,
                'phi': phi,
                'B': B_data['B_field'][B_ind],
                'profile': np.column_stack((
                    norm_bin_centers,
                    profile.n_macroparticles / np.max(profile.n_macroparticles)
                )),
                'phi_s': phi_s,
                'random_phase_shift': random_phase_shift
            }
        else:
            noise_arr = np.random.normal(
                0,
                noise * np.max(profile.n_macroparticles),
                len(profile.n_macroparticles)
            )
            noisy_profile = profile.n_macroparticles + noise_arr
            return {
                'status': 'created',
                'total_voltage': total_voltage,
                'v_ratio': v_ratio,
                'B_dot': B_dot,
                'B_ind': B_ind,
                'emittance': emittance * filling_factor,
                'filling_factor': filling_factor,
                'phi': phi,
                'B': B_data['B_field'][B_ind],
                'profile': np.column_stack((
                    norm_bin_centers,
                    noisy_profile / np.max(noisy_profile)
                )),
                'phi_s': phi_s,
                'random_phase_shift': random_phase_shift
            }
    
def cyclic_distance(x, y):
    diff = (x - y) % (2*np.pi)
    return np.abs(-np.pi + (diff + np.pi) % (2*np.pi))

def cyclic_distance_w_sign(x, y):
    diff = (x - y) % (2*np.pi)
    return -np.pi + (diff + np.pi) % (2*np.pi)

class BeamEnvironmentFixed:
    """Fixed version with stabilized reward function"""
    def __init__(self, simulate_func, optimal_phases, max_ep_len):
        self.simulate_profile = simulate_func
        self.optimal_phases = optimal_phases
        self.param_ids = list(optimal_phases.keys())
        self._order = np.arange(len(self.param_ids))
        # Simplified curriculum
        self.filling_factors = [0.2, 0.5, 0.75, 1.0, 1.25]
        np.random.shuffle(self._order)
        self._next_idx = 0
        self._completed_ffs = {idx: set() for idx in range(len(self.param_ids))}
        
        self.max_steps = max_ep_len
        
        # FIXED REWARD COEFFICIENTS - Much more conservative
        self.distance_coeff = 2.0        # Base distance penalty (was 5.0)
        self.progress_coeff = 1.0        # Progress reward (was 20.0!)
        self.action_penalty_coeff = 0.1  # Action penalty (was 0.5)
        self.convergence_bonus = 20.0    # Convergence bonus (was 100)
        self.smart_adjustment_bonus = 1.0 # Smart adjustment bonus (was 50)

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            
        # Parameter selection logic (simplified from original)
        idx = self._order[self._next_idx]
        available_ffs = [i for i in range(len(self.filling_factors)) 
                        if i not in self._completed_ffs[idx]]
        
        if not available_ffs:
            self._next_idx = (self._next_idx + 1) % len(self._order)
            if self._next_idx == 0:
                np.random.shuffle(self._order)
                self._completed_ffs = {idx: set() for idx in range(len(self.param_ids))}
            idx = self._order[self._next_idx]
            available_ffs = list(range(len(self.filling_factors)))
                
        ff_idx = np.random.choice(available_ffs)
        self._completed_ffs[idx].add(ff_idx)
                
        self.current_param = list(self.param_ids[idx])
        self.current_ff = self.filling_factors[ff_idx]
        self.sim_params = [1] + self.current_param + [self.current_ff]
        
        # Phase initialization
        self.random_phase_shift = np.random.uniform(-np.pi, np.pi)
        self.optimal_phase_rel = (
            self.optimal_phases[tuple(self.current_param)] - np.pi - self.random_phase_shift
        )
        
        # REMOVED ADAPTIVE NOISE: Always start with full random initialization
        self.current_phase_rel = self._phase_wrap(
            self.optimal_phase_rel + np.random.uniform(-np.pi, np.pi)
        )
        
        self.current_rel_dist = cyclic_distance(self.optimal_phase_rel, self.current_phase_rel)
        self.current_dist_w_sign = cyclic_distance_w_sign(self.optimal_phase_rel, self.current_phase_rel)
        self.best_dist = self.current_rel_dist
        # Generate profile
        result = self.simulate_profile(self.sim_params, self.current_phase_rel, self.random_phase_shift)
        try:
            self.current_profile = result['profile'][:, 1].reshape(1, -1)
            self.phi_s = result['phi_s'] / np.pi
        except:
            self.current_profile = np.zeros((1, 1000))
            self.phi_s = 0.0

        self.steps = 0
        return [self.current_profile, self.phi_s]

    def step(self, action):
        """FIXED REWARD FUNCTION - Much more stable"""
        prev_dist = self.current_rel_dist
        prev_dist_w_sign = self.current_dist_w_sign

        # First give the rewards that are based on the previous state and the action taken in the current step
        # 1. Smart adjustment bonus - only for fine corrections
        smart_bonus = 0
        if (self.current_rel_dist <= 5 * np.pi / 180 and # Close to convergence
            abs(action) <= self.current_rel_dist and # Action is not too large
            abs(action) >= 0.5*self.current_rel_dist and # Action is not too small
            np.sign(action) == np.sign(self.current_dist_w_sign)): # Action is in the right direction
            smart_bonus = self.smart_adjustment_bonus / ((1 + abs(self.current_rel_dist-abs(action)) * 180/np.pi)*(self.steps+1)) 

        

        # Apply phase adjustment
        new_phase_rel = self._phase_wrap(self.current_phase_rel + action)
        
        # Simulate new profile
        results = self.simulate_profile(self.sim_params, float(new_phase_rel), self.random_phase_shift)
        try:
            new_profile = np.expand_dims(results['profile'][:, 1], axis=0)
            simulation_penalty = 0
        except:
            new_profile = np.zeros((1, 1000))
            simulation_penalty = 10

        # Calculate new distance
        self.current_rel_dist = cyclic_distance(self.optimal_phase_rel, new_phase_rel)
        self.current_dist_w_sign = cyclic_distance_w_sign(self.optimal_phase_rel, new_phase_rel)
        # self.best_dist = min(self.best_dist, self.current_rel_dist)
        
        # STABILIZED REWARD COMPONENTS
        
        # 1. Smooth distance penalty
        distance_reward = -self.distance_coeff * (self.current_rel_dist / np.pi)
        
        # 2. FIXED Progress reward - capped with tanh to prevent explosive rewards
        progress = prev_dist - self.current_rel_dist 
        progress_reward = self.progress_coeff * np.tanh(progress / (np.pi/4))
        
        # 3. Gentle action penalty
        action_penalty = -self.action_penalty_coeff * (abs(action) / np.pi)
        
        # 4. If close to convergence, and we stay there, give a bonus
        convergence_bonus_staying = 0
        if (self.current_rel_dist < (np.pi * 0.5 / 180) and # Very close to convergence after the action
            np.sign(action) == np.sign(prev_dist_w_sign) and # Action was in the right direction 
            abs(action) <= prev_dist):  # Action was not too large (we dont care about small actions very close to convergence)
            convergence_bonus_staying = self.convergence_bonus * np.exp(-self.current_rel_dist * 10)
        
        # 4. Smooth convergence bonus
        # convergence_bonus = 0
        # if self.current_rel_dist < (np.pi * 0.5 / 180):  # Within 0.5 degrees
        #     convergence_bonus = self.convergence_bonus * np.exp(-self.current_rel_dist * 10)
        
        # Combine rewards
        reward = distance_reward + progress_reward + action_penalty + smart_bonus + convergence_bonus_staying - simulation_penalty
        
        # Update state
        self.current_phase_rel = new_phase_rel
        self.current_profile = new_profile
        self.steps += 1

        # Termination
        done_convergence = False # self.current_rel_dist < (np.pi * 0.5 / 180)
        done_steps = self.steps >= self.max_steps
        done = done_convergence or done_steps

        return [new_profile, self.phi_s], reward, done

    def _phase_wrap(self, phase):
        return (phase + np.pi) % (2 * np.pi) - np.pi

class GymBeamEnvFixedWithMetrics(gym.Env):
    """Fixed Gym wrapper with metrics tracking"""
    def __init__(self, optimal_phases, max_ep_len):
        super().__init__()
        self.env = BeamEnvironmentFixed(simulate_beam_profile_PSB, optimal_phases, max_ep_len)
        
        # Observation space
        self.observation_space = spaces.Box(low=0, high=1, shape=(1001,), dtype=np.float32)
        
        # REDUCED Action space for stability
        max_action = np.pi / 4  # Reduced from π
        self.action_space = spaces.Box(low=-max_action, high=max_action, shape=(1,), dtype=np.float32)
        
        # Metrics tracking
        self.episode_actions = []
        self.episode_converged = False
        
    def reset(self, seed=None):
        state = self.env.reset(seed)
        profile = state[0].reshape(-1)
        phi_s = [state[1]] if np.isscalar(state[1]) else state[1].reshape(-1)
        
        # Reset metrics
        self.episode_actions = []
        self.episode_converged = False
        
        return np.concatenate([profile, phi_s]).astype(np.float32), {}

    def step(self, action):
        # Track action magnitude
        self.episode_actions.append(abs(float(action[0])))
        
        next_state, reward, done = self.env.step(float(action[0]))
        
        # Check convergence
        if done:
            self.episode_converged = self.env.current_rel_dist < (np.pi * 0.5 / 180)
        
        profile = next_state[0].reshape(-1)
        phi_s = [next_state[1]] if np.isscalar(next_state[1]) else next_state[1].reshape(-1)
        obs = np.concatenate([profile, phi_s]).astype(np.float32)
        
        return obs, float(reward), done, False, {
            'avg_action': np.mean(self.episode_actions) if self.episode_actions else 0.0,
            'converged': self.episode_converged
        }
class TestBeamEnvironment:
    def __init__(self, simulate_func, optimal_phases, max_ep_len):
        self.simulate_profile = simulate_func
        self.optimal_phases = optimal_phases  # Dict {param_id: optimal_phase}
        self.param_ids = list(optimal_phases.keys())
        self.param_idxs = list(range(len(self.param_ids)))
        self._order = np.arange(len(self.param_ids))
        np.random.shuffle(self._order)
        self._next_idx = 0
        self.current_param = None
        self.current_phase_rel = None
        self.current_profile = None
        self.phi_s = None
        self.max_steps = max_ep_len
        self.convergence_threshold = 1  # Start with 1 deg 
        self.max_voltage = 24e3  # V
        self.max_B = 1.2  # T
        self.max_B_dot = 3.7  # T/s
        # FIXED REWARD COEFFICIENTS - Much more conservative
        self.distance_coeff = 2.0        # Base distance penalty (was 5.0)
        self.progress_coeff = 1.0        # Progress reward (was 20.0!)
        self.action_penalty_coeff = 0.1  # Action penalty (was 0.5)
        self.convergence_bonus = 20.0    # Convergence bonus (was 100)
        self.smart_adjustment_bonus = 1.0 # Smart adjustment bonus (was 50)

    def reset(self,seed = None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        # Select a random parameter combination
        idx = self._order[self._next_idx]
        self._next_idx += 1
        if self._next_idx >= len(self._order):
            np.random.shuffle(self._order)
            self._next_idx = 0
        self.current_param = list(self.param_ids[idx])
        self.current_ff = np.random.uniform(0.1, 3)
        self.sim_params = [1] + self.current_param + [self.current_ff]
        # Get initial phase (could be from dataset or perturb optimal)
        self.random_phase_shift = np.random.uniform(-np.pi, np.pi)
        self.optimal_phase_rel = (
            self.optimal_phases[tuple(self.current_param)]
            - np.pi
            - self.random_phase_shift
        )
        self.current_phase_rel = self._phase_wrap(
            self.optimal_phase_rel + np.random.uniform(-np.pi, np.pi)
        )
        self.current_rel_dist = cyclic_distance(
            self.optimal_phase_rel, self.current_phase_rel
        )
        self.current_dist_w_sign = cyclic_distance_w_sign(
            self.optimal_phase_rel, self.current_phase_rel
        )
        # Generate initial profile
        result = self.simulate_profile(
            self.sim_params,
            self.current_phase_rel,
            self.random_phase_shift,
        )
        try:
            self.current_profile = result['profile'][:, 1].unsqueeze(0)
            self.phi_s = result['phi_s'] / np.pi
        except:
            self.current_profile = np.zeros((1, 1000))
            self.phi_s = result['phi_s'] / np.pi

        self.steps = 0
        return [self.current_profile, self.phi_s]

    def reset_eval(self, params, optimal_phase):
        current_phase = np.random.uniform(-np.pi, np.pi)
        self.current_phase_rel = self._phase_wrap(current_phase - np.pi + optimal_phase)
        self.optimal_phase_rel = self._phase_wrap(optimal_phase - np.pi)
        
        self.current_dist_w_sign = cyclic_distance_w_sign(
            self.optimal_phase_rel, self.current_phase_rel
        )
        self.current_rel_dist = np.abs(self.current_dist_w_sign)
        # self.random_phase_shift = np.random.uniform(-np.pi, np.pi)
        # Generate initial profile
        self.current_ff = np.random.uniform(0.1, 1)
        self.sim_params = [1] + list(params) + [self.current_ff]
        self.random_phase_shift = 0.0
        result = self.simulate_profile(
            self.sim_params,
            self.current_phase_rel,
            self.random_phase_shift,
        )
        try:
            self.current_profile = result['profile'][:, 1].unsqueeze(0)
            self.phi_s = result['phi_s'] / np.pi
        except:
            self.current_profile = np.zeros((1, 1000))
            self.phi_s = result['phi_s'] / np.pi

        self.steps = 0
        return [self.current_profile, self.phi_s]

    def step(self, action):
        """FIXED REWARD FUNCTION - Much more stable"""
        prev_dist = self.current_rel_dist
        prev_dist_w_sign = self.current_dist_w_sign

        # First give the rewards that are based on the previous state and the action taken in the current step
        # 1. Smart adjustment bonus - only for fine corrections
        smart_bonus = 0
        if (self.current_rel_dist <= 5 * np.pi / 180 and # Close to convergence
            abs(action) <= self.current_rel_dist and # Action is not too large
            abs(action) >= 0.5*self.current_rel_dist and # Action is not too small
            np.sign(action) == np.sign(self.current_dist_w_sign)): # Action is in the right direction
            smart_bonus = self.smart_adjustment_bonus / ((1 + abs(self.current_rel_dist-abs(action)) * 180/np.pi)*(self.steps+1)) 

        

        # Apply phase adjustment
        new_phase_rel = self._phase_wrap(self.current_phase_rel + action)
        
        # Simulate new profile
        results = self.simulate_profile(self.sim_params, float(new_phase_rel), self.random_phase_shift)
        try:
            new_profile = np.expand_dims(results['profile'][:, 1], axis=0)
            simulation_penalty = 0
        except:
            new_profile = np.zeros((1, 1000))
            simulation_penalty = 10

        # Calculate new distance
        self.current_rel_dist = cyclic_distance(self.optimal_phase_rel, new_phase_rel)
        self.current_dist_w_sign = cyclic_distance_w_sign(self.optimal_phase_rel, new_phase_rel)
        # self.best_dist = min(self.best_dist, self.current_rel_dist)
        
        # STABILIZED REWARD COMPONENTS
        
        # 1. Smooth distance penalty
        distance_reward = -self.distance_coeff * (self.current_rel_dist / np.pi)
        
        # 2. FIXED Progress reward - capped with tanh to prevent explosive rewards
        progress = prev_dist - self.current_rel_dist 
        progress_reward = self.progress_coeff * np.tanh(progress / (np.pi/4))
        
        # 3. Gentle action penalty
        action_penalty = -self.action_penalty_coeff * (abs(action) / np.pi)
        
        # 4. If close to convergence, and we stay there, give a bonus
        convergence_bonus_staying = 0
        if (self.current_rel_dist < (np.pi * 0.5 / 180) and # Very close to convergence after the action
            np.sign(action) == np.sign(prev_dist_w_sign) and # Action was in the right direction 
            abs(action) <= prev_dist):  # Action was not too large (we dont care about small actions very close to convergence)
            convergence_bonus_staying = self.convergence_bonus * np.exp(-self.current_rel_dist * 10)
        
        # 4. Smooth convergence bonus
        # convergence_bonus = 0
        # if self.current_rel_dist < (np.pi * 0.5 / 180):  # Within 0.5 degrees
        #     convergence_bonus = self.convergence_bonus * np.exp(-self.current_rel_dist * 10)
        
        # Combine rewards
        reward = distance_reward + progress_reward + action_penalty + smart_bonus + convergence_bonus_staying - simulation_penalty
        
        # Update state
        self.current_phase_rel = new_phase_rel
        self.current_profile = new_profile
        self.steps += 1

        # Termination
        done_convergence = False # self.current_rel_dist < (np.pi * 0.5 / 180)
        done_steps = self.steps >= self.max_steps
        done = done_convergence or done_steps

        return [new_profile, self.phi_s], reward, done

    def _phase_wrap(self, phase):
        return (phase + np.pi) % (2 * np.pi) - np.pi
    

def _make_test_gym_env(optimal_phases_path, max_ep_len):
    class GymBeamEnv(gym.Env):  # noqa
        """
        Gym wrapper for the BeamEnvironment.
        Observation: concatenated beam profile (1000,) and phi_s (1,) -> shape (1001,)
        Action: phase adjustment in radians, shape (1,)
        """
        metadata = {'render_modes': ['human', 'rgb_array']}

        def __init__(self, optimal_phases_path, max_ep_len):
            super(GymBeamEnv, self).__init__()
            optimal_phases = pickle.load(open(optimal_phases_path, 'rb'))
            self.env = TestBeamEnvironment(simulate_beam_profile_PSB, optimal_phases, max_ep_len)
            # Observation: profile (1000,) [0,1] + phi_s (1,) [-1,1] => (1001,)
            obs_dim = 1001
            self.observation_space = spaces.Box(
                low=-1, high=1, shape=(obs_dim,), dtype=np.float32
            )
            # Action: one-dimensional phase adjustment in [-pi/2, pi/2]
            max_action = np.pi 
            self.action_space = spaces.Box(
                low=-max_action, high=max_action, shape=(1,), dtype=np.float32
            )
            # Support video recording: set render_mode for VecEnv
            self.render_mode = 'rgb_array'
        
        def reset(self, seed = None):
            self.state = self.env.reset(seed)

            profile= self.state[0].reshape(-1)
            phi_s = self.state[1].reshape(-1)

            return np.concatenate([profile, phi_s]).astype(np.float32), {}

        def step(self, action):
            # action: array of shape (1,)
            action_env = float(action[0])
            next_state, reward, done = self.env.step(np.array(action_env))
            self.state = next_state
            profile = next_state[0].reshape(-1)
            phi_s = next_state[1].reshape(-1)
            
            obs = np.concatenate([profile, phi_s]).astype(np.float32)
            # Convert reward to Python float
            try:
                reward = float(reward.get())
            except:
                reward = float(reward)
            return obs, reward, done, False, {}
        def render(self, mode='rgb_array'):
            import numpy as _np
            # Create a figure without blocking
            fig = plt.figure()
            plt.title(f"Synchronous Phase: {self.state[1]:.3f}")
            # Plot the current profile
            plt.plot(self.state[0][0])
            plt.ylim(0, 1.1)
            plt.xlim(0, 1000)
            plt.xlabel('Slice index')
            plt.ylabel('Normalized number of particles')
            # Draw the canvas and convert to RGB numpy array
            fig.canvas.draw()
            width, height = fig.canvas.get_width_height()
            img = _np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8').reshape(height, width, 4)
            # Convert to RGB by dropping the alpha channel
            img = img[..., 1:]
            plt.close(fig)
            if mode == 'human':
                # Display image for human viewer
                import cv2
                cv2.imshow('GymBeamEnv', img[..., ::-1])
                cv2.waitKey(1)
                return None
            # Return RGB array for video recording
            return img

    return GymBeamEnv(optimal_phases_path, max_ep_len)

def make_beam_env(optimal_phases_path, max_ep_len):
    """Create beam environment for LSTM-TD3"""
    optimal_phases = pickle.load(open(optimal_phases_path, 'rb'))
    return GymBeamEnvFixedWithMetrics(optimal_phases, max_ep_len)

# ===== END BEAM ENVIRONMENT =====

class RenderingCallback:
    """Callback to generate GIFs during training showing agent behavior"""
    def __init__(self, test_env, render_freq=500, 
    save_path="/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/training_gifs/",
     max_hist_len=10, verbose=0, 
     act_dim=1,
     obs_dim=1001,
     act_limit=np.pi/4):
        self.test_env = test_env
        self.render_freq = render_freq
        self.save_path = save_path
        self.gif_counter = 0
        self.verbose = verbose
        self.max_hist_len = max_hist_len
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.act_limit = act_limit
        # Create directory if it doesn't exist
        os.makedirs(save_path, exist_ok=True)
        
    def should_render(self, epoch):
        """Check if it's time to render"""
        return epoch % self.render_freq == 0 and epoch > 0
    
    def get_action(self, o, o_buff, a_buff, o_buff_len, noise_scale, ac, device=None):
        h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(device)
        h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(device)
        h_l = torch.tensor([o_buff_len]).float().to(device)
        with torch.no_grad():
            a = ac.act(torch.as_tensor(o, dtype=torch.float32).view(1, -1).to(device),
                       h_o, h_a, h_l).reshape(self.act_dim)
        a += noise_scale * np.random.randn(self.act_dim)
        return np.clip(a, -self.act_limit, self.act_limit)

    
    def generate_training_gif(self, ac, epoch, device):
        """Generate a GIF showing agent behavior during an episode"""
        if self.verbose > 0:
            print(f"Generating training GIF at step {epoch}...")
        
        # Reset environment
        obs, _ = self.test_env.reset()
        done = False
        step = 0
        
        frames = []
        actions_taken = []
        phase_errors = []
        rewards = []
        
        # Initialize history buffers (same as in test_agent)
        obs_dim = self.obs_dim
        act_dim = self.act_dim
        max_hist_len = self.max_hist_len  # This should match the model's expectation
        
        if max_hist_len > 0:
            o_buff = np.zeros([max_hist_len, obs_dim])
            a_buff = np.zeros([max_hist_len, act_dim])
            o_buff[0, :] = obs
            o_buff_len = 0
        else:
            o_buff = np.zeros([1, obs_dim])
            a_buff = np.zeros([1, act_dim])
            o_buff_len = 0
        
        while not done and step < 30:  # Max 30 steps
            # Get action from model (same as get_action function)
            action = self.get_action(obs, o_buff, a_buff, o_buff_len, 0, ac, device)
            actions_taken.append(float(action))
            
            # Take step
            obs2, reward, terminated, truncated, info = self.test_env.step(action)
            done = terminated or truncated
            rewards.append(float(reward))
            
            # Get current phase error
            try:
                if hasattr(self.test_env, 'env') and hasattr(self.test_env.env, 'current_rel_dist'):
                    current_error = self.test_env.env.current_rel_dist * 180 / np.pi
                    phase_errors.append(current_error)
                else:
                    phase_errors.append(0)
            except:
                phase_errors.append(0)
            
            # Create frame
            frame = self._create_frame(obs, step, actions_taken, phase_errors, rewards, done, epoch)
            frames.append(frame)
            
            # Update history buffers
            if max_hist_len != 0:
                if o_buff_len == max_hist_len:
                    o_buff[:max_hist_len - 1] = o_buff[1:]
                    a_buff[:max_hist_len - 1] = a_buff[1:]
                    o_buff[max_hist_len - 1] = list(obs)
                    a_buff[max_hist_len - 1] = list(action)
                else:
                    o_buff[o_buff_len + 1 - 1] = list(obs)
                    a_buff[o_buff_len + 1 - 1] = list(action)
                    o_buff_len += 1
            
            obs = obs2
            step += 1
        
        # Save GIF
        gif_path = os.path.join(self.save_path, f"training_step_{epoch:08d}_{self.gif_counter:02d}.gif")
        imageio.mimsave(gif_path, frames, fps=2, duration=0.5)
        if self.verbose > 0:
            print(f"Saved training GIF: {gif_path}")
        
        self.gif_counter += 1
    
    def _create_frame(self, obs, step, actions, phase_errors, rewards, done, epoch):
        """Create a single frame showing beam profile and metrics"""
        # Extract beam profile and phi_s from observation
        beam_profile = obs[:1000]  # First 1000 elements
        phi_s = obs[1000]  # Last element
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Plot 1: Beam profile
        ax1.plot(beam_profile, 'b-', linewidth=2)
        ax1.set_title(f'Beam Profile (Step {step})')
        ax1.set_xlabel('Slice Index')
        ax1.set_ylabel('Normalized Intensity')
        ax1.set_ylim(0, 1.1)
        ax1.grid(True, alpha=0.3)
        
        # Add convergence status
        if (done and len(phase_errors) > 0) or (len(phase_errors) > 0 and phase_errors[-1] < 0.5):
            status = "CONVERGING" if phase_errors[-1] < 0.5 else "MAX STEPS"
            ax1.text(0.02, 0.95, status, transform=ax1.transAxes, 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="green" if status == "CONVERGING" else "red", alpha=0.7),
                    fontsize=10, fontweight='bold', color='white')
        
        # Plot 2: Actions taken
        if actions:
            ax2.bar(range(len(actions)), np.array(actions) * 180 / np.pi, color='orange', alpha=0.7)
            ax2.set_title('Actions Taken (degrees)')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('Phase Adjustment (°)')
            ax2.grid(True, alpha=0.3)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        else:
            ax2.text(0.5, 0.5, 'No actions yet', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Actions Taken (degrees)')
        
        # Plot 3: Phase error evolution
        if phase_errors:
            ax3.plot(phase_errors, 'r-', marker='o', linewidth=2, markersize=4)
            ax3.set_title('Phase Error Evolution')
            ax3.set_xlabel('Step')
            ax3.set_ylabel('Phase Error (°)')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Target (0.5°)')
            ax3.legend()
            
            # Add current error text
            if phase_errors:
                ax3.text(0.02, 0.95, f'Current Error: {phase_errors[-1]:.2f}°', 
                        transform=ax3.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax3.text(0.5, 0.5, 'No error data yet', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Phase Error Evolution')
        
        # Plot 4: Rewards
        if rewards:
            ax4.plot(rewards, 'g-', marker='s', linewidth=2, markersize=4)
            ax4.set_title('Rewards')
            ax4.set_xlabel('Step')
            ax4.set_ylabel('Reward')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # Add cumulative reward
            cumulative_reward = sum(rewards)
            ax4.text(0.02, 0.95, f'Cumulative: {cumulative_reward:.2f}', 
                    transform=ax4.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        else:
            ax4.text(0.5, 0.5, 'No rewards yet', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Rewards')
        
        # Add overall title
        fig.suptitle(f'LSTM-TD3 Agent Training Progress - Timestep {epoch:,}\n'
                    f'Synchronous Phase: {phi_s:.3f}π', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        img = np.frombuffer(fig.canvas.tostring_argb(), dtype='uint8').reshape(height, width, 4)
        img = img[..., 1:]
        plt.close(fig)
        return img


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, max_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_size = max_size
        self.obs_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros((max_size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((max_size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(max_size, dtype=np.float32)
        self.done_buf = np.zeros(max_size, dtype=np.float32)
        self.ptr, self.size = 0, 0

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.obs2_buf[self.ptr] = list(next_obs)
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}

    def sample_batch_with_history(self, batch_size=32, max_hist_len=100):
        """

        :param batch_size:
        :param max_hist_len: the length of experiences before current experience
        :return:
        """
        idxs = np.random.randint(max_hist_len, self.size, size=batch_size)
        # History
        if max_hist_len == 0:
            hist_obs = np.zeros([batch_size, 1, self.obs_dim])
            hist_act = np.zeros([batch_size, 1, self.act_dim])
            hist_obs2 = np.zeros([batch_size, 1, self.obs_dim])
            hist_act2 = np.zeros([batch_size, 1, self.act_dim])
            hist_obs_len = np.zeros(batch_size)
            hist_obs2_len = np.zeros(batch_size)
        else:
            hist_obs = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs_len = max_hist_len * np.ones(batch_size)
            hist_obs2 = np.zeros([batch_size, max_hist_len, self.obs_dim])
            hist_act2 = np.zeros([batch_size, max_hist_len, self.act_dim])
            hist_obs2_len = max_hist_len * np.ones(batch_size)

            # Extract history experiences before sampled index
            for i, id in enumerate(idxs):
                hist_start_id = id - max_hist_len
                if hist_start_id < 0:
                    hist_start_id = 0
                # If exist done before the last experience (not including the done in id), start from the index next to the done.
                if len(np.where(self.done_buf[hist_start_id:id] == 1)[0]) != 0:
                    hist_start_id = hist_start_id + (np.where(self.done_buf[hist_start_id:id] == 1)[0][-1]) + 1
                hist_seg_len = id - hist_start_id
                hist_obs_len[i] = hist_seg_len
                hist_obs[i, :hist_seg_len, :] = self.obs_buf[hist_start_id:id]
                hist_act[i, :hist_seg_len, :] = self.act_buf[hist_start_id:id]
                # If the first experience of an episode is sampled, the hist lengths are different for obs and obs2.
                if hist_seg_len == 0:
                    hist_obs2_len[i] = 1
                else:
                    hist_obs2_len[i] = hist_seg_len
                hist_obs2[i, :hist_seg_len, :] = self.obs2_buf[hist_start_id:id]
                hist_act2[i, :hist_seg_len, :] = self.act_buf[hist_start_id+1:id+1]

        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs],
                     hist_obs=hist_obs,
                     hist_act=hist_act,
                     hist_obs2=hist_obs2,
                     hist_act2=hist_act2,
                     hist_obs_len=hist_obs_len,
                     hist_obs2_len=hist_obs2_len)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}
#######################################################################################

#######################################################################################


class MLPCritic(nn.Module):
    def __init__(self, obs_dim, act_dim,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPCritic, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()
        # Memory
        #    Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]

        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size)-1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h+1]),
                                           nn.ReLU()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim + act_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [1]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]),
                                      nn.Identity()]

    def forward(self, obs, act, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        #    History output mask to reduce disturbance cased by none history memory
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
                                    1).long()).squeeze(1)

        # Current Feature Extraction
        x = torch.cat([obs, act], dim=-1)
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        # squeeze(x, -1) : critical to ensure q has right shape.
        return torch.squeeze(x, -1), extracted_memory


class MLPActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit,
                 mem_pre_lstm_hid_sizes=(128,),
                 mem_lstm_hid_sizes=(128,),
                 mem_after_lstm_hid_size=(128,),
                 cur_feature_hid_sizes=(128,),
                 post_comb_hid_sizes=(128,),
                 hist_with_past_act=False):
        super(MLPActor, self).__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit
        self.hist_with_past_act = hist_with_past_act
        #
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()

        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()

        # Memory
        #    Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        #    LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]
        #   After-LSTM
        self.mem_after_lstm_layer_size = [self.mem_lstm_layer_sizes[-1]] + list(mem_after_lstm_hid_size)
        for h in range(len(self.mem_after_lstm_layer_size) - 1):
            self.mem_after_lstm_layers += [nn.Linear(self.mem_after_lstm_layer_size[h],
                                                     self.mem_after_lstm_layer_size[h + 1]),
                                           nn.ReLU()]

        # Current Feature Extraction
        cur_feature_layer_size = [obs_dim] + list(cur_feature_hid_sizes)
        for h in range(len(cur_feature_layer_size) - 1):
            self.cur_feature_layers += [nn.Linear(cur_feature_layer_size[h], cur_feature_layer_size[h + 1]),
                                        nn.ReLU()]

        # Post-Combination
        post_combined_layer_size = [self.mem_after_lstm_layer_size[-1] + cur_feature_layer_size[-1]] + list(
            post_comb_hid_sizes) + [act_dim]
        for h in range(len(post_combined_layer_size) - 2):
            self.post_combined_layers += [nn.Linear(post_combined_layer_size[h], post_combined_layer_size[h + 1]),
                                          nn.ReLU()]
        self.post_combined_layers += [nn.Linear(post_combined_layer_size[-2], post_combined_layer_size[-1]), nn.Tanh()]

    def forward(self, obs, hist_obs, hist_act, hist_seg_len):
        #
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory
        #    Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        #    LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        #    After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        hist_out = torch.gather(x, 1,
                                (tmp_hist_seg_len - 1).view(-1, 1).repeat(1, self.mem_after_lstm_layer_size[-1]).unsqueeze(
                                    1).long()).squeeze(1)

        # Current Feature Extraction
        x = obs
        for layer in self.cur_feature_layers:
            x = layer(x)

        # Post-Combination
        extracted_memory = hist_out
        x = torch.cat([extracted_memory, x], dim=-1)

        for layer in self.post_combined_layers:
            x = layer(x)
        return self.act_limit * x, extracted_memory


class MLPActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit=1,
                 critic_mem_pre_lstm_hid_sizes=(128,),
                 critic_mem_lstm_hid_sizes=(128,),
                 critic_mem_after_lstm_hid_size=(128,),
                 critic_cur_feature_hid_sizes=(128,),
                 critic_post_comb_hid_sizes=(128,),
                 critic_hist_with_past_act=False,
                 actor_mem_pre_lstm_hid_sizes=(128,),
                 actor_mem_lstm_hid_sizes=(128,),
                 actor_mem_after_lstm_hid_size=(128,),
                 actor_cur_feature_hid_sizes=(128,),
                 actor_post_comb_hid_sizes=(128,),
                 actor_hist_with_past_act=False):
        super(MLPActorCritic, self).__init__()
        self.q1 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.q2 = MLPCritic(obs_dim, act_dim,
                            mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                            mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                            mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                            cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                            post_comb_hid_sizes=critic_post_comb_hid_sizes,
                            hist_with_past_act=critic_hist_with_past_act)
        self.pi = MLPActor(obs_dim, act_dim, act_limit,
                           mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                           mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                           mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
                           cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                           post_comb_hid_sizes=actor_post_comb_hid_sizes,
                           hist_with_past_act=actor_hist_with_past_act)

    def act(self, obs, hist_obs=None, hist_act=None, hist_seg_len=None, device=None):
        if (hist_obs is None) or (hist_act is None) or (hist_seg_len is None):
            hist_obs = torch.zeros(1, 1, self.obs_dim).to(device)
            hist_act = torch.zeros(1, 1, self.act_dim).to(device)
            hist_seg_len = torch.zeros(1).to(device)
        with torch.no_grad():
            act, _, = self.pi(obs, hist_obs, hist_act, hist_seg_len)
            return act.cpu().numpy()


#######################################################################################

#######################################################################################
def lstm_td3(resume_exp_dir=None,
             env_name='', seed=0,
             steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
             polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
             start_steps=10000,
             update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
             noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
             batch_size=100,
             max_hist_len=100,
             partially_observable=False,
             pomdp_type = 'remove_velocity',
             flicker_prob=0.2, random_noise_sigma=0.1, random_sensor_missing_prob=0.1,
             use_double_critic = True,
             use_target_policy_smooth = True,
             critic_mem_pre_lstm_hid_sizes=(128,),
             critic_mem_lstm_hid_sizes=(128,),
             critic_mem_after_lstm_hid_size=(128,),
             critic_cur_feature_hid_sizes=(128,),
             critic_post_comb_hid_sizes=(128,),
             critic_hist_with_past_act=False,
             actor_mem_pre_lstm_hid_sizes=(128,),
             actor_mem_lstm_hid_sizes=(128,),
             actor_mem_after_lstm_hid_size=(128,),
             actor_cur_feature_hid_sizes=(128,),
             actor_post_comb_hid_sizes=(128,),
             actor_hist_with_past_act=False,
             # Beam environment parameters
             use_beam_env=False,
             optimal_phases_path='',
             # Logger parameters
             exp_name='lstm_td3',
             data_dir='spinup_data_lstm_gate',
             logger_kwargs=dict(), save_freq=1,
             # TensorBoard parameters
             use_tensorboard=False,
             tensorboard_log_freq=100,
             # Rendering parameters
             render_freq=0,
             # Testing parameters  
             test_freq=10):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act``
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of
            observations as inputs, and ``q1`` and ``q2`` should accept a batch
            of observations and a batch of actions as inputs. When called,
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target
            networks. Target networks are updated towards main networks
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long
            you wait between updates, the ratio of env steps to gradient steps
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target
            policy.

        noise_clip (float): Limit for absolute value of target policy
            smoothing noise.

        policy_delay (int): Policy will only be updated once every
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        
        use_beam_env (bool): Use beam environment instead of bullet environment.
        
        optimal_phases_path (str): Path to optimal phases pickle file for beam environment.
        
        exp_name (str): Experiment name for logging.
        
        data_dir (str): Directory for saving experiment data.

        logger_kwargs (dict): Keyword args for EpochLogger. If empty, will be 
            automatically generated using exp_name, seed, and data_dir.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
        
        use_tensorboard (bool): Whether to use TensorBoard for logging.
                
        render_freq (int): How often (in terms of gap between steps) to render the environment.
        
        test_freq (int): How often (in terms of gap between epochs) to run test episodes.

    """
    # Setup output directory and configuration saving
    if resume_exp_dir is None:
        # Set up logger kwargs if not provided
        if not logger_kwargs:
            logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir, datestamp=True)
        
        # Create output directory
        output_dir = logger_kwargs['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration to JSON file (replacing logger.save_config)
        config_dict = locals().copy()
        # Remove non-serializable items
        config_dict.pop('logger_kwargs', None)
        
        import json
        from lstm_td3.utils.serialization_utils import convert_json
        config_json = convert_json(config_dict)
        config_json['exp_name'] = exp_name
        
        config_path = osp.join(output_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(config_json, f, indent=4, sort_keys=True)
        print(f"Configuration saved to: {config_path}")
    else:
        output_dir = logger_kwargs['output_dir']

    # Initialize TensorBoard logging (required now)
    tensorboard_writer = None
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_dir = logger_kwargs.get('tensorboard_dir', osp.join(output_dir, 'tensorboard'))
            os.makedirs(tensorboard_dir, exist_ok=True)
            tensorboard_writer = SummaryWriter(tensorboard_dir)
            print(f"TensorBoard logging enabled: {tensorboard_dir}")
        except ImportError:
            print("Warning: TensorBoard not available. Install with: pip install tensorboard")
            tensorboard_writer = None
            use_tensorboard = False
    
    
    # Initialize epoch storage with running statistics instead of lists
    epoch_storage = {
        # Episode metrics - these are still accumulated normally (small numbers)
        'EpRet': [],
        'EpLen': [],
        'TestEpRet': [],
        'TestEpLen': [],
        
        # Training metrics - now use running statistics instead of lists
        'LossQ_count': 0,
        'LossQ_sum': 0.0,
        
        'LossPi_count': 0,
        'LossPi_sum': 0.0,
        
        # Q-values and memory - use running statistics
        'Q1Vals_count': 0,
        'Q1Vals_sum': 0.0,
        'Q1Vals_min': float('inf'),
        'Q1Vals_max': float('-inf'),
        
        'Q2Vals_count': 0,
        'Q2Vals_sum': 0.0,
        'Q2Vals_min': float('inf'),
        'Q2Vals_max': float('-inf'),
        
        'Q1ExtractedMemory_count': 0,
        'Q1ExtractedMemory_sum': 0.0,
        'Q1ExtractedMemory_min': float('inf'),
        'Q1ExtractedMemory_max': float('-inf'),
        
        'Q2ExtractedMemory_count': 0,
        'Q2ExtractedMemory_sum': 0.0,
        'Q2ExtractedMemory_min': float('inf'),
        'Q2ExtractedMemory_max': float('-inf'),
        
        'ActExtractedMemory_count': 0,
        'ActExtractedMemory_sum': 0.0,
        'ActExtractedMemory_min': float('inf'),
        'ActExtractedMemory_max': float('-inf'),
    }

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Environment creation - support both beam and bullet environments
    if use_beam_env:
        # Create beam environment
        env = make_beam_env(optimal_phases_path, max_ep_len)
        test_env = _make_test_gym_env(optimal_phases_path, max_ep_len)
        print(f"Created beam environment with observation space: {env.observation_space}")
        print(f"Created beam environment with action space: {env.action_space}")
        
        
    elif partially_observable:
        env = POMDPWrapper(env_name, pomdp_type, flicker_prob, random_noise_sigma, random_sensor_missing_prob)
        test_env = POMDPWrapper(env_name, pomdp_type, flicker_prob, random_noise_sigma, random_sensor_missing_prob)
    else:
        # env, test_env = gym.make(env_name), gym.make(env_name)
        env = make_bullet_task(env_name, dp_type='MDP')
        test_env = make_bullet_task(env_name, dp_type='MDP')
        env.seed(seed)
        test_env.seed(seed)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = MLPActorCritic(obs_dim, act_dim, act_limit,
                        critic_mem_pre_lstm_hid_sizes=critic_mem_pre_lstm_hid_sizes,
                        critic_mem_lstm_hid_sizes=critic_mem_lstm_hid_sizes,
                        critic_mem_after_lstm_hid_size=critic_mem_after_lstm_hid_size,
                        critic_cur_feature_hid_sizes=critic_cur_feature_hid_sizes,
                        critic_post_comb_hid_sizes=critic_post_comb_hid_sizes,
                        critic_hist_with_past_act=critic_hist_with_past_act,
                        actor_mem_pre_lstm_hid_sizes=actor_mem_pre_lstm_hid_sizes,
                        actor_mem_lstm_hid_sizes=actor_mem_lstm_hid_sizes,
                        actor_mem_after_lstm_hid_size=actor_mem_after_lstm_hid_size,
                        actor_cur_feature_hid_sizes=actor_cur_feature_hid_sizes,
                        actor_post_comb_hid_sizes=actor_post_comb_hid_sizes,
                        actor_hist_with_past_act=actor_hist_with_past_act)
    ac_targ = deepcopy(ac)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ac.to(device)
    ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, max_size=replay_size)

    # # Count variables (protip: try to get a feel for how different size networks behave!)
    # var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' % var_counts)
    # Initialize rendering callback if enabled
    rendering_callback = None
    if render_freq > 0 and use_beam_env:
        try:
            gif_save_path = osp.join(output_dir, 'training_gifs')
            rendering_callback = RenderingCallback(
                test_env=_make_test_gym_env(optimal_phases_path, max_ep_len),  # Will be set after environment creation
                render_freq=render_freq,
                save_path=gif_save_path,
                max_hist_len=max_hist_len,
                verbose=1,
                obs_dim=obs_dim,
                act_dim=act_dim,
                act_limit=act_limit
            )
            print(f"Rendering callback enabled: frequency={render_freq}, save_path={gif_save_path}")
        except Exception as e:
            print(f"Warning: Could not initialize rendering callback: {e}")
            rendering_callback = None
    elif render_freq > 0 and not use_beam_env:
        print("Warning: Rendering callback only supported for beam environments")
        rendering_callback = None



    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        h_o, h_a, h_o2, h_a2, h_o_len, h_o2_len = data['hist_obs'], data['hist_act'], data['hist_obs2'], data['hist_act2'], data['hist_obs_len'], data['hist_obs2_len']

        q1, q1_extracted_memory = ac.q1(o, a, h_o, h_a, h_o_len)
        q2, q2_extracted_memory = ac.q2(o, a, h_o, h_a, h_o_len)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ, _ = ac_targ.pi(o2, h_o2, h_a2, h_o2_len)

            # Target policy smoothing
            if use_target_policy_smooth:
                epsilon = torch.randn_like(pi_targ) * target_noise
                epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
                a2 = pi_targ + epsilon
                a2 = torch.clamp(a2, -act_limit, act_limit)
            else:
                a2 = pi_targ

            # Target Q-values
            q1_pi_targ, _ = ac_targ.q1(o2, a2, h_o2, h_a2, h_o2_len)
            q2_pi_targ, _ = ac_targ.q2(o2, a2, h_o2, h_a2, h_o2_len)

            if use_double_critic:
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            else:
                q_pi_targ = q1_pi_targ
            backup = r + gamma * (1 - d) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()

        if use_double_critic:
            loss_q = loss_q1 + loss_q2
        else:
            loss_q = loss_q1

        # Useful info for logging
        # import pdb; pdb.set_trace()
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy(),
                         Q1ExtractedMemory=q1_extracted_memory.mean(dim=1).detach().cpu().numpy(),
                         Q2ExtractedMemory=q2_extracted_memory.mean(dim=1).detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o, h_o, h_a, h_o_len = data['obs'], data['hist_obs'], data['hist_act'], data['hist_obs_len']
        a, a_extracted_memory = ac.pi(o, h_o, h_a, h_o_len)
        q1_pi, _ = ac.q1(o, a, h_o, h_a, h_o_len)
        loss_info = dict(ActExtractedMemory=a_extracted_memory.mean(dim=1).detach().cpu().numpy())
        return -q1_pi.mean(), loss_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    # Helper function to update running statistics
    def update_running_stats(prefix, values):
        """Update running statistics for a set of values"""
        if values is None or len(values) == 0:
            return
        
        # Convert tensor to numpy if necessary
        if hasattr(values, 'detach'):
            values = values.detach().cpu().numpy()
        
        flat_values = values.flatten()
        count = len(flat_values)
        sum_val = float(np.sum(flat_values))
        min_val = float(np.min(flat_values))
        max_val = float(np.max(flat_values))
        
        epoch_storage[f'{prefix}_count'] += count
        epoch_storage[f'{prefix}_sum'] += sum_val
        epoch_storage[f'{prefix}_min'] = min(epoch_storage[f'{prefix}_min'], min_val)
        epoch_storage[f'{prefix}_max'] = max(epoch_storage[f'{prefix}_max'], max_val)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Store loss info for epoch-end logging - use running statistics
        epoch_storage['LossQ_count'] += 1
        epoch_storage['LossQ_sum'] += loss_q.item()
        
        # Update Q-values running statistics
        update_running_stats('Q1Vals', loss_info['Q1Vals'])
        update_running_stats('Q2Vals', loss_info['Q2Vals'])
        update_running_stats('Q1ExtractedMemory', loss_info['Q1ExtractedMemory'])
        update_running_stats('Q2ExtractedMemory', loss_info['Q2ExtractedMemory'])

        # Possibly update pi and target networks
        if timer % policy_delay == 0:
            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi, loss_info_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Store policy loss info for epoch-end logging
            epoch_storage['LossPi_count'] += 1
            epoch_storage['LossPi_sum'] += loss_pi.item()
            
            # Update actor memory running statistics
            update_running_stats('ActExtractedMemory', loss_info_pi['ActExtractedMemory'])

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, o_buff, a_buff, o_buff_len, noise_scale, device=None):
        h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(device)
        h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(device)
        h_l = torch.tensor([o_buff_len]).float().to(device)
        with torch.no_grad():
            a = ac.act(torch.as_tensor(o, dtype=torch.float32).view(1, -1).to(device),
                       h_o, h_a, h_l).reshape(act_dim)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        test_returns = []
        for j in range(num_test_episodes):
            o, _ = test_env.reset()
            d, ep_ret, ep_len = False, 0, 0

            if max_hist_len > 0:
                o_buff = np.zeros([max_hist_len, obs_dim])
                a_buff = np.zeros([max_hist_len, act_dim])
                o_buff[0, :] = o
                o_buff_len = 0
            else:
                o_buff = np.zeros([1, obs_dim])
                a_buff = np.zeros([1, act_dim])
                o_buff_len = 0

            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a = get_action(o, o_buff, a_buff, o_buff_len, 0, device)
                o2, r, terminated, truncated, info = test_env.step(a)
                d = terminated or truncated

                ep_ret += r
                ep_len += 1
                # Add short history
                if max_hist_len != 0:
                    if o_buff_len == max_hist_len:
                        o_buff[:max_hist_len - 1] = o_buff[1:]
                        a_buff[:max_hist_len - 1] = a_buff[1:]
                        o_buff[max_hist_len - 1] = list(o)
                        a_buff[max_hist_len - 1] = list(a)
                    else:
                        o_buff[o_buff_len + 1 - 1] = list(o)
                        a_buff[o_buff_len + 1 - 1] = list(a)
                        o_buff_len += 1
                o = o2

            test_returns.append(ep_ret)
            # Store test metrics for epoch-end logging
            epoch_storage['TestEpRet'].append(ep_ret)
            epoch_storage['TestEpLen'].append(ep_len)
        
        # Return average test performance for best model tracking
        return np.mean(test_returns)

    # Prepare for interaction with environment
    start_time = time.time()
    past_t = 0
    start_epoch = 0
    o, _ = env.reset()
    ep_ret, ep_len = 0, 0
    
    # Best model tracking
    if resume_exp_dir is None:
        best_test_ret = float('-inf')
        best_model_saved = False
    # else: best_test_ret is set in resume section above
    
    # Latest test return tracking (for epochs when we don't test)
    latest_test_ret = float('-inf')

    if max_hist_len > 0:
        o_buff = np.zeros([max_hist_len, obs_dim])
        a_buff = np.zeros([max_hist_len, act_dim])
        o_buff[0, :] = o
        o_buff_len = 0
    else:
        o_buff = np.zeros([1, obs_dim])
        a_buff = np.zeros([1, act_dim])
        o_buff_len = 0

    if resume_exp_dir is not None:
        # Load the best model checkpoint
        resume_checkpoint_path = osp.join(resume_exp_dir, "pyt_save")
        
        # Try new format first (best-model.pt, best-context.pt)
        best_context_path = osp.join(resume_checkpoint_path, 'best-context.pt')
        best_model_path = osp.join(resume_checkpoint_path, 'best-model.pt')
        
        if osp.exists(best_context_path) and osp.exists(best_model_path):
            # New format - load best model
            print("Loading from best model checkpoints...")
            context_checkpoint = torch.load(best_context_path)
            model_checkpoint = torch.load(best_model_path)
            
            # Restore best model tracking
            best_test_ret = context_checkpoint.get('best_test_ret', float('-inf'))
            best_model_saved = True  # We have a saved best model
            print(f"Resuming from best model with test return: {best_test_ret:.3f}")
            
        else:
            # Fallback to old format for backward compatibility
            print("Loading from old format checkpoints...")
            checkpoint_files = os.listdir(resume_checkpoint_path)
            
            # Find latest checkpoint files with step numbers
            context_files = [f for f in checkpoint_files if 'context' in f and 'verified' in f]
            model_files = [f for f in checkpoint_files if 'model' in f and 'verified' in f]
            
            if not context_files or not model_files:
                raise FileNotFoundError("No valid checkpoint files found")
            
            # Get latest versions
            latest_context_version = max([int(f.split('-')[3]) for f in context_files])
            latest_model_version = max([int(f.split('-')[3]) for f in model_files])
            latest_version = min(latest_context_version, latest_model_version)
            
            latest_context_file = f'checkpoint-context-Step-{latest_version}-verified.pt'
            latest_model_file = f'checkpoint-model-Step-{latest_version}-verified.pt'
            
            context_checkpoint = torch.load(osp.join(resume_checkpoint_path, latest_context_file))
            model_checkpoint = torch.load(osp.join(resume_checkpoint_path, latest_model_file))
            
            # No best tracking in old format
            best_test_ret = float('-inf')
            best_model_saved = False

        # Restore experiment context
        if 'logger_epoch_dict' in context_checkpoint:
            # Convert old logger format to new epoch_storage format
            old_epoch_dict = context_checkpoint['logger_epoch_dict']
            for key in epoch_storage.keys():
                if key in old_epoch_dict:
                    epoch_storage[key] = old_epoch_dict[key]
        replay_buffer = context_checkpoint['replay_buffer']
        start_time = context_checkpoint['start_time']
        past_t = context_checkpoint['t'] + 1  # Crucial add 1 step to t to avoid repeating.
        start_epoch = context_checkpoint.get('epoch', 0) + 1  # Resume from next epoch

        # Restore model
        ac.load_state_dict(model_checkpoint['ac_state_dict'])
        ac_targ.load_state_dict(model_checkpoint['target_ac_state_dict'])
        pi_optimizer.load_state_dict(model_checkpoint['pi_optimizer_state_dict'])
        q_optimizer.load_state_dict(model_checkpoint['q_optimizer_state_dict'])

    print("past_t={}".format(past_t))
    print("start_epoch={}".format(start_epoch))
    print(f"Starting training loop. Total epochs: {epochs}, Steps per epoch: {steps_per_epoch}")
    print(f"Test frequency: every {test_freq} epochs (reduces testing overhead)")
    if render_freq > 0:
        print(f"Rendering frequency: every {render_freq} steps")
    else:
        print("Rendering disabled (render_freq=0)")
    
    # Main loop: collect experience in env and update/log each epoch
    t = past_t  # Global step counter
    
    for epoch in range(start_epoch, epochs):  # Loop over epochs
        print(f"\n📊 STARTING EPOCH {epoch}")
        epoch_step_count = 0
        
        # Inner loop: collect steps_per_epoch steps (or until some convergence condition)
        while epoch_step_count < steps_per_epoch:
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy (with some noise, via act_noise).
            if t > start_steps:
                a = get_action(o, o_buff, a_buff, o_buff_len, act_noise, device)
            else:
                a = env.action_space.sample()

            # Step the env
            o2, r, terminated, truncated, info = env.step(a)
            d = terminated or truncated

            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it's an artificial terminal signal
            # that isn't based on the agent's state)
            d = False if ep_len == max_ep_len else d

            # Store experience to replay buffer
            replay_buffer.store(o, a, r, o2, d)

            # Add short history
            if max_hist_len != 0:
                if o_buff_len == max_hist_len:
                    o_buff[:max_hist_len - 1] = o_buff[1:]
                    a_buff[:max_hist_len - 1] = a_buff[1:]
                    o_buff[max_hist_len - 1] = list(o)
                    a_buff[max_hist_len - 1] = list(a)
                else:
                    o_buff[o_buff_len + 1 - 1] = list(o)
                    a_buff[o_buff_len + 1 - 1] = list(a)
                    o_buff_len += 1

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == max_ep_len):
                # Store episode metrics for epoch-end logging
                epoch_storage['EpRet'].append(ep_ret)
                epoch_storage['EpLen'].append(ep_len)
                
                # TensorBoard logging for episode metrics
                if use_tensorboard and tensorboard_writer is not None:
                    tensorboard_writer.add_scalar('Episode/Return', ep_ret, t)
                    tensorboard_writer.add_scalar('Episode/Length', ep_len, t)
                    if use_beam_env and hasattr(env, 'env') and hasattr(env.env, 'current_rel_dist'):
                        phase_error_deg = env.env.current_rel_dist * 180 / np.pi
                        tensorboard_writer.add_scalar('Episode/PhaseError_deg', phase_error_deg, t)
                        converged = phase_error_deg < 0.5
                        tensorboard_writer.add_scalar('Episode/Converged', float(converged), t)
                
                o, _ = env.reset()
                ep_ret, ep_len = 0, 0

                if max_hist_len > 0:
                    o_buff = np.zeros([max_hist_len, obs_dim])
                    a_buff = np.zeros([max_hist_len, act_dim])
                    o_buff[0, :] = o
                    o_buff_len = 0
                else:
                    o_buff = np.zeros([1, obs_dim])
                    a_buff = np.zeros([1, act_dim])
                    o_buff_len = 0

            # Update handling
            if t >= update_after and t % update_every == 0:
                for j in range(update_every):
                    batch = replay_buffer.sample_batch_with_history(batch_size, max_hist_len)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    update(data=batch, timer=j)
                    
            

            # Increment counters
            t += 1
            epoch_step_count += 1
        
        # END OF EPOCH - This is now guaranteed to be at the proper epoch boundary
        print(f"📊 EPOCH {epoch} COMPLETED after {epoch_step_count} steps (total steps: {t})")
        # Rendering callback (independent of epochs)
        if rendering_callback is not None and rendering_callback.should_render(epoch):
            try:
                rendering_callback.generate_training_gif(ac, epoch, device)
            except Exception as e:
                print(f"Warning: Could not generate training GIF: {e}")
        # Test the performance only on testing epochs
        current_test_ret = None
        if epoch % test_freq == 0:
            print(f"🧪 TESTING at epoch {epoch} (test_freq={test_freq})")
            current_test_ret = test_agent()
            latest_test_ret = current_test_ret
            print(f"   Test return: {current_test_ret:.3f}")
        else:
            current_test_ret = latest_test_ret  # Use latest available
            print(f"   Skipping test (last test return: {latest_test_ret:.3f})")
            
        # TensorBoard logging for epoch-level metrics (replacing all logger functionality)
        if use_tensorboard and tensorboard_writer is not None:
            # Basic epoch metrics
            tensorboard_writer.add_scalar('Epoch/Epoch', epoch, epoch)
            # Only log test return when we have fresh data
            if epoch % test_freq == 0:
                tensorboard_writer.add_scalar('Epoch/TestReturn', current_test_ret, epoch)
            tensorboard_writer.add_scalar('Epoch/BestTestReturn', best_test_ret, epoch)
            tensorboard_writer.add_scalar('Epoch/TotalEnvInteracts', t, epoch)
            tensorboard_writer.add_scalar('Epoch/Time', time.time() - start_time, epoch)
            
            # Training episode returns with statistics
            if len(epoch_storage['EpRet']) > 0:
                ep_rets = np.array(epoch_storage['EpRet'])
                tensorboard_writer.add_scalar('Epoch/EpRet_Mean', np.mean(ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/EpRet_Std', np.std(ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/EpRet_Min', np.min(ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/EpRet_Max', np.max(ep_rets), epoch)
            
            # Test episode returns with statistics (only when we have fresh test data)
            if epoch % test_freq == 0 and len(epoch_storage['TestEpRet']) > 0:
                test_ep_rets = np.array(epoch_storage['TestEpRet'])
                tensorboard_writer.add_scalar('Epoch/TestEpRet_Mean', np.mean(test_ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/TestEpRet_Std', np.std(test_ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/TestEpRet_Min', np.min(test_ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/TestEpRet_Max', np.max(test_ep_rets), epoch)
            
            # Episode lengths (average only)
            if len(epoch_storage['EpLen']) > 0:
                tensorboard_writer.add_scalar('Epoch/EpLen_Mean', np.mean(epoch_storage['EpLen']), epoch)
            
            if epoch % test_freq == 0 and len(epoch_storage['TestEpLen']) > 0:
                tensorboard_writer.add_scalar('Epoch/TestEpLen_Mean', np.mean(epoch_storage['TestEpLen']), epoch)
            
            # Q-values and memory with statistics
            if epoch_storage['Q1Vals_count'] > 0:
                tensorboard_writer.add_scalar('Epoch/Q1Vals_Mean', epoch_storage['Q1Vals_sum'] / epoch_storage['Q1Vals_count'], epoch)
                tensorboard_writer.add_scalar('Epoch/Q1Vals_Min', epoch_storage['Q1Vals_min'], epoch)
                tensorboard_writer.add_scalar('Epoch/Q1Vals_Max', epoch_storage['Q1Vals_max'], epoch)
            
            if epoch_storage['Q2Vals_count'] > 0:
                tensorboard_writer.add_scalar('Epoch/Q2Vals_Mean', epoch_storage['Q2Vals_sum'] / epoch_storage['Q2Vals_count'], epoch)
                tensorboard_writer.add_scalar('Epoch/Q2Vals_Min', epoch_storage['Q2Vals_min'], epoch)
                tensorboard_writer.add_scalar('Epoch/Q2Vals_Max', epoch_storage['Q2Vals_max'], epoch)
            
            if epoch_storage['Q1ExtractedMemory_count'] > 0:
                tensorboard_writer.add_scalar('Epoch/Q1ExtractedMemory_Mean', epoch_storage['Q1ExtractedMemory_sum'] / epoch_storage['Q1ExtractedMemory_count'], epoch)
                tensorboard_writer.add_scalar('Epoch/Q1ExtractedMemory_Min', epoch_storage['Q1ExtractedMemory_min'], epoch)
                tensorboard_writer.add_scalar('Epoch/Q1ExtractedMemory_Max', epoch_storage['Q1ExtractedMemory_max'], epoch)
            
            if epoch_storage['Q2ExtractedMemory_count'] > 0:
                tensorboard_writer.add_scalar('Epoch/Q2ExtractedMemory_Mean', epoch_storage['Q2ExtractedMemory_sum'] / epoch_storage['Q2ExtractedMemory_count'], epoch)
                tensorboard_writer.add_scalar('Epoch/Q2ExtractedMemory_Min', epoch_storage['Q2ExtractedMemory_min'], epoch)
                tensorboard_writer.add_scalar('Epoch/Q2ExtractedMemory_Max', epoch_storage['Q2ExtractedMemory_max'], epoch)
            
            # Actor memory with statistics
            if epoch_storage['ActExtractedMemory_count'] > 0:
                tensorboard_writer.add_scalar('Epoch/ActExtractedMemory_Mean', epoch_storage['ActExtractedMemory_sum'] / epoch_storage['ActExtractedMemory_count'], epoch)
                tensorboard_writer.add_scalar('Epoch/ActExtractedMemory_Min', epoch_storage['ActExtractedMemory_min'], epoch)
                tensorboard_writer.add_scalar('Epoch/ActExtractedMemory_Max', epoch_storage['ActExtractedMemory_max'], epoch)
            
            # Loss metrics (average only)
            if epoch_storage['LossQ_count'] > 0:
                tensorboard_writer.add_scalar('Epoch/LossQ_Mean', epoch_storage['LossQ_sum'] / epoch_storage['LossQ_count'], epoch)
            
            if epoch_storage['LossPi_count'] > 0:
                tensorboard_writer.add_scalar('Epoch/LossPi_Mean', epoch_storage['LossPi_sum'] / epoch_storage['LossPi_count'], epoch)
        
        # Save best model only when we have a fresh test result and it improves
        if epoch % test_freq == 0 and current_test_ret > best_test_ret:
            best_test_ret = current_test_ret
            best_model_saved = True
            
            print(f"\n🎯 NEW BEST MODEL! Test return: {best_test_ret:.3f} (Epoch {epoch})")
            
            # Save the best model and context (overwrite previous)
            fpath = osp.join(output_dir, 'pyt_save')
            os.makedirs(fpath, exist_ok=True)
            
            # Simple filenames without step numbers - will overwrite previous best
            context_fname = osp.join(fpath, 'best-context.pt')
            model_fname = osp.join(fpath, 'best-model.pt')
            
            # Save context elements for the best model
            context_elements = {
                'env': env, 
                'replay_buffer': replay_buffer,
                'epoch_storage': epoch_storage,
                'start_time': start_time, 
                't': t,
                'epoch': epoch,
                'best_test_ret': best_test_ret
            }
            
            # Save model elements
            model_elements = {
                'ac_state_dict': ac.state_dict(),
                'target_ac_state_dict': ac_targ.state_dict(),
                'pi_optimizer_state_dict': pi_optimizer.state_dict(),
                'q_optimizer_state_dict': q_optimizer.state_dict(),
                'epoch': epoch,
                'best_test_ret': best_test_ret,
                'total_steps': t
            }
            
            # Save files
            torch.save(context_elements, context_fname, pickle_protocol=4)
            torch.save(model_elements, model_fname, pickle_protocol=4)
            
            context_size_mb = osp.getsize(context_fname) / (1024**2)
            model_size_mb = osp.getsize(model_fname) / (1024**2)
            print(f"   💾 Saved - Context: {context_size_mb:.1f}MB, Model: {model_size_mb:.1f}MB")

        # Clear epoch storage for next epoch (replacing logger functionality)
        # Reset episode lists (small data)
        epoch_storage['EpRet'] = []
        epoch_storage['EpLen'] = []
        # Only clear test metrics when we actually ran tests
        if epoch % test_freq == 0:
            epoch_storage['TestEpRet'] = []
            epoch_storage['TestEpLen'] = []
        
        # Reset running statistics counters
        epoch_storage['LossQ_count'] = 0
        epoch_storage['LossQ_sum'] = 0.0
        epoch_storage['LossPi_count'] = 0
        epoch_storage['LossPi_sum'] = 0.0
        
        # Reset Q-values statistics
        epoch_storage['Q1Vals_count'] = 0
        epoch_storage['Q1Vals_sum'] = 0.0
        epoch_storage['Q1Vals_min'] = float('inf')
        epoch_storage['Q1Vals_max'] = float('-inf')
        
        epoch_storage['Q2Vals_count'] = 0
        epoch_storage['Q2Vals_sum'] = 0.0
        epoch_storage['Q2Vals_min'] = float('inf')
        epoch_storage['Q2Vals_max'] = float('-inf')
        
        # Reset memory statistics
        epoch_storage['Q1ExtractedMemory_count'] = 0
        epoch_storage['Q1ExtractedMemory_sum'] = 0.0
        epoch_storage['Q1ExtractedMemory_min'] = float('inf')
        epoch_storage['Q1ExtractedMemory_max'] = float('-inf')
        
        epoch_storage['Q2ExtractedMemory_count'] = 0
        epoch_storage['Q2ExtractedMemory_sum'] = 0.0
        epoch_storage['Q2ExtractedMemory_min'] = float('inf')
        epoch_storage['Q2ExtractedMemory_max'] = float('-inf')
        
        epoch_storage['ActExtractedMemory_count'] = 0
        epoch_storage['ActExtractedMemory_sum'] = 0.0
        epoch_storage['ActExtractedMemory_min'] = float('inf')
        epoch_storage['ActExtractedMemory_max'] = float('-inf')
        
        # Print epoch summary to console
        print(f"Epoch {epoch} Summary:")
        if epoch % test_freq == 0:
            print(f"  Test Return: {current_test_ret:.3f} (Best: {best_test_ret:.3f})")
        else:
            print(f"  Latest Test Return: {latest_test_ret:.3f} (Best: {best_test_ret:.3f})")
        print(f"  Total Steps: {t}")
        print(f"  Time Elapsed: {time.time() - start_time:.1f}s")
    
    # Clean up TensorBoard writer
    if use_tensorboard and tensorboard_writer is not None:
        tensorboard_writer.close()
        print("TensorBoard logging completed.")


def str2bool(v):
    """Function used in argument parser for converting string to bool."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def list2tuple(v):
    return tuple(v)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_exp_dir', type=str, default=None, help="The directory of the resuming experiment.")
    parser.add_argument('--env_name', type=str, default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--max_hist_len', type=int, default=5)
    parser.add_argument('--partially_observable', type=str2bool, nargs='?', const=True, default=False, help="Using POMDP")
    parser.add_argument('--pomdp_type',
                        choices=['remove_velocity', 'flickering', 'random_noise', 'random_sensor_missing',
                                 'remove_velocity_and_flickering', 'remove_velocity_and_random_noise',
                                 'remove_velocity_and_random_sensor_missing', 'flickering_and_random_noise',
                                 'random_noise_and_random_sensor_missing', 'random_sensor_missing_and_random_noise'],
                        default='remove_velocity')
    parser.add_argument('--flicker_prob', type=float, default=0.2)
    parser.add_argument('--random_noise_sigma', type=float, default=0.1)
    parser.add_argument('--random_sensor_missing_prob', type=float, default=0.1)
    parser.add_argument('--use_double_critic', type=str2bool, nargs='?', const=True, default=True,
                        help="Using double critic")
    parser.add_argument('--use_target_policy_smooth', type=str2bool, nargs='?', const=True, default=True,
                        help="Using target policy smoothing")
    parser.add_argument('--critic_mem_pre_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--critic_mem_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--critic_mem_after_lstm_hid_size', type=int, nargs="+", default=[])
    parser.add_argument('--critic_cur_feature_hid_sizes', type=int, nargs="?", default=[128, 128])
    parser.add_argument('--critic_post_comb_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--critic_hist_with_past_act', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--actor_mem_pre_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_mem_lstm_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_mem_after_lstm_hid_size', type=int, nargs="+", default=[])
    parser.add_argument('--actor_cur_feature_hid_sizes', type=int, nargs="?", default=[128, 128])
    parser.add_argument('--actor_post_comb_hid_sizes', type=int, nargs="+", default=[128])
    parser.add_argument('--actor_hist_with_past_act', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_beam_env', type=str2bool, nargs='?', const=True, default=False, help="Use beam environment instead of bullet tasks")
    parser.add_argument('--optimal_phases_path', type=str, default='', help="Path to optimal phases pickle file for beam environment")
    parser.add_argument('--exp_name', type=str, default='lstm_td3')
    parser.add_argument("--data_dir", type=str, default='spinup_data_lstm_gate')
    parser.add_argument('--save_freq', type=int, default=1, help="How often (in terms of gap between epochs) to save the current policy and value function")
    parser.add_argument('--use_tensorboard', type=str2bool, nargs='?', const=True, default=False, help="Enable TensorBoard logging")
    parser.add_argument('--tensorboard_log_freq', type=int, default=100, help="TensorBoard logging frequency")
    parser.add_argument('--render_freq', type=int, default=0, help="Frequency of generating training GIFs (0 = disabled)")
    parser.add_argument('--test_freq', type=int, default=10, help="How often (in terms of gap between epochs) to run test episodes")
    args = parser.parse_args()

    # Interpret without current feature extraction.
    if args.critic_cur_feature_hid_sizes is None:
        args.critic_cur_feature_hid_sizes = []
    if args.actor_cur_feature_hid_sizes is None:
        args.actor_cur_feature_hid_sizes = []


    # Handle resuming experiment
    if args.resume_exp_dir is not None:
        # Load config_json
        resume_exp_dir = args.resume_exp_dir
        config_path = osp.join(args.resume_exp_dir, 'config.json')
        with open(osp.join(args.resume_exp_dir, "config.json"), 'r') as config_file:
            config_json = json.load(config_file)
        # Update resume_exp_dir value as default is None.
        config_json['resume_exp_dir'] = resume_exp_dir
        # Print config_json
        output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
        print(colorize('Loading config:\n', color='cyan', bold=True))
        print(output)
        # Restore the hyper-parameters
        logger_kwargs = config_json["logger_kwargs"]   # Restore logger_kwargs
        config_json.pop('logger', None)                # Remove logger from config_json
        args = json.loads(json.dumps(config_json), object_hook=lambda d: namedtuple('args', d.keys())(*d.values()))
    else:
        # Set default data directory for new experiments
        args.data_dir = osp.join(
            osp.dirname('/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/lstm_td3_logs'),
            args.data_dir)
        logger_kwargs = {}


    lstm_td3(resume_exp_dir=args.resume_exp_dir,
             env_name=args.env_name,
             gamma=args.gamma, seed=args.seed, epochs=args.epochs,
             max_hist_len=args.max_hist_len,
             partially_observable=args.partially_observable,
             pomdp_type=args.pomdp_type,
             flicker_prob=args.flicker_prob,
             random_noise_sigma=args.random_noise_sigma,
             random_sensor_missing_prob=args.random_sensor_missing_prob,
             use_double_critic=args.use_double_critic,
             use_target_policy_smooth=args.use_target_policy_smooth,
             critic_mem_pre_lstm_hid_sizes=tuple(args.critic_mem_pre_lstm_hid_sizes),
             critic_mem_lstm_hid_sizes=tuple(args.critic_mem_lstm_hid_sizes),
             critic_mem_after_lstm_hid_size=tuple(args.critic_mem_after_lstm_hid_size),
             critic_cur_feature_hid_sizes=tuple(args.critic_cur_feature_hid_sizes),
             critic_post_comb_hid_sizes=tuple(args.critic_post_comb_hid_sizes),
             actor_mem_pre_lstm_hid_sizes=tuple(args.actor_mem_pre_lstm_hid_sizes),
             actor_mem_lstm_hid_sizes=tuple(args.actor_mem_lstm_hid_sizes),
             actor_mem_after_lstm_hid_size=tuple(args.actor_mem_after_lstm_hid_size),
             actor_cur_feature_hid_sizes=tuple(args.actor_cur_feature_hid_sizes),
             actor_post_comb_hid_sizes=tuple(args.actor_post_comb_hid_sizes),
             actor_hist_with_past_act=args.actor_hist_with_past_act,
             use_beam_env=args.use_beam_env,
             optimal_phases_path=args.optimal_phases_path,
             exp_name=args.exp_name,
             data_dir=args.data_dir,
             logger_kwargs=logger_kwargs if args.resume_exp_dir else {},
             save_freq=args.save_freq,
             use_tensorboard=args.use_tensorboard,
             tensorboard_log_freq=args.tensorboard_log_freq,
             render_freq=args.render_freq,
             test_freq=args.test_freq)
