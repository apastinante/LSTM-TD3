from copy import deepcopy
import numpy as np
import gymnasium as gym
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from lstm_td3.utils.logx import EpochLogger, setup_logger_kwargs, colorize
import itertools
from lstm_td3.env_wrapper.pomdp_wrapper import POMDPWrapper
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

# Import vectorized environment support
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# ===== BEAM ENVIRONMENT FUNCTIONS AND CLASSES =====

# Global cache for magnetic field data to avoid concurrent file access
_magnetic_field_cache = {}
_local_file_paths = {}  # Global mapping of AFS paths to local paths

def copy_files_to_local():
    """Copy magnetic field files to local storage to avoid AFS concurrent access"""
    import shutil
    import tempfile
    
    # Create temporary directory
    local_dir = tempfile.mkdtemp(prefix='beam_data_')
    
    # Copy files to local storage
    afs_files = [
        '/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/programs/unnamed/Bfield.npy',
        '/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/programs/TOF/Bfield.npy',
        '/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/EX_02_Finemet.txt'
        
    ]
    
    local_files = {}
    for afs_file in afs_files:
        if os.path.exists(afs_file):
            local_file = os.path.join(local_dir, os.path.basename(afs_file))
            shutil.copy2(afs_file, local_file)
            local_files[afs_file] = local_file
            print(f"Copied {afs_file} → {local_file}")
    
    return local_files

def set_local_file_paths(local_files):
    """Set the global mapping of local file paths"""
    global _local_file_paths
    _local_file_paths = local_files

def get_local_path(afs_path):
    """Get local path for an AFS path, fallback to original if not found"""
    global _local_file_paths
    return _local_file_paths.get(afs_path, afs_path)

def load_magnetic_field(b_choice: int):
    """Load and process magnetic field program using local files"""
    global _magnetic_field_cache
    

    
    if b_choice == 1:
        afs_path = '/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/programs/unnamed/Bfield.npy'
        local_path = get_local_path(afs_path)
        sync_momentum = np.load(local_path)
        t_arr = sync_momentum[0].flatten()  # Time array [s]
        B_field = sync_momentum[1].flatten() * 1e-4  # Convert to Tesla

    elif b_choice == 2:
        afs_path = '/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/programs/TOF/Bfield.npy'
        local_path = get_local_path(afs_path)
        sync_momentum = np.load(local_path)
        t_arr = sync_momentum[0].flatten()  # Time array [s]
        B_field = sync_momentum[1].flatten() * 1e-4  # Convert to Tesla

    # Trim to injection/extraction window
    inj_idx = np.where(t_arr <= 275)[0][-1]
    ext_idx = np.where(t_arr >= 805)[0][0]

    result = {
        't_arr': (t_arr[inj_idx:ext_idx] - t_arr[inj_idx]) / 1e3,
        'B_field': B_field[inj_idx:ext_idx],
        'sync_momentum': B_field[inj_idx:ext_idx] * 8.239 * c
    }
    
    return result

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
        afs_finemet_path = '/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/EX_02_Finemet.txt'
        local_finemet_path = get_local_path(afs_finemet_path)
        F_C = np.loadtxt(
            local_finemet_path,
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
        self.steps_in_convergence = 0
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
            self.steps_in_convergence += 1
        
        
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

        return [new_profile, self.phi_s], reward, done, self.current_rel_dist , self.steps_in_convergence

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
        
    def reset(self, seed=None, options=None):
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
        
        next_state, reward, done , error, steps_in_convergence = self.env.step(float(action[0]))
        
        # Check convergence
        if done:
            self.episode_converged = self.env.current_rel_dist < (np.pi * 0.5 / 180)
        
        profile = next_state[0].reshape(-1)
        phi_s = [next_state[1]] if np.isscalar(next_state[1]) else next_state[1].reshape(-1)
        obs = np.concatenate([profile, phi_s]).astype(np.float32)
        
        return obs, float(reward), done, False, {
            'avg_action': np.mean(self.episode_actions) if self.episode_actions else 0.0,
            'converged': self.episode_converged,
            'phase_error': error*180/np.pi,
            'steps_in_convergence': steps_in_convergence
        }

def make_beam_env_factory(optimal_phases_path, max_ep_len):
    """Factory function for creating beam environments (needed for SubprocVecEnv)"""
    def _create_env():
        
        optimal_phases = pickle.load(open(optimal_phases_path, 'rb'))
        return GymBeamEnvFixedWithMetrics(optimal_phases, max_ep_len)
    return _create_env

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
        
    def should_render(self, epoch, n_envs):
        """Check if it's time to render"""
        real_epoch = epoch*n_envs
        epochs_done_in_step = np.linspace(real_epoch - n_envs + 1, real_epoch, n_envs, endpoint=True)
        return np.any(epochs_done_in_step % self.render_freq == 0) and epoch > 0
    
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
        max_hist_len = self.max_hist_len
        
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
            # Get action from model
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


class SeparateEnvReplayBuffer:
    """
    Much simpler approach: separate replay buffer for each environment.
    Pre-allocates max_size // n_envs to each environment buffer.
    """

    def __init__(self, obs_dim, act_dim, max_size, n_envs):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_envs = n_envs
        self.env_buffer_size = max_size // n_envs
        
        print(f"Creating {n_envs} separate replay buffers, {self.env_buffer_size} capacity each")
        
        # Create separate replay buffers for each environment
        self.env_buffers = []
        for env_id in range(n_envs):
            self.env_buffers.append(ReplayBuffer(obs_dim, act_dim, self.env_buffer_size))

    def store(self, obs, act, rew, next_obs, done, env_ids):
        """Store experiences in their corresponding environment buffers"""
        for i in range(len(obs)):
            env_id = env_ids[i]
            self.env_buffers[env_id].store(obs[i], act[i], rew[i], next_obs[i], done[i])

    def sample_batch(self, batch_size=32):
        """Standard sampling without history"""
        # Sample proportionally from all environments
        base_samples_per_env = batch_size // self.n_envs
        extra_samples = batch_size % self.n_envs
        
        all_batches = []
        
        for env_id in range(self.n_envs):
            env_samples = base_samples_per_env + (1 if env_id < extra_samples else 0)
            
            if env_samples > 0 and self.env_buffers[env_id].size > 0:
                batch = self.env_buffers[env_id].sample_batch(env_samples)
                all_batches.append(batch)
        
        if not all_batches:
            raise RuntimeError("No environment buffers have data for sampling")
        
        # Combine all batches
        combined_batch = {}
        for key in all_batches[0].keys():
            combined_batch[key] = torch.cat([batch[key] for batch in all_batches], dim=0)
        
        return combined_batch

    def sample_batch_with_history(self, batch_size=32, max_hist_len=100):
        """
        Sample batch with proper history from all environment buffers.
        Each environment contributes proportionally to the batch.
        """
        # Calculate how many samples each environment should contribute
        base_samples_per_env = batch_size // self.n_envs
        extra_samples = batch_size % self.n_envs
        
        all_batches = []
        
        for env_id in range(self.n_envs):
            # Some environments get one extra sample if batch_size doesn't divide evenly
            env_samples = base_samples_per_env + (1 if env_id < extra_samples else 0)
            
            # Only sample if this environment has enough experiences
            if env_samples > 0 and self.env_buffers[env_id].size > max_hist_len:
                try:
                    batch = self.env_buffers[env_id].sample_batch_with_history(env_samples, max_hist_len)
                    all_batches.append(batch)
                except:
                    # Skip this environment if sampling fails
                    continue
        
        # If no environments have enough data, fall back to any available data
        if not all_batches:
            for env_id in range(self.n_envs):
                if self.env_buffers[env_id].size > 0:
                    try:
                        # Sample whatever we can get
                        available_samples = min(batch_size // 2, self.env_buffers[env_id].size - 1)
                        if available_samples > 0:
                            batch = self.env_buffers[env_id].sample_batch_with_history(available_samples, max_hist_len)
                            all_batches.append(batch)
                            break
                    except:
                        continue
        
        if not all_batches:
            raise RuntimeError("No environment buffers have sufficient data for sampling")
        
        # Combine all batches from different environments
        combined_batch = {}
        for key in all_batches[0].keys():
            combined_batch[key] = torch.cat([batch[key] for batch in all_batches], dim=0)
        
        return combined_batch

    @property
    def size(self):
        """Total number of experiences across all environment buffers"""
        return sum(buffer.size for buffer in self.env_buffers)

# Keep the original ReplayBuffer for single-environment compatibility
class ReplayBuffer:
    """A simple FIFO experience replay buffer for agents."""

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
        """Sample batch with history for LSTM networks"""
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
                if hist_seg_len < max_hist_len:
                    hist_obs2_len[i] = hist_obs_len[i] + 1
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
        
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()
        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()
        
        # Memory - Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        # LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]

        # After-LSTM
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
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory - Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        # LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        # After-LSTM
        for layer in self.mem_after_lstm_layers:
            x = layer(x)
        # History output mask to reduce disturbance cased by none history memory
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
        
        self.mem_pre_lstm_layers = nn.ModuleList()
        self.mem_lstm_layers = nn.ModuleList()
        self.mem_after_lstm_layers = nn.ModuleList()
        self.cur_feature_layers = nn.ModuleList()
        self.post_combined_layers = nn.ModuleList()

        # Memory - Pre-LSTM
        if self.hist_with_past_act:
            mem_pre_lstm_layer_size = [obs_dim + act_dim] + list(mem_pre_lstm_hid_sizes)
        else:
            mem_pre_lstm_layer_size = [obs_dim] + list(mem_pre_lstm_hid_sizes)
        for h in range(len(mem_pre_lstm_layer_size) - 1):
            self.mem_pre_lstm_layers += [nn.Linear(mem_pre_lstm_layer_size[h],
                                                   mem_pre_lstm_layer_size[h + 1]),
                                         nn.ReLU()]
        # LSTM
        self.mem_lstm_layer_sizes = [mem_pre_lstm_layer_size[-1]] + list(mem_lstm_hid_sizes)
        for h in range(len(self.mem_lstm_layer_sizes) - 1):
            self.mem_lstm_layers += [
                nn.LSTM(self.mem_lstm_layer_sizes[h], self.mem_lstm_layer_sizes[h + 1], batch_first=True)]
        # After-LSTM
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
        tmp_hist_seg_len = deepcopy(hist_seg_len)
        tmp_hist_seg_len[hist_seg_len == 0] = 1
        if self.hist_with_past_act:
            x = torch.cat([hist_obs, hist_act], dim=-1)
        else:
            x = hist_obs
        # Memory - Pre-LSTM
        for layer in self.mem_pre_lstm_layers:
            x = layer(x)
        # LSTM
        for layer in self.mem_lstm_layers:
            x, (lstm_hidden_state, lstm_cell_state) = layer(x)
        # After-LSTM
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


# ===== PARALLEL ENVIRONMENT MANAGER =====

class ParallelEnvManager:
    """Manages parallel environments with history buffers for LSTM-TD3"""
    
    def __init__(self, n_envs, obs_dim, act_dim, max_hist_len):
        self.n_envs = n_envs
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.max_hist_len = max_hist_len
        
        # Initialize history buffers for each environment
        self.reset_history_buffers()
        
        # Episode tracking
        self.episode_returns = [0.0] * n_envs
        self.episode_lengths = [0] * n_envs
        self.episode_phase_errors = [0.0] * n_envs
        
    def reset_history_buffers(self):
        """Reset history buffers for all environments"""
        if self.max_hist_len > 0:
            self.o_buffs = [np.zeros([self.max_hist_len, self.obs_dim]) for _ in range(self.n_envs)]
            self.a_buffs = [np.zeros([self.max_hist_len, self.act_dim]) for _ in range(self.n_envs)]
            self.o_buff_lens = [0] * self.n_envs
        else:
            self.o_buffs = [np.zeros([1, self.obs_dim]) for _ in range(self.n_envs)]
            self.a_buffs = [np.zeros([1, self.act_dim]) for _ in range(self.n_envs)]
            self.o_buff_lens = [0] * self.n_envs
    
    def reset_env_history(self, env_idx, obs):
        """Reset history buffer for a specific environment"""
        if self.max_hist_len > 0:
            self.o_buffs[env_idx] = np.zeros([self.max_hist_len, self.obs_dim])
            self.a_buffs[env_idx] = np.zeros([self.max_hist_len, self.act_dim])
            self.o_buffs[env_idx][0, :] = obs
            self.o_buff_lens[env_idx] = 0
        else:
            self.o_buffs[env_idx] = np.zeros([1, self.obs_dim])
            self.a_buffs[env_idx] = np.zeros([1, self.act_dim])
            self.o_buff_lens[env_idx] = 0
        
        # Reset episode tracking
        self.episode_returns[env_idx] = 0.0
        self.episode_lengths[env_idx] = 0
        self.episode_phase_errors[env_idx] = 0.0
    
    def update_history(self, env_idx, obs, action):
        """Update history buffer for a specific environment"""
        if self.max_hist_len != 0:
            if self.o_buff_lens[env_idx] == self.max_hist_len:
                self.o_buffs[env_idx][:self.max_hist_len - 1] = self.o_buffs[env_idx][1:]
                self.a_buffs[env_idx][:self.max_hist_len - 1] = self.a_buffs[env_idx][1:]
                self.o_buffs[env_idx][self.max_hist_len - 1] = list(obs)
                self.a_buffs[env_idx][self.max_hist_len - 1] = list(action)
            else:
                self.o_buffs[env_idx][self.o_buff_lens[env_idx]] = list(obs)
                self.a_buffs[env_idx][self.o_buff_lens[env_idx]] = list(action)
                self.o_buff_lens[env_idx] += 1
    
    def get_history_tensors(self, env_idx, device):
        """Get history tensors for a specific environment"""
        h_o = torch.tensor(self.o_buffs[env_idx]).view(1, self.o_buffs[env_idx].shape[0], self.o_buffs[env_idx].shape[1]).float().to(device)
        h_a = torch.tensor(self.a_buffs[env_idx]).view(1, self.a_buffs[env_idx].shape[0], self.a_buffs[env_idx].shape[1]).float().to(device)
        h_l = torch.tensor([self.o_buff_lens[env_idx]]).float().to(device)
        return h_o, h_a, h_l
    
    def update_episode_stats(self, env_idx, reward, phase_error):
        """Update episode statistics"""
        self.episode_returns[env_idx] += reward
        self.episode_lengths[env_idx] += 1
        self.episode_phase_errors[env_idx] = phase_error
    
    def get_episode_stats(self, env_idx):
        """Get episode statistics for a specific environment"""
        return self.episode_returns[env_idx], self.episode_lengths[env_idx], self.episode_phase_errors[env_idx] 


def lstm_td3_parallel(resume_exp_dir=None,
                     env_name='', seed=0,
                     steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99,
                     polyak=0.995, pi_lr=1e-3, q_lr=1e-3,
                     start_steps=10000,
                     update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
                     noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
                     batch_size=100,
                     max_hist_len=100,
                     n_envs=max(1, os.cpu_count()-2),  # NEW: Number of parallel environments
                     use_double_critic=True,
                     use_target_policy_smooth=True,
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
                     exp_name='lstm_td3_parallel',
                     data_dir='spinup_data_lstm_gate',
                     logger_kwargs=dict(),
                     save_freq=1,
                     # TensorBoard parameters
                     use_tensorboard=False,
                     tensorboard_log_freq=100,
                     # Rendering parameters
                     render_freq=0,
                     # Testing parameters  
                     test_freq=10):
    """
    LSTM-TD3 with Parallel Environments
    
    Key improvements:
    - Multiple parallel environments for faster data collection
    - Asynchronous action sending and synchronous waiting
    - Proper history buffer management per environment
    - Local file copying to avoid AFS concurrent access issues
    - Maintains all original LSTM-TD3 functionality
    """
    
    # Setup output directory and configuration saving
    if resume_exp_dir is None:
        # Set up logger kwargs if not provided
        if not logger_kwargs:
            logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir, datestamp=True)
        
        # Create output directory
        output_dir = logger_kwargs['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        # Save configuration to JSON file
        config_dict = locals().copy()
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

    # Initialize TensorBoard logging
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
    
    # Initialize epoch storage with running statistics
    epoch_storage = {
        'EpRet': [],
        'EpLen': [],
        'TestEpRet': [],
        'TestEpLen': [],
        'LossQ_count': 0,
        'LossQ_sum': 0.0,
        'LossPi_count': 0,
        'LossPi_sum': 0.0,
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

    # COPY FILES TO LOCAL STORAGE FOR PARALLEL ACCESS
    if use_beam_env:
        print("📁 Setting up local file copying for parallel environments to avoid AFS concurrent access issues...")
        # Main process also needs local files for test environment
        local_files = copy_files_to_local()
        set_local_file_paths(local_files)
        print(f"✅ Copied {len(local_files)} files to local storage for main process")

        # PARALLEL ENVIRONMENT CREATION
        print(f"Creating {n_envs} parallel environments...")

        # Create parallel beam environments
        env_factory = make_beam_env_factory(optimal_phases_path, max_ep_len)
        
        if n_envs == 1:
            env = DummyVecEnv([env_factory])
            vec_env_type = "DummyVecEnv (single process)"
        else:
            env = SubprocVecEnv([env_factory for _ in range(n_envs)])
            vec_env_type = "SubprocVecEnv (multiprocess with local files)"
        
        # Single environment for testing (will use the same local files from main process)
        test_env = make_beam_env(optimal_phases_path, max_ep_len)
        
        print(f"Created {n_envs} parallel beam environments using {vec_env_type}")
        print(f"Observation space: {env.observation_space}")
        print(f"Action space: {env.action_space}")
    else:
        # For non-beam environments, you would create them similarly
        # This is a placeholder for other environment types
        raise NotImplementedError("Non-beam parallel environments not implemented yet")

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    # Create parallel environment manager
    env_manager = ParallelEnvManager(n_envs, obs_dim, act_dim, max_hist_len)

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

    # List of parameters for both Q-networks
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = SeparateEnvReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, max_size=replay_size, n_envs=n_envs)

    # Initialize rendering callback if enabled
    rendering_callback = None
    if render_freq > 0 and use_beam_env:
        try:
            gif_save_path = osp.join(output_dir, 'training_gifs')
            rendering_callback = RenderingCallback(
                test_env=test_env,
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

    # Set up loss functions
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
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy(),
                         Q1ExtractedMemory=q1_extracted_memory.mean(dim=1).detach().cpu().numpy(),
                         Q2ExtractedMemory=q2_extracted_memory.mean(dim=1).detach().cpu().numpy())

        return loss_q, loss_info

    def compute_loss_pi(data):
        o, h_o, h_a, h_o_len = data['obs'], data['hist_obs'], data['hist_act'], data['hist_obs_len']
        a, a_extracted_memory = ac.pi(o, h_o, h_a, h_o_len)
        q1_pi, _ = ac.q1(o, a, h_o, h_a, h_o_len)
        loss_info = dict(ActExtractedMemory=a_extracted_memory.mean(dim=1).detach().cpu().numpy())
        return -q1_pi.mean(), loss_info

    # Set up optimizers
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    # Helper function to update running statistics
    def update_running_stats(prefix, values):
        if values is None or len(values) == 0:
            return
        
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

        # Store loss info for epoch-end logging
        epoch_storage['LossQ_count'] += 1
        epoch_storage['LossQ_sum'] += loss_q.item()
        
        # Update Q-values running statistics
        update_running_stats('Q1Vals', loss_info['Q1Vals'])
        update_running_stats('Q2Vals', loss_info['Q2Vals'])
        update_running_stats('Q1ExtractedMemory', loss_info['Q1ExtractedMemory'])
        update_running_stats('Q2ExtractedMemory', loss_info['Q2ExtractedMemory'])

        # Possibly update pi and target networks
        if timer % policy_delay == 0:
            # Freeze Q-networks during policy update
            for p in q_params:
                p.requires_grad = False

            # Policy gradient step
            pi_optimizer.zero_grad()
            loss_pi, loss_info_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks
            for p in q_params:
                p.requires_grad = True

            # Store policy loss info
            epoch_storage['LossPi_count'] += 1
            epoch_storage['LossPi_sum'] += loss_pi.item()
            
            # Update actor memory running statistics
            update_running_stats('ActExtractedMemory', loss_info_pi['ActExtractedMemory'])

            # Update target networks by polyak averaging
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action_parallel(observations, env_manager, noise_scale, device):
        """Get actions for all parallel environments"""
        actions = []
        
        for env_idx in range(n_envs):
            obs = observations[env_idx]
            h_o, h_a, h_l = env_manager.get_history_tensors(env_idx, device)
            
            with torch.no_grad():
                a = ac.act(torch.as_tensor(obs, dtype=torch.float32).view(1, -1).to(device),
                          h_o, h_a, h_l).reshape(act_dim)
            
            a += noise_scale * np.random.randn(act_dim)
            action = np.clip(a, -act_limit, act_limit)
            actions.append(action)
        
        return np.array(actions)

    def test_agent():
        """Test the agent with single environment"""
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
                # Get action using single environment logic
                h_o = torch.tensor(o_buff).view(1, o_buff.shape[0], o_buff.shape[1]).float().to(device)
                h_a = torch.tensor(a_buff).view(1, a_buff.shape[0], a_buff.shape[1]).float().to(device)
                h_l = torch.tensor([o_buff_len]).float().to(device)
                
                with torch.no_grad():
                    a = ac.act(torch.as_tensor(o, dtype=torch.float32).view(1, -1).to(device),
                              h_o, h_a, h_l).reshape(act_dim)
                
                # No noise for testing
                a = np.clip(a, -act_limit, act_limit)
                
                o2, r, terminated, truncated, info = test_env.step(a)
                d = terminated or truncated

                ep_ret += r
                ep_len += 1
                
                # Update history
                if max_hist_len != 0:
                    if o_buff_len == max_hist_len:
                        o_buff[:max_hist_len - 1] = o_buff[1:]
                        a_buff[:max_hist_len - 1] = a_buff[1:]
                        o_buff[max_hist_len - 1] = list(o)
                        a_buff[max_hist_len - 1] = list(a)
                    else:
                        o_buff[o_buff_len] = list(o)
                        a_buff[o_buff_len] = list(a)
                        o_buff_len += 1
                o = o2

            test_returns.append(ep_ret)
            epoch_storage['TestEpRet'].append(ep_ret)
            epoch_storage['TestEpLen'].append(ep_len)
        
        return np.mean(test_returns)

    # MAIN TRAINING LOOP WITH PARALLEL ENVIRONMENTS
    print(f"Starting parallel training with {n_envs} environments...")
    print(f"Expected speedup: ~{n_envs}x for data collection")
    
    start_time = time.time()
    past_t = 0
    start_epoch = 0
    
    # Best model tracking
    if resume_exp_dir is None:
        best_test_ret = float('-inf')
        best_model_saved = False
    
    latest_test_ret = float('-inf')

    # Initialize parallel environments
    observations = env.reset()
    for env_idx in range(n_envs):
        env_manager.reset_env_history(env_idx, observations[env_idx])

    # Resume from checkpoint if specified
    if resume_exp_dir is not None:
        # Load checkpoint logic (same as original)
        resume_checkpoint_path = osp.join(resume_exp_dir, "pyt_save")
        
        best_context_path = osp.join(resume_checkpoint_path, 'best-context.pt')
        best_model_path = osp.join(resume_checkpoint_path, 'best-model.pt')
        
        if osp.exists(best_context_path) and osp.exists(best_model_path):
            print("Loading from best model checkpoints...")
            context_checkpoint = torch.load(best_context_path)
            model_checkpoint = torch.load(best_model_path)
            
            best_test_ret = context_checkpoint.get('best_test_ret', float('-inf'))
            best_model_saved = True
            print(f"Resuming from best model with test return: {best_test_ret:.3f}")
        else:
            print("No valid checkpoint found, starting from scratch")

        # Restore experiment context
        if 'logger_epoch_dict' in context_checkpoint:
            old_epoch_dict = context_checkpoint['logger_epoch_dict']
            for key in epoch_storage.keys():
                if key in old_epoch_dict:
                    epoch_storage[key] = old_epoch_dict[key]
        
        replay_buffer = context_checkpoint['replay_buffer']
        start_time = context_checkpoint['start_time']
        past_t = context_checkpoint['t'] + 1
        start_epoch = context_checkpoint.get('epoch', 0) + 1

        # Restore model
        ac.load_state_dict(model_checkpoint['ac_state_dict'])
        ac_targ.load_state_dict(model_checkpoint['target_ac_state_dict'])
        pi_optimizer.load_state_dict(model_checkpoint['pi_optimizer_state_dict'])
        q_optimizer.load_state_dict(model_checkpoint['q_optimizer_state_dict'])

    t = past_t
    
    print(f"Starting training loop from epoch {start_epoch}")
    print(f"Using {n_envs} parallel environments")
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total epochs: {epochs}")
    
    # MAIN TRAINING LOOP
    for epoch in range(start_epoch, epochs):
        print(f"\n📊 STARTING EPOCH {epoch} (Parallel: {n_envs} envs)")
        epoch_step_count = 0
        
        while epoch_step_count < steps_per_epoch:
            # PARALLEL ACTION SELECTION AND ENVIRONMENT STEPPING
            if t > start_steps:
                # Use policy with noise
                actions = get_action_parallel(observations, env_manager, act_noise, device)
            else:
                # Random actions for initial exploration
                actions = np.array([env.action_space.sample() for _ in range(n_envs)])
            
            # ASYNCHRONOUS STEP: Send actions to ALL environments simultaneously
            # This is the key parallel optimization - all environments step simultaneously
            next_observations, rewards, dones, infos = env.step(actions)
            
            # PROCESS RESULTS FROM ALL ENVIRONMENTS
            for env_idx in range(n_envs):
                obs = observations[env_idx]
                action = actions[env_idx]
                reward = rewards[env_idx]
                next_obs = next_observations[env_idx]
                done = dones[env_idx]
                phase_error = infos[env_idx]['phase_error']
                steps_in_convergence = infos[env_idx]['steps_in_convergence']
                # Update episode stats
                env_manager.update_episode_stats(env_idx, reward, phase_error)
                
                # Store transition in replay buffer  
                replay_buffer.store([obs], [action], [reward], [next_obs], [done], [env_idx])
                
                # Update history buffer
                env_manager.update_history(env_idx, obs, action)
                
                # Handle episode termination
                if done:
                    # Get episode stats
                    ep_ret, ep_len, ep_phase_error = env_manager.get_episode_stats(env_idx)
                    
                    # Store episode metrics
                    epoch_storage['EpRet'].append(ep_ret)
                    epoch_storage['EpLen'].append(ep_len)
                    
                    # TensorBoard logging
                    if use_tensorboard and tensorboard_writer is not None:
                        tensorboard_writer.add_scalar('Episode/Return', ep_ret, t)
                        tensorboard_writer.add_scalar('Episode/Length', ep_len, t)
                        tensorboard_writer.add_scalar('Episode/PhaseErrorDeg', ep_phase_error, t)
                        tensorboard_writer.add_scalar('Episode/StepsInConvergence', steps_in_convergence, t)
                    # Reset environment (happens automatically in VecEnv)
                    # But we need to reset our history tracking
                    env_manager.reset_env_history(env_idx, next_observations[env_idx])
            
            # Update observations for next iteration
            observations = next_observations
            
            # TRAINING UPDATES
            dummy_t = np.linspace(t +1 , t+n_envs +1 , n_envs, endpoint=False, dtype=int)
            if t >= update_after and np.any(dummy_t % update_every == 0):
                for j in range(update_every):
                    batch = replay_buffer.sample_batch_with_history(batch_size, max_hist_len)
                    batch = {k: v.to(device) for k, v in batch.items()}
                    update(data=batch, timer=j)
            
            # Increment counters
            t += 1*n_envs
            epoch_step_count += 1
        
        # END OF EPOCH
        print(f"📊 EPOCH {epoch} COMPLETED after {epoch_step_count} steps (total steps: {t})")
        print(f"   Data collection speedup: {n_envs}x faster with parallel environments")
        
        # Rendering callback
        if rendering_callback is not None and rendering_callback.should_render(epoch, n_envs):
            try:
                rendering_callback.generate_training_gif(ac, epoch, device)
            except Exception as e:
                print(f"Warning: Could not generate training GIF: {e}")
        
        # Testing
        current_test_ret = None
        if epoch % test_freq == 0:
            print(f"🧪 TESTING at epoch {epoch}")
            current_test_ret = test_agent()
            latest_test_ret = current_test_ret
            print(f"   Test return: {current_test_ret:.3f}")
        else:
            current_test_ret = latest_test_ret
            print(f"   Skipping test (last test return: {latest_test_ret:.3f})")
        
        # TensorBoard logging for epoch-level metrics
        if use_tensorboard and tensorboard_writer is not None:
            tensorboard_writer.add_scalar('Epoch/Epoch', epoch, epoch)
            if epoch % test_freq == 0:
                tensorboard_writer.add_scalar('Epoch/TestReturn', current_test_ret, epoch)
            tensorboard_writer.add_scalar('Epoch/BestTestReturn', best_test_ret, epoch)
            tensorboard_writer.add_scalar('Epoch/TotalEnvInteracts', t, epoch)
            tensorboard_writer.add_scalar('Epoch/Time', time.time() - start_time, epoch)
            tensorboard_writer.add_scalar('Epoch/DataCollectionSpeedup', n_envs, epoch)
            
            # Episode returns statistics
            if len(epoch_storage['EpRet']) > 0:
                ep_rets = np.array(epoch_storage['EpRet'])
                tensorboard_writer.add_scalar('Epoch/EpRet_Mean', np.mean(ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/EpRet_Std', np.std(ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/EpRet_Min', np.min(ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/EpRet_Max', np.max(ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/EpisodeCount', len(ep_rets), epoch)
            
            # Test episode statistics
            if epoch % test_freq == 0 and len(epoch_storage['TestEpRet']) > 0:
                test_ep_rets = np.array(epoch_storage['TestEpRet'])
                tensorboard_writer.add_scalar('Epoch/TestEpRet_Mean', np.mean(test_ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/TestEpRet_Std', np.std(test_ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/TestEpRet_Min', np.min(test_ep_rets), epoch)
                tensorboard_writer.add_scalar('Epoch/TestEpRet_Max', np.max(test_ep_rets), epoch)
            
            # Loss metrics
            if epoch_storage['LossQ_count'] > 0:
                tensorboard_writer.add_scalar('Epoch/LossQ_Mean', epoch_storage['LossQ_sum'] / epoch_storage['LossQ_count'], epoch)
            
            if epoch_storage['LossPi_count'] > 0:
                tensorboard_writer.add_scalar('Epoch/LossPi_Mean', epoch_storage['LossPi_sum'] / epoch_storage['LossPi_count'], epoch)
            
            # Q-values and memory statistics
            if epoch_storage['Q1Vals_count'] > 0:
                tensorboard_writer.add_scalar('Epoch/Q1Vals_Mean', epoch_storage['Q1Vals_sum'] / epoch_storage['Q1Vals_count'], epoch)
                tensorboard_writer.add_scalar('Epoch/Q1Vals_Min', epoch_storage['Q1Vals_min'], epoch)
                tensorboard_writer.add_scalar('Epoch/Q1Vals_Max', epoch_storage['Q1Vals_max'], epoch)
        
        # Save best model
        if epoch % test_freq == 0 and current_test_ret > best_test_ret:
            best_test_ret = current_test_ret
            best_model_saved = True
            
            print(f"\n🎯 NEW BEST MODEL! Test return: {best_test_ret:.3f} (Epoch {epoch})")
            print(f"   Achieved with {n_envs} parallel environments")
            
            # Save the best model
            fpath = osp.join(output_dir, 'pyt_save')
            os.makedirs(fpath, exist_ok=True)
            
            context_fname = osp.join(fpath, 'best-context.pt')
            model_fname = osp.join(fpath, 'best-model.pt')
            
            # Save context
            context_elements = {
                'env_manager': env_manager,
                'replay_buffer': replay_buffer,
                'epoch_storage': epoch_storage,
                'start_time': start_time, 
                't': t,
                'epoch': epoch,
                'best_test_ret': best_test_ret,
                'n_envs': n_envs  # Save number of environments used
            }
            
            # Save model
            model_elements = {
                'ac_state_dict': ac.state_dict(),
                'target_ac_state_dict': ac_targ.state_dict(),
                'pi_optimizer_state_dict': pi_optimizer.state_dict(),
                'q_optimizer_state_dict': q_optimizer.state_dict(),
                'epoch': epoch,
                'best_test_ret': best_test_ret,
                'total_steps': t,
                'n_envs': n_envs
            }
            
            torch.save(context_elements, context_fname, pickle_protocol=4)
            torch.save(model_elements, model_fname, pickle_protocol=4)
            
            context_size_mb = osp.getsize(context_fname) / (1024**2)
            model_size_mb = osp.getsize(model_fname) / (1024**2)
            print(f"   💾 Saved - Context: {context_size_mb:.1f}MB, Model: {model_size_mb:.1f}MB")

        # Clear epoch storage
        epoch_storage['EpRet'] = []
        epoch_storage['EpLen'] = []
        if epoch % test_freq == 0:
            epoch_storage['TestEpRet'] = []
            epoch_storage['TestEpLen'] = []
        
        # Reset running statistics
        for key in epoch_storage.keys():
            if key.endswith('_count'):
                epoch_storage[key] = 0
            elif key.endswith('_sum'):
                epoch_storage[key] = 0.0
            elif key.endswith('_min'):
                epoch_storage[key] = float('inf')
            elif key.endswith('_max'):
                epoch_storage[key] = float('-inf')
        
        # Print epoch summary
        print(f"Epoch {epoch} Summary:")
        if epoch % test_freq == 0:
            print(f"  Test Return: {current_test_ret:.3f} (Best: {best_test_ret:.3f})")
        else:
            print(f"  Latest Test Return: {latest_test_ret:.3f} (Best: {best_test_ret:.3f})")
        print(f"  Total Steps: {t}")
        print(f"  Parallel Environments: {n_envs}")
        print(f"  Time Elapsed: {time.time() - start_time:.1f}s")
        print(f"  Estimated Speedup: {n_envs}x")
    
    # Clean up
    if use_tensorboard and tensorboard_writer is not None:
        tensorboard_writer.close()
        print("TensorBoard logging completed.")
    
    # Clean up temporary files
    if use_beam_env and _local_file_paths:
        import shutil
        # Get the temporary directory from any local file path
        temp_dir = os.path.dirname(list(_local_file_paths.values())[0])
        try:
            shutil.rmtree(temp_dir)
            print(f"🗑️  Cleaned up temporary files from {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"Final best test return: {best_test_ret:.3f}")
    print(f"Total training time: {time.time() - start_time:.1f}s")
    print(f"Parallel environments used: {n_envs}")


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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume_exp_dir', type=str, default=None)
    parser.add_argument('--env_name', type=str, default='HalfCheetahBulletEnv-v0')
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--max_hist_len', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=4, help="Number of parallel environments")
    
    # Neural network architecture
    parser.add_argument('--use_double_critic', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--use_target_policy_smooth', type=str2bool, nargs='?', const=True, default=True)
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
    
    # Beam environment
    parser.add_argument('--use_beam_env', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--optimal_phases_path', type=str, default='')
    
    # Logging and saving
    parser.add_argument('--exp_name', type=str, default='lstm_td3_parallel')
    parser.add_argument("--data_dir", type=str, default='spinup_data_lstm_gate')
    parser.add_argument('--save_freq', type=int, default=1)
    parser.add_argument('--use_tensorboard', type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--tensorboard_log_freq', type=int, default=100)
    parser.add_argument('--render_freq', type=int, default=0)
    parser.add_argument('--test_freq', type=int, default=10)
    
    args = parser.parse_args()

    # Handle current feature extraction
    if args.critic_cur_feature_hid_sizes is None:
        args.critic_cur_feature_hid_sizes = []
    if args.actor_cur_feature_hid_sizes is None:
        args.actor_cur_feature_hid_sizes = []

    # Handle resuming experiment
    if args.resume_exp_dir is not None:
        resume_exp_dir = args.resume_exp_dir
        config_path = osp.join(args.resume_exp_dir, 'config.json')
        with open(config_path, 'r') as config_file:
            config_json = json.load(config_file)
        config_json['resume_exp_dir'] = resume_exp_dir
        
        output = json.dumps(config_json, separators=(',', ':\t'), indent=4, sort_keys=True)
        print(colorize('Loading config:\n', color='cyan', bold=True))
        print(output)
        
        logger_kwargs = config_json["logger_kwargs"]
        config_json.pop('logger', None)
        args = json.loads(json.dumps(config_json), object_hook=lambda d: namedtuple('args', d.keys())(*d.values()))
    else:
        args.data_dir = osp.join(
            osp.dirname('/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/lstm_td3_logs'),
            args.data_dir)
        logger_kwargs = {}

    # Determine number of environments
    if args.n_envs <= 0:
        args.n_envs = max(1, os.cpu_count() - 2)
    
    print(f"🚀 Starting LSTM-TD3 with {args.n_envs} parallel environments")
    print(f"Expected speedup: ~{args.n_envs}x for data collection")

    lstm_td3_parallel(
        resume_exp_dir=args.resume_exp_dir,
        env_name=args.env_name,
        gamma=args.gamma, 
        seed=args.seed, 
        epochs=args.epochs,
        max_hist_len=args.max_hist_len,
        n_envs=args.n_envs,
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