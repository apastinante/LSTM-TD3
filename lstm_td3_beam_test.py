#!/usr/bin/env python3
"""
LSTM-TD3 Beam Environment Test Script

This script demonstrates how to use the modified LSTM-TD3 implementation with 
the beam environment for particle accelerator beam profile optimization.

The LSTM provides temporal memory that should help reduce oscillations in 
beam phase adjustments by considering the history of beam profiles and actions.
"""

import sys
import os

# Add lstm_td3 to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'lstm_td3'))

from lstm_td3.lstm_td3_tb_main import lstm_td3 as lstm_td3_tb
from lstm_td3.lstm_td3_main import lstm_td3
from lstm_td3.lstm_td3_tb_main_parallel import lstm_td3_parallel as lstm_td3_tb_parallel
from lstm_td3.lstm_td3_tb_main_parallel_per import lstm_td3_parallel_per as lstm_td3_tb_parallel_per
from lstm_td3.lstm_td3_tb_main_parallel_per_simp_reward import lstm_td3_parallel_per_simple_reward as lstm_td3_tb_parallel_per_simple_reward
from lstm_td3.utils.logx import setup_logger_kwargs

def main():
    """
    Example usage of LSTM-TD3 with beam environment
    """
    
    # Path to optimal phases (you'll need to adjust this path)
    optimal_phases_path = os.path.join(os.path.dirname(__file__), 'optimal_phases_w_imp.pkl')
    
    # LSTM-TD3 hyperparameters optimized for beam environment
    # The key insight is that we want longer memory for temporal consistency
    # and smaller networks to avoid overfitting to the complex beam physics
    
    config = {
        'use_beam_env': True,
        'optimal_phases_path': optimal_phases_path,
        'seed': 30,
        'epochs': 20000,  # Amount of epochs
        'steps_per_epoch': 30,  # Amount of steps per epoch
        'max_hist_len': 5,  # Memory of last 5 steps for temporal consistency

        # Environment specific
        'max_ep_len': 30,  # Max steps per episode (has to match the max_ep_len)
        'num_test_episodes': 4,
        'profile_slices': 1000, # Number of bins in the profile
        'noise_level': 0.0, # Noise level for the returned profiles

        # Learning parameters 
        'gamma': 0.98,  # High discount for long-term stability
        'pi_lr': 1e-4,  # Learning rate for policy
        'q_lr': 2e-4,   # Learning rate for critic
        'polyak': 0.995,  # Slow target updates
        
        # Exploration parameters 
        'act_noise': 0.08,  # Action noise (equivalent to ~5 degrees)
        'target_noise': 0.03,  # Target noise
        'noise_clip': 0.1,
        
        # Training parameters
        'batch_size': 256,
        'start_steps': 20000,  # Random exploration steps
        'update_after': 10000, # Update after 10000 steps of getting data
        'update_every': 15,    # Update every 15 steps
        'policy_delay': 3,    # Policy delay (TD3 Specific)
        
        # LSTM Architecture - optimized for beam environment
        # Smaller networks to avoid overfitting, focus on temporal patterns
        
        # Critic LSTM parameters
        'critic_mem_pre_lstm_hid_sizes': (128,),  # Pre-processing
        'critic_mem_lstm_hid_sizes': (256,),     # Main LSTM memory
        'critic_mem_after_lstm_hid_size': (128,), # Post-LSTM processing
        'critic_cur_feature_hid_sizes': (256, 128), # Current observation features
        'critic_post_comb_hid_sizes': (128,),    # Final combination
        'critic_hist_with_past_act': True,       # Include past actions in memory (VERY IMPORTANT)
        
        # Actor LSTM parameters 
        'actor_mem_pre_lstm_hid_sizes': (128,),
        'actor_mem_lstm_hid_sizes': (256,),
        'actor_mem_after_lstm_hid_size': (128,),
        'actor_cur_feature_hid_sizes': (256, 128),
        'actor_post_comb_hid_sizes': (128,),
        'actor_hist_with_past_act': True,
        
        
        
        # Experiment name
        'exp_name': 'lstm_td3_beam_temporal_memory_per_1000_no_noise',

        # Replay buffer size
        'replay_size': 1_000_000,
        
        # TensorBoard logging (REQUIRED - replaces progress.txt logging)
        'use_tensorboard': True,
        'tensorboard_log_freq': 50,  # Log every 50 steps during training (less frequent for performance)
        'save_freq': 100,  # Save models every 100 epochs

        # Testing parameters
        'render_freq': 10000,  # Render every 10000 epochs (show performance)
        'test_freq': 50,  # Test every 50 epochs

        # Parallel training parameters
        'n_envs': 0, # Number of environments to run in parallel (0 means all but 2 cores are used)

        # PER parameters
        'use_per': True, # Use Prioritized Experience Replay (PER)
        'use_unified_buffer': True, # Use a single buffer for all environments (slight overhead but better performance)

        # Resume training parameters
        'resume_exp_dir': None, 
        
        
    }
    use_simple_reward = False
    # Setup logging
    logger_kwargs = setup_logger_kwargs(
        config['exp_name'], 
        config['seed'], 
        data_dir= os.path.join(os.path.dirname(__file__), 'logs'), 
        datestamp=True
    )
    
    print("=" * 60)
    print("LSTM-TD3 Beam Environment Training (TensorBoard Version)")
    print("=" * 60)
    print(f"Using beam environment with optimal phases from: {optimal_phases_path}")
    print(f"LSTM memory length: {config['max_hist_len']} steps")
    print(f"Training for {config['epochs']} epochs")
    print(f"Experiment name: {config['exp_name']}")
    print(f"TensorBoard logging: {'✅ REQUIRED (replaces progress.txt)' if config['use_tensorboard'] else '❌ DISABLED - NO LOGGING!'}")
    print(f"Logs directory: {logger_kwargs['output_dir']}")
    print("=" * 60)
    
    # Add TensorBoard log directory to logger kwargs
    if config['use_tensorboard']:
        tensorboard_dir = os.path.join(logger_kwargs['output_dir'], 'tensorboard')
        logger_kwargs['tensorboard_dir'] = tensorboard_dir
        print(f"TensorBoard logs: {tensorboard_dir}")
        print("Monitor training with: tensorboard --logdir=" + tensorboard_dir)
        print("=" * 60)
        if config['n_envs'] > 1:
            # config['update_every'] = config['update_every'] * config['n_envs']
            if config['use_per']:
                lstm_td3_tb_parallel_per(
                    logger_kwargs=logger_kwargs,
                    **config
                ) 
            else:
                lstm_td3_tb_parallel(
                    logger_kwargs=logger_kwargs,
                    **config
                )
        elif config['n_envs'] <= 0:
            config['n_envs'] = max(1, os.cpu_count()-2)
            # config['update_every'] = config['update_every'] * config['n_envs']
            if config['use_per']:
                if use_simple_reward:
                    lstm_td3_tb_parallel_per_simple_reward(
                        logger_kwargs=logger_kwargs,
                        **config
                    )
                else:
                    lstm_td3_tb_parallel_per(
                    logger_kwargs=logger_kwargs,
                    **config
                    )
            else:
                lstm_td3_tb_parallel(
                logger_kwargs=logger_kwargs,
                **config
            )
        else:
            # Run LSTM-TD3 training
            lstm_td3_tb(
                logger_kwargs=logger_kwargs,
                **config
            )
    else:
        # Parallel training is not supported without TensorBoard
        lstm_td3(
            logger_kwargs=logger_kwargs,
            **config
        )

    print("🎉 Training completed!")
    print(f"📁 Configuration saved to: {logger_kwargs['output_dir']}")
    if config['use_tensorboard']:
        print(f"📊 TensorBoard logs: {logger_kwargs['tensorboard_dir']}")
        print(f"🌐 View results with: tensorboard --logdir={logger_kwargs['tensorboard_dir']}")
    else:
        print("⚠️  WARNING: No logs saved - TensorBoard was disabled!")

if __name__ == '__main__':
    main() 