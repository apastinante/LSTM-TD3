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
from lstm_td3.utils.logx import setup_logger_kwargs

def main():
    """
    Example usage of LSTM-TD3 with beam environment
    """
    
    # Path to optimal phases (you'll need to adjust this path)
    optimal_phases_path = '/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/optimal_phases_w_imp.pkl'
    
    # LSTM-TD3 hyperparameters optimized for beam environment
    # The key insight is that we want longer memory for temporal consistency
    # and smaller networks to avoid overfitting to the complex beam physics
    
    config = {
        'use_beam_env': True,
        'optimal_phases_path': optimal_phases_path,
        'seed': 44,
        'epochs': round(5000*5/3),  # Reduced from default for testing
        'steps_per_epoch': 30,  # Reduced for faster testing
        'max_hist_len': 5,  # IMPORTANT: Memory of last 10 steps for temporal consistency
        
        # Learning parameters - conservative for stability
        'gamma': 0.99,  # High discount for long-term stability
        'pi_lr': 5e-4,  # Lower learning rate for policy
        'q_lr': 8e-4,   # Slightly higher for critic
        'polyak': 0.995,  # Slow target updates
        
        # Exploration parameters - smaller for beam environment
        'act_noise': 0.05,  # Reduced action noise (equivalent to ~3 degrees)
        'target_noise': 0.02,  # Very small target noise
        'noise_clip': 0.1,
        
        # Training parameters
        'batch_size': 128,
        'start_steps': 5000,  # Random exploration steps
        'update_after': 2000,
        'update_every': 2,    # More frequent updates
        'policy_delay': 2,    # Delayed policy updates
        
        # LSTM Architecture - optimized for beam environment
        # Smaller networks to avoid overfitting, focus on temporal patterns
        
        # Critic LSTM parameters
        'critic_mem_pre_lstm_hid_sizes': (64,),  # Pre-processing
        'critic_mem_lstm_hid_sizes': (128,),     # Main LSTM memory
        'critic_mem_after_lstm_hid_size': (64,), # Post-LSTM processing
        'critic_cur_feature_hid_sizes': (128, 64), # Current observation features
        'critic_post_comb_hid_sizes': (128,),    # Final combination
        'critic_hist_with_past_act': True,       # Include past actions in memory (VERY IMPORTANT)
        
        # Actor LSTM parameters (similar but slightly smaller)
        'actor_mem_pre_lstm_hid_sizes': (64,),
        'actor_mem_lstm_hid_sizes': (128,),
        'actor_mem_after_lstm_hid_size': (64,),
        'actor_cur_feature_hid_sizes': (128, 64),
        'actor_post_comb_hid_sizes': (64,),
        'actor_hist_with_past_act': True,
        
        # Environment specific
        'max_ep_len': 30,  # Max steps per episode (matches beam environment)
        'num_test_episodes': 4,
        
        # Experiment name
        'exp_name': 'lstm_td3_beam_temporal_memory_updated',

        # Replay buffer size
        'replay_size': 200_000,
        
        # TensorBoard logging (REQUIRED - replaces progress.txt logging)
        'use_tensorboard': True,
        'tensorboard_log_freq': 50,  # Log every 50 steps during training (less frequent for performance)
        'save_freq': 100,  # Save models every 100 epochs
        'render_freq': 500,  # Save models every 500 epochs
    }
    
    # Setup logging
    logger_kwargs = setup_logger_kwargs(
        config['exp_name'], 
        config['seed'], 
        data_dir='/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/lstm_td3_beam_logs_updated', 
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
    
    # Run LSTM-TD3 training
    if config['use_tensorboard']:
        lstm_td3_tb(
            logger_kwargs=logger_kwargs,
            **config
        )
    else:
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