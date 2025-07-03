#!/usr/bin/env python3
"""
load_trained_policy.py

Script to load a trained LSTM-TD3 policy and use it for inference.
Shows how to extract the actor network from saved checkpoints.
"""

import os
import sys
import torch
import numpy as np
import pickle

# Add the lstm_td3 module to the path
sys.path.append('/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator')

from lstm_td3.lstm_td3_main import MLPActorCritic, make_beam_env

class TrainedLSTMTD3Policy:
    """Wrapper for loading and using trained LSTM-TD3 policies"""
    
    def __init__(self, model_checkpoint_path, env_config=None):
        """
        Load trained policy from checkpoint
        
        Args:
            model_checkpoint_path: Path to checkpoint-model-Step-X-verified.pt file
            env_config: Optional environment configuration (obs_dim, act_dim, etc.)
        """
        self.model_path = model_checkpoint_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load checkpoint
        print(f"Loading model from: {model_checkpoint_path}")
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        
        # Extract model state
        self.ac_state_dict = checkpoint['ac_state_dict']
        
        # If env_config not provided, try to infer from state dict
        if env_config is None:
            env_config = self._infer_config_from_state_dict()
            
        self.obs_dim = env_config['obs_dim']
        self.act_dim = env_config['act_dim']
        self.act_limit = env_config['act_limit']
        
        # Create actor-critic with same architecture as training
        self.actor_critic = MLPActorCritic(
            obs_dim=self.obs_dim,
            act_dim=self.act_dim, 
            act_limit=self.act_limit,
            # Use default architecture - adjust if you used different settings
            critic_mem_pre_lstm_hid_sizes=(128,),
            critic_mem_lstm_hid_sizes=(128,),
            critic_mem_after_lstm_hid_size=(),
            critic_cur_feature_hid_sizes=(128, 128),
            critic_post_comb_hid_sizes=(128,),
            actor_mem_pre_lstm_hid_sizes=(128,),
            actor_mem_lstm_hid_sizes=(128,),
            actor_mem_after_lstm_hid_size=(),
            actor_cur_feature_hid_sizes=(128, 128),
            actor_post_comb_hid_sizes=(128,)
        )
        
        # Load trained weights
        self.actor_critic.load_state_dict(self.ac_state_dict)
        self.actor_critic.to(self.device)
        self.actor_critic.eval()  # Set to evaluation mode
        
        print(f"✅ Policy loaded successfully!")
        print(f"   Observation dim: {self.obs_dim}")
        print(f"   Action dim: {self.act_dim}")
        print(f"   Action limit: {self.act_limit}")
        print(f"   Device: {self.device}")
        
    def _infer_config_from_state_dict(self):
        """Try to infer environment config from model state dict"""
        # Look for input/output layer shapes to infer dimensions
        try:
            # Actor's current feature layer gives us obs_dim
            pi_first_layer = 'pi.cur_feature_layers.0.weight'
            if pi_first_layer in self.ac_state_dict:
                obs_dim = self.ac_state_dict[pi_first_layer].shape[1]
            else:
                obs_dim = 1001  # Default for beam environment
                
            # Actor's final layer gives us act_dim  
            pi_last_layer = 'pi.post_combined_layers.-1.weight'
            for key in self.ac_state_dict.keys():
                if 'pi.post_combined_layers' in key and 'weight' in key:
                    act_dim = self.ac_state_dict[key].shape[0]
                    break
            else:
                act_dim = 1  # Default
                
            act_limit = 1.0  # Default - may need adjustment
            
            print("ℹ️  Inferred configuration from model:")
            print(f"   obs_dim: {obs_dim}, act_dim: {act_dim}, act_limit: {act_limit}")
            
            return {'obs_dim': obs_dim, 'act_dim': act_dim, 'act_limit': act_limit}
            
        except Exception as e:
            print(f"⚠️  Could not infer config: {e}")
            # Return beam environment defaults
            return {'obs_dim': 1001, 'act_dim': 1, 'act_limit': 0.785}  # π/4 for beam env
    
    def predict(self, observation, history_obs=None, history_actions=None, history_length=None):
        """
        Get action from policy given observation and history
        
        Args:
            observation: Current observation (numpy array)
            history_obs: History of observations (numpy array, shape: [hist_len, obs_dim])
            history_actions: History of actions (numpy array, shape: [hist_len, act_dim])  
            history_length: Length of valid history (int)
            
        Returns:
            action: Action to take (numpy array)
        """
        with torch.no_grad():
            # Convert observation to tensor
            obs_tensor = torch.as_tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            # Handle history
            if history_obs is None or history_actions is None or history_length is None:
                # No history - use dummy history
                hist_obs = torch.zeros(1, 1, self.obs_dim).to(self.device)
                hist_act = torch.zeros(1, 1, self.act_dim).to(self.device)
                hist_len = torch.zeros(1).to(self.device)
            else:
                # Use provided history
                hist_obs = torch.as_tensor(history_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                hist_act = torch.as_tensor(history_actions, dtype=torch.float32).unsqueeze(0).to(self.device)
                hist_len = torch.as_tensor([history_length], dtype=torch.float32).to(self.device)
            
            # Get action from policy
            action, _ = self.actor_critic.pi(obs_tensor, hist_obs, hist_act, hist_len)
            return action.cpu().numpy().flatten()
    
    def act(self, observation, **kwargs):
        """Simple interface compatible with Gym environments"""
        return self.predict(observation, **kwargs)

def load_best_checkpoint(experiment_dir):
    """Find and load the best model checkpoint in an experiment directory"""
    checkpoint_dir = os.path.join(experiment_dir, 'pyt_save')
    
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"No checkpoint directory found: {checkpoint_dir}")
    
    # Try new format first (best-model.pt)
    best_model_path = os.path.join(checkpoint_dir, 'best-model.pt')
    
    if os.path.exists(best_model_path):
        print("Found best model checkpoint")
        return best_model_path
    
    # Fallback to old format for backward compatibility
    print("Best model not found, looking for latest checkpoint...")
    model_files = []
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith('checkpoint-model-') and filename.endswith('-verified.pt'):
            # Extract step number
            try:
                step = int(filename.split('-')[3])
                model_files.append((step, os.path.join(checkpoint_dir, filename)))
            except (IndexError, ValueError):
                continue
    
    if not model_files:
        raise FileNotFoundError(f"No model checkpoint files found in {checkpoint_dir}")
    
    # Get latest checkpoint
    model_files.sort(key=lambda x: x[0])  # Sort by step number
    latest_step, latest_file = model_files[-1]
    
    print(f"Found latest checkpoint: Step {latest_step}")
    return latest_file

def demo_policy_usage():
    """Demonstrate how to load and use a trained policy"""
    
    # Example: Load from specific experiment
    experiment_dir = "/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/lstm_td3_beam_logs/2025-06-30_lstm_td3_beam_temporal_memory/2025-06-30_15-57-34-lstm_td3_beam_temporal_memory_s42"
    
    try:
        # Load best checkpoint
        model_path = load_best_checkpoint(experiment_dir)
        
        # Create policy
        policy = TrainedLSTMTD3Policy(model_path)
        
        # Optional: Create environment to test
        optimal_phases_path = '/afs/cern.ch/work/a/apastina/DatasetGen/datasetgenerator/optimal_phases_w_imp.pkl'
        if os.path.exists(optimal_phases_path):
            env = make_beam_env(optimal_phases_path)
            
            print("\n🧪 Testing policy on environment:")
            obs, _ = env.reset()
            
            for step in range(5):
                action = policy.act(obs)
                print(f"   Step {step}: action = {action}")
                
                obs, reward, done, truncated, info = env.step(action)
                print(f"   → reward = {reward:.3f}, done = {done}")
                
                if done or truncated:
                    obs, _ = env.reset()
                    print("   Environment reset")
        else:
            print("⚠️  Optimal phases file not found - skipping environment test")
            
            # Test with dummy observation
            print("\n🧪 Testing policy with dummy observation:")
            dummy_obs = np.random.randn(1001)  # Beam environment observation size
            action = policy.act(dummy_obs)
            print(f"   Dummy observation → action = {action}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        print("Make sure the experiment directory path is correct and contains checkpoints")

if __name__ == "__main__":
    print("=" * 60)
    print("LSTM-TD3 Policy Loading Demo")
    print("=" * 60)
    
    demo_policy_usage() 