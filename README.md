# LSTM-TD3: Enhanced Implementation for Particle Accelerator Beam Optimization

This is an enhanced implementation of LSTM-TD3 (Long Short-Term Memory based Twin Delayed Deep Deterministic Policy Gradient) originally proposed in [Memory-based Deep Reinforcement Learning for POMDP](https://arxiv.org/pdf/2102.12344.pdf). 

The implementation has been significantly extended with **parallel processing**, **Prioritized Experience Replay (PER)**, **TensorBoard logging**, and a **custom beam environment** for particle accelerator physics applications, specifically for optimizing the 2nd harmonic phase (Φ₂) in double-harmonic RF cavity systems.

## 🚀 Key Improvements & Features

### Core Enhancements
- **Memory-Efficient Logging**: TensorBoard integration for reduced memory usage and better visualization
- **Parallel Environment Processing**: Multi-environment training for significantly faster data collection
- **Prioritized Experience Replay (PER)**: Both unified and separate buffer implementations with importance sampling
- **Custom Beam Environment**: Physics-based environment for particle accelerator beam profile optimization
- **POMDP Support**: Comprehensive wrapper for partial observability scenarios
- **Multiple Training Variants**: Different combinations of features for various use cases

### Physics Application
- **Domain**: Particle accelerator physics (Proton Synchrotron Booster at CERN)
- **Problem**: Optimize 2nd harmonic RF cavity phasing to compensate for space charge and beam loading effects
- **Goal**: Maintain uniform longitudinal beam profiles and prevent beam instabilities
- **Simulation**: Integration with BLonD (Beam Longitudinal Dynamics) code

## 📦 Installation

### Prerequisites
- Python 3.6+
- PyTorch (install from [official website](https://pytorch.org/get-started/locally/))
- BLonD simulation package (for beam environment)

### Install from Source
```bash
git clone https://github.com/apastinante/LSTM-TD3.git
cd LSTM-TD3
pip install -e .
```

### Dependencies
The setup.py automatically installs:
- torch
- gym/gymnasium 
- numpy
- pandas
- pybullet (for standard environments)
- blond (for beam physics simulations)
- joblib

## 🎯 Available Training Variants

### 1. Basic LSTM-TD3
**File**: `lstm_td3/lstm_td3_main.py`
- Original implementation with basic improvements
- Supports both standard gym and beam environments

```bash
python lstm_td3/lstm_td3_main.py
```

### 2. LSTM-TD3 with TensorBoard
**File**: `lstm_td3/lstm_td3_tb_main.py` 
- Added TensorBoard logging for better monitoring
- Memory-efficient logging system
- Enhanced testing and rendering capabilities

```bash
python lstm_td3/lstm_td3_tb_main.py
```

### 3. Parallel LSTM-TD3
**File**: `lstm_td3/lstm_td3_tb_main_parallel.py`
- **Multiple parallel environments** for faster training
- Asynchronous action execution across environments
- Proper history buffer management per environment
- Automatic CPU core detection for optimal parallelization

```bash
python lstm_td3/lstm_td3_tb_main_parallel.py
```

### 4. Parallel LSTM-TD3 with PER
**File**: `lstm_td3/lstm_td3_tb_main_parallel_per.py`
- All parallel features **plus Prioritized Experience Replay**
- Both unified and separate buffer implementations
- Configurable PER parameters (α, β, ε)
- Importance sampling with bias correction

```bash
python lstm_td3/lstm_td3_tb_main_parallel_per.py
```

### 5. Parallel LSTM-TD3 with PER (Simplified Reward)
**File**: `lstm_td3/lstm_td3_tb_main_parallel_per_simp_reward.py`
- Same as above but with simplified reward calculation
- Optimized for beam environment applications
- Reduced computational overhead

```bash
python lstm_td3/lstm_td3_tb_main_parallel_per_simp_reward.py
```

## 🔬 Beam Environment Usage

The beam environment is specifically designed for particle accelerator applications:

### Quick Start Example
```python
from lstm_td3.lstm_td3_tb_main_parallel_per import lstm_td3_parallel_per

# Configuration for beam environment
lstm_td3_parallel_per(
    use_beam_env=True,
    optimal_phases_path='optimal_phases_w_imp.pkl',
    n_envs=8,  # Use 8 parallel environments
    use_per=True,  # Enable Prioritized Experience Replay
    epochs=20000,
    steps_per_epoch=30,
    max_hist_len=5,  # LSTM memory length
    profile_slices=1000,  # Profile resolution
    noise_level=0.0  # Simulation noise
)
```

### Using the Test Script
```bash
python lstm_td3_beam_test.py
```

This script demonstrates optimal hyperparameters for beam applications.

## ⚙️ Configuration Parameters

### Parallel Processing
- `n_envs`: Number of parallel environments (default: CPU cores - 2)
- `use_unified_buffer`: Use unified vs. separate PER buffers

### Prioritized Experience Replay
- `use_per`: Enable/disable PER (default: True)
- `per_alpha`: Priority exponent (default: 0.6)
- `per_beta`: Importance sampling exponent (default: 0.4)
- `per_epsilon`: Small constant to avoid zero priorities (default: 1e-6)

### LSTM Architecture
- `max_hist_len`: LSTM memory length (default: 100)
- `critic_mem_lstm_hid_sizes`: LSTM hidden layer sizes
- `actor_mem_lstm_hid_sizes`: Actor LSTM configuration

### Beam Environment
- `use_beam_env`: Enable beam physics environment
- `optimal_phases_path`: Path to reference optimal phases
- `profile_slices`: Number of beam profile bins (default: 1000)
- `noise_level`: Simulation noise level (default: 0.0)

### Logging & Monitoring
- `use_tensorboard`: Enable TensorBoard logging
- `tensorboard_log_freq`: Logging frequency
- `exp_name`: Experiment name for organization

## 📊 Supported Environments

### Standard Environments
- HalfCheetahBulletEnv-v0
- AntBulletEnv-v0
- HopperBulletEnv-v0
- Walker2DBulletEnv-v0
- InvertedPendulumBulletEnv-v0
- InvertedDoublePendulumBulletEnv-v0
- InvertedPendulumSwingupBulletEnv-v0
- ReacherBulletEnv-v0

### POMDP Variants
The POMDP wrapper supports various types of partial observability:
- `remove_velocity`: Remove velocity information
- `flickering`: Random observation masking
- `random_noise`: Gaussian noise addition
- `random_sensor_missing`: Random sensor failures
- Combinations of the above

### Beam Environment
Custom physics-based environment for particle accelerator optimization:
- **State Space**: Beam profile characteristics, phase information
- **Action Space**: 2nd harmonic phase adjustments (Φ₂)
- **Reward**: Based on beam profile uniformity and stability metrics
- **Physics**: Integrated with BLonD simulation package

## 📁 Project Structure

```
LSTM-TD3/
├── lstm_td3/                          # Main package
│   ├── lstm_td3_main.py              # Basic implementation
│   ├── lstm_td3_tb_main.py           # With TensorBoard
│   ├── lstm_td3_tb_main_parallel.py  # Parallel environments
│   ├── lstm_td3_tb_main_parallel_per.py # Parallel + PER
│   ├── lstm_td3_tb_main_parallel_per_simp_reward.py # Simplified reward
│   ├── env_wrapper/                   # Environment wrappers
│   │   ├── env.py                    # Bullet environment utilities
│   │   └── pomdp_wrapper.py          # POMDP implementations
│   ├── utils/                         # Utilities
│   │   ├── logx.py                   # Logging utilities
│   │   ├── plot.py                   # Plotting functions
│   │   └── test_policy.py            # Policy testing
│   └── user_config.py                # User configuration
├── programs/                          # Accelerator data
│   ├── NORMHRS/                      # Normal operation data
│   ├── TOF/                          # Time-of-flight data
│   └── mod_NORMHRS/                  # Modified operation data
├── lstm_td3_beam_test.py             # Beam environment test script
├── optimal_phases_w_imp.pkl          # Reference optimal phases
└── setup.py                          # Installation configuration
```

## 🔧 Advanced Usage

### Custom Beam Environment Configuration
```python
# Advanced beam environment setup
config = {
    'use_beam_env': True,
    'optimal_phases_path': 'optimal_phases_w_imp.pkl',
    'max_ep_len': 30,           # Episode length (acceleration cycle steps)
    'profile_slices': 1000,     # Beam profile resolution
    'noise_level': 0.0,         # Environmental noise
    'n_envs': 16,               # High parallelization
    'use_per': True,            # Prioritized replay
    'per_alpha': 0.6,           # Priority strength
    'max_hist_len': 10,         # Extended LSTM memory
}
```

### Multi-Environment Training with Custom Rewards
```python
# Parallel training with custom reward shaping
lstm_td3_parallel_per_simple_reward(
    use_beam_env=True,
    n_envs=12,
    epochs=50000,
    batch_size=256,
    replay_size=int(5e5),
    use_tensorboard=True,
    tensorboard_log_freq=50
)
```

## 📝 Key Research Contributions

1. **Parallel RL Training**: Efficient multi-environment data collection
2. **Physics-Informed RL**: Custom environment for accelerator physics
3. **Memory-Efficient Implementation**: TensorBoard logging with reduced memory footprint
4. **Advanced Experience Replay**: Both unified and distributed PER implementations
5. **Temporal Consistency**: LSTM memory for stable beam phase control

## 🤝 Contributing

This codebase is designed for particle accelerator physics research. Key areas for contribution:
- Additional environments
- Enhanced parallel processing / GPU parallelization
- Alternative replay mechanisms.

## 📚 Citation

If you use this code for research, please cite the original LSTM-TD3 paper:
```bibtex
@article{lstm_td3_2021,
  title={Memory-based Deep Reinforcement Learning for POMDP},
  author={Meng, Lingheng and others},
  journal={arXiv preprint arXiv:2102.12344},
  year={2021}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
