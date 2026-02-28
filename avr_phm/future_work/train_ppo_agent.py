"""
train_ppo_agent.py — SOTA 2025 Prescriptive Maintenance RL Agent

This script implements Proximal Policy Optimization (PPO) via `Stable-Baselines3`
to train a Reinforcement Learning agent to make optimal maintenance decisions
based on the prognostic outputs of the Physics-Informed Neural Network (PINN).

References (late 2024 / 2025 SOTA):
- Transformer-Driven Deep Reinforcement Learning-Enabled Prescriptive Maintenance Framework
- Adaptive Reinforcement Learning Framework for Real-Time Tool Wear Optimization
- Deep Reinforcement Learning-based prescriptive maintenance approaches
"""

import os
import sys

# Crucial NumPy 2.x Hotfix:
# Stable-Baselines3 checks for TensorBoard on import, which crashes on Windows
# under PyTorch 2.x/NumPy 2.x due to a deprecated `np.complex_` alias in TensorFlow.
# By mocking the import natively, we bypass the crash without breaking PyTorch's backend.
from unittest.mock import MagicMock
sys.modules['tensorboard'] = MagicMock()
sys.modules['tensorboard.compat'] = MagicMock()
sys.modules['tensorboard.compat.tensorflow_stub'] = MagicMock()

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

# Add parent directory to path to allow importing the main project modules
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

# Import the architectural blueprint for the Environment from tier_1_concepts
from models.tier_1_concepts import AVRMaitenanceEnv

class MockPINN:
    """A dummy PINN that returns structured outputs simulating the real PINN."""
    def __init__(self):
        pass
    def __call__(self, sensor_data):
        return {
            "rul": 50000.0,
            "fault_1s_prob": 0.01
        }

def train_prescriptive_agent():
    # Setup directories
    log_dir = os.path.join(BASE_DIR, "future_work", "logs")
    model_dir = os.path.join(BASE_DIR, "future_work", "models")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("Initializing AVR Prescriptive Maintenance Environment...")
    
    # We purposefully vectorize the environment to allow the PPO agent to gather
    # asynchronous experiences faster across multiple simulation instances.
    n_envs = 4
    dummy_pinn = MockPINN()
    env = make_vec_env(lambda: AVRMaitenanceEnv(pinn_model=dummy_pinn), n_envs=n_envs)
    
    # Setup Evaluation Callback
    # This evaluates the agent every 10,000 steps and saves the best model
    # based exclusively on the highest reward (maximizing machinery uptime vs cost)
    eval_env = make_vec_env(lambda: AVRMaitenanceEnv(pinn_model=dummy_pinn), n_envs=1)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=model_dir,
        log_path=log_dir,
        eval_freq=max(10000 // n_envs, 1),
        deterministic=True,
        render=False
    )

    # Instantiate the modern PPO Algorithm (Proximal Policy Optimization) 
    # State-of-the-Art continuous/discrete action space solver as of 2024/2025.
    print(f"Instantiating PPO Agent. Device mapping: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99 # Discount factor emphasizes long-term machinery survival
    )

    # Train the agent
    total_timesteps = 100_000 # Configurable: Increase to 2M+ for published research
    print(f"\\n[TRAIN] Commencing PPO Optimization for {total_timesteps} steps...")
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=eval_callback,
        tb_log_name="PPO_AVR_Maintenance"
    )

    # Save final model
    final_path = os.path.join(model_dir, "ppo_avr_final")
    model.save(final_path)
    print(f"\\n[SUCCESS] Final model saved to {final_path}.zip")

    # Example: How to run inference with the trained agent
    print("\\n[TEST] Running trained agent on 10 steps of evaluation sequence:")
    obs = eval_env.reset()
    for i in range(10):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = eval_env.step(action)
        action_names = ["Do Nothing", "Schedule Maintenance", "Immediate Shutdown"]
        print(f"Step {i+1}: AI Predicted Action: {action_names[action[0]]} | Reward: {rewards[0]:.2f}")

if __name__ == "__main__":
    try:
        import stable_baselines3
    except ImportError:
        print("ERROR: stable-baselines3 is required to train the PPO agent.")
        print("Please run: pip install stable-baselines3")
        sys.exit(1)
        
    train_prescriptive_agent()
