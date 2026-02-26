"""
run_rl_experiment.py — Standalone Execution for Prescriptive Reinforcement Learning
Year: 2025 SOTA Concept

This script initializes the Gymnasium environment `AVRMaitenanceEnv` from 
`tier_1_concepts.py` and runs a theoretical "Random Agent" through simulating
a predictive maintenance pipeline over 100 days.

Hardware target: CPU
"""

import os
import time

# We handle the gymnasium import safely so the script doesn't completely crash 
# if the library hasn't been pip-installed by the user yet.
try:
    import gymnasium as gym
    from models.tier_1_concepts import AVRMaitenanceEnv
    GYM_AVAILABLE = True
except ImportError:
    GYM_AVAILABLE = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Mock PINN for the RL Environment ────────────────────────────────────────
class MockPINN:
    """A dummy PINN that returns structured outputs simulating the real PINN."""
    def __init__(self):
        pass
    def __call__(self, sensor_data):
        # Returns simulated RUL (Remaining Useful Life) decreasing over time
        return {
            "rul": 50000.0, # seconds
            "fault_1s_prob": 0.01
        }

def run_prescriptive_rl():
    if not GYM_AVAILABLE:
        print("[ERROR] Gymnasium is required to run the Prescriptive RL Environment.")
        print("Please run: `pip install gymnasium` and try again.")
        return

    print("Initializing PINN-Driven RL Maintenance Environment...")
    
    dummy_pinn = MockPINN()
    env = AVRMaitenanceEnv(pinn_model=dummy_pinn)
    
    # Run 5 simulated "Episodes" to failure or successful maintenance
    episodes = 5
    
    for episode in range(1, episodes + 1):
        state, info = env.reset()
        done = False
        total_reward = 0
        step_count = 0
        
        print(f"\n--- Episode {episode} ---")
        print(f"Initial State | V: {state[0]:.1f}, RUL: {state[1]:.0f}s, Fault Risk: {state[2]:.2f}")
        
        while not done:
            # We use a purely random agent to demonstrate the environment loop
            # Real agent would use Deep Q-Network or PPO to choose the action based on PINN RUL
            action = env.action_space.sample() 
            
            action_map = {0: "Operate", 1: "Throttle", 2: "Maintain"}
            
            # Step the environment forward
            next_state, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            step_count += 1
            
            if step_count % 10 == 0 or done:
                print(f"Step {step_count:03d} | Agent Action: '{action_map[action]:<8}' | "
                      f"RUL: {next_state[1]:.0f}s | Reward: {reward:+.1f}")
                time.sleep(0.05) # Slow down output for visual effect
                
        print(f"Episode {episode} Finished in {step_count} steps. Total Reward: {total_reward:+.1f}")
        
    print("\n[RL LOOP COMPLETE]")

if __name__ == "__main__":
    run_prescriptive_rl()
