
import gymnasium as gym
from custom_car_env import CustomCarRacing
from agents import Agent, RandomAgent, ManualAgent, HeuristicAgent, SmartAgent

class Trainer:
    """
    Manages the interaction between the Environment and the Agent.
    """
    def __init__(self, env: CustomCarRacing, agent: Agent, episodes: int = 5, training: bool = True):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.training = training

    def run(self):
        for ep in range(self.episodes):
            obs, _ = self.env.reset()
            self.agent.reset()
            total_reward = 0.0
            step = 0
            patience = 100 # Start with some patience
            
            # For manual agent, we need to ensure the window is active and events are processed
            # The env.render() usually handles the pygame event loop if it's in human mode
            # But let's make sure we pump events if our agent needs them.
            
            print(f"--- Episode {ep + 1} Start ---")
            
            while True:
                # Select action
                action = self.agent.select_action(obs)
                
                # Action Repeat (Frame Skip)
                repeat_reward = 0.0
                frame_skip = 4
                
                for _ in range(frame_skip):
                    # Step environment
                    next_obs, reward, terminated, truncated, info = self.env.step(action)
                    repeat_reward += reward
                    done = terminated or truncated
                    
                    if done:
                        break
                
                # Early Stopping Logic (Progress-Based Patience)
                # Instead of hard limits, we use 'patience'.
                # As long as the car is finding NEW track (repeat_reward > 0), we let it run.
                # If it stops finding track for too long (spinning, stuck, or grass), we kill it.
                
                if repeat_reward > 0:
                    patience = 100 # Reset patience if we made progress
                else:
                    patience -= 1  # Decay patience if no positive progress
                
                if patience <= 0:
                    print(f"Run out of patience (stuck/spinning). Aborting episode.")
                    repeat_reward -= 100.0 # Death penalty to prevent suicide strategy
                    done = True
                
                # Hard limit just in case (e.g. 5000 steps is extremely long for one lap)
                if step > 2000:
                   print("Hard time limit reached. Stopping.")
                   done = True

                # Learn (optional)
                # Note: We observe the total reward of the skipped frames, and the final state
                # CRITICAL Fix: We must observe AFTER applying the death penalty!
                
                if self.training:
                    # Tuning: Neutralize Time Penalty
                    # Real reward includes -0.1 per frame. We want the agent to see 0 for "living" and + for "progress".
                    # We add back 0.1 * frame_skip to the reward it learns from.
                    learning_reward = repeat_reward + (0.1 * frame_skip)
                    
                    self.agent.observe(obs, action, learning_reward, done, next_obs, info)

                obs = next_obs
                total_reward += repeat_reward
                step += 1 # We count 1 decision step (which is 'frame_skip' env steps)

                if done:
                    print(f"Episode {ep + 1} Done: steps={step} (x{frame_skip} frames), total_reward={total_reward:.2f}")
                    break

def main():
    print("==========================================")
    print("   WELCOME TO AWESOME CAR RACING OOP DEMO")
    print("==========================================")
    print("Select Mode:")
    print("1. Manual Drive (You control with Arrow Keys)")
    print("2. Auto Pilot (Heuristic Agent)")
    print("3. Random Chaos (Baseline)")
    print("4. Train Smart Agent (PPO)")
    print("5. Visualize Smart Agent (Load & Play)")
    
    choice = input("Enter choice (1-5): ").strip()
    
    # Polymorphism in action: We initialize specific agents but treat them uniformly
    env = CustomCarRacing(render_mode="human", lap_complete_percent=0.95)
    agent = None
    episodes = 3
    training_enabled = False # Default off for manual/heuristic/random
    
    if choice == '1':
        print("Starting Manual Mode... Use Arrow Keys to Drive!")
        agent = ManualAgent()
        episodes = 10 # Let user play more
    elif choice == '2':
        print("Starting Auto Pilot... Sit back and relax!")
        agent = HeuristicAgent(env)
        episodes = 5
    elif choice == '3':
        print("Starting Random Mode... Good luck!")
        agent = RandomAgent(env.action_space)
    elif choice == '4':
        print("Starting Training Mode... Models will be saved as 'smart_agent_model.pth'.")
        training_enabled = True # Enable training
        
        fast = input("Enable Fast Training (No Render)? (y/n): ").strip().lower()
        if fast == 'y':
            print("Switching to Headless Mode (rgb_array)...")
            env.close()
            # Re-init env with rgb_array to avoid window creation but still get image obs
            env = CustomCarRacing(render_mode="rgb_array", lap_complete_percent=0.95)
        
        # Check if we want to resume? For now start fresh or load if exists
        # We can ask user but for simplicity let's just instantiate
        agent = SmartAgent(env.action_space, model_path="smart_agent_model.pth")
        episodes = 5000 # Run for a long time
    elif choice == '5':
        print("Starting Visualization Mode... Loading 'smart_agent_model.pth'.")
        print("Training is DISABLED. Sit back and watch it drive!")
        training_enabled = False
        # Use human render mode (default)
        agent = SmartAgent(env.action_space, model_path="smart_agent_model.pth")
        episodes = 10
    else:
        print("Invalid choice, defaulting to Auto Pilot.")
        agent = HeuristicAgent(env)

    try:
        trainer = Trainer(env, agent, episodes=episodes, training=training_enabled)
        trainer.run()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
