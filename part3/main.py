
import gymnasium as gym
from custom_car_env import CustomCarRacing
from agents import Agent, RandomAgent, ManualAgent, HeuristicAgent

class Trainer:
    """
    Manages the interaction between the Environment and the Agent.
    """
    def __init__(self, env: CustomCarRacing, agent: Agent, episodes: int = 5):
        self.env = env
        self.agent = agent
        self.episodes = episodes

    def run(self):
        for ep in range(self.episodes):
            obs = self.env.reset()
            self.agent.reset()
            total_reward = 0.0
            step = 0
            
            # For manual agent, we need to ensure the window is active and events are processed
            # The env.render() usually handles the pygame event loop if it's in human mode
            # But let's make sure we pump events if our agent needs them.
            
            print(f"--- Episode {ep + 1} Start ---")
            
            while True:
                # Select action
                action = self.agent.select_action(obs)
                
                # Step environment
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                # Learn (optional)
                self.agent.observe(obs, action, reward, done, next_obs, info)

                obs = next_obs
                total_reward += reward
                step += 1

                if done:
                    print(f"Episode {ep + 1} Done: steps={step}, total_reward={total_reward:.2f}")
                    break

def main():
    print("==========================================")
    print("   WELCOME TO AWESOME CAR RACING OOP DEMO")
    print("==========================================")
    print("Select Mode:")
    print("1. Manual Drive (You control with Arrow Keys)")
    print("2. Auto Pilot (Heuristic Agent)")
    print("3. Random Chaos (Baseline)")
    
    choice = input("Enter choice (1-3): ").strip()
    
    # Polymorphism in action: We initialize specific agents but treat them uniformly
    env = CustomCarRacing(render_mode="human", lap_complete_percent=0.95)
    agent = None
    episodes = 3
    
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
    else:
        print("Invalid choice, defaulting to Auto Pilot.")
        agent = HeuristicAgent(env)

    try:
        trainer = Trainer(env, agent, episodes=episodes)
        trainer.run()
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    finally:
        env.close()
        print("Environment closed.")

if __name__ == "__main__":
    main()
