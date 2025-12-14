
import gymnasium as gym
import pygame
import numpy as np
import math
from custom_car_env import CustomCarRacing
from agents import Agent, RandomAgent, ManualAgent, SmartAgent, OpponentSmartAgent
import pygame, ctypes, time

class Trainer:
    """
    Manages the interaction between the Environment and the Agent.
    Includes Custom Rendering and Mini-map Support.
    """
    def __init__(self, env: CustomCarRacing, agent: Agent, episodes: int = 5, training: bool = True, render_enabled: bool = False):
        self.env = env
        self.agent = agent
        self.episodes = episodes
        self.training = training
        self.render_enabled = render_enabled
        self.track_points = None
        
        if self.render_enabled:
            pygame.init()
            # PiP Mode: Main Screen 800x600
            self.screen = pygame.display.set_mode((800, 600))
            pygame.display.set_caption("Car Racing RL - Neon Edition + Mini-Map")
            self.clock = pygame.time.Clock()
        
        # Sync training state to agent if supported
        if hasattr(self.agent, 'training'):
            self.agent.training = self.training
            
        # NPC Agent (Heuristic) for Car 2
        self.npc_agent = SmartAgent(env.action_space, model_path="smart_agent_model.pth")

        self.env.reset()
        if self.render_enabled:
            pygame.display.flip()
            pygame.event.pump()
            time.sleep(0.1)
            hwnd = pygame.display.get_wm_info()["window"]
            ctypes.windll.user32.ShowWindow(hwnd, 6)
            ctypes.windll.user32.ShowWindow(hwnd, 9)

    def run(self):
        # If in training mode OR visualization of SmartAgent, initialize Opponent Agent as well
        self.opponent_agent = None
        if isinstance(self.agent, SmartAgent):
            # Use dedicated OpponentSmartAgent
            self.opponent_agent = OpponentSmartAgent(self.env.action_space) 
            
            # Sync training state for opponent too
            if hasattr(self.opponent_agent, 'training'):
                 self.opponent_agent.training = self.training
                 
            print("Opponent AI (Blue) Initialized.")

        for ep in range(self.episodes):
            obs, _ = self.env.reset()
            
            # Reset Agents
            self.agent.reset()
            self.track_points = None # Force refresh of track points for new episode
            if self.opponent_agent:
                self.opponent_agent.reset()
            # self.npc_agent.reset() # Removed old logic
            
            total_reward = 0.0
            step = 0
            patience = 100 
            
            # Initial Opponent Observation
            obs_opp = None
            if self.opponent_agent:
                obs_opp = self.env.get_opponent_observation()
            
            print(f"--- Episode {ep + 1} Start ---")
            
            while True:

                if self.render_enabled:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise KeyboardInterrupt

                # Select action
                action = self.agent.select_action(obs)
                
                action_opp = None
                if self.opponent_agent:
                    action_opp = self.opponent_agent.select_action(obs_opp)
                
                # Action Repeat (Frame Skip)
                repeat_reward = 0.0
                frame_skip = 2
                
                # Opponent Cumulative Reward for this frame skip
                opp_repeat_reward = 0.0
                
                for _ in range(frame_skip):
                    # Pass both actions to environment
                    next_obs, reward, terminated, truncated, info = self.env.step(action, opponent_action=action_opp)
                    repeat_reward += reward
                    
                    # --- Opponent Reward Calculation ---
                    if self.opponent_agent and self.env.car2:
                        # Initialize Reward
                        r_opp = -0.1 # Living penalty
                        
                        # Cache Track Points
                        if self.track_points is None:
                             if hasattr(self.env, 'track') and self.env.track:
                                 self.track_points = np.array([t[2:4] for t in self.env.track])
                        
                        if self.track_points is not None:
                            # 1. RACING REWARD (Run the track!)
                            # ---------------------------------
                            p = self.env.car2.hull.position
                            diff = self.track_points - np.array([p[0], p[1]])
                            dists_sq = np.sum(diff**2, axis=1)
                            closest_idx = np.argmin(dists_sq)
                            min_dist = math.sqrt(dists_sq[closest_idx])
                            
                            # Off-track check
                            if min_dist > 10.0:
                                r_opp -= 3.0 # Penalty for being off-track
                            else:
                                # Alignment & Speed Reward
                                next_idx = (closest_idx + 1) % len(self.track_points)
                                next_pt = self.track_points[next_idx]
                                curr_pt = self.track_points[closest_idx]
                                
                                dx = next_pt[0] - curr_pt[0]
                                dy = next_pt[1] - curr_pt[1]
                                track_angle = math.atan2(dy, dx)
                                
                                vel = self.env.car2.hull.linearVelocity
                                speed = np.linalg.norm(vel)
                                v_angle = math.atan2(vel[1], vel[0])
                                
                                angle_diff = v_angle - track_angle
                                while angle_diff > math.pi: angle_diff -= 2*math.pi
                                while angle_diff < -math.pi: angle_diff += 2*math.pi
                                
                                # High weight for racing! (Speed ~50-100)
                                # Reward = Speed * Alignment
                                r_opp += (speed * math.cos(angle_diff)) / 2.0 
                                
                                # 1.1 STABILITY PENALTY (Anti-Spin)
                                # Penalize high angular velocity to prevent spinning for points
                                angular_vel = self.env.car2.hull.angularVelocity
                                r_opp -= abs(angular_vel) * 1.0

                                # 1.2 IDLE PENALTY
                                # Penalize if the car is moving too slowly (camping/stuck)
                                if speed < 5.0:
                                    r_opp -= 5.0
                        
                            # 2. BLOCKING REWARD (Block the Red Car!)
                            # ---------------------------------------
                            # Only if we are on track (don't block if we are lost in the grass)
                            if min_dist <= 10.0 and self.env.car:
                                p1 = np.array(self.env.car.hull.position) # Red Car
                                p2 = np.array(self.env.car2.hull.position) # Blue Car (Self)
                                dist_to_red = np.linalg.norm(p1 - p2)
                                
                                # Find Red Car's approximate track direction
                                diff_red = self.track_points - p1
                                dists_sq_red = np.sum(diff_red**2, axis=1)
                                idx_red = np.argmin(dists_sq_red)
                                next_pt_red = self.track_points[(idx_red + 1) % len(self.track_points)]
                                curr_pt_red = self.track_points[idx_red]
                                track_dir_red = next_pt_red - curr_pt_red
                                track_dir_red /= (np.linalg.norm(track_dir_red) + 1e-6)
                                
                                # Vector from Red to Blue
                                to_blue = p2 - p1
                                dot_prod = np.dot(to_blue, track_dir_red)
                                
                                # Blocking Condition: Ahead of Red AND Close
                                # PLUS: Must be stable (not spinning) and moving generally forward
                                is_stable = abs(angular_vel) < 1.0 and math.cos(angle_diff) > 0.5
                                if dot_prod > 0 and dist_to_red < 15.0 and is_stable:
                                    # Give Bonus (Lower weight than racing)
                                    r_opp += 0.5 
                                    
                        opp_repeat_reward += r_opp
                    # -----------------------------------

                    done = terminated or truncated
                    
                    # Terminate if Opponent is WAY Off-Track (Boundary check)
                    # "Boundary" here means significantly far from track (> 30.0 units)
                    if self.opponent_agent and self.env.car2 and self.env.is_off_track(self.env.car2, max_dist=50.0):
                        done = True
                        opp_repeat_reward -= 50.0 # Heavy Penalty for dying
                        print("Opponent went out of boundary. Episode Terminated.")
                    
                    if done:
                        if terminated:
                            print(f"System: Lap Completed 🏁 (Terminated)")
                        if truncated:
                            print(f"System: Car Out of Bounds / Truncated ☠️ (Truncated)")
                        break
                        

                
                # Get Next Observation for Opponent
                next_obs_opp = None
                if self.opponent_agent:
                    next_obs_opp = self.env.get_opponent_observation()

                # Visualization (Render Loop)
                # Visualization (Render Loop)
                if self.render_enabled:
                    # 1. Main Game View (High-Res from Env) - Full Screen 800x600
                    frame = self.env.render()
                    surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
                    # Already 800x600 natively due to our fix
                    self.screen.blit(surf, (0, 0))
                    
                    # 2. Opponent View (PiP) - Small Box in Bottom Right
                    if self.env.car2 and self.opponent_agent:
                         # Pass accumulate score if available
                         opp_score = getattr(self, 'total_reward_opp', 0.0)
                         # Normalize for display (Continuous reward is much larger than discrete)
                         # Scale down by ~30 to match Red Car's point scale roughly
                         frame_opp = self.env.get_opponent_render(score=opp_score / 30.0)
                         
                         if frame_opp is not None:
                             surf_opp = pygame.surfarray.make_surface(np.transpose(frame_opp, (1, 0, 2)))
                             # Scale down for PiP (e.g., 30% size -> 240x180)
                             pip_w, pip_h = 240, 180
                             surf_opp = pygame.transform.scale(surf_opp, (pip_w, pip_h))
                             
                             # Draw Border
                             border_rect = pygame.Rect(800 - pip_w - 20 - 2, 600 - pip_h - 20 - 2, pip_w + 4, pip_h + 4)
                             pygame.draw.rect(self.screen, (255, 255, 255), border_rect)
                             
                             # Blit PiP
                             self.screen.blit(surf_opp, (800 - pip_w - 20, 600 - pip_h - 20))


                    # 4. Mini-Map Implementation
                    # Move to Top-Left to balance PiP in Bottom-Right
                    map_size = 200
                    mini_map = pygame.Surface((map_size, map_size))
                    mini_map.fill((30, 30, 30)) # Dark background
                    mini_map.set_alpha(220) 
                    
                    if hasattr(self.env, 'track') and len(self.env.track) > 0:
                        # Extract x, z from track
                        track_points = np.array([t[2:4] for t in self.env.track])
                        min_x, max_x = track_points[:, 0].min(), track_points[:, 0].max()
                        min_y, max_y = track_points[:, 1].min(), track_points[:, 1].max()
                        
                        padding = 10
                        width = max_x - min_x
                        height = max_y - min_y
                        scale = (map_size - 2 * padding) / max(width, height) if max(width, height) > 0 else 1.0
                        
                        def to_map(x, y):
                             mx = (x - min_x) * scale + padding + (map_size - 2*padding - width*scale)/2
                             my = (max_y - y) * scale + padding + (map_size - 2*padding - height*scale)/2
                             return int(mx), int(my)

                        # Draw Track
                        for tx, ty in track_points:
                            mx, my = to_map(tx, ty)
                            pygame.draw.circle(mini_map, (200, 200, 200), (mx, my), 2)
                            
                        # Draw Car
                        if hasattr(self.env, 'car') and self.env.car:
                            cx, cy = self.env.car.hull.position
                            cmx, cmy = to_map(cx, cy)
                            pygame.draw.circle(mini_map, (255, 50, 50), (cmx, cmy), 5)
                            
                            # Direction Line
                            angle = self.env.car.hull.angle
                            end_x = cmx + 10 * np.cos(angle + np.pi / 2)
                            end_y = cmy - 10 * np.sin(angle + np.pi / 2)
                            pygame.draw.line(mini_map, (255, 255, 0), (cmx, cmy), (end_x, end_y), 2)

                        # Draw Opponent Car (NPC)
                        if hasattr(self.env, 'car2') and self.env.car2:
                            cx2, cy2 = self.env.car2.hull.position
                            cmx2, cmy2 = to_map(cx2, cy2)
                            pygame.draw.circle(mini_map, (50, 50, 255), (cmx2, cmy2), 5) # Blue dot for NPC
                            
                            # Direction Line for NPC
                            angle2 = self.env.car2.hull.angle
                            end_x2 = cmx2 + 10 * np.cos(angle2 + np.pi / 2)
                            end_y2 = cmy2 - 10 * np.sin(angle2 + np.pi / 2)
                            pygame.draw.line(mini_map, (100, 100, 255), (cmx2, cmy2), (end_x2, end_y2), 2)

                    # Blit Map to Top-Right
                    self.screen.blit(mini_map, (800 - map_size - 10, 10))
                    
                    pygame.display.flip()
                    
                    # 3. FPS Limiting
                    self.clock.tick(60)

                # Logic
                if repeat_reward > 0:
                    patience = 100 
                else:
                    patience -= 1
                
                if patience <= 0:
                    print(f"Run out of patience. Aborting.")
                    repeat_reward -= 100.0 
                    done = True
                
                if step > 2000:
                   print("Hard time limit reached.")
                   done = True

                if self.training:
                    # Train Player Agent
                    learning_reward = repeat_reward + (0.1 * frame_skip)
                    self.agent.observe(obs, action, learning_reward, done, next_obs, info)
                    
                    # Train Opponent Agent
                    if self.opponent_agent:
                        # Opponent treats proximity as good
                        self.opponent_agent.observe(obs_opp, action_opp, opp_repeat_reward, done, next_obs_opp, {})

                obs = next_obs
                if self.opponent_agent:
                    obs_opp = next_obs_opp
                    
                total_reward += repeat_reward
                if self.opponent_agent:
                    # Accumulate opponent score for display
                    if not hasattr(self, 'total_reward_opp'): self.total_reward_opp = 0.0
                    self.total_reward_opp += opp_repeat_reward
                    
                step += 1 

                if done:
                    # Reset opp score for next ep
                    if hasattr(self, 'total_reward_opp'): self.total_reward_opp = 0.0
                    print(f"Episode {ep + 1} Done: steps={step}, total_reward={total_reward:.2f}")
                    break

def main():
    print("==========================================")
    print("   WELCOME TO AWESOME CAR RACING OOP DEMO")
    print("==========================================")
    print("Select Mode:")
    print("1. Manual Drive")
    print("2. Random Chaos")
    print("3. Train Smart Agent (PPO)")
    print("4. Visualize Smart Agent")
    
    choice = input("Enter choice (1-4): ").strip()
    
    # Use rgb_array for custom rendering loop
    render_mode = "rgb_array"
    env = CustomCarRacing(render_mode=render_mode, lap_complete_percent=0.95)
    
    agent = None
    episodes = 3
    training_enabled = False
    render_enabled = True # Default On
    
    if choice == '1':
        agent = ManualAgent()
        episodes = 10
    elif choice == '2':
        agent = RandomAgent(env.action_space)
    elif choice == '3':
        print("Training Mode...")
        training_enabled = True
        fast = input("Enable Fast Training (No Render)? (y/n): ").strip().lower()
        if fast == 'y':
            render_enabled = False
        
        agent = SmartAgent(env.action_space, model_path="smart_agent_model.pth")
        episodes = 5000 
    elif choice == '4':
        print("Visualization Mode...")
        # Force render enabled for visualization
        render_enabled = True
        training_enabled = False
        agent = SmartAgent(env.action_space, model_path="smart_agent_model.pth")
        episodes = 10
    else:
        agent = SmartAgent(env.action_space, model_path="smart_agent_model.pth")
    
    # NPC Agent Configuration
    # Default to Heuristic
    npc_agent = SmartAgent(env.action_space, model_path="smart_agent_model.pth")
    
    # If Training Mode, let NPC learn too!
    if choice == '4':
        print("Initializing NPC Smart Agent...")
        # Share the Brain! NPC uses same policy as Main Agent.
        # model_path=None prevents NPC from saving the model (avoid file lock), 
        # but the Main Agent will save the shared policy updates.
        npc_agent = SmartAgent(env.action_space, model_path=None, policy=agent.policy)

    try:
        trainer = Trainer(env, agent, episodes=episodes, training=training_enabled, render_enabled=render_enabled)
        # Override the default npc_agent with our smart one if needed
        trainer.npc_agent = npc_agent
        trainer.run()
    except KeyboardInterrupt:
        print("\nSimulation stopped.")
    finally:
        env.close()
        pygame.quit()
        print("Closed.")

if __name__ == "__main__":
    main()
