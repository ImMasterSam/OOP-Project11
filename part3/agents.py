
from abc import ABC, abstractmethod
import numpy as np
import pygame
import torch
from collections import deque
from cnn_network import CNNNetwork

class Agent(ABC):
    """
    Abstract Base Class for all Agents.
    Demonstrates Abstraction.
    """
    @abstractmethod
    def select_action(self, observation):
        """Return an action based on the observation."""
        pass

    def reset(self):
        """Reset internal state if needed."""
        pass

    def observe(self, obs, action, reward, done, next_obs, info):
        """Optional: Learn from experience."""
        pass


class RandomAgent(Agent):
    """
    An agent that selects actions randomly.
    """
    def __init__(self, action_space):
        self.action_space = action_space

    def select_action(self, observation):
        return self.action_space.sample()


class ManualAgent(Agent):
    """
    An agent that allows the user to control the car using keyboard arrows.
    Demonstrates Interaction with the Environment via Hardware Input.
    Features: Arrow keys for steering/gas/brake, SPACE for boost.
    """
    def __init__(self):
        self.steering = 0.0
        self.gas = 0.0
        self.brake = 0.0
        # Boost system - short burst for "push forward" feel
        self.boost_duration = 0  # Frames remaining for boost
        self.boost_cooldown = 0  # Frames until boost is available again
        self.BOOST_LENGTH = 30    # Very short burst (8 frames = ~0.16 sec)
        self.BOOST_COOLDOWN = 60 # Cooldown between boosts (~1.2 sec)
    
    def reset(self):
        self.boost_duration = 0
        self.boost_cooldown = 0

    def select_action(self, observation):
        # Update cooldowns
        if self.boost_cooldown > 0:
            self.boost_cooldown -= 1
        if self.boost_duration > 0:
            self.boost_duration -= 1

        # Reset base actions
        self.steering = 0.0
        self.gas = 0.0
        self.brake = 0.0

        # Get key states
        keys = pygame.key.get_pressed()

        if keys[pygame.K_LEFT]:
            self.steering = -1.0
        elif keys[pygame.K_RIGHT]:
            self.steering = 1.0
        
        if keys[pygame.K_UP]:
            self.gas = 1.0
        else:
            # Natural deceleration when not pressing gas
            # Apply light braking to slow down the car
            self.brake = 0.05
        
        if keys[pygame.K_DOWN]:
            self.brake = 0.8 # Full brake overrides natural decel
        
        # BOOST: Press SPACE for a short burst push!
        if keys[pygame.K_SPACE] and self.boost_cooldown == 0 and self.boost_duration == 0:
            self.boost_duration = self.BOOST_LENGTH
            self.boost_cooldown = self.BOOST_COOLDOWN
            print("🚀 BOOST!")
        
        # Apply boost effect - short intense burst
        if self.boost_duration > 0:
            self.gas = 1.0   # Full gas during boost
            self.brake = 0.0 # No braking during boost

        # Return continuous action: [steer, gas, brake]
        # steer: -1 to 1, gas: 0 to 1, brake: 0 to 1
        return np.array([self.steering, self.gas, self.brake], dtype=np.float32)


class HeuristicAgent(Agent):
    """
    A rule-based agent that drives automatically using Pure Pursuit algorithm.
    Demonstrates encapsulation of driving logic.
    """
    def __init__(self, env, target_car="car"):
        self.env = env
        self.target_car_attr = target_car
        self.last_idx = 0
        self.step_count = 0
    
    def reset(self):
        self.last_idx = 0
        self.step_count = 0

    def select_action(self, observation):
        """
        Pure Pursuit: Steer towards a target point ahead on the track.
        """
        self.step_count += 1
        
        # Access the target car dynamically (e.g., 'car' or 'car2')
        car = getattr(self.env, self.target_car_attr, None)
        
        if not car or not hasattr(self.env, 'track') or len(self.env.track) == 0:
            return np.array([0, 0, 0], dtype=np.float32)
        
        # Startup pause
        if self.step_count < 20:
            return np.array([0, 0, 0], dtype=np.float32)

        track = self.env.track
        car_x, car_y = car.hull.position
        car_angle = car.hull.angle
        speed = np.linalg.norm(car.hull.linearVelocity)
        omega = car.hull.angularVelocity

        # Find closest track point
        n = len(track)
        fx, fy = np.cos(car_angle), np.sin(car_angle)  # 車頭方向

        if self.step_count <= 5:
            # 剛開始，還沒有穩定的 last_idx，就先整圈掃
            candidate_indices = range(n)
        else:
            window = 40
            candidate_indices = [(self.last_idx + k) % n for k in range(-window, window + 1)]
        
        best_idx = None
        best_dist = float("inf")

        for i in candidate_indices:
            tx, ty = track[i][2], track[i][3]
            dx = tx - car_x
            dy = ty - car_y

            # dot > 0 才是在車子的前半平面
            dot = dx * fx + dy * fy
            if dot <= 0:
                continue

            d2 = dx * dx + dy * dy
            if d2 < best_dist:
                best_dist = d2
                best_idx = i

        # 如果前面一個都沒找到（極端情況），退回原本「最近點」邏輯
        if best_idx is None:
            best_dist = float("inf")
            for i in range(n):
                tx, ty = track[i][2], track[i][3]
                d2 = (car_x - tx) ** 2 + (car_y - ty) ** 2
                if d2 < best_dist:
                    best_dist = d2
                    best_idx = i

        self.last_idx = best_idx

        # Look ahead - use more lookahead at higher speeds
        # lookahead = int(10 + 0.5 * speed)
        # lookahead = max(5, min(lookahead, 25))
        lookahead = 15
        target_idx = (best_idx + lookahead) % len(track)
        
        target_x = track[target_idx][2]
        target_y = track[target_idx][3]
        
        # Calculate angle to target in world frame
        dx = target_x - car_x
        dy = target_y - car_y
        target_angle = np.arctan2(dy, dx)
        
        # Calculate angle error (how much we need to turn)
        angle_error = target_angle - car_angle
        
        # Normalize to [-pi, pi]
        while angle_error > np.pi: angle_error -= 2*np.pi
        while angle_error < -np.pi: angle_error += 2*np.pi
        
        # PID steering
        Kp = 1.2   # 先小一點，不要太暴力
        Kd = 0.4
        steer = -Kp * angle_error - Kd * omega
        steer = float(np.clip(steer, -1.0, 1.0))
        
        # Speed control
        base_speed = 22.0
        curve_penalty = 15.0 * abs(steer)   # 轉越大就降速越多
        target_speed = base_speed - curve_penalty
        target_speed = np.clip(target_speed, 8.0, 25.0)

        if speed < target_speed - 2:
            gas, brake = 0.6, 0.0
        elif speed > target_speed + 2:
            gas, brake = 0.0, 0.3
        else:
            gas, brake = 0.2, 0.0
            
        return np.array([steer, gas, brake], dtype=np.float32)


class SmartAgent(Agent):
    """
    PPO Agent using CNN.
    Auto-trains while driving.
    """
    """
    PPO Agent using CNN.
    Auto-trains while driving.
    """
    def __init__(self, action_space, model_path=None, policy=None):
        self.action_space = action_space
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SmartAgent initialized on device: {self.device}")
        self.model_path = model_path
        self.training = True
        
        if policy:
             # Shared policy mode
             self.policy = policy
             print("Using shared policy.")
        else:
             self.policy = CNNNetwork(action_dim=3).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=3e-4) # Standard PPO lr
        
        # Only load if we are NOT sharing (or if sharing, we assume main agent loaded it)
        # Actually, if we share, we should just use the policy as is.
        if model_path and not policy:
            self.load(model_path)
            
        # PPO Hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_rate = 0.2
        # Tuning: Increase entropy to force exploration (stop driving straight only)
        # Increased to 0.05 to make the agent "try more" (User Request)
        self.ent_coef = 0.05 
        self.vf_coef = 0.5
        self.batch_size = 64
        self.n_epochs = 10
        self.target_kl = 0.02
        
        # Frame Stacking
        self.frames = deque(maxlen=4)
        
        # Buffer
        self.buffer_size = 2000
        self.reset_buffer()
        
    def preprocess(self, obs):
        # RGB (96, 96, 3) -> Gray (96, 96)
        # Formula: L = 0.299*R + 0.587*G + 0.114*B
        gray = np.dot(obs[...,:3], [0.299, 0.587, 0.114])
        
        # Tuning: Binarization for high contrast
        # Road (Grey ~60) -> 1.0, Grass (Dark ~30) -> 0.0
        # Threshold = 45 allows perfect separation
        gray = np.where(gray > 45.0, 1.0, 0.0)
        
        if len(self.frames) == 0:
            for _ in range(4):
                self.frames.append(gray)
        else:
            self.frames.append(gray)
            
        # Stack -> (4, 96, 96)
        return np.array(self.frames)

    def reset_buffer(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        
        self.last_raw_action = None
        self.last_log_prob = None
        self.last_value = None
    
    def reset(self):
        self.frames.clear()
    
    def load(self, path):
        try:
            state_dict = torch.load(path, map_location=self.device)
            # Check shape compatibility
            current_shape = self.policy.features[0].weight.shape
            loaded_shape = state_dict['features.0.weight'].shape
            if current_shape != loaded_shape:
                print(f"Shape mismatch: Current {current_shape} vs Loaded {loaded_shape}. Starting fresh.")
                return

            self.policy.load_state_dict(state_dict)
            print(f"Loaded model from {path}")
        except FileNotFoundError:
            print(f"No model found at {path}, starting fresh.")
        except Exception as e:
            print(f"Error loading model: {e}. Starting fresh.")

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
        print(f"Saved model to {path}")

    def select_action(self, observation):
        # Observation from env is (96, 96, 3)
        # Preprocess -> (4, 96, 96)
        state = self.preprocess(observation)
        
        # Torch: (1, 4, 96, 96)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            mean, log_std, value = self.policy(state_tensor)
            
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            
            # Sample raw action
            raw_action = dist.sample()
            log_prob = dist.log_prob(raw_action).sum(axis=-1)
            
        self.last_raw_action = raw_action.cpu().numpy()[0]
        self.last_log_prob = log_prob.cpu().numpy()[0]
        self.last_value = value.item()
        
        # Store state for learning (stack of 4)
        self.last_state = state
        
        # Process action for environment (Squash)
        # Using Tanh for Steer [-1,1], Sigmoid for Gas/Brake [0,1]
        raw = self.last_raw_action
        
        # Tuning: Steering Sensitivity Multiplier
        # Multiply by 3.0 to make steering much sharper (small output -> big turn)
        steer = np.tanh(raw[0] * 3.0)
        
        # Tuning: Action Biasing
        # Brake: -5.0 bias -> Sigmoid(-5.0) ~= 0.0 (Release Brake). Cures "Riding the Brakes".
        # Gas: +0.2 bias -> Sigmoid(0.2) ~= 0.55 (Slightly press gas).
        gas = 1 / (1 + np.exp(-(raw[1] + 0.3))) 
        brake = 1 / (1 + np.exp(-(raw[2] - 2.4)))
        
        # Optional: Add deadzone for gas/brake to avoid constant small acceleration?
        # For now, keep it smooth.
        
        return np.array([steer, gas, brake], dtype=np.float32)

    def observe(self, obs, action, reward, done, next_obs, info):
        # Store experience
        # NOTE: obs in arguments is original raw obs, but we want the processed state we used.
        # self.last_state populated in select_action contains the state used to generate this action.
        
        if self.last_raw_action is not None and hasattr(self, 'last_state'):
            self.states.append(self.last_state)
            self.actions.append(self.last_raw_action)
            self.log_probs.append(self.last_log_prob)
            
            # Tuning: Scale reward to stabilize training
            # CarRacing rewards are big (~1000), PPO likes small numbers (~1.0)
            scaled_reward = reward / 100.0
            self.rewards.append(scaled_reward)
            
            self.dones.append(done)
            self.values.append(self.last_value)
            
            if len(self.states) >= self.buffer_size:
                self.learn(next_obs, done)

    def learn(self, next_obs, done):
        print("Training update...")
        
        # Get next value
        # Process next_obs (it might not be in stack yet if we just stepped)
        # Actually learn is called after observe.
        # If done, next_obs is terminal. If not, we need to stack it.
        # But wait, `preprocess` updates internal stack state. 
        # We need a temporary stack for next_state? actually next_obs passed here is raw.
        # CAUTION: preprocess modifies self.frames. We should avoid modifying it here just for value estimation if it's not the next step yet?
        # Actually, if we are in loop, step -> next_obs. next_obs will be the obs for NEXT select_action.
        # So it is safe to act as if we are processing it? 
        # BUT `learn` is called inside `observe`, which is before `select_action`.
        # However, for Value bootstrapping we need V(s').
        # Let's just create a temporary stack attempt.
        
        temp_frames = self.frames.copy()
        temp_gray = np.dot(next_obs[...,:3], [0.299, 0.587, 0.114])
        # Same Binarization here
        temp_gray = np.where(temp_gray > 45.0, 1.0, 0.0)
        temp_frames.append(temp_gray)
        next_state = np.array(temp_frames)
        
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            next_value = self.policy.get_value(next_state_tensor).item()
            
        # Convert buffer to tensors
        # states is list of (4, 96, 96) arrays
        states_tensor = torch.FloatTensor(np.array(self.states)).to(self.device) # (B, 4, 96, 96)
        actions_tensor = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values + [next_value])
        
        # Compute GAE (Advantage and Return)
        advantages = np.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                nextnonterminal = 1.0 - done # If current step is done, next is 0 (but wait, next_obs is potentially terminal?)
                # Actually, `dones[t]` is whether step t is done.
                # If step t is done, then next_value should be ignored (or masked).
                # But `values` has `len(rewards) + 1` entries.
                # If t is done, then values[t+1] is effectively 0 for the purpose of this episode, 
                # but physically it might be the start of new ep?
                # Simplified GAE: mask delta with (1-done)
                pass

            # Correct logic:
            # delta = r + gamma * V(s') * (1-d) - V(s)
            next_val = values[t+1]
            non_terminal = 1.0 - dones[t]
            
            delta = rewards[t] + self.gamma * next_val * non_terminal - values[t]
            lastgaelam = delta + self.gamma * self.gae_lambda * non_terminal * lastgaelam
            advantages[t] = lastgaelam
            
        returns = advantages + values[:-1]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        
        # PPO Update Loop
        dataset_size = len(self.states)
        indices = np.arange(dataset_size)
        
        for epoch in range(self.n_epochs):
            np.random.shuffle(indices)
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                batch_states = states_tensor[idx]
                batch_actions = actions_tensor[idx]
                batch_old_log_probs = old_log_probs_tensor[idx]
                batch_advantages = advantages_tensor[idx]
                batch_returns = returns_tensor[idx]
                
                # Forward pass
                mean, log_std, value = self.policy(batch_states)
                std = torch.exp(log_std)
                dist = torch.distributions.Normal(mean, std)
                
                new_log_prob = dist.log_prob(batch_actions).sum(axis=-1)
                entropy = dist.entropy().sum(axis=-1).mean()
                
                # Ratio
                ratio = torch.exp(new_log_prob - batch_old_log_probs)
                
                # Clip Policy Loss
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_rate, 1.0 + self.clip_rate) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = ((value.squeeze() - batch_returns) ** 2).mean()
                
                loss = policy_loss + self.vf_coef * value_loss - self.ent_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                # Tuning: Gradient Clipping to prevent explosion
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.optimizer.step()
                
            # Optional: Check KL divergence for early stopping?
            
        print("Training update complete.")
        self.reset_buffer()

        

        if self.model_path:
            self.save(self.model_path)

class OpponentSmartAgent(SmartAgent):
    """
    PPO Agent for the Opponent Car (Blue).
    This agent learns to be aggressive or whatever objective we set for it.
    It functions identically to SmartAgent but saves/loads a different model.
    """
    def __init__(self, action_space, model_path="smart_opponent_model.pth"):
        # Initialize SmartAgent with a different model path
        super().__init__(action_space, model_path=model_path)
