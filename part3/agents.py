
from abc import ABC, abstractmethod
import numpy as np
import pygame

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
    def __init__(self, env):
        self.env = env
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
        
        car = self.env.car
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
