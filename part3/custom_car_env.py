
import numpy as np
import math
import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.envs.box2d.car_dynamics import Car
import pygame
from pygame import gfxdraw

# Constants (Local copies for our HD render)
# We force 800x600 to match our split screen half-width
STATE_W = 96
STATE_H = 96
WINDOW_W = 800
WINDOW_H = 600
VIDEO_W = 800
VIDEO_H = 600

SCALE = 6.0
ZOOM = 2.7
FPS = 50 

class CustomCarRacing(CarRacing):
    """
    A custom subclass of CarRacing that implements a 'Neon' visual theme.
    Demonstrates Inheritance and Encapsulation of visual configurations.
    Also adds a second Opponent Car (NPC).
    """
    def __init__(self, render_mode: str | None = None, verbose: bool = False, lap_complete_percent: float = 0.95):
        # Call the parent constructor
        super().__init__(render_mode=render_mode, verbose=verbose, lap_complete_percent=lap_complete_percent)
        print("DEBUG: CustomCarRacing __init__ called")
        self.car2 = None

    def reset(self, *, seed=None, options=None):
        # Clear cache from previous episode
        self.track_points = None
        
        obs, info = super().reset(seed=seed, options=options)
        
        # Instantiate Second Car (NPC)
        # Offset start position slightly to the right (x + delta) or adjust angle?
        # track[0] is (alpha, beta, x, y)
        if self.track and len(self.track) > 0:
            # Use the track's angle for correct orientation
            init_angle = self.track[0][1]
            init_x = self.track[0][2]
            init_y = self.track[0][3]
            
            # Calculate perpendicular offset for side-by-side start
            # +90 degrees is to the "left" relative to track direction
            lane_offset = 8
            stagger_dist = 5.0 # Distance for the second car to be behind
            
            # Global Shift to the Right
            center_bias = 3.0 # Shift everything 3.0 units to the right
            
            # Normal Vector (Left)
            norm_dx = math.cos(init_angle + math.pi / 2)
            norm_dy = math.sin(init_angle + math.pi / 2)
            
            dx = lane_offset * norm_dx
            dy = lane_offset * norm_dy
            
            # Shift Vector (Right = -Normal)
            shift_dx = -center_bias * norm_dx
            shift_dy = -center_bias * norm_dy
            
            # Additional Backward Vector (Tangent)
            # -Tangent direction
            back_dx = -stagger_dist * math.cos(init_angle)
            back_dy = -stagger_dist * math.sin(init_angle)

            # Car 1: Left of center + Shift
            c1_x = init_x - dx + shift_dx
            c1_y = init_y - dy + shift_dy
            
            # Car 2: Right of center and Behind + Shift
            c2_x = init_x + dx + back_dx + shift_dx
            c2_y = init_y + dy + back_dy + shift_dy
            
            # Move Agent Car (self.car)
            self.car.destroy()
            self.car = Car(self.world, init_angle, c1_x, c1_y)
            self.car.hull.color = (0.8, 0.0, 0.0) 
            
            # Create NPC Car (self.car2)
            self.car2 = Car(self.world, init_angle, c2_x, c2_y)
            self.car2.hull.color = (0.2, 0.2, 0.8)

        return obs, info


    def step(self, action: np.ndarray | int, opponent_action: np.ndarray | int = None):
        # 0. Apply Opponent Action (if any)
        if opponent_action is not None and self.car2:
            if self.continuous:
                opponent_action = opponent_action.astype(np.float64)
                self.car2.steer(-opponent_action[0])
                self.car2.gas(opponent_action[1])
                self.car2.brake(opponent_action[2])
            else:
                 # Discrete logic for car2 if needed (assuming continuous for now)
                 pass
            self.car2.step(1.0 / FPS)

        # 1. Step the main agent (done by parent)
        # Parent step() calls world.Step() which steps physics for ALL bodies.
        result = super().step(action)
        # result is (obs, reward, terminated, truncated, info)

        # 2. Add Opponent Reward/Info if needed?
        # Typically environment just returns main agent's experience.
        # Opponent experience needs to be fetched separately or calculated.
        
        return result

    def apply_npc_action(self, action):
        """
        External control method for the NPC car.
        action: [steer, gas, brake]
        """
        if self.car2:
            steer, gas, brake = action
            # Box2D SWIG wrapper hates numpy floats, cast to python float
            self.car2.steer(-float(steer))
            self.car2.gas(float(gas))
            self.car2.brake(float(brake))

    def get_opponent_observation(self):
        """
        Render the environment from the Opponent Car's perspective.
        HACK: Swap self.car and self.car2, render, then swap back.
        """
        if not self.car2:
            return np.zeros((96, 96, 3), dtype=np.uint8) # No opponent

        # 1. Swap
        temp_car = self.car
        self.car = self.car2
        self.car2 = temp_car
        
        # 2. Render State
        # We need to ensure we use the correct state rendering logic
        # self._render("state_pixels") uses self.car
        obs = self._render("state_pixels")
        
        # 3. Swap Back
        self.car2 = self.car
        self.car = temp_car
        
        return obs

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode: str):
        assert mode in self.metadata["render_modes"]

        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert self.car is not None
        # computing transformations
        angle = -self.car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(self.car.hull.position[0]) * zoom
        scroll_y = -(self.car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        self.car.draw(
            self.surf,
            zoom,
            trans,
            angle,
            mode not in ["state_pixels_list", "state_pixels"],
        )
        
        # --- Draw Opponent Car (NPC) ---
        if self.car2:
            self.car2.draw(
                self.surf,
                zoom, # Use same zoom
                trans, # Use same translation (relative to car1)
                angle, # Use same camera angle (relative to car1)
                mode not in ["state_pixels_list", "state_pixels"],
            )
        # -------------------------------

        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward, True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def close(self):
        if self.car2:
            self.car2.destroy()
            self.car2 = None
        super().close()
        
    def _init_colors(self):
        """
        Override the parent's color initialization to create a cool Neon theme.
        This demonstrates Polymorphism (overriding a method to change behavior).
        """
        # Neon Dark Theme
        # Background: Dark Purple/Blue
        self.bg_color = np.array([20, 20, 40]) 
        
        # Grass: Slightly lighter dark blue/grid-like feel (functionally same as bg but we can tweak)
        self.grass_color = np.array([30, 30, 50])
        
        # Road: Dark Grey
        self.road_color = np.array([60, 60, 60])
        
        # We don't have direct access to border colors easily without overriding more, 
        # but changing road and bg makes a big difference.
        
        # Note: The original code uses these attributes directly in _render_road.

    def get_centered_observation(self, target_car):
        """
        Render the environment from the perspective of target_car.
        Temporarily swaps self.car with target_car to trick the camera.
        """
        if not target_car or not self.car:
            return self.render()
            
        real_car = self.car
        
        # Swap
        self.car = target_car
        
        # Render
        # Force "state_pixels" mode to get the 96x96 observation required by the agent
        obs = self._render("state_pixels")
        
        # Swap back
        self.car = real_car
        
        return obs

    def get_opponent_render(self, score=None):
        """
        Get high-res render from opponent perspective.
        Also visualizes the Red car (as car2) and correct score.
        """
        if not self.car2 or not self.car:
            return None
            
        # 1. Backup State
        real_car = self.car
        real_car2 = self.car2
        real_reward = self.reward
        
        # 2. Swap Roles
        # Blue becomes Main Car (Ego)
        self.car = real_car2
        # Red becomes NPC (Other)
        self.car2 = real_car 
        
        # 3. Swap Score (if provided)
        # This tricks the environment's UI to display the opponent's score
        if score is not None:
            self.reward = score
        
        # 4. Render
        img = self.render()
        
        # 5. Restore State
        self.car = real_car
        self.car2 = real_car2
        self.reward = real_reward
        
        return img



    def is_off_track(self, car, max_dist=10.0):
        """
        Check if a car is off-track (distance > max_dist units).
        """
        if not car: return False
        
        # Cache check
        if not hasattr(self, 'track_points') or self.track_points is None:
             if hasattr(self, 'track') and self.track:
                self.track_points = np.array([t[2:4] for t in self.track])
             else:
                return False

        p = car.hull.position
        diff = self.track_points - np.array([p[0], p[1]])
        dists_sq = np.sum(diff**2, axis=1)
        min_dist = math.sqrt(np.min(dists_sq))
        
        return min_dist > max_dist
