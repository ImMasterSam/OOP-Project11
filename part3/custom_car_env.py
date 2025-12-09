
import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing

class CustomCarRacing(CarRacing):
    """
    A custom subclass of CarRacing that implements a 'Neon' visual theme.
    Demonstrates Inheritance and Encapsulation of visual configurations.
    """
    def __init__(self, render_mode: str | None = None, verbose: bool = False, lap_complete_percent: float = 0.95):
        # Call the parent constructor
        super().__init__(render_mode=render_mode, verbose=verbose, lap_complete_percent=lap_complete_percent)
        
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
