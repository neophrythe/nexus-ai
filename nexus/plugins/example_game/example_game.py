import numpy as np
from gymnasium import spaces
from typing import Any, Dict, Optional

from nexus.core.base import GamePlugin
from nexus.environments import GameEnvironment, GamePhase


class ExampleGame(GamePlugin):
    
    async def initialize(self) -> None:
        self.game_state = {
            "score": 0,
            "lives": 3,
            "level": 1,
            "enemies": [],
            "player_pos": [400, 300]
        }
        self.screen_width = 800
        self.screen_height = 600
    
    async def shutdown(self) -> None:
        self.game_state = None
    
    async def validate(self) -> bool:
        return True
    
    def get_game_state(self) -> Dict[str, Any]:
        return self.game_state.copy()
    
    def get_observation_space(self) -> spaces.Space:
        return spaces.Dict({
            "screen": spaces.Box(
                low=0, 
                high=255, 
                shape=(self.screen_height, self.screen_width, 3),
                dtype=np.uint8
            ),
            "score": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            "lives": spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            "level": spaces.Box(low=1, high=100, shape=(1,), dtype=np.int32)
        })
    
    def get_action_space(self) -> spaces.Space:
        return spaces.MultiDiscrete([
            3,
            3,
            2
        ])


class ExampleGameEnvironment(GameEnvironment):
    
    def __init__(self, *args, **kwargs):
        super().__init__("ExampleGame", *args, **kwargs)
        self.screen_buffer = np.zeros((600, 800, 3), dtype=np.uint8)
        self.player_x = 400
        self.player_y = 300
        self.score = 0
        self.lives = 3
        self.level = 1
    
    def _build_observation_space(self) -> spaces.Space:
        return spaces.Dict({
            "screen": spaces.Box(0, 255, (600, 800, 3), dtype=np.uint8),
            "player_pos": spaces.Box(-np.inf, np.inf, (2,), dtype=np.float32),
            "game_info": spaces.Dict({
                "score": spaces.Box(0, np.inf, (1,), dtype=np.float32),
                "lives": spaces.Discrete(10),
                "level": spaces.Discrete(100)
            })
        })
    
    def _build_action_space(self) -> spaces.Space:
        return spaces.Discrete(8)
    
    def _get_observation(self) -> Dict[str, Any]:
        self._render_game()
        
        return {
            "screen": self.screen_buffer.copy(),
            "player_pos": np.array([self.player_x, self.player_y], dtype=np.float32),
            "game_info": {
                "score": np.array([self.score], dtype=np.float32),
                "lives": self.lives,
                "level": self.level
            }
        }
    
    def _calculate_reward(self, observation: Any, action: int) -> float:
        reward = 0.0
        
        reward += 0.01
        
        old_score = self.score
        self._update_game_state(action)
        
        if self.score > old_score:
            reward += (self.score - old_score) * 10
        
        if self.lives < 3:
            reward -= 100
        
        return reward
    
    def _is_terminated(self, observation: Any) -> bool:
        return self.lives <= 0
    
    def _is_truncated(self, observation: Any) -> bool:
        return self.frame_count >= 10000
    
    def _execute_action(self, action: int) -> None:
        move_speed = 10
        
        action_map = {
            0: (0, -move_speed),
            1: (move_speed, -move_speed),
            2: (move_speed, 0),
            3: (move_speed, move_speed),
            4: (0, move_speed),
            5: (-move_speed, move_speed),
            6: (-move_speed, 0),
            7: (-move_speed, -move_speed)
        }
        
        if action in action_map:
            dx, dy = action_map[action]
            self.player_x = np.clip(self.player_x + dx, 20, 780)
            self.player_y = np.clip(self.player_y + dy, 20, 580)
    
    def _detect_game_phase(self, observation: Any) -> GamePhase:
        if self.lives <= 0:
            return GamePhase.ENDED
        return GamePhase.PLAYING
    
    def _render_game(self) -> None:
        self.screen_buffer.fill(0)
        
        self.screen_buffer[50:100, 100:700, :] = [0, 100, 0]
        
        px, py = int(self.player_x), int(self.player_y)
        self.screen_buffer[py-10:py+10, px-10:px+10, :] = [0, 0, 255]
        
        self.screen_buffer[10:30, 10:10+self.lives*20, :] = [255, 0, 0]
        
        score_width = min(200, int(self.score / 10))
        self.screen_buffer[10:30, 600:600+score_width, :] = [255, 255, 0]
    
    def _update_game_state(self, action: int) -> None:
        if np.random.random() < 0.01:
            self.score += 10
        
        if self.score > 100 * self.level:
            self.level += 1
        
        if np.random.random() < 0.001:
            self.lives -= 1