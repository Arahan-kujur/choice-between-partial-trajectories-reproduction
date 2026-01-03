"""
Grid World MDP Environment

Implements a simple 5x5 grid world with deterministic transitions.
Based on the paper: "Choice Between Partial Trajectories: Disentangling Goals from Beliefs"
"""

import numpy as np
from typing import Tuple, List, Optional


class GridWorld:
    """
    5x5 grid world MDP with deterministic transitions.
    
    States: (row, col) coordinates from (0,0) to (4,4)
    Actions: 0=up, 1=down, 2=left, 3=right
    Terminal state: goal position (default: (4,4))
    """
    
    def __init__(self, size: int = 5, goal_pos: Tuple[int, int] = (4, 4), 
                 goal_reward: float = 10.0, step_penalty: float = -0.1,
                 seed: Optional[int] = None):
        """
        Initialize grid world.
        
        Args:
            size: Grid size (default: 5x5)
            goal_pos: Position of terminal goal state
            goal_reward: Reward for reaching goal
            step_penalty: Penalty per step (negative reward)
            seed: Random seed for reproducibility
        """
        self.size = size
        self.goal_pos = goal_pos
        self.goal_reward = goal_reward
        self.step_penalty = step_penalty
        
        # Action mapping: 0=up, 1=down, 2=left, 3=right
        self.actions = [0, 1, 2, 3]
        self.action_names = ['up', 'down', 'left', 'right']
        self.n_actions = len(self.actions)
        
        # State space: all (row, col) pairs
        self.states = [(r, c) for r in range(size) for c in range(size)]
        self.n_states = len(self.states)
        
        # Ground truth reward function: r(s, a) = step_penalty for all non-goal states,
        # goal_reward when reaching goal
        # We'll store this as a state-based reward for simplicity
        self.true_reward = self._initialize_true_reward()
        
        if seed is not None:
            np.random.seed(seed)
    
    def _initialize_true_reward(self) -> np.ndarray:
        """
        Initialize ground truth reward function.
        Returns a |S| x |A| array where reward[s_idx, a] = r(s, a).
        """
        reward = np.full((self.n_states, self.n_actions), self.step_penalty)
        
        # Goal state gets goal_reward for any action (terminal)
        goal_idx = self.state_to_idx(self.goal_pos)
        reward[goal_idx, :] = self.goal_reward
        
        return reward
    
    def state_to_idx(self, state: Tuple[int, int]) -> int:
        """Convert (row, col) state to linear index."""
        return state[0] * self.size + state[1]
    
    def idx_to_state(self, idx: int) -> Tuple[int, int]:
        """Convert linear index to (row, col) state."""
        return (idx // self.size, idx % self.size)
    
    def is_terminal(self, state: Tuple[int, int]) -> bool:
        """Check if state is terminal (goal)."""
        return state == self.goal_pos
    
    def get_reward(self, state: Tuple[int, int], action: int) -> float:
        """
        Get reward for state-action pair.
        
        Args:
            state: Current state (row, col)
            action: Action taken
            
        Returns:
            Reward value
        """
        if self.is_terminal(state):
            return self.goal_reward
        
        return self.step_penalty
    
    def step(self, state: Tuple[int, int], action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        Execute action in state (deterministic transition).
        
        Args:
            state: Current state (row, col)
            action: Action to take (0=up, 1=down, 2=left, 3=right)
            
        Returns:
            (next_state, reward, is_terminal)
        """
        row, col = state
        
        # Deterministic transitions
        if action == 0:  # up
            next_row = max(0, row - 1)
            next_col = col
        elif action == 1:  # down
            next_row = min(self.size - 1, row + 1)
            next_col = col
        elif action == 2:  # left
            next_row = row
            next_col = max(0, col - 1)
        elif action == 3:  # right
            next_row = row
            next_col = min(self.size - 1, col + 1)
        else:
            raise ValueError(f"Invalid action: {action}")
        
        next_state = (next_row, next_col)
        reward = self.get_reward(state, action)
        is_terminal = self.is_terminal(next_state)
        
        return next_state, reward, is_terminal
    
    def get_state_features(self, state: Tuple[int, int]) -> np.ndarray:
        """
        Get feature representation of state.
        For simplicity, we use one-hot encoding per grid cell.
        
        Args:
            state: State (row, col)
            
        Returns:
            Feature vector of length n_states
        """
        features = np.zeros(self.n_states)
        state_idx = self.state_to_idx(state)
        features[state_idx] = 1.0
        return features
    
    def get_reward_params_shape(self) -> Tuple[int, int]:
        """
        Get shape of reward parameterization.
        Returns (n_states, n_actions) for state-action reward.
        """
        return (self.n_states, self.n_actions)

