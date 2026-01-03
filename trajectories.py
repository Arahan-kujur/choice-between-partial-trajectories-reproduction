"""
Trajectory Generation and Partial Trajectory Creation

Generates full rollouts and partial trajectories (prefixes) from the grid world.
"""

import numpy as np
from typing import List, Tuple, Optional
from env import GridWorld


class Trajectory:
    """
    Represents a trajectory: sequence of (state, action, reward) tuples.
    """
    
    def __init__(self, states: List[Tuple[int, int]], 
                 actions: List[int], 
                 rewards: List[float]):
        """
        Initialize trajectory.
        
        Args:
            states: List of states [(row, col), ...]
            actions: List of actions [action_idx, ...]
            rewards: List of rewards [reward, ...]
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.length = len(states)
    
    def get_partial(self, length: int) -> 'Trajectory':
        """
        Get partial trajectory (prefix) of specified length.
        
        Args:
            length: Length of partial trajectory (must be <= self.length)
            
        Returns:
            Partial trajectory
        """
        if length > self.length:
            length = self.length
        
        return Trajectory(
            states=self.states[:length],
            actions=self.actions[:length],
            rewards=self.rewards[:length]
        )
    
    def get_total_return(self) -> float:
        """Compute total return (sum of rewards)."""
        return sum(self.rewards)
    
    def __len__(self) -> int:
        return self.length


def generate_trajectory(env: GridWorld, 
                       start_state: Optional[Tuple[int, int]] = None,
                       policy: Optional[np.ndarray] = None,
                       max_steps: int = 100,
                       seed: Optional[int] = None) -> Trajectory:
    """
    Generate a trajectory by following a policy in the environment.
    
    Args:
        env: GridWorld environment
        start_state: Starting state (default: random)
        policy: Policy matrix of shape (n_states, n_actions) with probabilities.
                If None, uses random policy.
        max_steps: Maximum trajectory length
        seed: Random seed
        
    Returns:
        Trajectory object
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize starting state
    if start_state is None:
        # Random non-terminal state
        non_terminal_states = [s for s in env.states if not env.is_terminal(s)]
        start_state = tuple(non_terminal_states[np.random.randint(len(non_terminal_states))])
    
    states = [start_state]
    actions = []
    rewards = []
    
    current_state = start_state
    
    for _ in range(max_steps):
        # Check if terminal
        if env.is_terminal(current_state):
            break
        
        # Select action
        if policy is None:
            # Random policy
            action = np.random.randint(env.n_actions)
        else:
            state_idx = env.state_to_idx(current_state)
            action_probs = policy[state_idx, :]
            action = np.random.choice(env.n_actions, p=action_probs)
        
        # Take step
        next_state, reward, is_terminal = env.step(current_state, action)
        
        actions.append(action)
        rewards.append(reward)
        states.append(next_state)
        
        current_state = next_state
        
        if is_terminal:
            break
    
    return Trajectory(states, actions, rewards)


def generate_optimal_trajectory(env: GridWorld,
                                start_state: Optional[Tuple[int, int]] = None,
                                value_function: Optional[np.ndarray] = None) -> Trajectory:
    """
    Generate trajectory using optimal policy (greedy w.r.t. value function).
    
    Args:
        env: GridWorld environment
        start_state: Starting state (default: random)
        value_function: Pre-computed value function V(s) of shape (n_states,).
                       If None, computes it via value iteration.
    
    Returns:
        Trajectory following optimal policy
    """
    if value_function is None:
        value_function = compute_value_function(env)
    
    # Initialize starting state
    if start_state is None:
        non_terminal_states = [s for s in env.states if not env.is_terminal(s)]
        start_state = tuple(non_terminal_states[np.random.randint(len(non_terminal_states))])
    
    states = [start_state]
    actions = []
    rewards = []
    
    current_state = start_state
    
    for _ in range(env.size * env.size):  # Max steps to prevent infinite loops
        if env.is_terminal(current_state):
            break
        
        # Greedy action selection
        state_idx = env.state_to_idx(current_state)
        best_action = None
        best_value = float('-inf')
        
        for action in env.actions:
            next_state, reward, _ = env.step(current_state, action)
            next_state_idx = env.state_to_idx(next_state)
            # Q(s,a) = r(s,a) + V(s')
            q_value = reward + value_function[next_state_idx]
            
            if q_value > best_value:
                best_value = q_value
                best_action = action
        
        if best_action is None:
            break
        
        # Take step
        next_state, reward, is_terminal = env.step(current_state, best_action)
        
        actions.append(best_action)
        rewards.append(reward)
        states.append(next_state)
        
        current_state = next_state
        
        if is_terminal:
            break
    
    return Trajectory(states, actions, rewards)


def compute_value_function(env: GridWorld, gamma: float = 1.0, 
                          tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
    """
    Compute optimal value function using value iteration.
    
    Args:
        env: GridWorld environment
        gamma: Discount factor (default: 1.0 for undiscounted)
        tol: Convergence tolerance
        max_iter: Maximum iterations
        
    Returns:
        Value function V(s) of shape (n_states,)
    """
    V = np.zeros(env.n_states)
    
    for iteration in range(max_iter):
        V_new = np.zeros(env.n_states)
        
        for state in env.states:
            state_idx = env.state_to_idx(state)
            
            if env.is_terminal(state):
                # Terminal state: value is goal reward
                V_new[state_idx] = env.goal_reward
            else:
                # Bellman update: V(s) = max_a [r(s,a) + gamma * V(s')]
                best_value = float('-inf')
                
                for action in env.actions:
                    next_state, reward, _ = env.step(state, action)
                    next_state_idx = env.state_to_idx(next_state)
                    q_value = reward + gamma * V[next_state_idx]
                    
                    if q_value > best_value:
                        best_value = q_value
                
                V_new[state_idx] = best_value
        
        # Check convergence
        if np.max(np.abs(V_new - V)) < tol:
            break
        
        V = V_new
    
    return V


def generate_partial_trajectories(env: GridWorld, 
                                  n_trajectories: int = 100,
                                  min_length: int = 2,
                                  max_length: int = 10,
                                  policy: Optional[np.ndarray] = None,
                                  seed: Optional[int] = None) -> List[Trajectory]:
    """
    Generate a collection of partial trajectories of varying lengths.
    
    Args:
        env: GridWorld environment
        n_trajectories: Number of trajectories to generate
        min_length: Minimum partial trajectory length
        max_length: Maximum partial trajectory length
        policy: Policy to follow (None = random)
        seed: Random seed
        
    Returns:
        List of partial trajectories
    """
    if seed is not None:
        np.random.seed(seed)
    
    trajectories = []
    
    for _ in range(n_trajectories):
        # Generate full trajectory
        full_traj = generate_trajectory(env, policy=policy, seed=None)
        
        # Randomly truncate to partial length
        if full_traj.length > 0:
            partial_length = np.random.randint(min_length, min(max_length + 1, full_traj.length + 1))
            partial_traj = full_traj.get_partial(partial_length)
            trajectories.append(partial_traj)
    
    return trajectories

