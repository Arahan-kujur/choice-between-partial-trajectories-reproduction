"""
Scoring Models for Partial Trajectories

Implements three models for scoring partial trajectories:
1. Partial Return: sum of observed rewards
2. Cumulative Advantage: advantage relative to optimal continuation
3. Bootstrapped Return: partial return + estimated value of final state
"""

import numpy as np
from typing import Tuple, Optional
from env import GridWorld
from trajectories import Trajectory, compute_value_function


def partial_return_score(trajectory: Trajectory) -> float:
    """
    Compute partial return score: sum of observed rewards so far.
    
    This is the simplest model - just sums up the rewards seen in the trajectory.
    
    Args:
        trajectory: Partial trajectory
        
    Returns:
        Score (sum of rewards)
    """
    return trajectory.get_total_return()


def cumulative_advantage_score(trajectory: Trajectory, 
                                env: GridWorld,
                                value_function: np.ndarray) -> float:
    """
    Compute cumulative advantage score: advantage relative to optimal continuation.
    
    The advantage measures how much better/worse the trajectory is compared to
    the optimal policy from the starting state.
    
    Formally: A(τ) = R(τ) - V(s₀), where:
    - R(τ) is the partial return
    - V(s₀) is the optimal value from the starting state
    
    Args:
        trajectory: Partial trajectory
        env: GridWorld environment
        value_function: Optimal value function V(s) of shape (n_states,)
        
    Returns:
        Advantage score
    """
    if trajectory.length == 0:
        return 0.0
    
    # Get starting state
    start_state = trajectory.states[0]
    start_state_idx = env.state_to_idx(start_state)
    
    # Partial return
    partial_return = trajectory.get_total_return()
    
    # Optimal value from starting state
    optimal_value = value_function[start_state_idx]
    
    # Advantage = partial return - optimal value
    advantage = partial_return - optimal_value
    
    return advantage


def bootstrapped_return_score(trajectory: Trajectory,
                              env: GridWorld,
                              value_function: np.ndarray) -> float:
    """
    Compute bootstrapped return score: partial return + estimated value of final state.
    
    This model combines the observed rewards with an estimate of future value
    from the final state of the partial trajectory.
    
    Formally: B(τ) = R(τ) + γ * V(s_T), where:
    - R(τ) is the partial return
    - V(s_T) is the value of the final state
    - γ is the discount factor (default: 1.0)
    
    Args:
        trajectory: Partial trajectory
        env: GridWorld environment
        value_function: Value function V(s) of shape (n_states,)
        
    Returns:
        Bootstrapped return score
    """
    if trajectory.length == 0:
        return 0.0
    
    # Partial return
    partial_return = trajectory.get_total_return()
    
    # Get final state
    final_state = trajectory.states[-1]
    final_state_idx = env.state_to_idx(final_state)
    
    # Value of final state
    final_value = value_function[final_state_idx]
    
    # Bootstrapped return = partial return + value of final state
    # Using discount factor gamma = 1.0 (undiscounted)
    bootstrapped_return = partial_return + final_value
    
    return bootstrapped_return


def score_trajectory(trajectory: Trajectory,
                    model_type: str,
                    env: GridWorld,
                    value_function: Optional[np.ndarray] = None) -> float:
    """
    Score a trajectory using the specified model.
    
    Args:
        trajectory: Partial trajectory to score
        model_type: One of 'partial_return', 'cumulative_advantage', 'bootstrapped_return'
        env: GridWorld environment
        value_function: Value function (required for advantage and bootstrapped models)
        
    Returns:
        Score value
    """
    if model_type == 'partial_return':
        return partial_return_score(trajectory)
    
    elif model_type == 'cumulative_advantage':
        if value_function is None:
            value_function = compute_value_function(env)
        return cumulative_advantage_score(trajectory, env, value_function)
    
    elif model_type == 'bootstrapped_return':
        if value_function is None:
            value_function = compute_value_function(env)
        return bootstrapped_return_score(trajectory, env, value_function)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_choice_probability(score1: float, score2: float, 
                              temperature: float = 1.0) -> float:
    """
    Compute choice probability using softmax (logit model).
    
    P(choose τ₁ | τ₁, τ₂) = exp(score₁ / T) / (exp(score₁ / T) + exp(score₂ / T))
    
    Args:
        score1: Score of trajectory 1
        score2: Score of trajectory 2
        temperature: Temperature parameter (higher = more random)
        
    Returns:
        Probability of choosing trajectory 1
    """
    # Softmax with temperature
    exp_score1 = np.exp(score1 / temperature)
    exp_score2 = np.exp(score2 / temperature)
    
    prob1 = exp_score1 / (exp_score1 + exp_score2)
    
    return prob1

