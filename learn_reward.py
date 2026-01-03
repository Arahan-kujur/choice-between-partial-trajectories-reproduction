"""
Preference-Based Reward Learning

Learns a reward function from choice data using maximum likelihood estimation.
The reward function is parameterized as a state-action reward matrix.
"""

import numpy as np
from typing import List, Optional, Tuple
from env import GridWorld
from trajectories import Trajectory, compute_value_function
from preference_data import Choice
from scoring import score_trajectory, compute_choice_probability


def compute_trajectory_score_from_reward(trajectory: Trajectory,
                                        reward_params: np.ndarray,
                                        env: GridWorld,
                                        model_type: str,
                                        value_function: Optional[np.ndarray] = None) -> float:
    """
    Compute trajectory score using learned reward parameters.
    
    Args:
        trajectory: Partial trajectory
        reward_params: Learned reward parameters of shape (n_states, n_actions)
        env: GridWorld environment
        model_type: Scoring model ('partial_return', 'cumulative_advantage', 'bootstrapped_return')
        value_function: Value function (computed from reward_params if None)
        
    Returns:
        Score value
    """
    if model_type == 'partial_return':
        # Sum of rewards using learned reward function
        score = 0.0
        for i in range(len(trajectory.actions)):
            state = trajectory.states[i]
            action = trajectory.actions[i]
            state_idx = env.state_to_idx(state)
            score += reward_params[state_idx, action]
        return score
    
    elif model_type == 'cumulative_advantage':
        # Need value function from learned rewards
        if value_function is None:
            value_function = compute_value_function_from_reward(env, reward_params)
        
        # Partial return
        partial_return = 0.0
        for i in range(len(trajectory.actions)):
            state = trajectory.states[i]
            action = trajectory.actions[i]
            state_idx = env.state_to_idx(state)
            partial_return += reward_params[state_idx, action]
        
        # Optimal value from starting state
        start_state = trajectory.states[0]
        start_state_idx = env.state_to_idx(start_state)
        optimal_value = value_function[start_state_idx]
        
        return partial_return - optimal_value
    
    elif model_type == 'bootstrapped_return':
        # Need value function from learned rewards
        if value_function is None:
            value_function = compute_value_function_from_reward(env, reward_params)
        
        # Partial return
        partial_return = 0.0
        for i in range(len(trajectory.actions)):
            state = trajectory.states[i]
            action = trajectory.actions[i]
            state_idx = env.state_to_idx(state)
            partial_return += reward_params[state_idx, action]
        
        # Value of final state
        final_state = trajectory.states[-1]
        final_state_idx = env.state_to_idx(final_state)
        final_value = value_function[final_state_idx]
        
        return partial_return + final_value
    
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def compute_value_function_from_reward(env: GridWorld,
                                      reward_params: np.ndarray,
                                      gamma: float = 1.0,
                                      n_iter: int = 50) -> np.ndarray:
    """
    Compute value function from reward parameters using value iteration.
    Uses fixed number of iterations for speed.
    
    Args:
        env: GridWorld environment
        reward_params: Reward parameters of shape (n_states, n_actions)
        gamma: Discount factor
        n_iter: Fixed number of iterations (default: 50)
        
    Returns:
        Value function V(s) of shape (n_states,)
    """
    V = np.zeros(env.n_states)
    
    for iteration in range(n_iter):
        V_new = np.zeros(env.n_states)
        
        for state in env.states:
            state_idx = env.state_to_idx(state)
            
            if env.is_terminal(state):
                # Terminal state: use reward from reward_params
                V_new[state_idx] = reward_params[state_idx, 0]  # Use first action
            else:
                # Bellman update
                best_value = float('-inf')
                
                for action in env.actions:
                    next_state, _, _ = env.step(state, action)
                    next_state_idx = env.state_to_idx(next_state)
                    
                    # Use learned reward
                    reward = reward_params[state_idx, action]
                    q_value = reward + gamma * V[next_state_idx]
                    
                    if q_value > best_value:
                        best_value = q_value
                
                V_new[state_idx] = best_value
        
        V = V_new
    
    return V


def compute_log_likelihood(choices: List[Choice],
                          reward_params: np.ndarray,
                          env: GridWorld,
                          model_type: str,
                          temperature: float = 1.0,
                          value_function: Optional[np.ndarray] = None) -> float:
    """
    Compute log-likelihood of choices under learned reward parameters.
    
    Args:
        choices: List of choice data
        reward_params: Learned reward parameters
        env: GridWorld environment
        model_type: Scoring model type
        temperature: Temperature parameter for softmax
        value_function: Pre-computed value function (to avoid recomputing)
        
    Returns:
        Log-likelihood value
    """
    # Use provided value function or compute if needed
    if value_function is None and model_type in ['cumulative_advantage', 'bootstrapped_return']:
        value_function = compute_value_function_from_reward(env, reward_params, n_iter=50)
    
    log_likelihood = 0.0
    
    for choice in choices:
        # Compute scores
        score1 = compute_trajectory_score_from_reward(
            choice.trajectory1, reward_params, env, model_type, value_function
        )
        score2 = compute_trajectory_score_from_reward(
            choice.trajectory2, reward_params, env, model_type, value_function
        )
        
        # Compute choice probability
        prob1 = compute_choice_probability(score1, score2, temperature)
        
        # Log-likelihood contribution
        if choice.choice == 0:
            log_likelihood += np.log(prob1 + 1e-10)  # Add small epsilon for numerical stability
        else:
            log_likelihood += np.log(1 - prob1 + 1e-10)
    
    return log_likelihood


def learn_reward(choices: List[Choice],
                env: GridWorld,
                model_type: str,
                temperature: float = 1.0,
                learning_rate: float = 0.01,
                max_iter: int = 1000,
                tol: float = 1e-6,
                seed: Optional[int] = None) -> Tuple[np.ndarray, List[float]]:
    """
    Learn reward function from choice data using gradient ascent.
    
    Uses simple gradient ascent to maximize log-likelihood.
    
    Args:
        choices: Training choice data
        env: GridWorld environment
        model_type: Scoring model type
        temperature: Temperature parameter for softmax
        learning_rate: Learning rate for gradient ascent
        max_iter: Maximum iterations
        tol: Convergence tolerance
        seed: Random seed for initialization
        
    Returns:
        (learned_reward_params, log_likelihood_history)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Initialize reward parameters (small random values)
    reward_params = np.random.normal(0, 0.1, size=(env.n_states, env.n_actions))
    
    log_likelihood_history = []
    
    for iteration in range(max_iter):
        # Pre-compute value function once per iteration (not per choice!)
        value_function = None
        if model_type in ['cumulative_advantage', 'bootstrapped_return']:
            value_function = compute_value_function_from_reward(env, reward_params, n_iter=50)
        
        # Compute current log-likelihood
        ll = compute_log_likelihood(choices, reward_params, env, model_type, temperature, value_function)
        log_likelihood_history.append(ll)
        
        # Print progress every 10 iterations
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}/{max_iter}, log-likelihood: {ll:.4f}")
        
        # Compute gradient (finite differences approximation)
        gradient = np.zeros_like(reward_params)
        epsilon = 1e-5
        
        # Only compute gradient for a subset of parameters to speed up (sample-based)
        # For full gradient, uncomment the nested loops below
        n_params = env.n_states * env.n_actions
        sample_size = min(20, n_params)  # Sample 20 parameters or all if fewer
        
        if sample_size < n_params:
            # Sample random parameters
            param_indices = np.random.choice(n_params, size=sample_size, replace=False)
        else:
            param_indices = np.arange(n_params)
        
        for param_idx in param_indices:
            state_idx = param_idx // env.n_actions
            action = param_idx % env.n_actions
            
            # Perturb parameter
            reward_params_plus = reward_params.copy()
            reward_params_plus[state_idx, action] += epsilon
            
            # Recompute value function if needed
            value_function_plus = None
            if model_type in ['cumulative_advantage', 'bootstrapped_return']:
                value_function_plus = compute_value_function_from_reward(env, reward_params_plus, n_iter=50)
            
            ll_plus = compute_log_likelihood(
                choices, reward_params_plus, env, model_type, temperature, value_function_plus
            )
            
            # Finite difference gradient
            gradient[state_idx, action] = (ll_plus - ll) / epsilon
        
        # Gradient ascent step
        reward_params += learning_rate * gradient
        
        # Check convergence
        if iteration > 0 and abs(log_likelihood_history[-1] - log_likelihood_history[-2]) < tol:
            break
    
    return reward_params, log_likelihood_history


def compute_reward_recovery_error(learned_reward: np.ndarray,
                                 true_reward: np.ndarray) -> float:
    """
    Compute MSE between learned and true reward functions.
    
    Args:
        learned_reward: Learned reward parameters
        true_reward: True reward parameters
        
    Returns:
        Mean squared error
    """
    mse = np.mean((learned_reward - true_reward) ** 2)
    return mse


def compute_choice_accuracy(choices: List[Choice],
                           reward_params: np.ndarray,
                           env: GridWorld,
                           model_type: str,
                           temperature: float = 1.0) -> float:
    """
    Compute prediction accuracy on choice data.
    
    Args:
        choices: Choice data to evaluate
        reward_params: Learned reward parameters
        env: GridWorld environment
        model_type: Scoring model type
        temperature: Temperature parameter
        
    Returns:
        Accuracy (fraction of correctly predicted choices)
    """
    # Pre-compute value function if needed
    value_function = None
    if model_type in ['cumulative_advantage', 'bootstrapped_return']:
        value_function = compute_value_function_from_reward(env, reward_params)
    
    correct = 0
    total = len(choices)
    
    for choice in choices:
        # Compute scores
        score1 = compute_trajectory_score_from_reward(
            choice.trajectory1, reward_params, env, model_type, value_function
        )
        score2 = compute_trajectory_score_from_reward(
            choice.trajectory2, reward_params, env, model_type, value_function
        )
        
        # Predict choice
        predicted_choice = 0 if score1 > score2 else 1
        
        if predicted_choice == choice.choice:
            correct += 1
    
    return correct / total if total > 0 else 0.0

