"""
Synthetic Choice Data Generation

Generates synthetic human choices between partial trajectories using
the three scoring models (partial return, cumulative advantage, bootstrapped return).
"""

import numpy as np
from typing import List, Tuple, Optional
from env import GridWorld
from trajectories import Trajectory, generate_partial_trajectories, compute_value_function
from scoring import score_trajectory, compute_choice_probability


class Choice:
    """
    Represents a choice between two partial trajectories.
    """
    
    def __init__(self, trajectory1: Trajectory, trajectory2: Trajectory, 
                 choice: int, model_type: str):
        """
        Initialize choice.
        
        Args:
            trajectory1: First trajectory option
            trajectory2: Second trajectory option
            choice: 0 if trajectory1 chosen, 1 if trajectory2 chosen
            model_type: Model used to generate choice ('partial_return', etc.)
        """
        self.trajectory1 = trajectory1
        self.trajectory2 = trajectory2
        self.choice = choice  # 0 or 1
        self.model_type = model_type
    
    def get_chosen_trajectory(self) -> Trajectory:
        """Get the chosen trajectory."""
        return self.trajectory1 if self.choice == 0 else self.trajectory2
    
    def get_rejected_trajectory(self) -> Trajectory:
        """Get the rejected trajectory."""
        return self.trajectory2 if self.choice == 0 else self.trajectory1


def generate_choice(trajectory1: Trajectory,
                   trajectory2: Trajectory,
                   env: GridWorld,
                   model_type: str,
                   value_function: Optional[np.ndarray] = None,
                   temperature: float = 1.0,
                   seed: Optional[int] = None) -> Choice:
    """
    Generate a synthetic choice between two partial trajectories.
    
    The choice is generated using a softmax (logit) model based on the
    specified scoring model.
    
    Args:
        trajectory1: First trajectory option
        trajectory2: Second trajectory option
        env: GridWorld environment
        model_type: Scoring model ('partial_return', 'cumulative_advantage', 'bootstrapped_return')
        value_function: Value function (required for advantage and bootstrapped models)
        temperature: Temperature parameter for softmax (higher = more random)
        seed: Random seed
        
    Returns:
        Choice object
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Compute scores
    score1 = score_trajectory(trajectory1, model_type, env, value_function)
    score2 = score_trajectory(trajectory2, model_type, env, value_function)
    
    # Compute choice probability
    prob1 = compute_choice_probability(score1, score2, temperature)
    
    # Sample choice
    choice = 0 if np.random.random() < prob1 else 1
    
    return Choice(trajectory1, trajectory2, choice, model_type)


def generate_choice_dataset(true_env: GridWorld,
                            belief_env: GridWorld,
                            n_choices: int = 1000,
                            model_type: str = 'partial_return',
                            min_traj_length: int = 2,
                            max_traj_length: int = 10,
                            temperature: float = 1.0,
                            seed: Optional[int] = None) -> List[Choice]:
    """
    Generate a dataset of synthetic choices between partial trajectories.
    
    Uses belief_env for preference generation (human has wrong beliefs).
    Uses true_env for trajectory generation (trajectories are from true environment).
    
    Args:
        true_env: True GridWorld environment (for generating trajectories)
        belief_env: Human belief GridWorld environment (for scoring/preferences)
        n_choices: Number of choice pairs to generate
        model_type: Scoring model to use for generating choices
        min_traj_length: Minimum trajectory length
        max_traj_length: Maximum trajectory length
        temperature: Temperature parameter for softmax
        seed: Random seed
        
    Returns:
        List of Choice objects
    """
    if seed is not None:
        np.random.seed(seed)
    
    # MANDATORY DEBUG PRINT: Confirm belief mismatch is active
    print("USING HUMAN BELIEF GOAL:", belief_env.goal_pos)
    
    # Pre-compute value function if needed (using BELIEF environment)
    value_function = None
    if model_type in ['cumulative_advantage', 'bootstrapped_return']:
        value_function = compute_value_function(belief_env)
    
    choices = []
    
    for i in range(n_choices):
        # Generate two random partial trajectories (using TRUE environment)
        traj1 = generate_partial_trajectories(
            true_env, n_trajectories=1, 
            min_length=min_traj_length, 
            max_length=max_traj_length,
            seed=None
        )[0]
        
        traj2 = generate_partial_trajectories(
            true_env, n_trajectories=1,
            min_length=min_traj_length,
            max_length=max_traj_length,
            seed=None
        )[0]
        
        # Generate choice (using BELIEF environment for scoring)
        choice = generate_choice(
            traj1, traj2, belief_env, model_type,
            value_function=value_function,
            temperature=temperature,
            seed=None
        )
        
        choices.append(choice)
    
    return choices


def split_dataset(choices: List[Choice], 
                 train_ratio: float = 0.8,
                 seed: Optional[int] = None) -> Tuple[List[Choice], List[Choice]]:
    """
    Split choice dataset into train and test sets.
    
    Args:
        choices: List of choices
        train_ratio: Fraction of data for training
        seed: Random seed for shuffling
        
    Returns:
        (train_choices, test_choices)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Shuffle
    indices = np.arange(len(choices))
    np.random.shuffle(indices)
    
    # Split
    n_train = int(len(choices) * train_ratio)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_choices = [choices[i] for i in train_indices]
    test_choices = [choices[i] for i in test_indices]
    
    return train_choices, test_choices

