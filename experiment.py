"""
Main Experiment Runner

Runs the computational experiment comparing partial return, cumulative advantage,
and bootstrapped return models for learning reward functions from choices.
"""

import numpy as np
from typing import Dict, List, Tuple
import os

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Plots will be skipped.")

from env import GridWorld
from preference_data import generate_choice_dataset, split_dataset
from learn_reward import learn_reward, compute_reward_recovery_error, compute_choice_accuracy


def run_experiment(seed: int = 43,
                  n_choices: int = 1000,
                  train_ratio: float = 0.8,
                  temperature: float = 1.0,
                  learning_rate: float = 0.01,
                  max_iter: int = 500) -> Dict:
    """
    Run the main experiment comparing the three models.
    
    Args:
        seed: Random seed for reproducibility
        n_choices: Number of choice pairs to generate
        train_ratio: Fraction of data for training
        temperature: Temperature parameter for softmax
        learning_rate: Learning rate for reward learning
        max_iter: Maximum iterations for learning
        
    Returns:
        Dictionary with results for each model
    """
    print("=" * 60)
    print("Computational Experiment: Choice Between Partial Trajectories")
    print("=" * 60)
    print(f"Random seed: {seed}")
    print(f"Number of choices: {n_choices}")
    print(f"Train/test split: {train_ratio:.1%} / {1-train_ratio:.1%}")
    print()
    
    # Initialize environments: true environment and human belief environment
    true_env = GridWorld(goal_pos=(4, 4), seed=seed)
    belief_env = GridWorld(goal_pos=(3, 4), seed=seed)  # WRONG ON PURPOSE - belief mismatch
    true_reward = true_env.true_reward
    
    # Models to compare
    models = ['partial_return', 'cumulative_advantage', 'bootstrapped_return']
    
    results = {}
    
    for model_type in models:
        print(f"\n{'='*60}")
        print(f"Model: {model_type.replace('_', ' ').title()}")
        print(f"{'='*60}")
        
        # Generate choice data using this model
        # Use belief_env for preference generation (human has wrong beliefs)
        print("Generating synthetic choice data...")
        choices = generate_choice_dataset(
            true_env, belief_env, n_choices=n_choices, model_type=model_type,
            temperature=temperature, seed=seed
        )
        
        # Split into train/test
        train_choices, test_choices = split_dataset(choices, train_ratio, seed=seed)
        print(f"Training choices: {len(train_choices)}")
        print(f"Test choices: {len(test_choices)}")
        
        # Learn reward function (use true_env)
        print("Learning reward function...")
        learned_reward, ll_history = learn_reward(
            train_choices, true_env, model_type,
            temperature=temperature,
            learning_rate=learning_rate,
            max_iter=max_iter,
            seed=seed
        )
        
        # Evaluate (use true_env)
        print("Evaluating...")
        
        # Reward recovery error
        reward_error = compute_reward_recovery_error(learned_reward, true_reward)
        
        # Choice prediction accuracy (train) - use true_env
        train_accuracy = compute_choice_accuracy(
            train_choices, learned_reward, true_env, model_type, temperature
        )
        
        # Choice prediction accuracy (test) - use true_env
        test_accuracy = compute_choice_accuracy(
            test_choices, learned_reward, true_env, model_type, temperature
        )
        
        # Store results
        results[model_type] = {
            'learned_reward': learned_reward,
            'true_reward': true_reward,
            'reward_error': reward_error,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'log_likelihood_history': ll_history,
            'n_train': len(train_choices),
            'n_test': len(test_choices)
        }
        
        print(f"Reward recovery error (MSE): {reward_error:.6f}")
        print(f"Train accuracy: {train_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        print(f"Final log-likelihood: {ll_history[-1]:.4f}")
    
    return results


def print_results_table(results: Dict):
    """
    Print results in a formatted table.
    
    Args:
        results: Results dictionary from run_experiment
    """
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Model':<25} {'Reward Error (MSE)':<20} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 60)
    
    for model_type, result in results.items():
        model_name = model_type.replace('_', ' ').title()
        print(f"{model_name:<25} {result['reward_error']:<20.6f} "
              f"{result['train_accuracy']:<12.4f} {result['test_accuracy']:<12.4f}")
    
    print("=" * 60)


def plot_results(results: Dict, output_dir: str = 'plots'):
    """
    Generate plots comparing the three models.
    
    Args:
        results: Results dictionary from run_experiment
        output_dir: Directory to save plots
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = list(results.keys())
    model_names = [m.replace('_', ' ').title() for m in models]
    
    # Reward error
    reward_errors = [results[m]['reward_error'] for m in models]
    axes[0].bar(model_names, reward_errors, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('Reward Recovery Error (MSE)')
    axes[0].set_title('Reward Recovery Error')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Train accuracy
    train_accs = [results[m]['train_accuracy'] for m in models]
    axes[1].bar(model_names, train_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Train Accuracy')
    axes[1].set_ylim([0, 1])
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Test accuracy
    test_accs = [results[m]['test_accuracy'] for m in models]
    axes[2].bar(model_names, test_accs, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2].set_ylabel('Accuracy')
    axes[2].set_title('Test Accuracy')
    axes[2].set_ylim([0, 1])
    axes[2].tick_params(axis='x', rotation=45)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison plot to {output_dir}/model_comparison.png")
    
    # Plot 2: Learning curves
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    for model_type, result in results.items():
        ll_history = result['log_likelihood_history']
        model_name = model_type.replace('_', ' ').title()
        ax.plot(ll_history, label=model_name, linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log-Likelihood')
    ax.set_title('Learning Curves: Log-Likelihood vs Iteration')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_curves.png'), dpi=150, bbox_inches='tight')
    print(f"Saved learning curves to {output_dir}/learning_curves.png")
    
    plt.close('all')


def main():
    """Main entry point for the experiment."""
    # Set random seed for reproducibility
    seed = 42
    
    # Run experiment
    results = run_experiment(
        seed=seed,
        n_choices=1000,
        train_ratio=0.8,
        temperature=1.0,
        learning_rate=0.01,
        max_iter=500
    )
    
    # Print results table
    print_results_table(results)
    
    # Generate plots
    plot_results(results, output_dir='plots')
    
    print("\nExperiment complete!")


if __name__ == '__main__':
    main()

