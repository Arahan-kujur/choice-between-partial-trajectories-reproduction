# Choice Between Partial Trajectories: Computational Experiment

This repository implements the computational experiment from the paper:
**"Choice Between Partial Trajectories: Disentangling Goals from Beliefs"** (arXiv:2410.22690)

## Overview

This experiment compares three models for learning reward functions from choices between partial trajectories:

1. **Partial Return**: Sum of observed rewards so far
2. **Cumulative Advantage**: Advantage relative to optimal continuation
3. **Bootstrapped Return**: Partial return + estimated value of final state

## Project Structure

```
.
├── env.py              # Grid world MDP environment
├── trajectories.py     # Trajectory generation and partial trajectory creation
├── scoring.py          # Three scoring models (partial/advantage/bootstrapped)
├── preference_data.py  # Synthetic choice generation
├── learn_reward.py     # Preference-based reward learning
├── experiment.py       # Main experiment runner
├── plots/              # Output directory for figures
└── README.md           # This file
```

## Requirements

- Python 3.7+
- numpy
- matplotlib

## Installation

```bash
pip install numpy matplotlib
```

## Usage

Run the main experiment:

```bash
python experiment.py
```

This will:
1. Generate synthetic choice data for each model
2. Learn reward functions from the choice data
3. Evaluate reward recovery error and choice prediction accuracy
4. Print results in a table
5. Generate comparison plots in the `plots/` directory

## Experiment Details

### Environment
- 5×5 grid world
- Deterministic transitions
- Terminal goal state at (4,4) with positive reward
- Step penalty elsewhere

### Synthetic Human Model
- Uses ground-truth reward function
- Generates choices using softmax (logit) model based on trajectory scores
- Temperature parameter controls choice randomness

### Learning Task
- Learns reward function via maximum likelihood estimation
- Uses gradient ascent to maximize log-likelihood of choices
- Reward parameterized as state-action matrix

### Evaluation Metrics
- **Reward Recovery Error**: MSE between learned and true reward
- **Choice Prediction Accuracy**: Fraction of correctly predicted choices

## Results

The experiment outputs:
- Numerical results table comparing the three models
- Comparison bar charts (reward error, train/test accuracy)
- Learning curves (log-likelihood vs iteration)

## Reproducibility

The experiment uses a fixed random seed (42) for full reproducibility. All results should be deterministic when run with the same seed.

## References

Paper: "Choice Between Partial Trajectories: Disentangling Goals from Beliefs" (arXiv:2410.22690)

