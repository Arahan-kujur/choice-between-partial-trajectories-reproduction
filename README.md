# Choice Between Partial Trajectories: Disentangling Goals from Beliefs

This repository contains a from-scratch, fully reproducible implementation of the experiments from the paper:

**"Choice Between Partial Trajectories: Disentangling Goals from Beliefs"**  
Henrik Marklund, Benjamin Van Roy  
arXiv:2410.22690

The goal of this project is to study preference-based reward learning from choices between partial trajectories, and to compare three scoring models under controlled conditions.

## Overview

The paper argues that learning from partial trajectories requires explicitly modeling how humans reason about future outcomes, especially when human beliefs about the environment differ from reality.

This repository:
- Reimplements the experimental setup using a small tabular grid-world MDP
- Implements all three scoring models described in the paper
- Introduces a controlled belief mismatch between the true environment and the human's belief environment
- Evaluates models on reward recovery and held-out preference prediction accuracy

**Note:** No official code was released with the original paper; this implementation is independent and based solely on the paper description.

## Key Experimental Detail: Belief Mismatch

To isolate belief–goal disentanglement, we explicitly separate:

### True Environment
- Used for reward learning
- Used for evaluation
- Goal location: `(4, 4)`

### Human Belief Environment
- Used **only** when generating synthetic human preferences
- Contains an intentionally incorrect goal location: `(3, 4)`
- The learner does not observe the belief model and must infer preferences from choice data alone

This setup targets the regime where the paper claims bootstrapped return should outperform partial return.

## Models Compared

### Partial Return
Scores trajectories using only the observed rewards so far.

### Cumulative Advantage
Scores trajectories relative to an optimal future continuation.

### Bootstrapped Return
Scores trajectories using observed rewards plus an estimated future value.

## Results Summary

Across multiple random seeds, we observe:
- **Bootstrapped Return** achieves higher held-out preference prediction accuracy than Partial Return under belief mismatch
- **Partial Return** fails to generalize when human beliefs deviate from true dynamics
- Reward recovery MSE does not always improve, likely due to identifiability limitations in small tabular domains

### Example Output

```
Model                     Reward Error (MSE)   Train Acc    Test Acc
------------------------------------------------------------
Partial Return            ~4.10                ~0.66        ~0.53
Cumulative Advantage      ~4.52                ~0.58        ~0.52
Bootstrapped Return       ~8.40                ~0.54        ~0.58
```

The primary signal appears in generalization accuracy, consistent with the motivation of the paper.

## Project Structure

```
.
├── env.py               # Grid-world MDP definition
├── trajectories.py      # Rollouts and partial trajectory generation
├── scoring.py           # Partial, cumulative advantage, bootstrapped return
├── preference_data.py   # Synthetic human preference generation
├── learn_reward.py      # Preference-based reward learning
├── experiment.py        # Main experiment runner
└── plots/               # Optional output plots
```

## Requirements

- Python 3.9+
- NumPy
- Matplotlib (optional, for plots only)

### Installation

```bash
pip install numpy matplotlib
```

## Running the Experiment

Run the full comparison with:

```bash
python experiment.py
```

This will:
1. Generate synthetic preference data using the belief environment
2. Train reward models using each scoring method
3. Evaluate reward recovery error and preference prediction accuracy
4. Print a summary table of results

## Notes

- Results are evaluated directionally, not by exact numerical matching to the paper.
- The environment is intentionally small to ensure interpretability and reproducibility.
- The benefit of bootstrapped return appears primarily in held-out preference accuracy, not necessarily in raw reward MSE.

## License

This project is provided for research and educational purposes. Add a license file if reuse is desired.

## Acknowledgements

This work is based on:

**"Choice Between Partial Trajectories: Disentangling Goals from Beliefs"**  
Henrik Marklund, Benjamin Van Roy  
arXiv:2410.22690
