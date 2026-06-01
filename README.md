# Permutation Alignment

Resolved neuron alignment in independently trained VGG models using Earth Mover's Distance (EMD)-based optimal transport and activation matching.

## Overview

This project implements a novel approach to align neurons across independently trained neural networks by leveraging optimal transport theory and permutation symmetries. By discovering the bijective mappings between neurons in different models, we enable effective model merging and knowledge transfer without retraining.

## Key Features

- **Bijective Permutation Solver**: Efficiently discovers one-to-one neuron correspondences across models using optimal transport
- **Forward Hooks Integration**: Leverages PyTorch forward hooks for seamless activation matching and feature extraction
- **Model Merging**: Exploits permutation symmetries to merge independently trained models
- **EMD-Based Alignment**: Uses Earth Mover's Distance for principled neuron matching based on activation patterns

## Motivation

Training separate neural networks often results in models with functionally equivalent but structurally different neuron arrangements. Traditional approaches treat this as a computational barrier, but this project recognizes it as an opportunity. By discovering the permutation symmetries that relate different models, we can:

1. Merge independently trained networks
2. Transfer knowledge between models
3. Study neural network loss landscapes
4. Create ensemble methods that respect neuron correspondence

## Technical Approach

### Core Algorithm

The method extends concepts from:
- **Git Re-Basin** ([Frankle et al., 2022](https://arxiv.org/abs/2209.04173)): Finding mode connectivity in loss landscapes
- **Find the Lady** ([Pnevmatikakis et al., 2023](https://arxiv.org/abs/2303.13184)): Neuron alignment through optimal transport

### Implementation Details

1. **Feature Extraction**: Forward hooks capture activations from intermediate layers of both source and target models
2. **Distance Computation**: Calculate pairwise Earth Mover's Distance between neuron activation distributions
3. **Optimal Transport Solver**: Solve the assignment problem to find the optimal bijective permutation
4. **Model Merging**: Apply discovered permutations to align weights and enable model combination

## Installation

```bash
git clone https://github.com/sana-hgz/permutation-alignment.git
cd permutation-alignment
pip install -r requirements.txt
```

## Project Structure

```
permutation-alignment/
├── README.md                  # Project documentation
├── build_model.py             # Model building utilities
├── outils.py                  # General utility functions
├── permutation.py             # Core permutation solver
├── permutation3.py            # Extended permutation solver (v3)
├── permutation4.py            # Extended permutation solver (v4)
├── permute.py                 # Model permutation application
├── transformation.py          # Neural network transformations
├── train2.py                  # Training script
├── test.py                    # Test and evaluation script
├── job2.sh                    # Batch job submission script
└── .gitignore                 # Git ignore file
```

## Key Files

- **permutation3.py / permutation4.py**: Main implementations of the alignment algorithm
- **build_model.py**: Constructs VGG models for alignment
- **train2.py**: Trains independent VGG models
- **test.py**: Evaluates alignment quality and model merging results
- **outils.py**: Utility functions including EMD computations and activation matching
- **permute.py**: Applies discovered permutations to model weights

## Requirements

- Python 3.8+
- PyTorch 1.9+
- NumPy
- SciPy (for optimal transport solvers)
- POT (Python Optimal Transport library)

See `requirements.txt` for complete dependencies.

## Performance

The alignment quality depends on:
- **Layer selection**: Early layers typically have more generalizable features
- **Dataset size**: Larger datasets provide better activation statistics
- **Model similarity**: Models trained with similar initialization and hyperparameters align more easily
- **EMD resolution**: Trade-off between accuracy and computational cost
