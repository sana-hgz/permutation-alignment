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

## Usage

### Basic Example

```python
from permutation_alignment import PermutationSolver
import torch

# Load two independently trained VGG models
model1 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)
model2 = torch.hub.load('pytorch/vision:v0.10.0', 'vgg16', pretrained=True)

# Initialize the solver
solver = PermutationSolver(source_model=model1, target_model=model2)

# Compute alignment using activation matching
permutation = solver.align(layer_names=['features.0', 'features.3', 'features.6'])

# Apply permutation to align neurons
aligned_model2 = solver.apply_permutation(model2, permutation)

# Merge models
merged_model = solver.merge_models(model1, aligned_model2)
```

## Project Structure

```
permutation-alignment/
├── README.md
├── requirements.txt
├── permutation_alignment/
│   ├── __init__.py
│   ├── solver.py           # Core permutation solver
│   ├── transport.py        # EMD and optimal transport utilities
│   ├── hooks.py            # PyTorch forward hooks for activation capture
│   └── merger.py           # Model merging utilities
├── examples/
│   ├── align_vgg_models.py
│   └── model_merging_demo.py
└── tests/
    ├── test_solver.py
    └── test_alignment.py
```

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
