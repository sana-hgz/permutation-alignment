import torch
import numpy as np
import random
from torch.utils.data import DataLoader
#from model import build_model
import torch.nn as nn
from typing import List, Dict, Optional
from copy import deepcopy
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from transformation import GetTransforms
def set_seed(seed: int):
    """
    Make experiments reproducible by fixing all RNGs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device) -> float:
    """
    Computes classification accuracy of `model` over `data_loader` on `device`.
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            preds  = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total
def save_checkpoint(checkpoint: dict, path: str):
    torch.save(checkpoint, path)

def load_checkpoint(path: str, device="cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=True)
def average_state_dicts(
    state_dicts: List[Dict[str, torch.Tensor]],
    weights: Optional[List[float]] = None
) -> Dict[str, torch.Tensor]:
    avg = deepcopy(state_dicts[0])
    for k in avg:
        stacked = torch.stack([sd[k].float() for sd in state_dicts], dim=0)
        if weights is None:
            avg[k] = stacked.mean(0)
        else:
            w = torch.tensor(weights, device=stacked.device).view(-1, *[1]*(stacked.ndim-1))
            avg[k] = (w * stacked).sum(0)
    return avg
def average_models_with_bn(model1: nn.Module, model2: nn.Module, alpha: float = 0.5) -> nn.Module:
    """
    Properly average two models with BatchNorm layers.
    
    Args:
        model1, model2: models to average
        alpha: interpolation factor (0 = model1, 1 = model2, 0.5 = equal average)
    
    Returns:
        averaged model
    """
    averaged_model = deepcopy(model1)
    
    with torch.no_grad():
        # Average all parameters
        for (name1, param1), (name2, param2), (name_avg, param_avg) in zip(
            model1.named_parameters(), 
            model2.named_parameters(), 
            averaged_model.named_parameters()
        ):
            assert name1 == name2 == name_avg, f"Parameter names don't match: {name1}, {name2}, {name_avg}"
            param_avg.data.copy_((1 - alpha) * param1.data + alpha * param2.data)
        
        # Average BatchNorm running statistics
        for (name1, module1), (name2, module2), (name_avg, module_avg) in zip(
            model1.named_modules(), 
            model2.named_modules(), 
            averaged_model.named_modules()
        ):
            assert name1 == name2 == name_avg, f"Module names don't match: {name1}, {name2}, {name_avg}"
            
            if isinstance(module1, nn.BatchNorm2d):
                # Average running statistics
                if module1.running_mean is not None and module2.running_mean is not None:
                    module_avg.running_mean.data.copy_(
                        (1 - alpha) * module1.running_mean.data + alpha * module2.running_mean.data # type: ignore
                    )
                
                if module1.running_var is not None and module2.running_var is not None:
                    module_avg.running_var.data.copy_(
                        (1 - alpha) * module1.running_var.data + alpha * module2.running_var.data # type: ignore
                    )
                
                # Keep num_batches_tracked from model1 (or could average, but integer averaging is tricky)
                if hasattr(module1, 'num_batches_tracked') and hasattr(module_avg, 'num_batches_tracked'):
                    module_avg.num_batches_tracked.data.copy_(module1.num_batches_tracked.data) # type: ignore
    
    return averaged_model
"""
def evaluate_checkpoint(ckpt_path: str,
                        model_cfg: dict,
                        val_loader,
                        device) -> float:
  
    #Loads a checkpoint from disk, builds the corresponding model,
    #and returns its validation accuracy.

    # 1) Load the raw state_dict
    sd = load_checkpoint(ckpt_path,device)

    # 2) Build & load the model
    model = build_model(**model_cfg).to(device)
    model.load_state_dict(sd)

    # 3) Evaluate on your val_loader
    return evaluate(model, val_loader, device)
"""

def build_val_loader(val_frac: float, batch_size: int, seed: int = 42):
    """
    Create a validation DataLoader from CIFAR-10‘s training split,
    using *test* transforms (no randomness).
    """
    # 1) Use the *test* pipeline (no jitter)
    val_tf = GetTransforms().testparams()
    # 2) Load the full CIFAR-10 training set with test transforms
    full = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=val_tf # type: ignore
    )
    # 3) Split into (train_unused, val_ds)
    n_val   = int(val_frac * len(full))
    n_train = len(full) - n_val
    _, val_ds = random_split(
        full,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    # 4) Return DataLoader over just the validation portion
    return DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)
def weighted_average_state_dicts(sds: list[dict], weights: list[float]) -> dict:
    """
    sds:        list of state_dicts (all same keys & tensor shapes)
    weights:    list of positive scalars (e.g. validation accuracies)
    
    Returns   ∑_i w_i * sd_i[key]  / ∑_i w_i  for each tensor.
    """
    total_w = sum(weights)
    avg_sd = {}
    for k, v0 in sds[0].items():
        # force a float accumulator
        avg_sd[k] = torch.zeros_like(v0, dtype=torch.float32)
    for sd, w in zip(sds, weights):
        factor = w / total_w
        for k, v in sd.items():
            # cast v to float before scaling
            avg_sd[k] += v.to(torch.float32) * factor
    return avg_sd

def evaluate_bert(model, data_loader, device):
    """
    Compute accuracy of a HuggingFace sequence‐classification model.

    Args:
        model:       an AutoModelForSequenceClassification (already on `device`)
        data_loader: yields dicts with keys 'input_ids', 'attention_mask', 'label'
        device:      torch.device

    Returns:
        float accuracy over the entire loader.
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for batch in data_loader:
            # move inputs to device
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["label"].to(device)

            # forward
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits

            # predictions and stats
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / total