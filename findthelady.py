import torch
import torch.nn as nn
from copy import deepcopy
import ot # type: ignore
import numpy as np

def compute_l2_distance(W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise L2 distance matrix D between flattened output-channel rows of W1 and W2.
    W1, W2: weight tensors where the first dimension indexes output channels,
    and the remaining dimensions are flattened into a feature vector.
    Returns D of shape [C_out, C_out] where D[i,j] = ||row_i(W1) - row_j(W2)||_2.
    """
    rows1 = W1.view(W1.size(0), -1) 
    rows2 = W2.view(W2.size(0), -1) 
    # compute squared distances: ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    norms1 = (rows1**2).sum(dim=1, keepdim=True)  
    norms2 = (rows2**2).sum(dim=1, keepdim=True)  
    dot = rows1 @ rows2.t()                     
    
    D2 = norms1 + norms2.t() - 2.0 * dot
    # ensure non-negative
    D2 = torch.clamp(D2, min=0.0)
    return torch.sqrt(D2)
def _to_long_index(idx, device):
    idx = torch.as_tensor(idx, dtype=torch.long, device=device)
    return idx.view(-1)
def permute_linear_with_bias(layer: nn.Linear, perm: torch.Tensor):
    with torch.no_grad():
        perm = _to_long_index(perm, layer.weight.device)
        # permute output neurons (rows)
        layer.weight.copy_(layer.weight.index_select(0, perm))
        if layer.bias is not None:
            layer.bias.copy_(layer.bias.index_select(0, perm))

def greedy_permutation_dist(D: torch.Tensor) -> torch.Tensor:
    """
    Given a distance matrix D of shape [N, N], greedily match each row i to the column j
    with smallest D[i,j] that is not yet assigned.
    Returns perm tensor of shape [N] where perm[i] = matched j.
    """
    N = D.size(0)
    perm = torch.full((N,), -1, dtype=torch.long, device=D.device)
    used = torch.zeros(N, dtype=torch.bool, device=D.device)
    for i in range(N):
        # pick smallest distance in row i among unused columns
        row = D[i].clone()
        row[used] = float('inf')
        j = torch.argmin(row).item()
        perm[i] = j
        used[j] = True # type: ignore
    return perm
def greedy_permutation_second_best(D: torch.Tensor) -> torch.Tensor:
    """
    For each row i, pick the **second**-smallest distance D[i,j], ensuring
    you don’t just match i→i (distance=0).
    """
    N = D.size(0)
    perm = torch.full((N,), -1, dtype=torch.long, device=D.device)
    used = torch.zeros(N, dtype=torch.bool, device=D.device)
    for i in range(N):
        row = D[i].clone()
        row[used] = float('inf')
        # get the two smallest entries' indices
        vals, idxs = torch.sort(row)        # ascending order
        # vals[0] is the smallest , take vals[1]
        perm[i] = idxs[1].item() 
        j= perm[i]
        used[j] = True           # second‐best match
    return perm
def permute_batchnorm(bn: nn.BatchNorm2d, perm: torch.Tensor):
    bn.weight.data.copy_(bn.weight.data[perm])
    bn.bias.data.copy_(bn.bias.data[perm])
    bn.running_mean.data.copy_(bn.running_mean.data[perm]) # type: ignore
    bn.running_var.data.copy_(bn.running_var.data[perm]) # type: ignore


def permute_conv_with_bias(conv: nn.Conv2d, perm: torch.Tensor):
    # Shuffle output-channel rows of weight
    w = conv.weight.data
    w = w.index_select(0, perm)
    conv.weight.data.copy_(w)
    # Shuffle bias entries
    if conv.bias is not None:
        b = conv.bias.data
        conv.bias.data.copy_(b[perm])
"""

def permute_layer_weights(model: nn.Module, layer_name: str, perm: torch.Tensor, preserve_next: bool = True):
    
    Apply permutation to the specified layer's out-channels and optionally permute next layer's in-channels.
    If a BatchNorm2d follows the Conv2d, its parameters are permuted accordingly.
    
    layer_name: name as in model.named_modules().
    perm: tensor of shape [C_out] mapping new index -> old index.
  
    modules = dict(model.named_modules())
    assert layer_name in modules, f"Layer '{layer_name}' not found"
    layer = modules[layer_name]
    assert isinstance(layer, nn.Conv2d), "layer_name must reference a Conv2d"

    # permute weight & bias of this conv
    permute_conv_with_bias(layer, perm)

    # find the next layers after this conv
    names = list(model.named_modules())
    idx = next(i for i, (n, _) in enumerate(names) if n == layer_name)

    for n_next, m_next in names[idx + 1:]:
        if isinstance(m_next, nn.BatchNorm2d):
            # Permute the batch norm parameters
            permute_batchnorm(m_next, perm)
            continue  # continue in case both BN and Conv2d follow

        if preserve_next and isinstance(m_next, (nn.Conv2d, nn.Linear)):
            # Permute input channels of next Conv2d or Linear
            w2 = m_next.weight.data
            if isinstance(m_next, nn.Conv2d):
                w2 = w2.index_select(1, perm)  # input channels axis
            elif isinstance(m_next, nn.Linear):
                w2 = w2.index_select(1, perm)  # input features axis
            m_next.weight.data.copy_(w2)
            break
"""
def permute_layer_weights(model: nn.Module, layer_name: str, perm: torch.Tensor, preserve_next: bool = True):
    """
    Apply permutation to the specified layer's out-channels/neurons and
    optionally permute the next layer's input channels/features.
    If a BatchNorm2d follows a Conv2d, its parameters are permuted accordingly.

    layer_name: name as in model.named_modules()
    perm: tensor/list of shape [C_out] mapping new_index -> old_index
    """
    modules = dict(model.named_modules())
    assert layer_name in modules, f"Layer '{layer_name}' not found"
    layer = modules[layer_name]
    assert isinstance(layer, (nn.Conv2d, nn.Linear)), "layer must be Conv2d or Linear"

    # 1) Permute this layer's outputs (rows)
    if isinstance(layer, nn.Conv2d):
        permute_conv_with_bias(layer, perm)
    else:  # Linear
        permute_linear_with_bias(layer, perm)

    # 2) Walk forward to find the immediate consumer(s)
    names = list(model.named_modules())
    idx = next(i for i, (n, _) in enumerate(names) if n == layer_name)

    # Convert perm to proper tensor once (reuse for next layers)
    perm_t = _to_long_index(perm, next(model.parameters()).device)

    for n_next, m_next in names[idx + 1:]:
        # Handle BN right after a Conv2d
        if isinstance(layer, nn.Conv2d) and isinstance(m_next, nn.BatchNorm2d):
            permute_batchnorm(m_next, perm_t)
            continue  # keep searching for the real consumer

        if not preserve_next:
            break

        # ---- Cases for the next consumer ----
        if isinstance(layer, nn.Conv2d) and isinstance(m_next, nn.Conv2d):
            # Permute input channels of next conv (dim=1)
            with torch.no_grad():
                w2 = m_next.weight
                # guard grouped/depthwise convs
                groups = getattr(m_next, "groups", 1)
                if groups != 1:
                    raise ValueError(f"Grouped conv not supported when propagating permutation to {n_next}")
                m_next.weight.copy_(w2.index_select(1, perm_t))
            break

        if isinstance(layer, nn.Conv2d) and isinstance(m_next, nn.Linear):
            # Conv2d -> Flatten -> Linear
            # Inputs to the Linear are laid out as [c0(HW), c1(HW), ...]
            with torch.no_grad():
                w2 = m_next.weight
                in_features = w2.size(1)
                C_out = layer.weight.size(0)
                assert in_features % C_out == 0, (
                    f"{n_next}.in_features={in_features} is not divisible by channels={C_out}. "
                    "Need known flatten shape to propagate permutation."
                )
                block = in_features // C_out  # equals H*W after flatten
                # build expanded index for flattened features
                # for each channel index p, take its block [p*block : (p+1)*block]
                base = torch.arange(block, device=perm_t.device)
                perm_expanded = (perm_t.view(-1, 1) * block + base).reshape(-1)
                m_next.weight.copy_(w2.index_select(1, perm_expanded))
            break

        if isinstance(layer, nn.Linear) and isinstance(m_next, nn.Linear):
            # Linear -> Linear: permute input features of next linear (dim=1)
            with torch.no_grad():
                w2 = m_next.weight
                m_next.weight.copy_(w2.index_select(1, perm_t))
            break

        if isinstance(layer, nn.Linear) and isinstance(m_next, nn.Conv2d):
            # Rare: requires knowledge of reshape mapping (in_features -> C*H*W).
            # Not safe to guess without a recorded (C,H,W); raise to avoid silent misalignment.
            raise ValueError(
                f"Cannot safely propagate permutation from Linear '{layer_name}' to Conv2d '{n_next}' "
                "without a known reshape; provide (C,H,W) or a hook-based mapper."
            )

        # Skip layers that don't consume channels (e.g., ReLU, Pooling, Dropout)
        # and keep scanning until we hit the next parametric consumer.
        # If you have branches, you may need to propagate into *all* consumers.
        continue
def _flatten_filters(w: torch.Tensor) -> np.ndarray:
    # w: [out, ...] -> (out, feat)
    return w.detach().float().cpu().view(w.shape[0], -1).numpy()


def align_models_l2(model_ref: nn.Module,
                    model_alt: nn.Module,
                    layer_names: list) -> nn.Module:
    """
    Align model_alt to model_ref by greedy layer-wise permutation that minimizes L2 distance.
    layer_names: list of layer names to align in order.
    Returns aligned copy of model_alt.
    """
    m_out = deepcopy(model_alt)
    for layer_name in layer_names[:-1]:  
        # get weights to align
        #w_ref = dict(model_ref.named_modules())[layer_name].weight.data.clone()  # type: ignore
        #w_alt = dict(m_out.named_modules())[layer_name].weight.data.clone()   # type: ignore
        w_ref = dict(model_ref.named_modules())[layer_name].weight
        w_alt = dict(m_out.named_modules())[layer_name].weight
        # flatten filters to 2D: (out, feat)
        #w_ref = w_ref.view(w_ref.size(0), -1)  # [C_out, C_in * H * W]
        #w_alt = w_alt.view(w_alt.size(0), -1)  # [C_out, C_in * H * W]
        # compute distance matrix
        w_ref = _flatten_filters(w_ref) # type: ignore
        w_alt = _flatten_filters(w_alt) # type: ignore
        M = ot.dist(w_ref, w_alt)
        # compute L2 distance matrix
        #D = compute_l2_distance(w_ref, w_alt)
        G0 = ot.emd([], [], M, numItermax=300000) ## solver for the EMD
        permutation = np.argmax(G0, axis=0) ## get the permutation from the EMD matrix
        #perm = greedy_permutation_dist(D)  # greedy permutation based on L2 distance
        device = next(m_out.parameters()).device
        perm_t = torch.as_tensor(permutation, dtype=torch.long, device=device)
        print(f"Aligning {layer_name}:  (perm={permutation.tolist()[:20]})")
        # apply permutation
        permute_layer_weights(m_out, layer_name, perm_t, preserve_next=True)
    return m_out


