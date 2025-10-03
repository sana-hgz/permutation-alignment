import torch
import torch.nn as nn
from copy import deepcopy
import ot # type: ignore
import numpy as np
import gc


def _to_long_index(idx, device):
    return torch.as_tensor(idx, dtype=torch.long, device=device).view(-1)

@torch.no_grad()
def permute_linear_with_bias(layer: nn.Linear, perm):
    idx = _to_long_index(perm, layer.weight.device)
    C = layer.weight.size(0); assert idx.numel() == C
    layer.weight.copy_(layer.weight.index_select(0, idx))
    if layer.bias is not None:
        layer.bias.copy_(layer.bias.index_select(0, idx))

@torch.no_grad()
def permute_conv_with_bias(conv: nn.Conv2d, perm):
    idx = _to_long_index(perm, conv.weight.device)
    C = conv.weight.size(0); assert idx.numel() == C
    conv.weight.copy_(conv.weight.index_select(0, idx))
    if conv.bias is not None:
        conv.bias.copy_(conv.bias.index_select(0, idx))

@torch.no_grad()
def permute_batchnorm(bn: nn.BatchNorm2d, perm):
    dev = (bn.weight.device if bn.affine else bn.running_mean.device) # type: ignore
    idx = _to_long_index(perm, dev)
    if bn.affine:
        bn.weight.copy_(bn.weight.index_select(0, idx))
        bn.bias.copy_(bn.bias.index_select(0, idx))
    bn.running_mean.copy_(bn.running_mean.index_select(0, idx)) # type: ignore
    bn.running_var.copy_(bn.running_var.index_select(0, idx)) # type: ignore





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

import torch
import torch.nn as nn
from copy import deepcopy
import ot # type: ignore
import numpy as np


class SimpleBatchedActivationCapture:
    """Capture and aggregate activations across multiple batches - no quantiles"""
    def __init__(self, max_samples_per_channel=10000):
        self.max_samples_per_channel = max_samples_per_channel
        self.activation_lists = {}  # layer_name -> list of activation tensors
        self.hooks = []
    
    def register_hooks(self, model, layer_names):
        """Register hooks for specified layers"""
        def get_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation = output.detach().cpu()  # Move to CPU immediately
                else:
                    activation = output[0].detach().cpu()
                
                # Store this batch's activation
                if name not in self.activation_lists:
                    self.activation_lists[name] = []
                
                self.activation_lists[name].append(activation)
                    
            return hook
        
        modules = dict(model.named_modules())
        for layer_name in layer_names:
            if layer_name in modules:
                hook = modules[layer_name].register_forward_hook(
                    get_activation_hook(layer_name)
                )
                self.hooks.append(hook)
    
    def get_aggregated_activations(self, layer_name, method='channel_wise'):
        """
        Get aggregated activations across all batches
        
        Returns:
            numpy array of shape [channels/features, total_samples]
        """
        if layer_name not in self.activation_lists:
            raise ValueError(f"No data for layer {layer_name}")
        
        print(f"  Aggregating {len(self.activation_lists[layer_name])} batches...")
        
        # Concatenate all batches along batch dimension
        all_activations = torch.cat(self.activation_lists[layer_name], dim=0)
        print(f"  Total aggregated shape: {all_activations.shape}")
        
        # Apply subsampling if too many samples
        all_activations = self._subsample_if_needed(all_activations)
        
        # Convert to distributions per channel/feature
        return self._compute_distributions(all_activations, method)
    
    def _subsample_if_needed(self, activations):
        """Subsample activations if there are too many samples per channel"""
        if len(activations.shape) == 4:  # Conv: [B, C, H, W]
            B, C, H, W = activations.shape
            total_samples_per_channel = B * H * W
            
            if total_samples_per_channel > self.max_samples_per_channel:
                print(f"  Too many samples ({total_samples_per_channel}), subsampling...")
                
                # Simple strategy: just reduce number of batches to fit limit
                max_batches = self.max_samples_per_channel // (H * W)
                if max_batches < 1:
                    max_batches = 1
                
                if max_batches < B:
                    # Randomly subsample batches
                    batch_indices = torch.randperm(B)[:max_batches]
                    activations = activations[batch_indices]
                    print(f"  Kept {max_batches} batches, final shape: {activations.shape}")
                    
                    # Calculate actual samples per channel
                    final_samples = max_batches * H * W
                    print(f"  Final samples per channel: {final_samples}")
        
        elif len(activations.shape) == 2:  # Linear: [B, F]
            B, F = activations.shape
            if B > self.max_samples_per_channel:
                # Randomly subsample batch dimension
                batch_indices = torch.randperm(B)[:self.max_samples_per_channel]
                activations = activations[batch_indices, :]
                print(f"  Subsampled to shape: {activations.shape}")
        
        return activations
    
    def _compute_distributions(self, activation, method):
        """Convert aggregated activations to per-channel distributions"""
        if len(activation.shape) == 4:  # Conv layer: [B, C, H, W]
            if method == 'channel_wise':
                # For each channel, flatten spatial dims and concatenate batches
                # Result: [C, B*H*W]
                B, C, H, W = activation.shape
                distributions = activation.permute(1, 0, 2, 3).contiguous().view(C, -1)
            elif method == 'spatial_mean':
                # Average over spatial dimensions first: [B, C] -> [C, B]
                distributions = activation.mean(dim=(2, 3)).transpose(0, 1)
            else:  # flatten
                # Flatten everything except batch: [B, C*H*W] -> [C*H*W, B]
                distributions = activation.view(activation.shape[0], -1).transpose(0, 1)
        
        elif len(activation.shape) == 3:  # Subsampled conv: [B, C, spatial_samples]
            # This happens when we subsample spatial dimensions irregularly
            if method == 'channel_wise':
                # Each channel has B * spatial_samples values
                # Result: [C, B*spatial_samples]
                B, C, spatial_samples = activation.shape
                distributions = activation.permute(1, 0, 2).contiguous().view(C, -1)
            elif method == 'spatial_mean':
                # Average over spatial samples: [B, C] -> [C, B] 
                distributions = activation.mean(dim=2).transpose(0, 1)
            else:  # flatten
                # Flatten to [B*C*spatial_samples] then reshape to [C*spatial_samples, B]
                B, C, spatial_samples = activation.shape
                flattened = activation.view(B, -1)  # [B, C*spatial_samples]
                distributions = flattened.transpose(0, 1)  # [C*spatial_samples, B]
        
        elif len(activation.shape) == 2:  # Linear layer: [B, F]
            # Transpose to get [F, B]
            distributions = activation.transpose(0, 1)
        
        else:
            raise ValueError(f"Unsupported activation shape: {activation.shape}")
        
        return distributions.numpy()
    
    def clear_data(self):
        """Clear stored data to free memory"""
        self.activation_lists.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def align_models_simple_batched(model_ref: nn.Module,
                              model_alt: nn.Module,
                              layer_names: list,
                              dataloader,
                              num_batches: int = 10,
                              max_samples_per_channel: int = 10000,
                              distribution_method: str = 'channel_wise') -> nn.Module:
    """
    Align models using simple batch aggregation (no quantiles)
    
    Args:
        model_ref: reference model
        model_alt: model to be aligned  
        layer_names: list of layer names to align in order
        dataloader: DataLoader to get batches from
        num_batches: number of batches to aggregate over
        max_samples_per_channel: maximum samples per channel (for memory)
        distribution_method: 'channel_wise', 'spatial_mean', or 'flatten'
    
    Returns:
        aligned copy of model_alt
    """
    with torch.no_grad():
        m_out = deepcopy(model_alt)
        
        # Set models to eval mode
        model_ref.eval()
        m_out.eval()
        
        device = next(model_ref.parameters()).device
        
        for layer_name in layer_names[:-1]:  # Don't align the last layer
            print(f"Aligning layer: {layer_name}")
            
            # Setup simple batched activation capture
            capture_ref = SimpleBatchedActivationCapture(max_samples_per_channel)
            capture_alt = SimpleBatchedActivationCapture(max_samples_per_channel)
            
            capture_ref.register_hooks(model_ref, [layer_name])
            capture_alt.register_hooks(m_out, [layer_name])
            
            # Process multiple batches
            print(f"  Collecting activations from {num_batches} batches...")
            batch_count = 0
            for batch_idx, (data, _) in enumerate(dataloader):
                if batch_count >= num_batches:
                    break
                
                data = data.to(device)
                
                # Forward pass to capture activations
                with torch.no_grad():
                    _ = model_ref(data)
                    _ = m_out(data)
                
                batch_count += 1
                if batch_count % 5 == 0 or batch_count == num_batches:
                    print(f"    Collected batch {batch_count}/{num_batches}")
            
            # Get aggregated distributions
            print(f"  Computing distributions...")
            dist_ref = capture_ref.get_aggregated_activations(layer_name, distribution_method)
            dist_alt = capture_alt.get_aggregated_activations(layer_name, distribution_method)
            
            n_channels = dist_ref.shape[0]
            print(f"  Final distribution shapes - Ref: {dist_ref.shape}, Alt: {dist_alt.shape}")
            print(f"  Computing EMD matrix for {n_channels} channels...")
            
            # Compute EMD cost matrix
            M = ot.dist(dist_ref, dist_alt)
            
            print(f"  EMD range: [{M.min():.6f}, {M.max():.6f}]")
            
            # Solve optimal transport problem
            print("  Solving optimal transport...")
            G0 = ot.emd([], [], M, numItermax=300000)
            permutation = np.argmax(G0, axis=1)
            
            print(f"  Permutation computed: {permutation[:10]}...")
            
            # Apply permutation 
            perm_t = torch.as_tensor(permutation, dtype=torch.long, device=device)
            permute_layer_weights(m_out, layer_name, perm_t, preserve_next=True)
            
            # Cleanup
            capture_ref.remove_hooks()
            capture_alt.remove_hooks()
            capture_ref.clear_data()
            capture_alt.clear_data()
            
            print(f"  Layer {layer_name} aligned successfully")
        
        return m_out

