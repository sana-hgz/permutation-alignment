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


class ActivationCapture:
    """Helper class to capture activations with hooks"""
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self, model, layer_names):
        """Register hooks for specified layers"""
        def get_activation_hook(name):
            def hook(module, input, output):
                # Store activation on CPU to save GPU memory
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach().cpu()
                else:
                    # Handle tuple outputs 
                    self.activations[name] = output[0].detach().cpu()
            return hook
        
        modules = dict(model.named_modules())
        for layer_name in layer_names:
            if layer_name in modules:
                hook = modules[layer_name].register_forward_hook(
                    get_activation_hook(layer_name)
                )
                self.hooks.append(hook)
    
    def clear_activations(self):
        """Clear stored activations to free memory"""
        self.activations.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def compute_activation_distributions(activation: torch.Tensor, method='channel_wise'):
    """
    Compute distributions from activations for EMD calculation
    
    Args:
        activation: tensor of shape [batch, channels, ...] or [batch, features]
        method: 'channel_wise', 'spatial_mean', or 'flatten'
    
    Returns:
        numpy array of shape [channels/features, distribution_size]
    """
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
    
    elif len(activation.shape) == 2:  # Linear layer: [B, F]
        # Transpose to get [F, B]
        distributions = activation.transpose(0, 1)
    
    else:
        raise ValueError(f"Unsupported activation shape: {activation.shape}")
    
    return distributions.numpy()

"""
def compute_emd_between_distributions(dist1: np.ndarray, dist2: np.ndarray, 
                                    max_samples: int = 1000) -> float:
    ""
    Compute EMD between two 1D distributions using POT library
    
    Args:
        dist1, dist2: 1D arrays representing distributions
        max_samples: maximum number of samples to use (for efficiency)
    
    Returns:
        EMD distance as float
    ""
    # Subsample if too many points
    if len(dist1) > max_samples:
        indices1 = np.random.choice(len(dist1), max_samples, replace=False)
        dist1 = dist1[indices1]
    
    if len(dist2) > max_samples:
        indices2 = np.random.choice(len(dist2), max_samples, replace=False)
        dist2 = dist2[indices2]
    
    # Create uniform weights
    a = np.ones(len(dist1)) / len(dist1)
    b = np.ones(len(dist2)) / len(dist2)
    
    # Reshape for cost matrix computation
    x_a = dist1.reshape(-1, 1)
    x_b = dist2.reshape(-1, 1)
    
    # Compute cost matrix (L1 distance for 1D case)
    M = ot.dist(x_a, x_b, metric='euclidean')
    
    # Compute EMD
    return ot.emd2(a, b, M)
"""

class StreamingQuantileTracker:
    """Track quantiles of streaming data to represent activation distributions"""
    def __init__(self, num_quantiles=100, max_samples=10000):
        self.num_quantiles = num_quantiles
        self.max_samples = max_samples
        self.samples = []
        self.quantiles = None
        self.is_finalized = False
    
    def add_samples(self, data):
        """Add new samples to the tracker"""
        if self.is_finalized:
            raise ValueError("Cannot add samples after finalization")
        
        data_flat = data.flatten().cpu().numpy()
        self.samples.extend(data_flat)
        
        # Keep only recent samples to prevent memory explosion
        if len(self.samples) > self.max_samples:
            # Randomly subsample to maintain diversity
            indices = np.random.choice(len(self.samples), self.max_samples, replace=False)
            self.samples = [self.samples[i] for i in sorted(indices)]
    
    def finalize(self):
        """Compute final quantiles from accumulated samples"""
        if len(self.samples) == 0:
            self.quantiles = np.zeros(self.num_quantiles)
        else:
            self.quantiles = np.quantile(
                self.samples, 
                np.linspace(0, 1, self.num_quantiles)
            )
        self.is_finalized = True
        # Clear samples to save memory
        self.samples = []
        return self.quantiles
    
    def get_distribution(self):
        """Get the quantile-based distribution representation"""
        if not self.is_finalized:
            self.finalize()
        return self.quantiles


class BatchedActivationCapture:
    """Enhanced activation capture that aggregates across multiple batches"""
    def __init__(self, use_quantiles=True, num_quantiles=100, max_samples_per_channel=5000):
        self.use_quantiles = use_quantiles
        self.num_quantiles = num_quantiles
        self.max_samples_per_channel = max_samples_per_channel
        
        # Storage for different methods
        if use_quantiles:
            self.quantile_trackers = {}  # layer_name -> {channel_idx: StreamingQuantileTracker}
        else:
            self.activation_buffers = {}  # layer_name -> list of tensors
        
        self.hooks = []
        self.layer_shapes = {}  # Store shapes for later processing
    
    def register_hooks(self, model, layer_names):
        """Register hooks for specified layers"""
        def get_activation_hook(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    activation = output.detach()
                else:
                    activation = output[0].detach()
                
                # Store shape info
                self.layer_shapes[name] = activation.shape
                
                if self.use_quantiles:
                    self._process_activation_quantiles(name, activation)
                else:
                    self._store_activation_batch(name, activation)
                    
            return hook
        
        modules = dict(model.named_modules())
        for layer_name in layer_names:
            if layer_name in modules:
                hook = modules[layer_name].register_forward_hook(
                    get_activation_hook(layer_name)
                )
                self.hooks.append(hook)
    
    def _process_activation_quantiles(self, layer_name, activation):
        """Process activations using streaming quantile method"""
        if layer_name not in self.quantile_trackers:
            self.quantile_trackers[layer_name] = {}
        
        if len(activation.shape) == 4:  # Conv: [B, C, H, W]
            B, C, H, W = activation.shape
            for c in range(C):
                if c not in self.quantile_trackers[layer_name]:
                    self.quantile_trackers[layer_name][c] = StreamingQuantileTracker(
                        self.num_quantiles, self.max_samples_per_channel
                    )
                # Add all spatial locations for this channel across batch
                channel_data = activation[:, c, :, :]  # [B, H, W]
                self.quantile_trackers[layer_name][c].add_samples(channel_data)
        
        elif len(activation.shape) == 2:  # Linear: [B, F]
            B, F = activation.shape
            for f in range(F):
                if f not in self.quantile_trackers[layer_name]:
                    self.quantile_trackers[layer_name][f] = StreamingQuantileTracker(
                        self.num_quantiles, self.max_samples_per_channel
                    )
                # Add all batch samples for this feature
                feature_data = activation[:, f]  # [B]
                self.quantile_trackers[layer_name][f].add_samples(feature_data)
    
    def _store_activation_batch(self, layer_name, activation):
        """Store activation batch for later aggregation"""
        if layer_name not in self.activation_buffers:
            self.activation_buffers[layer_name] = []
        
        # Store on CPU to save GPU memory
        self.activation_buffers[layer_name].append(activation.cpu())
    
    def get_aggregated_distributions(self, layer_name, method='channel_wise'):
        """Get aggregated distributions for a layer"""
        if self.use_quantiles:
            return self._get_quantile_distributions(layer_name)
        else:
            return self._get_batched_distributions(layer_name, method)
    
    def _get_quantile_distributions(self, layer_name):
        """Get quantile-based distributions"""
        if layer_name not in self.quantile_trackers:
            raise ValueError(f"No data for layer {layer_name}")
        
        trackers = self.quantile_trackers[layer_name]
        num_channels = len(trackers)
        
        # Finalize all trackers and collect quantiles
        distributions = []
        for c in range(num_channels):
            if c in trackers:
                dist = trackers[c].get_distribution()
            else:
                # Handle missing channels (shouldn't happen but be safe)
                dist = np.zeros(self.num_quantiles)
            distributions.append(dist)
        
        return np.array(distributions)  # [C, num_quantiles]
    
    def _get_batched_distributions(self, layer_name, method):
        """Get distributions from batched activations"""
        if layer_name not in self.activation_buffers:
            raise ValueError(f"No data for layer {layer_name}")
        
        # Concatenate all batches
        all_activations = torch.cat(self.activation_buffers[layer_name], dim=0)
        print(f"    Aggregated activation shape: {all_activations.shape}")
        
        # Use existing distribution computation
        return compute_activation_distributions(all_activations, method)
    
    def clear_data(self):
        """Clear stored data to free memory"""
        if self.use_quantiles:
            self.quantile_trackers.clear()
        else:
            self.activation_buffers.clear()
        self.layer_shapes.clear()
        gc.collect()
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()




def align_models_activations_batched(model_ref: nn.Module,
                                   model_alt: nn.Module,
                                   layer_names: list,
                                   dataloader,
                                   num_batches: int = 10,
                                   use_quantiles: bool = True,
                                   distribution_method: str = 'channel_wise') -> nn.Module:
    """
    Align model_alt to model_ref using batched activation aggregation.
    
    Args:
        model_ref: reference model
        model_alt: model to be aligned
        layer_names: list of layer names to align in order
        dataloader: DataLoader to get batches from
        num_batches: number of batches to aggregate over
        use_quantiles: whether to use streaming quantiles or full batch storage
        distribution_method: how to compute distributions (if not using quantiles)
    
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
            
            # Setup batched activation capture for both models
            capture_ref = BatchedActivationCapture(
                use_quantiles=use_quantiles,
                max_samples_per_channel=5000 if use_quantiles else 10000
            )
            capture_alt = BatchedActivationCapture(
                use_quantiles=use_quantiles,
                max_samples_per_channel=5000 if use_quantiles else 10000
            )
            
            capture_ref.register_hooks(model_ref, [layer_name])
            capture_alt.register_hooks(m_out, [layer_name])
            
            # Process multiple batches
            print(f"  Processing {num_batches} batches for robust statistics...")
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
                if batch_count % 5 == 0:
                    print(f"    Processed batch {batch_count}/{num_batches}")
            
            print(f"  Computing distributions from {batch_count} batches...")
            
            # Get aggregated distributions
            dist_ref = capture_ref.get_aggregated_distributions(layer_name, distribution_method)
            dist_alt = capture_alt.get_aggregated_distributions(layer_name, distribution_method)
            
            n_channels = dist_ref.shape[0]
            print(f"  Distribution shapes - Ref: {dist_ref.shape}, Alt: {dist_alt.shape}")
            print(f"  Computing EMD matrix for {n_channels} channels/neurons...")
            
            # Compute EMD cost matrix
            M = ot.dist(dist_ref, dist_alt)
            
            print(f"  EMD matrix computed, shape: {M.shape}")
            print(f"  EMD range: [{M.min():.6f}, {M.max():.6f}]")
            
            # Solve optimal transport problem
            print("  Solving optimal transport...")
            G0 = ot.emd([], [], M, numItermax=300000)
            permutation = np.argmax(G0, axis=1)
            
            print(f"  Permutation computed: {permutation[:10]}... (showing first 10)")
            
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

def align_models_activations(model_ref: nn.Module,
                           model_alt: nn.Module,
                           layer_names: list,
                           input_data: torch.Tensor,
                           distribution_method: str = 'channel_wise') -> nn.Module:
    """
    Align model_alt to model_ref by comparing activation distributions using EMD.
    
    Args:
        model_ref: reference model
        model_alt: model to be aligned
        layer_names: list of layer names to align in order
        input_data: input tensor to pass through models for activation capture
        distribution_method: how to compute distributions from activations
    
    Returns:
        aligned copy of model_alt
    
    """
    with torch.no_grad():
        m_out = deepcopy(model_alt)
    
        # Set models to eval mode
        model_ref.eval()
        m_out.eval()
    
        device = next(model_ref.parameters()).device
        input_data = input_data.to(device)
        
        for layer_name in layer_names[:-1]:  # Don't align the last layer
            print(f"Aligning layer: {layer_name}")
            
            # Setup activation capture for both models
            capture_ref = ActivationCapture()
            capture_alt = ActivationCapture()
            
            capture_ref.register_hooks(model_ref, [layer_name])
            capture_alt.register_hooks(m_out, [layer_name])
            
            # Forward pass to capture activations
            with torch.no_grad():
                _ = model_ref(input_data)
                _ = m_out(input_data)
            
            # Get activations
            act_ref = capture_ref.activations[layer_name]
            act_alt = capture_alt.activations[layer_name]
            
            print(f"  Activation shapes - Ref: {act_ref.shape}, Alt: {act_alt.shape}")
            
            # Compute distributions for each channel/neuron
            dist_ref = compute_activation_distributions(act_ref, distribution_method)
            dist_alt = compute_activation_distributions(act_alt, distribution_method)
            
            n_channels = dist_ref.shape[0]
            print(f"  Computing EMD matrix for {n_channels} channels/neurons...")
            
            # Compute EMD cost matrix between all pairs of channels/neurons
            M = np.zeros((n_channels, n_channels))
            M = ot.dist(dist_ref, dist_alt)    
            """
            for i in range(n_channels):
                for j in range(n_channels):
                    try:
                        emd_dist = compute_emd_between_distributions(
                            dist_ref[i], dist_alt[j]
                        )
                        M[i, j] = emd_dist
                    except Exception as e:
                        print(f"  Warning: EMD computation failed for ({i},{j}): {e}")
                        M[i, j] = np.inf
            """
            print(f"  EMD matrix computed, shape: {M.shape}")
            print(f"  EMD range: [{M.min():.6f}, {M.max():.6f}]")
            
            # Solve optimal transport problem
            print("  Solving optimal transport...")
            with torch.no_grad():
                # Use POT to compute the optimal transport plan
                # G0: [n_channels, n_channels] matrix of transport plans
                G0 = ot.emd([], [], M, numItermax=500000)
                permutation = np.argmax(G0, axis=1)  # Note: axis=1 for row-to-col matching
            
            print(f"  Permutation computed: {permutation[:10]}... (showing first 10)")
            
            # Apply permutation
            device = next(m_out.parameters()).device
            perm_t = torch.as_tensor(permutation, dtype=torch.long, device=device)
            permute_layer_weights(m_out, layer_name, perm_t, preserve_next=True)
            
            # Cleanup
            capture_ref.remove_hooks()
            capture_alt.remove_hooks()
            capture_ref.clear_activations()
            capture_alt.clear_activations()
            
            print(f"  Layer {layer_name} aligned successfully")
        
        return m_out

