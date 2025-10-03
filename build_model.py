import torch
import torch.nn as nn
from torchvision import models
import torchvision.models as models
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import vgg16_bn,  VGG16_Weights ,vgg16, vgg11
def load_vgg16_cifar10(weights_path: str,
                       device: torch.device = None) -> nn.Module: # type: ignore
    """
    Builds a VGG‑16 for CIFAR‑10, loads the given weights, and returns
    the model in eval mode on the specified device.

    Args:
        weights_path: path to a .pth file of state_dict from training.
        device:       torch.device (e.g. 'cuda' or 'cpu'). If None, will auto‑detect.

    Returns:
        model: a VGG‑16 nn.Module ready for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build VGG16 with no pre-trained weights
    model = models.vgg11(weights=None, progress=True)
    
    in_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_feats, 10) # type: ignore

    
    state_dict = torch.load(weights_path, map_location=device, weights_only = True)
    model.load_state_dict(state_dict)

    
    model = model.to(device)
    model.eval()

    return model
