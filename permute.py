import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights
from outils import evaluate



def load_vgg16_cifar10(weights_path: str, device: torch.device=None) -> nn.Module: # type: ignore
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
    in_feats = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_feats, 10) # type: ignore

    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)

    model = model.to(device)
    model.eval()
    return model

def permute_conv_with_bias(conv: nn.Conv2d, perm: torch.Tensor):
    # Shuffle output‑channel rows of weight
    w = conv.weight.data
    w = w.index_select(0, perm)
    conv.weight.data.copy_(w)
    # Shuffle bias entries
    if conv.bias is not None:
        b = conv.bias.data
        conv.bias.data.copy_(b[perm])

def permute_layer(model: nn.Module,
                  layer_name: str,
                  preserve_fn: bool = False) -> torch.Tensor:
    """
    Permute output channels of `layer_name` (a Conv2d in model.named_modules()).
    If preserve_fn=True, also permutes the *input* channels of the very next
    Conv2d or Linear to keep the function aligned.
    Returns the permutation tensor.
    """
    
    modules = dict(model.named_modules())
    assert layer_name in modules, f"Layer '{layer_name}' not found"
    layer = modules[layer_name]
    assert isinstance(layer, nn.Conv2d), "layer_name must reference a Conv2d"

    # random perm of its out_channels
    C_out = layer.weight.size(0)
    perm  = torch.randperm(C_out, device=layer.weight.device)

    # permute weight & bias of this conv
    permute_conv_with_bias(layer, perm)

    if preserve_fn:
        # find the next conv or linear in forward order
        names = list(model.named_modules())
        idx   = next(i for i,(n,_) in enumerate(names) if n == layer_name)
        for n_next, m_next in names[idx+1:]:
            if isinstance(m_next, nn.Conv2d):
                # shuffle its input channels 
                w2 = m_next.weight.data
                w2 = w2.index_select(1, perm)
                m_next.weight.data.copy_(w2)
                break
            elif isinstance(m_next, nn.Linear):
                w2 = m_next.weight.data  # shape [out, in]
                w2 = w2.index_select(1, perm)
                m_next.weight.data.copy_(w2)
                break

    return perm
#eval

def evaluate_permutation(weights_path: str,
                         layer_name: str,
                         preserve_fn: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf_test = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406],
                             std =[0.229,0.224,0.225]),
    ])
    test_ds    = datasets.CIFAR10("./data", train=False, download=True, transform=tf_test)
    test_loader= DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=2)

    model = load_vgg16_cifar10(weights_path, device)
    base_acc = evaluate(model, test_loader, device)
    print(f"Baseline Test Acc: {base_acc*100:.2f}%")

    # 2) permuted
    model_p = load_vgg16_cifar10(weights_path, device)
    perm = permute_layer(model_p, layer_name, preserve_fn=preserve_fn)
    perm_acc = evaluate(model_p, test_loader, device)
    print(f"After permuting '{layer_name}' out‑channels by {perm.tolist()[:8]} → {perm_acc*100:.2f}%")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python permute_vgg16.py <weights.pth> <layer_name> [preserve_fn]")
        sys.exit(1)

    weights, layer = sys.argv[1], sys.argv[2]
    preserve = (len(sys.argv)>3 and sys.argv[3].lower() in ("true","1"))
    evaluate_permutation(weights, layer, preserve_fn=preserve)

    # e.g. python permute.py final_weights.pth features.0 False