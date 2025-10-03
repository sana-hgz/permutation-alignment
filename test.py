from train_vgg import eval_model
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from build_model import load_vgg16_cifar10
import torch
torch.cuda.empty_cache()
from findthelady import align_models_l2
from outils import evaluate , average_state_dicts, set_seed 
from permutation3 import align_models_activations_batched, align_models_activations
from permutation4 import align_models_simple_batched
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
batch_size = 256

model1  = load_vgg16_cifar10("vgg11_cifar10_12.pth", device)
model2  = load_vgg16_cifar10("vgg11_cifar10_aligned.pth", device)
#model3  = load_vgg16_cifar10("vgg16_cifar10_aligned.pth", device)
#input_data = torch.randn(batch_size, 3, 224, 224, device=device)

conv_layers = [
    name
    for name, module in model1.named_modules()
    if isinstance(module, nn.Conv2d)
]
#print(model)
"""
def get_conv_and_fc_layer_names(model: nn.Module):
    conv_names, fc_names = [], []
    for name, m in model.named_modules():
        if not name:  # skip the root module
            continue
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            conv_names.append(name)
        elif isinstance(m, nn.Linear):
            fc_names.append(name)
    return conv_names, fc_names
"""
def get_all_conv_fc_in_order(model: nn.Module):
    #Single list in definition order: convs then linears as they appear.
    names = []
    for name, m in model.named_modules():
        if not name: 
            continue
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
            names.append(name)
    return names

#conv_layers, fc_layers = get_conv_and_fc_layer_names(model1)
layer_names = get_all_conv_fc_in_order(model1)

test_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010))
])
criterion = nn.CrossEntropyLoss()
test_ds      = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
#batch_data, _ = next(iter(test_loader))  # Prend le premier batch
from itertools import islice
batch_data, batch_labels = next(islice(test_loader, 2,3)) # Prend le deuxième batch
layer_names = get_all_conv_fc_in_order(model1) 
model3= align_models_activations(
    model_ref=model1,
    model_alt=model2,
    layer_names=layer_names,
    input_data=batch_data,
    distribution_method = 'channel_wise'
)

torch.save(model3.state_dict(), "vgg11_cifar10_aligned2.pth")

sd2 = model2.state_dict()
sd3 = model3.state_dict()

all_equal1 = all(torch.equal(sd2[k], sd3[k]) for k in sd2)
print(all_equal1) 

"""
#model1  = load_vgg16_cifar10("vgg16_cifar10_1.pth", device)
for name, _ in model1.named_modules():
    print(name)
for idx, layer in enumerate(model1.features): # type: ignore
    print(f"features[{idx:2d}] → {layer.__class__.__name__}")
for idx, layer in enumerate(model1.classifier): # type: ignore
    print(f"classifier[{idx:2d}] → {layer.__class__.__name__}")
    
"""
test_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010))
])
batch_size = 32
criterion = nn.CrossEntropyLoss()
test_ds      = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=2)
def evaluate_batch(model: nn.Module, batch_data: torch.Tensor, batch_labels: torch.Tensor, device: torch.device) -> float:
    """
    Computes accuracy of `model` on a single batch.
    """
    model.eval()
    with torch.no_grad():
        X, y = batch_data.to(device), batch_labels.to(device)
        logits = model(X)
        preds = logits.argmax(dim=1)
        correct = (preds == y).sum().item()
        total = y.size(0)
    return correct / total
test_acc1 = evaluate(model1, test_loader, device)
acc1 = evaluate_batch(model1, batch_data, batch_labels, device)
test_acc3 = evaluate(model3, test_loader, device)
acc3 = evaluate_batch(model3, batch_data, batch_labels, device)
test_acc2 = evaluate(model2, test_loader, device)
acc2 = evaluate_batch(model2, batch_data, batch_labels, device)
print(f"Batch accuracy model1: {acc1:.4f}")
print(f"Batch accuracy model2: {acc2:.4f}")
print(f"Batch accuracy model3: {acc3:.4f}")

print(f"Test acc 3={test_acc3:.4f}")

print(f"Test acc 2={test_acc2:.4f}")   
print(f"Test acc 1={test_acc1:.4f}")  

def uniform_soup(ckpt_paths, out_path):
    sds = [load_vgg16_cifar10(p).state_dict() for p in ckpt_paths]
    avg_sd = average_state_dicts(sds) 
    torch.save(avg_sd, out_path)
    print(f"Uniform soup saved to {out_path}")
ckpt_paths =["vgg11_cifar10_aligned2.pth", "vgg11_cifar10_12.pth"]
ckpt_paths2 =["vgg11_cifar10_12.pth", "vgg11_cifar10_13.pth"]

uniform_soup(ckpt_paths, "vgg11_cifar10_soup1.pth")
model4 = load_vgg16_cifar10("vgg11_cifar10_soup1.pth", device)
test_acc4=eval_model(model4, test_loader,criterion, device)
print(f"Test acc soup aligned ={test_acc4:.4f}")
acc = evaluate_batch(model4, batch_data, batch_labels, device)
print(f"Batch accuracy: {acc:.4f}")
uniform_soup(ckpt_paths2, "vgg11_cifar10_soup2.pth")
model5 = load_vgg16_cifar10("vgg11_cifar10_soup2.pth", device)
test_acc5=eval_model(model5, test_loader,criterion, device)
print(f"Test acc soup not aligned={test_acc5:.4f}")

sd4 = model4.state_dict()
sd5 = model5.state_dict()

all_equal = all(torch.equal(sd4[k], sd5[k]) for k in sd4)
print(all_equal) 
sd2 = model2.state_dict()
sd3 = model3.state_dict()

all_equal1 = all(torch.equal(sd2[k], sd3[k]) for k in sd2)
print(all_equal1) 
def state_dicts_close(sd_a, sd_b, rtol=1e-5, atol=1e-8):
    if sd_a.keys() != sd_b.keys():
        return False
    for k in sd_a:
        a, b = sd_a[k], sd_b[k]
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if torch.is_floating_point(a):
            if not torch.allclose(a, b, rtol=rtol, atol=atol):
                return False
        else:
            if not torch.equal(a, b):
                return False
    return True
print("State dicts close:", state_dicts_close(sd4, sd5))
print("State dicts close:", state_dicts_close(sd2, sd3))

