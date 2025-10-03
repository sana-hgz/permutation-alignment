import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split

# 1) Settings
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
lr         = 0.01
epochs     = 30
val_frac   = 0.1

# 2) Transforms
train_tf = transforms.Compose([
    transforms.Resize(224),                 # VGG16 expects 224×224
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(224, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010))
])
test_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914,0.4822,0.4465),
                         (0.2023,0.1994,0.2010))
])

# 3) Datasets & Loaders
full_train = datasets.CIFAR10(root="./data", train=True,  download=True, transform=train_tf)
n_val      = int(len(full_train) * val_frac)
n_train    = len(full_train) - n_val
train_ds, val_ds = random_split(full_train, [n_train, n_val],
                                generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
test_ds      = datasets.CIFAR10(root="./data", train=False, download=True, transform=test_tf)
test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=4)


# 4) Build & adapt VGG16
model = models.vgg16(weights=None)  # no pre-trained weights
# replace the final classifier to output 10 classes
model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10) # type: ignore
model = model.to(device)


# 5) Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


# 6) Training & evaluation helpers
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        preds = logits.argmax(dim=1)
        running_loss += loss.item() * X.size(0)
        running_correct += (preds == y).sum().item()
        total += X.size(0)

    return running_loss / total, running_correct / total

def eval_model(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)

            preds = logits.argmax(dim=1)
            running_loss += loss.item() * X.size(0)
            running_correct += (preds == y).sum().item()
            total += X.size(0)

    return  running_correct / total
"""

# 7) Main training loop
best_val_acc = 0.0
for epoch in range(1, epochs+1):
    tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = eval_model(model, val_loader, criterion, device)
    scheduler.step()

    print(f"Epoch {epoch:02d}/{epochs}  "
          f"Train: loss={tr_loss:.4f}, acc={tr_acc:.4f}  "
          f"Val: loss={val_loss:.4f}, acc={val_acc:.4f}")

    # save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "vgg16_cifar10_best.pth")

# 8) Load best & test
model.load_state_dict(torch.load("vgg16_cifar10_best.pth", map_location=device))
test_loss, test_acc = eval_model(model, test_loader, criterion, device)
print(f"\nBest Val Acc: {best_val_acc:.4f}")
print(f"Test: loss={test_loss:.4f}, acc={test_acc:.4f}")
"""