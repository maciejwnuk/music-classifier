import os
import torch

from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import seaborn as sns
import matplotlib.pyplot as plt

from model import AudioCNN
from spectrograms import get_dataset_stats

from config import (
    get_device,
    IMG_SIZE, NUM_CLASSES,
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    TEST_SPLIT,
    SPECTROGRAMS_DIR, BASE_DIR,
)

# Raise macOS open file descriptors limit
import resource

_, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(
    resource.RLIMIT_NOFILE,
    (min(4096, hard), hard)
)

class DatasetWrapper(Dataset):
    def __init__(self, subset, transform = None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.subset)

def get_transform():
    mean, std = get_dataset_stats()

    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.Grayscale(1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = mean,
            std = std
        )
    ])

def get_train_transform():
    mean, std = get_dataset_stats()

    return transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.Grayscale(1),
        transforms.RandomAffine(
            degrees = 0,
            translate = (0.1, 0.05),
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = mean,
            std = std
        ),
        # Add missing patches
        transforms.RandomErasing(
            p = 0.5,
            scale = (0.02, 0.33),
            ratio = (0.3, 3.3),
            value = 0
        ),
        # Add frequency masking
        transforms.RandomErasing(
            p = 0.5,
            scale = (0.02, 0.15),
            ratio = (5.0, 10.0),
            value = 0
        ),
    ])

def save_confusion_matrix(cm, labels, path):
    plt.figure(figsize = (10, 8))

    sns.heatmap(
        cm,
        annot = True, fmt = "d", cmap = "Blues",
        xticklabels = labels, yticklabels = labels
    )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    plt.savefig(path)

    plt.close()

def main():
    device = get_device()

    print(f"Using device: {device}")
    print("Loading dataset...")

    dataset = datasets.ImageFolder(str(SPECTROGRAMS_DIR))

    songs_target = {}

    for path, target in dataset.samples:
        name = os.path.basename(path).split("_seg")[0]
        songs_target[name] = target

    songs = list(songs_target.keys())
    targets = list(songs_target.values())

    train_songs, test_songs = train_test_split(
        songs,
        test_size = TEST_SPLIT,
        stratify = targets,
        random_state = 42
    )

    # Using sets to speed up lookup (O(1) set vs O(n) list)
    train_songs_set = set(train_songs)
    test_songs_set = set(test_songs)

    train_idx = [
        i for i, (path, _) in enumerate(dataset.samples)
        if os.path.basename(path).split("_seg")[0] in train_songs_set
    ]
    test_idx = [
        i for i, (path, _) in enumerate(dataset.samples)
        if os.path.basename(path).split("_seg")[0] in test_songs_set
    ]

    print(f"Train songs: {len(train_songs)}")
    print(f"Test songs:  {len(test_songs)}")

    train_subset = torch.utils.data.Subset(dataset, train_idx)  # pyright: ignore[reportArgumentType]
    test_subset = torch.utils.data.Subset(dataset, test_idx)    # pyright: ignore[reportArgumentType]

    train_dataset = DatasetWrapper(
        train_subset,
        transform = get_train_transform()
    )
    test_dataset = DatasetWrapper(
        test_subset,
        transform = get_transform()
    )

    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True,
        num_workers = num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        num_workers = num_workers
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size:  {len(test_dataset)}")

    model = AudioCNN(NUM_CLASSES).to(device)

    train_targets = [dataset.targets[i] for i in train_idx]

    class_counts = torch.zeros(NUM_CLASSES)

    for t in train_targets:
        class_counts[t] += 1

    class_weights = class_counts.sum() / class_counts
    class_weights = class_weights.to(device)

    loss_fn = nn.CrossEntropyLoss(
        weight = class_weights,
        label_smoothing = 0.1
    )

    optimizer = optim.AdamW(
        params = filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr = LEARNING_RATE,
        weight_decay = 1e-2
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = NUM_EPOCHS,
        eta_min = 1e-6
    )

    best_acc = 0.0

    model_path = BASE_DIR / "model.pt"

    print("Starting training...")

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1:02d}/{NUM_EPOCHS}:")

        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm = 1.0
            )

            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total

        print(f"Train loss: {epoch_train_loss:.4f}, accuracy: {epoch_train_acc:.4f}")

        # Validation
        with torch.no_grad():
            model.eval()

            test_loss = 0.0
            test_correct = 0
            test_total = 0

            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)

                test_loss    += loss.item() * inputs.size(0)
                _, predicted  = torch.max(outputs, 1)
                test_total   += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        epoch_test_loss = test_loss / test_total
        epoch_test_acc = test_correct / test_total

        print(f"Test loss: {epoch_test_loss:.4f}, accuracy: {epoch_test_acc:.4f}")

        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc

            torch.save({
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "accuracy": best_acc,
                "classes": dataset.class_to_idx
            }, model_path)

            print("Saved model with better accuracy")

    print("Training completed. Evaluating model...")

    checkpoint = torch.load(model_path, weights_only = False)

    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("Per-class Accuracy:")

    cm = confusion_matrix(all_labels, all_preds)

    class_names = list(checkpoint["classes"].keys())

    for i in range(NUM_CLASSES):
        class_acc = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0

        print(f"- {class_names[i]}: {class_acc:.4f}")

    cm_path = BASE_DIR / "confusion_matrix.png"

    save_confusion_matrix(cm, class_names, cm_path)

    print(f"Saved confusion matrix plot to {cm_path}")

if __name__ == "__main__":
    main()
