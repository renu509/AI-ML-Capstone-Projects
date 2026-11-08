
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Dataset location
DATASET_DIR = Path(
    r"C:\Users\Hp\Downloads\archive\chest_xray"
)

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

# Training settings
IMAGE_SIZE = 160
BATCH_SIZE = 4
EPOCHS = 1
LEARNING_RATE = 0.0001
MAX_TRAIN_IMAGES = 800

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


def create_model(num_classes):
    weights = models.DenseNet121_Weights.DEFAULT
    model = models.densenet121(weights=weights)

    # Freeze the pretrained feature extractor
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace the final classifier
    model.classifier = nn.Linear(
        model.classifier.in_features,
        num_classes
    )

    return model


def main():

    # Image transformations for training
    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # Image transformations for validation and testing
    test_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    # Load training images
    train_aug = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_transform
    )

    # Same images without random augmentation
    train_eval = datasets.ImageFolder(
        TRAIN_DIR,
        transform=test_transform
    )

    # Select a smaller, stratified sample
    labels = train_aug.targets
    indices = list(range(len(labels)))

    sample_indices, _ = train_test_split(
        indices,
        train_size=MAX_TRAIN_IMAGES,
        random_state=42,
        stratify=labels
    )

    # Split sampled images into training and validation
    sample_labels = [
        labels[i] for i in sample_indices
    ]

    train_indices, val_indices = train_test_split(
        sample_indices,
        test_size=0.15,
        random_state=42,
        stratify=sample_labels
    )

    train_data = Subset(
        train_aug,
        train_indices
    )

    val_data = Subset(
        train_eval,
        val_indices
    )

    # Load untouched test images
    test_data = datasets.ImageFolder(
        TEST_DIR,
        transform=test_transform
    )

    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    classes = train_aug.classes

    print("Classes:", classes)
    print("Training images:", len(train_data))
    print("Validation images:", len(val_data))
    print("Test images:", len(test_data))
    print("Device:", DEVICE)

    # Create model
    model = create_model(len(classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.classifier.parameters(),
        lr=LEARNING_RATE
    )

    # Training loop
    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs = model(images)
                predictions = outputs.argmax(dim=1)

                correct += (
                    predictions == labels
                ).sum().item()

                total += labels.size(0)

        validation_accuracy = (
            100 * correct / total
        )

        average_loss = (
            total_loss / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Loss: {average_loss:.4f} | "
            f"Validation Accuracy: "
            f"{validation_accuracy:.2f}%"
        )

    # Evaluate on the untouched test set
    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:

            images = images.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.tolist()
            )

    print("\nTEST SET CLASSIFICATION REPORT")

    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=classes,
            zero_division=0
        )
    )

    print("Confusion Matrix:")

    print(
        confusion_matrix(
            all_labels,
            all_predictions
        )
    )

    # Save model
    model_path = Path(
        "models/densenet121_xray.pth"
    )

    torch.save(
        {
            "model_state_dict": model.cpu().state_dict(),
            "classes": classes
        },
        model_path
    )

    print("\nModel saved:", model_path)


if __name__ == "__main__":
    main()
