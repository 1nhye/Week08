import torch
from torch import optim
from torch.nn import CrossEntropyLoss

from i3d_model import InceptionI3d
from dataset_loader import get_dataloaders
from train_validate import train_one_epoch, validate
from config import *

def main():
    # Initialize the model
    model = InceptionI3d(num_classes=101, in_channels=16)
    model.replace_logits(num_classes=101)
    model = model.to(DEVICE)

    # Load the dataset
    train_loader, val_loader = get_dataloaders(DATA_DIR, ANNOTATION_PATH, BATCH_SIZE, NUM_WORKERS, TRAIN_SPLIT)

    # Define the optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = CrossEntropyLoss()

    # Training loop
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        # Train and validate
        train_loss, train_accuracy = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_accuracy = validate(model, val_loader, criterion, DEVICE)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

    # Save the model
    torch.save(model.state_dict(), "i3d_model.pth")
    print("Model saved as i3d_model.pth")

if __name__ == "__main__":
    main()
