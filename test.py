import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch import nn
from PIL import Image

# Define Dataset Class
class WoundClassificationDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Path to the dataset directory.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_names = sorted(os.listdir(root_dir))  # Sorted list of class names for consistency

        # Traverse subdirectories to gather image paths and labels
        for label, class_dir in enumerate(self.class_names):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for file_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, file_name)
                    if img_path.endswith(('.jpg', '.png', '.jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, label

# Define Training Function
def train_model():
    # Dataset paths
    dataset_dir = "D://Desktop//Wound_dataset"  # Replace with the path to your Wound_dataset directory
    batch_size = 8
    num_epochs = 10
    learning_rate = 0.001

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset and DataLoader
    dataset = WoundClassificationDataset(root_dir=dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.class_names))  # Adjust the output layer for the number of classes
    model = model.to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0

        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate metrics
            epoch_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()

        # Calculate accuracy
        accuracy = correct / len(dataset)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "wound_classification_model.pth")
    print("Model saved!")

# Define Evaluation Function
def evaluate_model():
    dataset_dir = "D://Desktop//Wound_dataset"  # Replace with the path to your Wound_dataset directory
    batch_size = 8

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Dataset and DataLoader
    dataset = WoundClassificationDataset(root_dir=dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.class_names))  # Adjust the output layer
    model.load_state_dict(torch.load("wound_classification_model.pth"))
    model = model.to(device)
    model.eval()

    # Evaluation Loop
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()

    accuracy = correct / len(dataset)
    print(f"Evaluation Accuracy: {accuracy:.4f}")

# Main Function
if __name__ == "__main__":
    # Train the model
    train_model()

    # Evaluate the model
    evaluate_model()