import torch
from torchvision import transforms

def compute_accuracy(model, data_loader, device):
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            
            _, predicted = torch.max(outputs.data, 1)  # Get index of max value
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    
    return (correct_predictions / total_predictions) * 100

data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing to [-1, 1]
])

data_transforms = transforms.Compose([
    transforms.RandomRotation(360),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalizing to [-1, 1]
])