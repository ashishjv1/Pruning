# test.py
import torch
import torch.nn as nn

def test_model(model, test_loader, criterion, device):
    model.eval()  
    correct_predictions = 0
    total_samples = 0
    running_loss = 0.0

    with torch.no_grad():  
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    
        avg_loss = running_loss / len(test_loader)
        accuracy = (correct_predictions / total_samples) * 100
    
        # Print results
        print(f'Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%')
        # return avg_loss, accuracy

def evaluate_model(model, test_loader, criterion, device='cuda'):
    # Move model to device
    model.to(device)
    
    return test_model(model, test_loader, criterion, device)
