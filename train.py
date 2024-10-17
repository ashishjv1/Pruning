import torch
import torch.nn as nn
import torch.optim as optim

# Function to ensure pruned weights remain zero during fine-tuning
def fine_tune_with_pruned_weights(model, mask_dict):
    # Apply masks to the pruned layers
    for name, param in model.named_parameters():
        if name in mask_dict:
            param.data[~mask_dict[name]] = 0  # Re-apply the mask to zero out pruned weights
    return model

def train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device, mask_dict=None):
    model.train()
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)
        
        if mask_dict:
            # Apply the mask after every step to prevent pruned weights from being updated
            model = fine_tune_with_pruned_weights(model, mask_dict)
        
        if scheduler:
            scheduler.step()
        
        # Compute average loss and accuracy
        avg_loss = running_loss / len(train_loader)
        accuracy = (correct_predictions / total_samples) * 100
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def train_and_save_model(model, train_loader, num_epochs=10, optimizer_name='Adam', scheduler_name='StepLR', lr=0.001, step_size=5, gamma=0.1, model_path='model.pth', device="cuda", mask_dict=None):
    criterion = nn.CrossEntropyLoss()
    
    if optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    if scheduler_name == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_name == 'ReduceLROnPlateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=gamma, patience=5)
    else:
        scheduler = None

    # Train the model with optional mask for pruned weights
    train_model(model, train_loader, criterion, optimizer, scheduler, num_epochs, device, mask_dict)
    save_model(model, model_path)
    
    return model_path
