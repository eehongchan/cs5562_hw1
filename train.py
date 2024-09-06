import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from datetime import datetime
import os

ADV_BATCH_SIZE = 5
ADV_BATCH_NUM = 5
ADV_IMG_FILE = './results/adv_images'

BATCH_SIZE = 10
BATCH_NUM = (ADV_BATCH_NUM * ADV_BATCH_SIZE) / BATCH_SIZE
LR = 0.0001
EPOCHS = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
save_directory = "./models/"

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.labels[index]

# Load adv images
loaded_dataset = torch.load(ADV_IMG_FILE)
data = loaded_dataset['adv_images']
labels = loaded_dataset['labels']

# Split into training & testing datasets
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2)
train_dataset = CustomDataset(train_data, train_labels)
test_dataset = CustomDataset(test_data, test_labels)

# Create dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialize model
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)
model = model.to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Train and val function
def train_model(model, loss_fn, optimizer, epochs):
    best_acc = 0.0
    best_loss = float('inf')
    last_saved_model = None
    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-------------------------------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = test_loader
            
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0

            # Iterate over data
            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_fn(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # Calculate statistics
                total_loss += loss.item()
                correct_predictions += torch.sum(preds == labels).item()
                total_samples += labels.size(0)


            epoch_loss = total_loss / len(dataloader.dataset)
            epoch_acc = correct_predictions / total_samples

            print(f'{phase} Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

            if epoch_acc > best_acc or (epoch_acc == best_acc and epoch_loss < best_loss):
                # Delete old best model
                if last_saved_model is not None and os.path.exists(last_saved_model):
                    print("Deleting old model")
                    os.remove(last_saved_model)
                    print(f"Deleted old model: {last_saved_model}")
                best_acc = epoch_acc
                best_loss = epoch_loss
                # Save the trained model
                current_time = datetime.now().strftime("%d%m%Y_%H%M%S")
                filename = os.path.join(save_directory, f"{current_time}_best_model.pth")
                last_saved_model = filename
                torch.save(model.state_dict(), filename)
                print(f"Model saved as {filename}")
        
    return model

# Train and evaluate model
model = train_model(model, loss_fn, optimizer, EPOCHS)