# train.py
import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import tifffile
from tqdm import tqdm

# Import the U-Net model from the other file
from unet_model import UNet

# ---- 1. DEFINE THE DATASET CLASS ----
class NeuronDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        # Get sorted lists of image and mask file paths
        self.image_paths = sorted(glob.glob(os.path.join(image_dir, '*.tif')))
        self.mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.tif')))
        
    def __len__(self):
        # The number of samples is the number of images
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load the image and mask
        image = tifffile.imread(self.image_paths[idx])
        mask = tifffile.imread(self.mask_paths[idx])
        
        # --- Pre-processing ---
        # Normalize image to [0, 1]
        image = image.astype(np.float32) / np.max(image)
        
        # Add a channel dimension (H, W) -> (C, H, W)
        image = np.expand_dims(image, axis=0)
        
        # Convert to PyTorch tensors
        image_tensor = torch.from_numpy(image)
        # Masks should be LongTensor for CrossEntropyLoss
        mask_tensor = torch.from_numpy(mask).long() 
        
        return image_tensor, mask_tensor

# ---- 2. SET UP TRAINING PARAMETERS ----
# Hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 4 # Adjust based on your GPU memory
NUM_EPOCHS = 25 # Start with a smaller number to test, then increase

# Data paths
IMAGE_DIR = 'data/images'
MASK_DIR = 'data/masks'

# ---- 3. INITIALIZE MODEL, DATALOADER, LOSS, AND OPTIMIZER ----
# Set device (use GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# NOTE: The number of classes should be max_neuron_id + 1 (for background)
# Let's assume you won't have more than 50 neurons in a single frame.
# If you do, increase this number.
NUM_CLASSES = 51 

# Initialize the U-Net model
# n_channels=1 because our images are grayscale
model = UNet(n_channels=1, n_classes=NUM_CLASSES).to(device)

# Create the dataset and dataloader
dataset = NeuronDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)
data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Loss function and optimizer
# CrossEntropyLoss is good for multi-class segmentation
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


# ---- 4. THE TRAINING LOOP ----
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train() # Set the model to training mode
    epoch_loss = 0
    
    # Use tqdm for a nice progress bar
    for images, masks in tqdm(data_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        # Move data to the selected device
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, masks)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(data_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_epoch_loss:.4f}")

# ---- 5. SAVE THE TRAINED MODEL ----
torch.save(model.state_dict(), 'neuron_model.pth')
print("Training finished. Model saved to neuron_model.pth")
