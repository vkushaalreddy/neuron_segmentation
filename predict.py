# predict.py
import torch
import numpy as np
import tifffile
import argparse
from unet_model import UNet

def predict(model_path, input_path, output_path, device):
    print("Loading model...")
    # NOTE: Make sure n_classes matches what you used in train.py
    n_classes = 51 
    model = UNet(n_channels=1, n_classes=n_classes)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # Set the model to evaluation mode

    print(f"Loading input image: {input_path}")
    # Load the single input image
    image = tifffile.imread(input_path)
    
    # Pre-process the image (same as in the Dataset class)
    image_for_model = image.astype(np.float32) / (np.max(image) + 1e-8)
    image_for_model = np.expand_dims(image_for_model, axis=0) # Add channel dimension
    image_for_model = np.expand_dims(image_for_model, axis=0) # Add batch dimension
    
    image_tensor = torch.from_numpy(image_for_model).to(device).float()

    print("Running prediction...")
    with torch.no_grad(): # Disable gradient calculation for inference
        output = model(image_tensor)
    
    # Post-process the output
    # The output is (B, C, H, W). We take the argmax along the class dimension C.
    predicted_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    
    print(f"Saving predicted mask to: {output_path}")
    tifffile.imwrite(output_path, predicted_mask.astype(np.uint16))
    print("Prediction finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict neuron masks using a trained U-Net model.')
    parser.add_argument('--model', type=str, required=True, help='Path to the trained model (.pth file)')
    parser.add_argument('--input', type=str, required=True, help='Path to the input image (.tif file)')
    parser.add_argument('--output', type=str, default='prediction.tif', help='Path to save the output mask')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    predict(args.model, args.input, args.output, device)