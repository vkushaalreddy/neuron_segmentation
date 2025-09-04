# predict.py
import torch
import numpy as np
import tifffile
import cv2
import argparse
import os
from unet_model import UNet

def predict(model, image_path, device):
    """Runs a single image through the model and returns the prediction."""
    # 1. Load and pre-process the image
    image = tifffile.imread(image_path).astype(np.float32)
    image = image / np.max(image) if np.max(image) > 0 else image
    image = np.expand_dims(image, axis=(0, 1))
    image_tensor = torch.from_numpy(image).to(device)

    # 2. Run the model
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    # 3. Post-process the output
    prediction = torch.argmax(output, dim=1).squeeze(0)
    prediction_np = prediction.cpu().numpy().astype(np.uint8)
    
    return prediction_np

def main():
    parser = argparse.ArgumentParser(description="Predict neuron masks from an image.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input .tif image.")
    parser.add_argument('--output', type=str, default='predictions', help="Folder to save the output images.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # NOTE: Ensure NUM_CLASSES matches what you used in train.py
    NUM_CLASSES = 51 
    model = UNet(n_channels=1, n_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    
    print(f"Model loaded. Predicting on {args.input}...")
    
    predicted_mask = predict(model, args.input, device)

    # --- VISUALIZATION STEP: FROM MASK TO CIRCLES ---
    original_image_8bit = tifffile.imread(args.input)
    original_image_8bit = (original_image_8bit / np.max(original_image_8bit) * 255).astype(np.uint8)
    output_visual = cv2.cvtColor(original_image_8bit, cv2.COLOR_GRAY2BGR)

    contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(output_visual, (cX, cY), 5, (0, 255, 0), 1) 

    output_filename = os.path.join(args.output, os.path.basename(args.input).replace('.tif', '_prediction_circles.png'))
    cv2.imwrite(output_filename, output_visual)

    print(f"Prediction complete. Visualization with circles saved to {output_filename}")


# --- THIS IS THE CRUCIAL ENTRY POINT BLOCK ---
# It must be at the end of the file and not indented.
if __name__ == '__main__':
    main()
