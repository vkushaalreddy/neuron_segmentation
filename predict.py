# predict.py (Upgraded for Video Stacks)
import torch
import numpy as np
import tifffile
import cv2
import argparse
import os
from unet_model import UNet
from tqdm import tqdm

def predict(model, image_array, device):
    """Runs a single image numpy array through the model and returns the prediction."""
    # Pre-process the single frame
    image = image_array.astype(np.float32)
    max_val = np.max(image)
    if max_val > 0:
        image = image / max_val
    
    image = np.expand_dims(image, axis=(0, 1)) # (H, W) -> (1, 1, H, W)
    image_tensor = torch.from_numpy(image).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image_tensor)

    prediction = torch.argmax(output, dim=1).squeeze(0)
    prediction_np = prediction.cpu().numpy().astype(np.uint8)
    
    return prediction_np

def main():
    parser = argparse.ArgumentParser(description="Predict neuron masks from an image or video stack.")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained .pth model file.")
    parser.add_argument('--input', type=str, required=True, help="Path to the input .tif image or video stack.")
    parser.add_argument('--output', type=str, default='predictions', help="Folder to save the output images.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    NUM_CLASSES = 51 
    model = UNet(n_channels=1, n_classes=NUM_CLASSES)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    
    print(f"Model loaded. Predicting on {args.input}...")
    
    # Load the entire image/stack
    video_stack = tifffile.imread(args.input)
    
    # If it's a single 2D image, add a dimension to make it a 1-frame stack
    if video_stack.ndim == 2:
        video_stack = np.expand_dims(video_stack, axis=0)

    # Get the base name for output files
    base_name = os.path.splitext(os.path.basename(args.input))[0]

    # Loop through each frame in the stack
    for i, frame in enumerate(tqdm(video_stack, desc="Processing frames")):
        # Get the prediction for the current frame
        predicted_mask = predict(model, frame, device)

        # --- VISUALIZATION STEP ---
        max_val_orig = np.max(frame)
        if max_val_orig > 0:
            original_image_8bit = (frame / max_val_orig * 255).astype(np.uint8)
        else:
            original_image_8bit = frame.astype(np.uint8)
            
        output_visual = cv2.cvtColor(original_image_8bit, cv2.COLOR_GRAY2BGR)

        contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.circle(output_visual, (cX, cY), 5, (0, 255, 0), 1) 

        output_filename = os.path.join(args.output, f"{base_name}_prediction_frame_{i:04d}.png")
        cv2.imwrite(output_filename, output_visual)

    print(f"\nPrediction complete for all {len(video_stack)} frames. Visualizations saved to '{args.output}' folder.")


if __name__ == '__main__':
    main()
