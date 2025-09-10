import torch
import torchvision.transforms as standard_transforms
import numpy as np
from PIL import Image
import cv2
from models import build_model
import os
import warnings
warnings.filterwarnings('ignore')

# ========================================================================================
# --- SETTINGS ---
# This section contains all the settings. You can change them here easily.

WEIGHT_PATH = './weights/SHTechA.pth'
IMAGE_PATH = './download.jpg'
OUTPUT_DIR = './logs'
DEVICE = 'cpu'  # We force the script to use the CPU, fixing the CUDA error.

# --- END SETTINGS ---
# ========================================================================================

def main():
    # Print a starting message
    print(f"Starting P2PNet evaluation on the CPU...")
    print(f"Using model weights from: {WEIGHT_PATH}")
    print(f"Processing image: {IMAGE_PATH}")

    # --- 1. Create the Model ---
    # Create a mock 'args' object to pass to the model builder
    class MockArgs:
        backbone = 'vgg16_bn'
        row = 2
        line = 2
    args = MockArgs()

    # Build the P2PNet model
    model = build_model(args)
    # Set the device (CPU)
    device = torch.device(DEVICE)
    # Move the model to the CPU
    model.to(device)
    
    # --- 2. Load the Trained Weights ---
    # Load the saved checkpoint file
    # map_location='cpu' is important to load a GPU-trained model onto a CPU
    checkpoint = torch.load(WEIGHT_PATH, map_location='cpu')
    # Load the weights into the model
    model.load_state_dict(checkpoint['model'])
    # Set the model to evaluation mode
    model.eval()

    # --- 3. Prepare the Image ---
    # Define the image transformations
    transform = standard_transforms.Compose([
        standard_transforms.ToTensor(), 
        standard_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load the image using the path from the SETTINGS
    img_raw = Image.open(IMAGE_PATH).convert('RGB')
    
    # Resize the image to be divisible by 128, as the model requires
    width, height = img_raw.size
    new_width = width // 128 * 128
    new_height = height // 128 * 128
    img_raw = img_raw.resize((new_width, new_height), Image.ANTIALIAS)
    
    # Apply the transformations
    img = transform(img_raw)
    
    # --- 4. Run Inference ---
    # Prepare the image tensor for the model
    samples = torch.Tensor(img).unsqueeze(0)
    samples = samples.to(device)

    # Run the model. torch.no_grad() is for efficiency.
    with torch.no_grad():
        outputs = model(samples)

    # --- 5. Process the Output ---
    # Get the predicted scores and points from the model's output
    outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
    outputs_points = outputs['pred_points'][0]

    # Set a confidence threshold to filter out weak predictions
    threshold = 0.5
    # Get the points that are above the confidence threshold
    points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
    # Count the number of detected people
    predict_cnt = len(points)
    
    print(f"Crowd count: {predict_cnt}")
    
    return predict_cnt

if __name__ == '__main__':
    main()