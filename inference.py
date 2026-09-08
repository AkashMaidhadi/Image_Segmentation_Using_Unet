# inference.py

import cv2
import torch
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import sys

# Add the current directory to sys.path to find local modules
sys.path.append('.') # Or the appropriate path to your project root if running from a subdirectory

from config import CSV_FILE, DEVICE, IMG_SIZE
from augmentations import get_valid_aug
from dataset import SegementationDataset
from model import SegmentationModel

def run_inference(model_path='best_model.pth', idx=20):
  # Load data for validation set
  df = pd.read_csv(CSV_FILE)
  _, valid_df = train_test_split(df, test_size=0.2, random_state=42)
  validset = SegementationDataset(valid_df, get_valid_aug())

  # Initialize model and load weights
  model = SegmentationModel()
  # Ensure model is on the correct device for loading (usually CPU first then move)
  model.to('cpu') # Load to CPU first
  model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
  model.to(DEVICE) # Then move to the desired device for inference
  model.eval() # Set model to evaluation mode

  # Get an image and mask from the validset
  image, mask = validset[idx]

  # Move image to device and add a batch dimension
  image_for_inference = image.to(DEVICE).unsqueeze(0)

  # Perform inference
  with torch.no_grad():
    logits_mask = model(image_for_inference)
    pred_mask = torch.sigmoid(logits_mask)
    pred_mask = (pred_mask > 0.5).float() # Threshold to get binary mask

  # Visualize the results
  plt.figure(figsize=(15, 5))

  plt.subplot(1, 3, 1)
  plt.imshow(image.squeeze(0).permute(1, 2, 0).cpu().numpy())
  plt.title("Original Image")
  plt.axis('off')

  plt.subplot(1, 3, 2)
  plt.imshow(mask.squeeze(0).cpu().numpy(), cmap='gray')
  plt.title("Original Mask")
  plt.axis('off')

  plt.subplot(1, 3, 3)
  plt.imshow(pred_mask.squeeze(0).cpu().numpy(), cmap='gray')
  plt.title("Predicted Mask")
  plt.axis('off')

  plt.show()

if __name__ == '__main__':
  run_inference(idx=20) # Example usage