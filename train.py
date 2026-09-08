# train.py

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
import numpy as np
import sys

# Add the current directory to sys.path to find local modules
sys.path.append('.') # Or the appropriate path to your project root if running from a subdirectory

from config import CSV_FILE, DEVICE, EPOCHS, LR, BATCH_SIZE
from augmentations import get_train_aug, get_valid_aug
from dataset import SegementationDataset
from model import SegmentationModel
from engine import train_fn, valid_fn

def main():
  # Load data
  df = pd.read_csv(CSV_FILE)
  train_df, valid_df = train_test_split(df, test_size=0.2, random_state=42)

  # Create datasets and dataloaders
  trainset = SegementationDataset(train_df, get_train_aug())
  validset = SegementationDataset(valid_df, get_valid_aug())

  trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
  validloader = DataLoader(validset, batch_size=BATCH_SIZE)

  # Initialize model, optimizer
  model = SegmentationModel()
  model.to(DEVICE)
  optimizer = torch.optim.Adam(model.parameters(), lr=LR)

  # Training loop
  best_valid_loss = np.inf
  for i in range(EPOCHS):
    train_loss = train_fn(trainloader, model, optimizer, DEVICE)
    valid_loss = valid_fn(validloader, model, DEVICE)

    if valid_loss < best_valid_loss:
      torch.save(model.state_dict(), 'best_model.pth')
      print('SAVED MODEL')
      best_valid_loss = valid_loss

    print(f'Epoch:{i+1} Train_loss: {train_loss:.4f}   Valid_loss: {valid_loss:.4f}')

if __name__ == '__main__':
  main()