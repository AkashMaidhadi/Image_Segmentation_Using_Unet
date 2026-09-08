# engine.py

import torch
from tqdm import tqdm

def train_fn(data_loader, model, optimizer, device):

  model.train()
  total_loss = 0.0

  for images, masks in tqdm(data_loader): # tqdm -> to track the no.of batches
    images = images.to(device)
    masks = masks.to(device)

    optimizer.zero_grad()
    logits, loss = model(images, masks)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
  return total_loss / len(data_loader)

def valid_fn(data_loader, model, device):

  model.eval()
  total_loss = 0.0

  with torch.no_grad():
    for images, masks in tqdm(data_loader): # tqdm -> to track the no.of batches
      images = images.to(device)
      masks = masks.to(device)

      logits, loss = model(images, masks)

      total_loss += loss.item()
  return total_loss / len(data_loader)