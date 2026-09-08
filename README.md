Human Segmentation Project
This repository contains a deep learning project focused on performing human segmentation on images using a U-Net architecture. The goal is to accurately delineate human figures from the background, producing a binary mask for each input image.

Project Overview
Human segmentation is a critical task in computer vision with applications ranging from augmented reality and virtual try-ons to autonomous driving and surveillance. This project leverages the power of convolutional neural networks to achieve pixel-level classification, identifying which pixels belong to a human subject.

Features
U-Net Architecture: Implements a U-Net model, a popular and effective architecture for image segmentation tasks, capable of capturing both high-level semantic information and fine-grained spatial details.
EfficientNet Encoder: Utilizes a pre-trained timm-efficientnet-b8 as the encoder backbone, benefiting from transfer learning and achieving high performance with fewer parameters.
Albumentations for Data Augmentation: Employs albumentations for on-the-fly data augmentation, enhancing model generalization and robustness to variations in input data (e.g., resizing, horizontal/vertical flips).
Custom Dataset Handling: Includes a custom torch.utils.data.Dataset implementation (SegmentationDataset) to efficiently load and preprocess image and mask data.
PyTorch Framework: Built entirely using PyTorch, a flexible and powerful deep learning library.
Dice Loss + BCEWithLogitsLoss: Combines Dice Loss and Binary Cross-Entropy with Logits Loss for robust training, addressing common challenges in segmentation tasks like class imbalance.
Training and Inference Scripts: Provides separate modules for training the model and performing inference on new images.
Dataset
The project uses the "Human Segmentation Dataset" ([redacted link]). This dataset consists of pairs of images and their corresponding human segmentation masks.

Setup and Installation
Clone the repository:

git clone <your-repository-url>
cd <your-repository-name>
Install dependencies:

pip install torch torchvision numpy pandas matplotlib scikit-learn segmentation-models-pytorch albumentations opencv-python
(Note: Some packages like albumentations might require specific versions or a direct install from GitHub if facing issues with pip.)

Download the dataset: Ensure the Human-Segmentation-Dataset-master folder is accessible. If running in Colab, you might clone it directly:

!git clone https://github.com/parth1620/Human-Segmentation-Dataset-master.git /content/Human-Segmentation-Dataset-master
(Adjust paths in config.py as necessary).

Usage
Training
To train the model, run the train.py script:

python train.py
The trained model weights (best_model.pth) will be saved in the project root directory.

Inference
To perform inference using a trained model, run the inference.py script:

python inference.py
This script will load the best_model.pth and display segmentation predictions on a sample from the validation set.

Project Structure
. # Project Root Directory
├── config.py             # Configuration parameters (image size, batch size, epochs, etc.)
├── augmentations.py      # Data augmentation pipelines (Albumentations)
├── dataset.py            # Custom PyTorch Dataset for loading images and masks
├── model.py              # U-Net model definition using segmentation_models_pytorch
├── engine.py             # Training and validation loop functions
├── train.py              # Main script to run the training process
├── inference.py          # Script to load a trained model and perform inference
└── best_model.pth        # (Generated) Saved weights of the best performing mo
