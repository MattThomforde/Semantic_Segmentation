import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

from predict import predict
from mask import apply_mask
from utils import load_checkpoint
import config
from model import UNET


def preprocess(img_filename, save=False, save_path="masked.png"):
    """
	This function takes an image file from the validation directory and predicts a hand mask, 
    applies the predicted mask to the original image, and then returns the masked image.

	:param img_filename: the filename (1807.png) of an image in the validation directory
	
	:returns: out_img: the original image with the predicted mask applied as a tensor
	"""

    # Load image
    # images = os.listdir(config.VAL_IMG_DIR)
    image_path = os.path.join(config.VAL_IMG_DIR, img_filename)

    # Get model 
    model = UNET(device=config.DEVICE).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    
    # Make prediction
    img, predicted_mask = predict(model, image_path, compare=save)

    # Apply predicted mask to the original image
    masked_img = apply_mask(img=img, mask=predicted_mask)

    # Save a copy
    if save:
        cv2.imwrite(save_path, masked_img)

    # Convert to tensor
    transformer = transforms.ToTensor()
    out_img = transformer(masked_img)

    return out_img 

if __name__ == '__main__':

    processed_image = preprocess("1500.png", save=True)

    print(processed_image.shape)
