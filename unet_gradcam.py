import torch
import torch.nn as nn
import torch.optim as optim
import config
from utils import load_checkpoint, get_loaders
from torchvision import transforms
from tqdm import tqdm
from model import UNETGradCAM
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
from predict import remove_components


def gradcam_predict(model, image, compare=False):
	"""
	This function loads an image and uses a model to make a prediction
	for a segmentation mask.

	:param model: the trained UNET model
	:param image_path: file path to the input image for which the prediction is to be made
	
	:returns: predicted: the mask predicted by the model (single channel of 0 (background) & 1 (mask))
	"""
	# Put model into evaluation mode
	model.eval()
	
	# with torch.no_grad():
	    
	# Load image
	img = cv2.imread(image_path)
	image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	# Transform image for inference
	transformations = transforms.Compose([transforms.ToPILImage(), 
										transforms.Resize((config.IMAGE_HEIGHT,config.IMAGE_WIDTH)),
										transforms.ToTensor()])	

	transformed_image = transformations(image).unsqueeze(0).to(config.DEVICE)

	# Resize the original image for comparison
	scaled_original = cv2.resize(image, (config.IMAGE_HEIGHT, config.IMAGE_HEIGHT))
	original = scaled_original.copy()

	# Get prediction
	predicted = model(transformed_image)
	predicted = torch.sigmoid(predicted)

	# predicted_mask = predicted.numpy().squeeze()
	# predicted_mask = predicted.detach().numpy().squeeze()

	# print(np.max(predicted_mask))

	# # Filter predictions so only those above threshold are kept
	# predicted = (predicted_mask > config.THRESHOLD) * 255

	# # Postprocess to keep only largest connected component
	# predicted = remove_components(predicted)

	# # Get ground truth mask to compare if doing a comparison
	# if compare == True:
		
	# 	image_file = image_path.split('/')[-1]
	# 	true_mask_path = os.path.join(config.VAL_MASK_DIR, image_file)
	# 	# print(true_mask_path)

	# 	# Load and scale ground-truth mask
	# 	true_mask = cv2.imread(true_mask_path, 0)
	# 	true_mask = cv2.resize(true_mask, (config.IMAGE_HEIGHT, config.IMAGE_HEIGHT))
	# 	true_mask = true_mask
		
	# 	# Make comparison plot
	# 	plot_comparison(original, true_mask, predicted)
	return img, predicted



if __name__ == '__main__':

    # Load image
    img_filename = "1500.png"
    # images = os.listdir(config.VAL_IMG_DIR)
    image_path = os.path.join(config.VAL_IMG_DIR, img_filename)

    # Get model 
    model = UNETGradCAM(device=config.DEVICE).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # Load Image
    img = cv2.imread(image_path)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	
	# Transform image for inference
    transformations = transforms.Compose([transforms.ToPILImage(), 
										transforms.Resize((config.IMAGE_HEIGHT,config.IMAGE_WIDTH)),
										transforms.ToTensor()])	

    transformed_image = transformations(image).unsqueeze(0).to(config.DEVICE)

	# Resize the original image for comparison
    scaled_original = cv2.resize(image, (config.IMAGE_HEIGHT, config.IMAGE_HEIGHT))

    model.eval()
    # model

    # Make prediction
    transformed_image.requires_grad_()
    print(transformed_image.requires_grad)
    predicted = model(transformed_image)
    predicted = torch.sigmoid(predicted)

    i = np.argmax(predicted.detach().numpy()[0])
    ind = np.unravel_index(np.argmax(predicted.detach().numpy()[0], axis=None), predicted.shape) 
    print(ind)   
    print(f'i: {i}')
    print(predicted[ind])
    print(predicted.shape)
    # predicted[:].backward()

    # print(predicted.shape)
    # print(type(predicted))

    # # predicted_mask.toTensor().backward()

    # # pull the gradients out of the model
    # gradients = model.get_activations_gradient()
    # print(gradients.shape)
    # # print(gradients.shape)
    # # # pool the gradients across the channels
    # # pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    # activations = model.get_activations(transformed_image).detach()
    # print(activations.shape)