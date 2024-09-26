import torch
import config
from utils import load_checkpoint, plot_heatmap
from torchvision import transforms
from tqdm import tqdm
from model import UNET
from mask import apply_mask
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def plot_comparison(original_image, original_mask, predicted_mask):
	"""
	This function plots a comparison between the original image,
	the ground-truth segmentation, and the segmentation predicted
	by the model.

	:param original_image: the original image
	:param original_mask: the ground-truth image mask (single channel of 0 (background) & 255 (mask))
	:param predicted_mask: the mask predicted by the model (single channel of 0 (background) & 255 (mask))
	"""

	# Initialize the figure
	fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))

	# Plot the images
	ax[0].imshow(original_image)
	ax[1].imshow(original_mask)
	ax[2].imshow(predicted_mask)

	# Add titles and set layout
	ax[0].set_title("Image")
	ax[1].set_title("True Mask")
	ax[2].set_title("Predicted Mask")
	fig.tight_layout()
	fig.savefig('comparison.png')

def remove_components(image):
	"""
	This function postprocesses an image to keep only a single (largest)
	connected component

	:param image: image of mask prediction which may or may not include undesirable extra masked areas
	
	:returns: out_img: image [0,255] of image mask with only a single connected component remaining
	"""
	# Ensure image is of uint8 type
	image = image.astype(np.uint8)

	# Get connected component states
	nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=4)

	# Get sizes of connected components (last column of 'stats' is CC_STAT_AREA)
	sizes = stats[:, -1]

	# Get label and size of largest connected component (0 is background)
	desired_label = 1
	component_size = sizes[1]

	# Loop over components
	for i in range(1, nb_components):
		if sizes[i] > component_size:
			desired_label = i
			component_size = sizes[i]
	
	# Create new image for only single connected component
	out_img = np.zeros(output.shape)

	# Set area of desired connected component to value of 255
	out_img[output == desired_label] = 255

	return out_img


def predict(model, image_path, compare=False):
	"""
	This function loads an image and uses a model to make a prediction
	for a segmentation mask.

	:param model: the trained UNET model
	:param image_path: file path to the input image for which the prediction is to be made
	
	:returns: predicted: the mask predicted by the model (single channel of 0 (background) & 1 (mask))
	"""
	# Put model into evaluation mode
	model.eval()
	
	with torch.no_grad():
	    
	    # Load image
		img = cv2.imread(image_path)
		image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    	
		# Transform image for inference
		transformations = transforms.Compose([transforms.ToPILImage(), 
                                          	transforms.Resize((config.IMAGE_HEIGHT,config.IMAGE_WIDTH)),
	                                        transforms.ToTensor()])	

		transformed_image = transformations(image).unsqueeze(0).to(config.DEVICE)
		transformed_image.requires_grad_()

		# Resize the original image for comparison
		scaled_original = cv2.resize(image, (config.IMAGE_HEIGHT, config.IMAGE_HEIGHT))
		original = scaled_original.copy()

		# Get prediction
		predicted = model(transformed_image)
		predicted = torch.sigmoid(predicted)
		predicted_mask = predicted.numpy().squeeze()
		print(np.max(predicted_mask))

		plot_heatmap(scaled_original, predicted_mask)

		# Filter predictions so only those above threshold are kept
		predicted = (predicted_mask > config.THRESHOLD) * 255

		# Postprocess to keep only largest connected component
		predicted = remove_components(predicted)

		# Get ground truth mask to compare if doing a comparison
		if compare == True:
			
			image_file = image_path.split('/')[-1]
			true_mask_path = os.path.join(config.VAL_MASK_DIR, image_file)
			# print(true_mask_path)

    		# Load and scale ground-truth mask
			true_mask = cv2.imread(true_mask_path, 0)
			true_mask = cv2.resize(true_mask, (config.IMAGE_HEIGHT, config.IMAGE_HEIGHT))
			true_mask = true_mask
    		
		    # Make comparison plot
			plot_comparison(original, true_mask, predicted)

	return img, predicted

def DICE_COE(mask1, mask2):
    intersect = np.sum(mask1*mask2)
    fsum = np.sum(mask1)
    ssum = np.sum(mask2)
    dice = (2 * intersect ) / (fsum + ssum)
    dice = np.mean(dice)
    dice = round(dice, 3) # for easy reading
    return dice  

if __name__ == '__main__':

	# Load image
	images = os.listdir(config.VAL_IMG_DIR)
	image_path = os.path.join(config.VAL_IMG_DIR, images[3])

	# Get model    
	# model = UNET(device=config.DEVICE).to(config.DEVICE)
	model = UNET(device=config.DEVICE).to(config.DEVICE)
	checkpoint = torch.load(config.MODEL_PATH, map_location=config.DEVICE)
	model.load_state_dict(checkpoint['state_dict'])

	scores = []

	for i, img in enumerate(images):
		image_path = os.path.join(config.VAL_IMG_DIR, images[i])

		if images[i] == '5533.png':
			print('Found it!')
			# Make prediction
			img, predicted_mask = predict(model, image_path, compare=True)

			print(predicted_mask.shape)


			true_mask_path = os.path.join(config.VAL_MASK_DIR, images[i])
			# print(true_mask_path)

			# Load and scale ground-truth mask
			true_mask = cv2.imread(true_mask_path, 0)
			true_mask = cv2.resize(true_mask, (config.IMAGE_HEIGHT, config.IMAGE_HEIGHT))
			# Compute Dice Score
			dice = DICE_COE(predicted_mask // 255, true_mask //255)
			# print(dice)
			scores.append(dice)

mean_dice = sum(scores) / len(scores)
print(mean_dice)