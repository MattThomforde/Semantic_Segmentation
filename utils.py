import torch
from torch.utils.data import DataLoader
from dataset import SegmentationDataset, StatisticDataset
import config
import matplotlib.pyplot as plt

# get_loaders used from example found here:
# https://www.youtube.com/watch?v=IHq1t7NxS8k

def get_loaders(
				train_img_dir, 
				train_mask_dir, 
				val_img_dir, 
				val_mask_dir,
				batch_size,
				train_transform,
				val_transform,
				pin_memory=True
				):

	train_dataset = SegmentationDataset(
										image_dir=train_img_dir,
										mask_dir=train_mask_dir,
										transform=train_transform
										)
	
	train_loader = DataLoader(
							train_dataset,
							batch_size=batch_size,
							pin_memory=pin_memory,
							shuffle=True
							)
	
	validation_dataset = SegmentationDataset(
											image_dir=val_img_dir,
											mask_dir=val_mask_dir,
											transform=val_transform
											)
	
	validation_loader = DataLoader(
									validation_dataset,
									batch_size=batch_size,
									pin_memory=pin_memory,
									shuffle=False
									)

	return train_loader, validation_loader

def save_checkpoint(model, filename=config.MODEL_PATH):
	print('--> Saving checkpoint..')
	torch.save(model, filename)

def load_checkpoint(model, optimizer, filepath=config.MODEL_PATH):
	print('--> Loading checkpoint..')
	checkpoint = torch.load(filepath)
	model.load_state_dict(checkpoint['state_dicdt'])
	optimizer.load_state_dict(checkpoint['optimizer'])

	return model, optimizer, checkpoint['epoch']

def img_statistics_loader(train_img_dir, batch_size, statistic_transforms=None, pin_memory=True):
	img_dataset = StatisticDataset(image_dir=train_img_dir, transform=statistic_transforms)
	
	statistics_loader = DataLoader(
							img_dataset,
							batch_size=batch_size,
							pin_memory=pin_memory,
							)

	return statistics_loader

def get_batch_statistics(loader):
    # function borrowed from: https://www.binarystudy.com/2022/04/how-to-normalize-image-dataset-inpytorch.html
	cnt = 0
	fst_moment = torch.empty(3)
	snd_moment = torch.empty(3)
	
	for batch_idx, images in enumerate(loader):
		b, c, h, w = images.shape
		nb_pixels = b * h * w
		sum_ = torch.sum(images, dim=[0, 2, 3])
		sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
		fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
		snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
		cnt += nb_pixels
	
	mean, std = fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

	return mean,std

def plot_heatmap(image, mask):

	# Getting the min and max values from the grid
	x_min, x_max = 0, 255

	# Passing the grid into the imshow function
	plt.imshow(image)
	plt.imshow(mask, alpha=0.3, cmap='jet', interpolation='nearest')
	plt.tick_params(left = False, right = False , labelleft = False ,
                labelbottom = False, bottom = False)
	cbar = plt.colorbar(alpha=1)
	cbar.solids.set(alpha=1)

	# plt.colorbar(img)
	plt.title("U-Net Probability that a Pixel 'Hand' Class")
	plt.savefig('heatmap.png')

	plt.show()
