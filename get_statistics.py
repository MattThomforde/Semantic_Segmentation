import torch
import torch.nn as nn
import torch.optim as optim
import config
from utils import img_statistics_loader, get_batch_statistics
from torchvision import transforms
from tqdm import tqdm
from model import UNET
import os


def main():
    
    print(f'Device: {config.DEVICE}')
    # Define Transformations
    stats_transformations = transforms.Compose([transforms.ToPILImage(), 
                                                transforms.Resize((config.IMAGE_HEIGHT,config.IMAGE_WIDTH)),
	                                            transforms.ToTensor()])

    stats_loader = img_statistics_loader(
        config.TRAIN_IMG_DIR,
        config.BATCH_SIZE,
        statistic_transforms=stats_transformations,
        pin_memory=config.PIN_MEMORY
    )

    print('Getting Statistics...')
    
    mean, std = get_batch_statistics(stats_loader)
    print("mean and std: \n", mean, std)


if __name__ == '__main__':
    main()
