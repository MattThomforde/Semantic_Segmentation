import torch
import torch.nn as nn
import torch.optim as optim
import config
from utils import load_checkpoint, save_checkpoint, get_loaders
from torchvision import transforms
from tqdm import tqdm
from model import UNET
import os


def main():
    
    # Define Transformations
    train_transformations = transforms.Compose([transforms.ToPILImage(), 
                                          transforms.Resize((config.IMAGE_HEIGHT,config.IMAGE_WIDTH)),
                                          transforms.ColorJitter(brightness=.2, contrast=.2),
                                          transforms.RandomHorizontalFlip(p=0.1),
                                          transforms.RandomRotation(15),
	                                        transforms.ToTensor()])
    
    val_transformations = transforms.Compose([transforms.ToPILImage(), 
                                          transforms.Resize((config.IMAGE_HEIGHT,config.IMAGE_WIDTH)),
	                                        transforms.ToTensor()])
    print(f'Device: {config.DEVICE}')
    model = UNET(device=config.DEVICE).to(config.DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    train_loader, val_loader = get_loaders(
        config.TRAIN_IMG_DIR,
        config.TRAIN_MASK_DIR,
        config.VAL_IMG_DIR,
        config.VAL_MASK_DIR,
        config.BATCH_SIZE,
        train_transformations,
        val_transformations,
        config.PIN_MEMORY
    )

    if config.LOAD_MODEL:
        model, optimizer, start_epoch = load_checkpoint(model, optimizer)
        model.to(config.DEVICE)

    train_steps = len(os.listdir(config.TRAIN_IMG_DIR)) // config.BATCH_SIZE
    val_steps = len(os.listdir(config.VAL_IMG_DIR)) // config.BATCH_SIZE
    history = {"train_loss": [], 
                "val_loss": []}

    # Loop through epochs
    print("Training the UNET model...")

    for e in tqdm(range(config.NUM_EPOCHS)):

      # Put model into training mode
      model.train()

      # Initialize loss variables
      total_train_loss = 0
      total_val_loss = 0

      # Loop through the training set
      for (batch, (x, y)) in enumerate(train_loader):
        
        # send the input to the device
        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

        # Forward pass
        prediction = model(x)
        loss = loss_fn(prediction, y)

        # Backprop and parameter update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        total_train_loss += loss

        # Stop tracking gradients
        with torch.no_grad():

          # Put model into evaluation mode
          model.eval()

          # Loop through the validation set
          for (x, y) in val_loader:

            (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))

            # Get prediction and validation loss
            prediction = model(x)
            total_val_loss += loss_fn(prediction, y)

      # Compute average training and validation losses
      avg_train_loss = total_train_loss / train_steps
      avg_val_loss = total_val_loss / val_steps

      # Update the training history
      history["train_loss"].append(avg_train_loss.cpu().detach().numpy())
      history["val_loss"].append(avg_val_loss.cpu().detach().numpy())

      # Print information to terminal
      print("\n[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
      print("Train loss: {:.6f}, Validation loss: {:.4f}".format(avg_train_loss, avg_val_loss))

      if (e +1) % 5 == 0:
        # Save the model
        checkpoint = {
                      'epoch': e + 1,
                      'state_dict': model.state_dict(),
                      'optimizer': optimizer.state_dict()
                      }
        save_checkpoint(checkpoint)

      # scheduler.step()

    # Save the model
    checkpoint = {
                  'epoch': e + 1,
                  'state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict()
                  }
    save_checkpoint(checkpoint)


if __name__ == '__main__':
    main()
