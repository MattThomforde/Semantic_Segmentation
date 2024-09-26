import cv2
import numpy as np

def apply_mask(img, mask):

    """
    Applies a mask to an original image.

    :param img: an image of size (H x W x C) where H == C
    :param mask: a single channel image of size (D x D) where background=0 and mask=255
    :returns: the original image with the mask applied
    """

    # Get image size for scaling mask
    H = img.shape[0]
    W = img.shape[1]

    # Scale mask up to original image size
    mask = cv2.resize(mask, (W, H))

    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Confirm image and mask are same datatype
    img = img.astype(np.uint8)
    mask = mask.astype(np.uint8)

    # Apply mask to original image
    masked_img = cv2.bitwise_and(img, img, mask=mask)
	
    return masked_img
