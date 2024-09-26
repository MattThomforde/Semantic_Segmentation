import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import config

class convolution_block(nn.Module):

    """
    The contracting path (left side) and expansion path (right side) of the UNET each consists 
    of the repeated application of blocks of the 3x3 convolutions, each followed by a ReLU.
    """

    def __init__(self, in_channels, out_channels, device='cpu'):

        """
        :param in_channels: the number of input channel.
        :param out_channels: the number of channels output by the convolution block.
        """
        super().__init__()
        
        self.device = device

        # Use padding=1 for same convolution to avoid need to interpolate final output
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1).to(self.device)
        self.norm = nn.BatchNorm2d(out_channels).to(self.device)
        self.relu = nn.ReLU().to(self.device)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1).to(self.device)

    def forward(self, inputs):
        """
        This function computes the forward pass for the convolution block.

        :param input: a PyTorch tensor of input feature map.

        :returns: the convolution block output feature map.
        """
        # First convolution/relu
        out = self.conv1(inputs)
        out = self.norm(out)
        out = self.relu(out)

        # Second convolution/relu
        out = self.conv2(out)
        out = self.norm(out)
        out = self.relu(out)

        return out

class contraction(nn.Module):

    """
    The contracting path (left side) of UNET includes downsampling consisting of a 2x2 Max Pooling
    operation with a stride of 2, doubling the number of features.
    """

    def __init__(self, in_channels, out_channels, device='cpu'):
        """
        :param in_channels: the number of input channel.
        :param out_channels: the number of channels output by the downsampling block.
        """
        super().__init__()

        self.device = device

        self.convolution = convolution_block(in_channels, out_channels).to(self.device)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2).to(self.device)

    def forward(self, inputs):
        """
        This function computes the forward pass for the downsampling block.

        :param input: a PyTorch tensor of input feature map.

        :returns: 
        """
        skip = self.convolution(inputs)
        pooled = self.max_pool(skip)

        return skip, pooled


class expansion(nn.Module):
    """
    The expansive path (right side) of UNET includes upsampling followed by 2x2 up-convolutions
    operation with a stride of 2, doubling the number of features.
    """

    def __init__(self, in_channels, out_channels, device='cpu'):
        """
        :param in_channels: the number of input channel.
        :param out_channels: the number of channels output by the upsampling block.
        """
        super().__init__()

        self.device = device

        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2).to(self.device)
        self.convolution_block = convolution_block((out_channels*2), out_channels).to(self.device)

    def forward(self, inputs, skip_connection):
        """
        This function computes the forward pass for the upsampling block.

        :param input: a PyTorch tensor of input feature map.

        :returns: the upsampling block output feature map.
        """
        # Up-Convolution
        out = self.up_conv(inputs)

        # Concatenate w/ Skip Connection
        cropped = self.crop(out, skip_connection)
        # print(input.shape)
        # print(out.shape)
        # print(skip_connection.shape)
        # print(cropped.shape)
        out = torch.cat([out, cropped], axis=1)

        # Convolution Block
        out = self.convolution_block(out)

        return out


    def crop(self, input_img, skip_img):
        """
        This function crops the skip connection to match the size
        of the upconvolution output from previous layer to 
        allow concatenation.

        :param input_image: the input from the previous up-convolution
        :param skip_image: the skip connection from the contracting side

        :returns: the upsampling block output feature map.
        """
        # Not necessary with padding=1 on contracting side, but is if padding=0
        _, _, h, w = input_img.shape
        cropped = torchvision.transforms.CenterCrop([h, w])(skip_img)
        # print(cropped.size)

        return cropped


class UNET(nn.Module):

    def __init__(self, device='cpu', retain_dim=True, out_size=(config.IMAGE_HEIGHT,config.IMAGE_WIDTH)):
        super(UNET, self).__init__()
        self.device = device
        self.out_size = out_size
        self.retain_dim = retain_dim

        # Contracting Path (Left Side)
        self.contraction_1 = contraction(3, 64).to(self.device)
        self.contraction_2 = contraction(64, 128).to(self.device)
        self.contraction_3 = contraction(128, 256).to(self.device)
        self.contraction_4 = contraction(256, 512).to(self.device)

        # Bottleneck
        self.bottleneck = convolution_block(512, 1024).to(self.device)

        # Expansive Path (Right Side)
        self.expansion_1 = expansion(1024, 512).to(self.device)
        self.expansion_2 = expansion(512, 256).to(self.device)
        self.expansion_3 = expansion(256, 128).to(self.device)
        self.expansion_4 = expansion(128, 64).to(self.device)

        # Binary Classifier
        self.classifier = nn.Conv2d(64, 1, kernel_size=1, padding=0).to(self.device)

    def forward(self, inputs):

        # Contraction Path (Left Side)
        skip_1, pooled_1 = self.contraction_1(inputs)
        skip_2, pooled_2 = self.contraction_2(pooled_1)
        skip_3, pooled_3 = self.contraction_3(pooled_2)
        skip_4, pooled_4 = self.contraction_4(pooled_3)

        # Bottleneck
        bottleneck = self.bottleneck(pooled_4)

        # Expansive Path (Right Side)
        exp_1 = self.expansion_1(bottleneck, skip_4)
        exp_2 = self.expansion_2(exp_1, skip_3)
        exp_3 = self.expansion_3(exp_2, skip_2)
        exp_4 = self.expansion_4(exp_3, skip_1)

        # Mask classification
        label = self.classifier(exp_4)

        # Unneccessary with padding changed to padding=1
        # With padding=0 (original paper) would ensure output matches input size
        # if self.retain_dim:
        #     label = F.interpolate(label, self.out_size)

        return label


class UNETGradCAM(nn.Module):

    def __init__(self, device='cpu', retain_dim=True, out_size=(config.IMAGE_HEIGHT,config.IMAGE_WIDTH)):
        super(UNETGradCAM, self).__init__()
        self.device = device
        self.out_size = out_size
        self.retain_dim = retain_dim
        self.gradients = None

        # Contracting Path (Left Side)
        self.contraction_1 = contraction(3, 64).to(self.device)
        self.contraction_2 = contraction(64, 128).to(self.device)
        self.contraction_3 = contraction(128, 256).to(self.device)
        self.contraction_4 = contraction(256, 512).to(self.device)

        # Bottleneck
        self.bottleneck = convolution_block(512, 1024).to(self.device)

        # Expansive Path (Right Side)
        self.expansion_1 = expansion(1024, 512).to(self.device)
        self.expansion_2 = expansion(512, 256).to(self.device)
        self.expansion_3 = expansion(256, 128).to(self.device)
        self.expansion_4 = expansion(128, 64).to(self.device)

        # Binary Classifier
        self.classifier = nn.Conv2d(64, 1, kernel_size=1, padding=0).to(self.device)

    def forward(self, inputs):
        # Contraction Path (Left Side)
        skip_1, pooled_1 = self.contraction_1(inputs)
        skip_2, pooled_2 = self.contraction_2(pooled_1)
        skip_3, pooled_3 = self.contraction_3(pooled_2)
        skip_4, pooled_4 = self.contraction_4(pooled_3)

        # Bottleneck
        bottleneck = self.bottleneck(pooled_4)

        # Expansive Path (Right Side)
        exp_1 = self.expansion_1(bottleneck, skip_4)
        exp_2 = self.expansion_2(exp_1, skip_3)
        exp_3 = self.expansion_3(exp_2, skip_2)
        exp_4 = self.expansion_4(exp_3, skip_1)
        
        # Mask classification
        label = self.classifier(exp_4)
        # label.requires_grad_()
        h = label.register_hook(self.activations_hook)

        return label

    def activations_hook(self, grad):
        self.gradients = grad

    # method for the gradient extraction
    def get_activations_gradient(self):
        return self.gradients
    
    # method for the activation exctraction
    def get_activations(self, X):
        # Contraction Path (Left Side)
        skip_1, pooled_1 = self.contraction_1(X)
        skip_2, pooled_2 = self.contraction_2(pooled_1)
        skip_3, pooled_3 = self.contraction_3(pooled_2)
        skip_4, pooled_4 = self.contraction_4(pooled_3)

        # Bottleneck
        bottleneck = self.bottleneck(pooled_4)

        # Expansive Path (Right Side)
        exp_1 = self.expansion_1(bottleneck, skip_4)
        exp_2 = self.expansion_2(exp_1, skip_3)
        exp_3 = self.expansion_3(exp_2, skip_2)
        exp_4 = self.expansion_4(exp_3, skip_1)
        
        # Mask classification
        label = self.classifier(exp_4)
        
        return label