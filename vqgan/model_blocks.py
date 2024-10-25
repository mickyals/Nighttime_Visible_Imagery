#import libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    """
    Swish activation function.

    References:
        https://arxiv.org/abs/1710.05941
    """
    def __init__(self):
        """
        Initialize the Swish activation function.

        The Swish activation function is defined as:

        f(x) = x * g(x)

        where g(x) is the sigmoid function.
        """
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Swish activation function.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor. The output tensor is the result of multiplying the input tensor by the sigmoid of the input tensor.
        """
        # Compute the sigmoid of the input tensor
        sigmoid_x = torch.sigmoid(x)

        # Multiply the input tensor by the sigmoid of the input tensor
        output = x * sigmoid_x

        # Return the output tensor
        return output
    
class GroupNorm(nn.Module):
    """
    Group normalization layer.

    References:
        https://arxiv.org/abs/1803.08494

    Parameters
    ----------
    in_channels : int
        Number of input channels.

    Attributes
    ----------
    groupnorm : nn.GroupNorm
        Group normalization module.
    """

    def __init__(self, in_channels: int) -> None:
        """
        Initialize the group normalization layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        """
        super().__init__()

        # Initialize the group normalization module
        self.groupnorm = nn.GroupNorm(
            num_groups=32,  # Number of groups, 32 is the default value
            num_channels=in_channels,  # Number of input channels
            eps=1e-6,  # Small value added to variance for numerical stability
            affine=True  # Whether to learn the affine parameters
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of group normalization layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """
        return self.groupnorm(x)


class ResBlock(nn.Module):
    """
    Residual block.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout : float, optional
        Dropout rate. Defaults to 0.0.
    """

    def __init__(self, in_channels:int, out_channels:int, dropout:float=0.0):
        """
        Initialize the residual block.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        dropout : float, optional
            Dropout rate. Defaults to 0.0.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.block = nn.Sequential(
            # Group normalization
            GroupNorm(in_channels),
            # Swish activation
            Swish(),
            # 3x3 convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # Group normalization
            GroupNorm(out_channels),
            # Swish activation
            Swish(),
            # 3x3 convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # Dropout
            nn.Dropout(dropout),
            # 1x1 convolution
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=1),
        )

        if in_channels != out_channels:
            # Convolution for shortcut path
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the residual block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor.
        """
        if self.in_channels != self.out_channels:
            # Shortcut path
            return self.block(x) + self.conv_shortcut(x)
        else:
            # Residual path
            return self.block(x) + x        
    
    
class Downsample(nn.Module):
    """
    Downsample the input image by a factor of 2.

    This module first pads the input with zeros on the right and bottom sides by one pixel, and then applies a 3x3 convolutional layer with stride 2.
    """

    def __init__(self, in_channels: int) -> None:
        """
        Initialize the downsample module.

        This module first pads the input with zeros on the right and bottom sides by one pixel,
        and then applies a 3x3 convolutional layer with stride 2.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        """
        super().__init__()

        # Pad the input with zeros on the right and bottom sides by one pixel
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), value=0)

        # 3x3 convolutional layer with stride 2
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the downsample module.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Downsampled tensor.
        """
        x = self.pad(x)
        return self.conv(x)
    


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        """
        Initialize the upsample module.

        This module performs nearest neighbor upsampling with a scale factor of 2.
        The upsampling is done by padding the input with zeros, and then applying a 3x3 convolutional layer.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        """
        super().__init__()

        # Pad the input with zeros
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), value=0)

        # 3x3 convolutional layer with stride 1
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the upsample module.

        This module performs nearest neighbor upsampling with a scale factor of 2.
        The upsampling is done by padding the input with zeros, and then applying a 3x3 convolutional layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Upsampled tensor.
        """
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        # Pad the upsampled tensor with zeros
        x = self.pad(x)
        # Apply a 3x3 convolutional layer
        return self.conv(x)


class NonLocalAttention(nn.Module):
    """
    Non-local attention module.

    This module applies non-local attention as described in the paper "Non-local Neural Networks" by Wang et al.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    """
    def __init__(self, in_channels:int):
        """
        Initialize the non-local attention module.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        """
        super().__init__()

        self.in_channels = in_channels

        # Group normalization layer
        self.groupnorm = GroupNorm(in_channels)

        # Query, key and value convolutional layers
        self.query = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # self.query: (N, C, H, W) -> (N, C, H, W)
        self.key = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # self.key: (N, C, H, W) -> (N, C, H, W)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # self.value: (N, C, H, W) -> (N, C, H, W)

        # Output projection layer
        self.project_out = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # self.project_out: (N, C, H, W) -> (N, C, H, W)

        # Softmax layer
        self.softmax = nn.Softmax(dim=2)
        # self.softmax: (N, C, H, W) -> (N, C, H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the non-local attention module.

        The non-local attention module applies non-local attention as described in the paper "Non-local Neural Networks" by Wang et al.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor with applied non-local attention.
        """
        batch, _, height, width = x.size()

        # Group normalization layer to normalize the input
        x = self.groupnorm(x)

        # Query, key and value convolutional layers to compute the attention weights
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Reshape the query, key and value to fit the matrix multiplication
        query = query.reshape(batch, self.in_channels, height * width)
        key = key.reshape(batch, self.in_channels, height * width)
        value = value.reshape(batch, self.in_channels, height * width)

        # Permute the query to fit the matrix multiplication
        query = query.permute(0, 2, 1)

        # Compute the attention weights
        score = torch.bmm(query, key) * (self.in_channels ** -0.5)
        weights = self.softmax(score)

        # Compute the output
        weights = weights.permute(0, 2, 1)
        attention = torch.bmm(value, weights)

        attention = attention.reshape(batch, self.in_channels, height, width)

        # Output projection layer to project the output back to the original shape
        out = self.project_out(attention)

        return x + out
