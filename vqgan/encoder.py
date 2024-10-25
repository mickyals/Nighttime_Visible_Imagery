import torch
import torch.nn as nn
from vqgan.model_blocks import Downsample, NonLocalAttention, ResBlock, Swish, GroupNorm


class Encoder(nn.Module):
    """
    Encoder model for VQGAN.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    latent_dim : int, optional
        Dimension of the latent space. Defaults to 256.
    image_size : int, optional
        Size of the input image. Defaults to 512.
    dropout : float, optional
        Dropout rate. Defaults to 0.0.
    intermediate_channels : list, optional
        List of intermediate channels. Defaults to [128,128,256,256,512].
    m_blocks : int, optional
        Number of ResBlocks at each scale. Defaults to 2.
    attention_resolution : list, optional
        List of resolutions for non-local attention. Defaults to [32].
    """

    def __init__(self, in_channels:int,
                 latent_dim:int=256,
                 image_size:int=512,
                 dropout:float=0.0,
                 intermediate_channels:list = [128,128,256,256,512],
                 m_blocks:int = 2,
                 attention_resolution:list =[32]
    ):
        """
        Initialize the Encoder model.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        latent_dim : int, optional
            Dimension of the latent space. Defaults to 256.
        image_size : int, optional
            Size of the input image. Defaults to 512.
        dropout : float, optional
            Dropout rate. Defaults to 0.0.
        intermediate_channels : list, optional
            List of intermediate channels. Defaults to [128,128,256,256,512].
        m_blocks : int, optional
            Number of ResBlocks at each scale. Defaults to 2.
        attention_resolution : list, optional
            List of resolutions for non-local attention. Defaults to [32].
        """
        super().__init__()

        intermediate_channels.insert(0, intermediate_channels[0])

        layers = [nn.Conv2d(in_channels, intermediate_channels[0], kernel_size=3, stride=1, padding=1)]

        for n in range(1, len(intermediate_channels) - 1):
            in_channels = intermediate_channels[n]
            out_channels = intermediate_channels[n + 1]

            # Add m_blocks of ResBlocks
            for _ in range(m_blocks):
                layers.append(ResBlock(in_channels, out_channels, dropout))
                in_channels = out_channels

                # Add non-local attention at the specified resolutions
                if image_size in attention_resolution:
                    layers.append(NonLocalAttention(in_channels))

            # Downsample at each scale except the last one
            if n != len(intermediate_channels) - 2:
                layers.append(Downsample(in_channels = intermediate_channels[n+1]))
                image_size = image_size // 2

        in_channels = intermediate_channels[-1]
        layers.extend(
            [
                # Final blocks
                ResBlock(in_channels, in_channels, dropout),
                NonLocalAttention(in_channels),
                ResBlock(in_channels, in_channels, dropout),
                GroupNorm(in_channels),
                Swish(),
                nn.Conv2d(in_channels, latent_dim, kernel_size=3, stride=1, padding=1),
            ]
        )

        self.encoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Encoded tensor.
        """
        return self.encoder(x)

            


