import torch
import torch.nn as nn
from vqgan.model_blocks import NonLocalAttention, ResBlock, Swish, GroupNorm, Upsample


class Decoder(nn.Module):
    """
    Decoder model for VQGAN.

    Args:
        final_img_channels (int): Number of channels in the final output image.
        latent_dim (int, optional): Dimension of the latent space. Defaults to 256.
        latent_img_size (int, optional): Size of the latent image. Defaults to 32.
        intermediate_channels (list[int], optional): List of intermediate channels. Defaults to [128,128,256,256,512].
        m_blocks (int, optional): Number of ResBlocks in each scale. Defaults to 3.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        attention_resolution (list[int], optional): List of resolutions at which to apply attention. Defaults to [32].
    """
    def __init__(self,
                 final_img_channels:int,
                 latent_dim:int=256,
                 latent_img_size:int = 32,
                 intermediate_channels:list = [128,128,256,256,512],
                 m_blocks:int = 3,
                 dropout:float = 0.0,
                 attention_resolution:list = [32],
                 ):
        """
        Initialize the Decoder model.

        Parameters
        ----------
        final_img_channels : int
            Number of channels in the final output image.
        latent_dim : int, optional
            Dimension of the latent space. Defaults to 256.
        latent_img_size : int, optional
            Size of the latent image. Defaults to 32.
        intermediate_channels : list[int], optional
            List of intermediate channels. Defaults to [128,128,256,256,512].
        m_blocks : int, optional
            Number of ResBlocks at each scale. Defaults to 3.
        dropout : float, optional
            Dropout rate. Defaults to 0.0.
        attention_resolution : list[int], optional
            List of resolutions at which to apply attention. Defaults to [32].
        """
        super().__init__()

        # Reverse the list of intermediate channels
        intermediate_channels = intermediate_channels[::-1]

        layers = []
        in_channels = intermediate_channels[0]

        # First ResBlock
        layers.extend(
            [
                # Conv2d to convert latent space to first intermediate channel
                nn.Conv2d(latent_dim, intermediate_channels[0], kernel_size=3, stride=1, padding=1),
                # First ResBlock
                ResBlock(in_channels, in_channels, dropout),
                # First non-local attention
                NonLocalAttention(in_channels),
                # Second ResBlock
                ResBlock(in_channels, in_channels, dropout),
            ]
        )

        # Loop over the list of intermediate channels
        for n in range(len(intermediate_channels)):
            out_channels = intermediate_channels[n]

            # Loop over the number of ResBlocks at each scale
            for _ in range(m_blocks):
                layers.append(ResBlock(in_channels, out_channels, dropout))
                in_channels = out_channels

                # Apply non-local attention at the specified resolutions
                if latent_img_size in attention_resolution:
                    layers.append(NonLocalAttention(in_channels))

            # Upsample at each scale except the last one
            if n != 0:
                layers.append(Upsample(in_channels))
                latent_img_size = latent_img_size * 2

        # Last GroupNorm and Swish
        layers.extend(
            [  
                GroupNorm(in_channels),
                Swish(),
                # Conv2d to convert to final image channels
                nn.Conv2d(in_channels, final_img_channels, kernel_size=3, stride=1, padding=1),
            ]
        )

        self.decoder = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
