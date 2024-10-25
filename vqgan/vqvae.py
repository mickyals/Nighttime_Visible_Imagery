import torch
import torch.nn as nn

from decoder import Decoder
from encoder import Encoder
from codebooks import Codebook, CodebookEMA



class VQVAE(nn.Module):
    """
    VQVAE model with a encoder, decoder and a codebook.

    Parameters:
        input_image_channels (int): Number of channels in the input image.
        output_image_channels (int): Number of channels in the output image.
        img_size (int): Size of the input image.
        latent_dim (int): Dimensionality of the latent space.
        latent_img_size (int): Size of the latent image.
        intermediate_channels (list): List of intermediate channels.
        m_blocks_encode (int): Number of ResBlocks in the encoder.
        m_blocks_decode (int): Number of ResBlocks in the decoder.
        dropout (float): Dropout rate.
        attention_resolution (list): List of resolutions at which to apply attention.
        num_codebook_vectors (int): Number of vectors in the codebook.
        ema_decay (float): Decay rate for the exponential moving average.
        epsilon (float): Epsilon value for the codebook.
        beta (float): Beta value for the codebook.
    """
    def __init__(self,
                 input_image_channels:int = 3,
                 output_image_channels:int = 3,
                 img_size:int = 512,
                 latent_dim:int = 256,
                 latent_img_size:int = 32,
                 intermediate_channels:list = [128,128,256,256,512],
                 m_blocks_encode:int = 2,
                 m_blocks_decode:int = 3,
                 dropout:float = 0.0,
                 attention_resolution:list = [32],
                 num_codebook_vectors:int = 1024,
                 ema_decay:float = 0.99,
                 epsilon:float = 1e-5,
                 beta = 1.0
    ):
        """
        Initializes the VQVAE model.

        Parameters:
            input_image_channels (int): Number of channels in the input image.
            output_image_channels (int): Number of channels in the output image.
            img_size (int): Size of the input image.
            latent_dim (int): Dimensionality of the latent space.
            latent_img_size (int): Size of the latent image.
            intermediate_channels (list): List of intermediate channels.
            m_blocks_encode (int): Number of ResBlocks in the encoder.
            m_blocks_decode (int): Number of ResBlocks in the decoder.
            dropout (float): Dropout rate.
            attention_resolution (list): List of resolutions at which to apply attention.
            num_codebook_vectors (int): Number of vectors in the codebook.
            ema_decay (float): Decay rate for the exponential moving average.
            epsilon (float): Epsilon value for the codebook.
            beta (float): Beta value for the codebook.
        """
        super().__init__()

        self.input_image_channels = input_image_channels
        self.output_image_channels = output_image_channels
        self.num_codebook_vectors = num_codebook_vectors

        self.encoder = Encoder(
            in_channels= input_image_channels,
            latent_dim = latent_dim,
            image_size = img_size,
            dropout = dropout,
            intermediate_channels = intermediate_channels,
            m_blocks = m_blocks_encode,
            attention_resolution = attention_resolution
        )

        self.decoder = Decoder(
            final_img_channels = output_image_channels,
            latent_dim = latent_dim,
            latent_img_size = latent_img_size,
            intermediate_channels = intermediate_channels,
            m_blocks = m_blocks_decode,
            dropout = dropout,
            attention_resolution = attention_resolution
        )

        if ema_decay > 0.0:
            self.codebook = CodebookEMA(
                num_of_embeddings = num_codebook_vectors,
                latent_dim = latent_dim,
                beta = beta,
                ema_decay = ema_decay,
                epsilon=epsilon
            )
        else:
            self.codebook = Codebook(
                num_of_embeddings = num_codebook_vectors,
                latent_dim = latent_dim,
                beta = beta
            )

        self.quant_conv = nn.Conv2d(latent_dim, latent_dim, 1)
        self.post_quant_conv = nn.Conv2d(latent_dim, latent_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VQVAE.

        Parameters:
            x (torch.Tensor): Input image.

        Returns:
            decoded_images (torch.Tensor): Decoded image.
            encoding_indices (torch.Tensor): Indices of the codebook vectors.
            codebook_loss (torch.Tensor): Codebook loss.
            perplexity (torch.Tensor): Perplexity of the codebook vectors.
        """
        # Encode input image to latent space
        encoded_images = self.encoder(x)

        # Apply pre-quantization convolution
        quant_encoded = self.quant_conv(encoded_images)

        # Quantize encoded images using codebook
        z_quantized, codebook_loss, perplexity, encoding_indices = self.codebook(quant_encoded)

        # Apply post-quantization convolution
        post_quantized = self.post_quant_conv(z_quantized)

        # Decode quantized latent space back to image space
        decoded_images = self.decoder(post_quantized)

        return decoded_images, encoding_indices, codebook_loss, perplexity
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode the input image.

        This function takes the input image and returns the quantized latent space,
        codebook loss, perplexity of the codebook vectors, and the indices of the
        codebook vectors.

        Parameters:
            x (torch.Tensor): Input image.

        Returns:
            z_quantized (torch.Tensor): Quantized latent space.
            codebook_loss (torch.Tensor): Codebook loss.
            perplexity (torch.Tensor): Perplexity of the codebook vectors.
            encoding_indices (torch.Tensor): Indices of the codebook vectors.
        """
        # Encode the input image to the latent space
        encoded_images = self.encoder(x)

        # Apply the pre-quantization convolution
        quant_encoded = self.quant_conv(encoded_images)

        # Quantize the encoded images using the codebook
        z_quantized, codebook_loss, perplexity, encoding_indices = self.codebook(quant_encoded)

        # Return the quantized latent space, codebook loss, perplexity, and encoding indices
        return z_quantized, codebook_loss, perplexity, encoding_indices
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode the latent space to generate an image.

        Parameters:
            z (torch.Tensor): Latent space.

        Returns:
            decoded_images (torch.Tensor): Decoded image.
        """
        # Apply the post-quantization convolution to the latent space
        z_quantized = self.post_quant_conv(z)

        # Decode the quantized latent space to an image
        decoded_images = self.decoder(z_quantized)

        return decoded_images
    
    def calculate_lambda(self, perceptual_loss: torch.Tensor, gan_loss: torch.Tensor) -> float:
        """
        Calculate the lambda value for the perceptual loss.

        This function calculates the lambda value by taking the norm of the perceptual loss
        gradients and dividing it by the norm of the GAN loss gradients.

        Parameters:
            perceptual_loss (torch.Tensor): Perceptual loss.
            gan_loss (torch.Tensor): GAN loss.

        Returns:
            lambda (float): Lambda value.
        """
        # Get the weights of the last layer of the decoder
        last_layer = self.decoder[-1]
        last_layer_weight = last_layer.weight

        # Calculate the gradients of the perceptual loss with respect to the weights
        perceptual_loss_grads = torch.autograd.grad(perceptual_loss, last_layer_weight, retain_graph=True)[0]

        # Calculate the gradients of the GAN loss with respect to the weights
        gan_loss_grads = torch.autograd.grad(gan_loss, last_layer_weight, retain_graph=True)[0]

        # Calculate the lambda value
        lmda = torch.norm(perceptual_loss_grads) / (torch.norm(gan_loss_grads) + 1e-4)

        # Clamp the lambda value to ensure it is within a reasonable range
        lmda = torch.clamp(lmda, 0, 1e4).detach()

        # Return the lambda value
        return 0.8 * lmda

    
    @staticmethod
    def adopt_weight(disc_factor: float, iteration: int, threshold: int, value: float = 0.0) -> float:
        """
        Adopt the weight of the discriminator based on the current iteration.

        The weight of the discriminator is adopted based on the current iteration number.
        If the iteration number is less than the threshold, the weight is set to the value.
        Otherwise, the weight is not changed.

        Parameters:
            disc_factor (float): Weight of the discriminator.
            iteration (int): Current iteration.
            threshold (int): Threshold for the weight.
            value (float, optional): Value to return if the iteration is less than the threshold. Defaults to 0.0.

        Returns:
            adopted_weight (float): Adopted weight of the discriminator.
        """
        if iteration < threshold:
            disc_factor = value
        return disc_factor
    
    def load_checkpoint(self, path: str) -> None:
        """
        Load the checkpoint of the model.

        Parameters:
            path (str): Path to the checkpoint file.
        """
        # Load the state dict from the checkpoint file
        state_dict = torch.load(path)
        # Load the state dict into the model
        self.load_state_dict(state_dict)

    def save_checkpoint(self, path: str) -> None:
        """
        Save the checkpoint of the model.

        Parameters:
            path (str): Path to the checkpoint file.
        """
        # Save the state dict of the model to the checkpoint file
        torch.save(self.state_dict(), path)
    
                 