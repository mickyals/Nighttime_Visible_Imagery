import torch
import torch.nn as nn
import torch.nn.functional as F


class Codebook(nn.Module):
    """
    Vector Quantizer.

    This module is used to quantize the output of the encoder to the nearest code
    in the latent space. The quantization is done by calculating the distances
    between the input and the codes in the latent space, and then encoding the
    input to the nearest code.

    Attributes:
        embedding_dim_D: The embedding dimensionality of the latent space.
        num_embeddings_K: The number of embeddings in the latent space.
        codebook: The embedding layer containing the latent space.
        beta: A hyperparameter controlling the commitment loss.
    """

    def __init__(self, num_of_embeddings:int = 1024,
                  latent_dim:int = 256,
                    beta:float = 0.25):
        """
        Initializes the vector quantizer.

        Args:
            num_of_embeddings: The number of codes in the latent space.
            embedding_dim: The length of each code in the latent space.
            beta: A hyperparameter controlling the commitment loss.
        """
        super().__init__()

        # The embedding dimensionality of the latent space
        self.embedding_dim_D = latent_dim

        # The number of embeddings in the latent space
        self.num_embeddings_K = num_of_embeddings
        # Initialize the latent space as an embedding layer
        self.codebook = nn.Embedding(self.num_embeddings_K, self.embedding_dim_D)

        # Initialize the latent space weights to be uniform
        self.codebook.weight.data.uniform_(-1.0 / self.num_embeddings_K, 1.0 / self.num_embeddings_K)

        # Store the hyperparameter controlling the commitment loss
        self.beta = beta

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the VectorQuantizer.

        This function takes the output of the encoder and quantizes it to the
        nearest code in the latent space. The output is the quantized version of
        the input, the loss, the perplexity of the latent space codes, and the
        encoding of the input.

        Args:
            inputs: The output of the encoder.

        Returns:
            z_quantized: The quantized version of the input.
            loss: The loss of the quantization, which is the sum of the codebook
                loss and the commitment loss.
            perplexity: The perplexity of the latent space codes.
            encodings: The encoding of the input.
        """
        # convert z_encoded from (batch, channel, height, width) to (batch, height, width, channels)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()

        input_shape = inputs.shape

        inputs_flattened = inputs.view(-1, self.embedding_dim_D)

        # Calculate distances
        distances = (torch.sum(inputs_flattened ** 2, dim=1, keepdim=True)) + \
        torch.sum(self.codebook.weight ** 2, dim=1) - \
        2 * torch.matmul(inputs_flattened, self.codebook.weight.t())

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.embedding_dim_D, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        z_quantized = torch.matmul(encodings, self.codebook.weight).view(input_shape)

        # Loss
        codebook_loss = F.mse_loss(z_quantized.detach(), inputs)
        commitment_loss = F.mse_loss(z_quantized, inputs.detach())
        loss = codebook_loss + self.beta * commitment_loss

        z_quantized = inputs + (z_quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from (batch, height, width, channel) to (batch, channel, height, width)
        z_quantized = z_quantized.permute(0, 3, 1, 2).contiguous()

        return z_quantized, loss, perplexity, encoding_indices


class CodebookEMA(nn.Module):
    """
    Vector Quantizer with Exponential Moving Average.

    This module is used to quantize the output of the encoder to the nearest code
    in the latent space. The quantization is done by calculating the distances
    between the input and the codes in the latent space, and then encoding the
    input to the nearest code. The exponential moving average is used to update
    the latent space.

    Attributes:
        embedding_dim_D: The embedding dimensionality of the latent space.
        num_embeddings_K: The number of embeddings in the latent space.
        latent_space: The embedding layer containing the latent space.
        ema_cluster_size: The exponential moving average of the cluster size.
        ema_weight: The exponential moving average of the weight.
        decay: The exponential moving average decay rate.
        epsilon: Epsilon for division.

    """

    def __init__(self, num_of_embeddings:int = 1024,
                 latent_dim:int = 256,
                 beta:float = 0.25,
                 ema_decay:float = 0.99,
                 epsilon=1e-5):
        """
        Initializes the vector quantizer.

        Args:
            num_of_embeddings: The number of codes in the latent space.
            latent_dim: The length of each code in the latent space.
            beta: A hyperparameter controlling the commitment loss.
            ema_decay: The exponential moving average decay rate.

        """
        super().__init__()

        # The embedding dimensionality of the latent space
        self.embedding_dim_D = latent_dim

        # The number of embeddings in the latent space
        self.num_embeddings_K = num_of_embeddings

        # Initialize the latent space as an embedding layer
        self.latent_space = nn.Embedding(self.num_embeddings_K, self.embedding_dim_D)

        # Initialize the latent space weights to be uniform
        self.latent_space.weight.data.normal_()
        # Store the hyperparameter controlling the commitment loss
        self.beta = beta

        # Exponential moving average variables
        self.register_buffer('ema_cluster_size', torch.zeros(self.num_embeddings_K))
        self.ema_weight = nn.Parameter(torch.Tensor(self.num_embeddings_K, self.embedding_dim_D))
        self.ema_weight.data.normal_()

        # Exponential moving average decay rate
        self.decay = ema_decay

        # Epsilon for division
        self.epsilon = epsilon

    def forward(self, inputs):
        """
        Forward pass of the VectorQuantizerEMA.

        This function takes the output of the encoder and quantizes it to the
        nearest code in the latent space. The output is the quantized version of
        the input, the loss, the perplexity of the latent space codes, and the
        encoding of the input.

        Args:
            inputs: The output of the encoder.

        Returns:
            z_quantized: The quantized version of the input.
            loss: The loss of the quantization, which is the sum of the codebook
                loss and the commitment loss.
            perplexity: The perplexity of the latent space codes.
            encodings: The encoding of the input.
        """
        # Permute the input to (batch, height, width, channel)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten the input to (batch * height * width, channel)
        input_flattened = inputs.view(-1, self.embedding_dim_D)

        # Calculate the distances between the input and the codes in the latent space
        distances = (torch.sum(input_flattened**2, dim=1, keepdim=True) +
                    torch.sum(self.latent_space.weight**2, dim=1) -
                    2 * torch.matmul(input_flattened, self.latent_space.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings_K, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        z_quantized = torch.matmul(encodings, self.latent_space.weight).view(input_shape)

        if self.training:
            # Update the exponential moving average variables
            self.ema_cluster_size = self.ema_cluster_size * self.decay + \
                                    (1 - self.decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self.ema_cluster_size.data)
            self.ema_cluster_size = (
                (self.ema_cluster_size + self.epsilon) / (n + self.num_embeddings_K * self.epsilon) * n)

            dw = torch.matmul(encodings.t(), input_flattened)
            self.ema_weight = nn.Parameter(self.ema_weight * self.decay + (1 - self.decay) * dw)

            self.latent_space.weight = nn.Parameter(self.ema_weight / self.ema_cluster_size.unsqueeze(1))

        # Calculate the loss
        commitment_loss = F.mse_loss(z_quantized, inputs)
        loss = self.beta * commitment_loss

        # Detach the quantized values
        z_quantized = inputs + (z_quantized - inputs).detach()

        # Calculate the perplexity
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        # convert quantized from (batch, height, width, channel) to (batch, channel, height, width)
        z_quantized = z_quantized.permute(0, 3, 1, 2).contiguous()

        return z_quantized, loss, perplexity, encoding_indices



