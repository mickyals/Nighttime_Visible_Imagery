import argparse
import torch
import torch.functional as F
from torchvision.utils import make_grid
import wandb as wb
from vqgan.discriminator import Discriminator
from utils.utils import set_seed, weight_init
from utils.dataset_builder import get_dataloader, GoesNumpyDataset
from piq import LPIPS

set_seed(42)


class VQGANTrainer:
    def __init__(self, 
                 #model parameters
                 model:torch.nn.Module,
                 disc_input_channels:int,
                 x_idxs:list,
                 y_idxs:list,
                 dataset_path:str,
                 
                 #experiment tracking 
                 project:str,
                 entity:str,
                 api_key:str,
                 hyperparameters:dict,

                 #hyperparameters
                 learning_rate:float,
                 beta1:float,
                 beta2:float,
                 perceptual_loss_weight:float,
                 recon_loss_weight:float,
                 disc_factor:float,
                 disc_start:int,

                 #data and misc
                 device:str = "cuda" if torch.cuda.is_available() else "cpu"):
        
        # initialize wandb
        try:
            if api_key:
                wb.login(key=api_key)
            else:
                wb.login()
        except Exception as e:
            print(f'Error initializing wandb: {e}')

        self.project = project
        self.entity = entity
        self.hyperparameters = hyperparameters

        # initialize model
        self.device = device
        self.vqvae = model
        self.discriminator = Discriminator(image_channels=disc_input_channels).to(self.device)
        self.discriminator.apply(weight_init)

        self.perceptual_loss = LPIPS().eval().to(self.device)

        self.opt_vqvae, self.opt_disc, self.scheduler_vqvae, self.scheduler_disc = self.configure_optimizers(learning_rate, beta1, beta2)

        # datset params
        self.data_dir = dataset_path
        self.x_idxs = x_idxs
        self.y_idxs = y_idxs

        # hyperparameters
        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.perceptual_loss_weight = perceptual_loss_weight
        self.recon_loss_weight = recon_loss_weight


    def configure_optimizers(self, learning_rate: float, beta1: float, beta2: float) -> Tuple[torch.optim.Optimizer, torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler, torch.optim.lr_scheduler._LRScheduler]:
        """
        Configure the optimizers and learning rate schedulers for the VQVAE and discriminator.

        Args:
            learning_rate (float): The initial learning rate.
            beta1 (float): The beta1 parameter for the Adam optimizer.
            beta2 (float): The beta2 parameter for the Adam optimizer.

        Returns:
            A tuple of four objects: the optimizer for the VQVAE, the optimizer for the discriminator, the learning rate scheduler for the VQVAE, and the learning rate scheduler for the discriminator.

        """
        # Define the optimizer for the VQVAE
        opt_vqvae = torch.optim.Adam(
            # The parameters to optimize are the encoder, decoder, quantizer, quant_conv, and post_quant_conv
            list(self.vqgan.encoder.parameters()
                + self.vqgan.decoder.parameters()
                + self.vqgan.quantizer.parameters()
                + self.vqgan.quant_conv.parameters()
                + self.vqgan.post_quant_conv.parameters()),
            # The initial learning rate
            lr=learning_rate, 
            # The beta1 and beta2 parameters for the Adam optimizer
            betas=(beta1, beta2)
        )

        # Define the optimizer for the discriminator
        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

        # Define the learning rate scheduler for the VQVAE
        scheduler_vqvae = torch.optim.lr_scheduler.CosineAnnealingLR(opt_vqvae, T_0=10, T_mult=1)

        # Define the learning rate scheduler for the discriminator
        scheduler_disc = torch.optim.lr_scheduler.CosineAnnealingLR(opt_disc, T_0=10, T_mult=1)

        # Return the optimizers and learning rate schedulers
        return opt_vqvae, opt_disc, scheduler_vqvae, scheduler_disc

        

    def train(self, epochs:int):
        """
        Train the VQVAE and discriminator using the data loader.

        Args:
            epochs (int): The number of epochs to train for.

        """
        train_data = GoesNumpyDataset(self.data_dir, self.x_idxs, self.y_idxs, self.transform, True)
        train_loader, val_loader = get_dataloader(train_data, batch_size=1, shuffle=True)

        steps_per_epoch = len(train_loader)

        for epoch in range(epochs):
            with wb.init(project=self.project, entity=self.entity, config=self.hyperparameters, name=f"Epoch {epoch}"):
                for i, (input_images, output_images) in enumerate(train_loader):
                    """
                    Train the VQVAE and discriminator on a single batch of data.

                    Args:
                        input_images (torch.Tensor): The input images.
                        output_images (torch.Tensor): The output images.

                    """
                    input_images = input_images.to(self.device)
                    output_images = output_images.to(self.device)

                    decoded_images, _ , codebook_loss, perplexity = self.vqvae(input_images)

                    disc_real = self.discriminator(output_images)
                    disc_fake = self.discriminator(decoded_images)

                    disc_factor = self.adopt_weight(self.disc_factor, epoch*steps_per_epoch + i, self.disc_start)

                    # Calculate the perceptual loss, reconstruction loss, and generator loss
                    perceptual_loss = self.perceptual_loss(input_images, decoded_images)
                    recon_loss = F.mse_loss(decoded_images, output_images)
                    perceptual_recon_loss = self.perceptual_loss_weight * perceptual_loss + self.recon_loss_weight * recon_loss
                    perceptual_recon_loss = perceptual_recon_loss.mean()
                    generator_loss = -torch.mean(disc_fake)

                    lmda = self.calculate_lambda(perceptual_recon_loss, generator_loss)
                    vq_loss = perceptual_recon_loss + codebook_loss + disc_factor * lmda * generator_loss


                    d_loss_real = torch.mean(F.relu(1.0 - disc_real))
                    d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
                    gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)
# ======================================================================================= #
                    # Backpropagate the loss and update the optimizers
                    self.opt_vqvae.zero_grad()
                    vq_loss.backward(retain_graph=True)

                    self.opt_disc.zero_grad()
                    gan_loss.backward()

                    self.opt_vqvae.step()
                    self.opt_disc.step()

                    # Update scheduler at each iterations
                    self.scheduler_vqvae.step()
                    self.scheduler_disc.step()

# ================================== LOGGING ========================================= #
                    wb.log({
                            "vq_loss": vq_loss.item(),
                            "gan_loss": gan_loss.item(),
                            "g_loss": generator_loss.item(),
                            "perplexity": perplexity.mean().item()
                            })
                    

# ================================================================================================== #
# =========================================== VALIDATION ============================================= #
# ================================================================================================== #
                    with torch.no_grad():
                        self.vqvae.eval()  # Set the model to evaluation mode
                        self.discriminator.eval()  # Set the discriminator to evaluation mode
                        val_vq_loss = 0.0
                        val_gan_loss = 0.0
                        val_perplexity = 0.0

                        for i, (input_images, output_images) in enumerate(val_loader):
                            input_images = input_images.to(self.device)
                            output_images = output_images.to(self.device)
                            decoded_images, _ , codebook_loss, perplexity = self.vqvae(input_images)
                            disc_real = self.discriminator(output_images)
                            disc_fake = self.discriminator(decoded_images)
                            perceptual_loss = self.perceptual_loss(input_images, decoded_images)
                            recon_loss = F.mse_loss(decoded_images, output_images)
                            perceptual_recon_loss = self.perceptual_loss_weight * perceptual_loss + self.recon_loss_weight * recon_loss
                            perceptual_recon_loss = perceptual_recon_loss.mean()
                            generator_loss = -torch.mean(disc_fake)

                            lmda = self.calculate_lambda(perceptual_recon_loss, generator_loss)
                            vq_loss = perceptual_recon_loss + codebook_loss + disc_factor * lmda * generator_loss

                            d_loss_real = torch.mean(F.relu(1.0 - disc_real))
                            d_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
                            gan_loss = disc_factor * 0.5*(d_loss_real + d_loss_fake)

                            val_vq_loss += vq_loss.item()
                            val_gan_loss += gan_loss.item()
                            val_perplexity += perplexity.mean().item()

                            if i == 0:
                                wb.log({"val_input": wb.Image(input_images[0].cpu().detach().numpy()),
                                        "val_output": wb.Image(output_images[0].cpu().detach().numpy()),
                                        "val_decoded": wb.Image(decoded_images[0].cpu().detach().numpy())})

                        val_vq_loss /= len(val_loader)
                        val_gan_loss /= len(val_loader)
                        val_perplexity /= len(val_loader)
                        wb.log({"val_vq_loss": val_vq_loss, "val_gan_loss": val_gan_loss, "val_perplexity": val_perplexity})
# ================================================================================================== #





                        






        