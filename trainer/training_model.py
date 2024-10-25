import argparse
import torch
import torch.functional as F
import wandb as wb
from vqgan.discriminator import Discriminator
from utils.utils import set_seed, weight_init
from utils.dataset_builder import get_dataloader
from piq import LPIPS

set_seed(42)
class VQGANTrainer:
    def __init__(self, 
                 #model parameters
                 model:torch.nn.Module,
                 disc_input_channels:int,
                 
                 #experiment tracking 
                 project:str,
                 entity:str,
                 api_key:str,
                 config:dict,

                 #hyperparameters
                 learning_rate:float,
                 beta1:float,
                 beta2:float,
                 perceptual_loss_weight:float,
                 recon_loss_weight:float,
                 disc_factor:float,
                 disc_start:int,

                 #data and misc
                 device:str = "cuda" if torch.cuda.is_available() else "cpu",
                 train_dataset_path:str = "/data/train",
                 test_dataset_path:str = "/data/test",
                 batch_size:int = 32
    ):
        
        try:
            if api_key:
                wb.login(key=api_key)
            else:
                wb.login()
            self.experiment = wb.init(project=project, entity=entity)
            self.config = config
        except Exception as e:
            print(f'Error initializing wandb: {e}')


        self.device = device
        self.vqvae = model
        self.discriminator = Discriminator(image_channels=disc_input_channels).to(self.device)
        self.discriminator.apply(weight_init)

        self.perceptual_loss = LPIPS().eval().to(self.device)

        self.opt_vqvae, self.opt_disc = self.configure_optimizers(learning_rate, beta1, beta2)


        self.disc_factor = disc_factor
        self.disc_start = disc_start
        self.perceptual_loss_weight = perceptual_loss_weight
        self.recon_loss_weight = recon_loss_weight

        self.GlobalStep = 0


    def configure_optimizers(self, learning_rate: float, beta1: float, beta2: float):

        opt_vqvae = torch.optim.Adam(
            list(self.vqgan.encoder.parameters()
                + self.vqgan.decoder.parameters()
                + self.vqgan.quantizer.parameters()
                + self.vqgan.quant_conv.parameters()
                + self.vqgan.post_quant_conv.parameters()),
            lr=learning_rate, betas=(beta1, beta2)
        )

        opt_disc = torch.optim.Adam(self.discriminator.parameters(), lr=learning_rate, betas=(beta1, beta2))

        return opt_vqvae, opt_disc

        

    def training_step(self, images: torch.Tensor):

        decoded_images, encoding_indices, codebook_loss, perplexity = self.vqvae(images)

        perceptual_loss = self.perceptual_loss(images, decoded_images)
        recon_loss = torch.abs(images - decoded_images)

        perceptual_recon_loss = (
            self.perceptual_loss_weight * perceptual_loss
            + self.recon_loss_weight * recon_loss)

        disc_real = self.discriminator(images)
        disc_fake = self.discriminator(decoded_images)

        disc_factor = self.vqgan.adopt_weight(disc_factor=self.disc_factor,
                                              iteration = self.GlobalStep,
                                              threshold=self.disc_start)
        
        g_loss = -torch.mean(disc_fake)

        lmda = self.vqvae.calc_lambda(perceptual_recon_loss, g_loss)

        vq_loss = perceptual_recon_loss + codebook_loss + disc_factor * lmda * g_loss

        disc_loss_real = torch.mean(F.relu(1.0 - disc_real))
        disc_loss_fake = torch.mean(F.relu(1.0 + disc_fake))
        disc_loss = disc_loss_real + disc_loss_fake
        gan_loss = disc_factor * 0.5 * disc_loss

        mean_perplexity = torch.mean(perplexity)









if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="VIS_MODEL")

    # encoder and decoder arguments
    parser.add_argument('--in_channels', type=int, default=[32,64,128,256], help='list of feature channels for down and up sampling (default: [32,64,128,256]')
    parser.add_argument('--m_blocks', type=int, default=3, help='number of down/up blocks (default: 3)')
    parser.add_argument('--input_channels', type=int, default=3, help='number of input channels (default: 3)')
    parser.add_argument('--latent_dim', type=int, default=256, help='latent dimension (default: 256)')

    # coodebook arguments





    args = parser.parse_args()
    