import torch
import torch.nn as nn

class DDPM_model(nn.Module):
    def __init__(self, config):
        super(DDPM_model, self).__init__()
        self.config = config
        
        betas = torch.linspace(1e-4, 0.01, config.T)
        self.alphas_cumprod = torch.cumprod(1.0 - betas, dim=0)
        self.time_embedding = nn.Embedding(config.T, config.hidden_channels)
        self.label_embedding = nn.Embedding(config.num_classes, config.hidden_channels)
        # convdown formula: 
        self.encoder = nn.Sequential(
            nn.Conv2d(config.in_channels, config.hidden_channels, kernel_size=3), # input size: 1x28x28 Output size: 26x26
            nn.MaxPool2d(kernel_size=2, stride=2), # input size: 26x26 Output size: 13x13
            nn.ReLU(),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3), # input size: 13x13 Output size: 11x11
            nn.MaxPool2d(kernel_size=2, stride=2), # input size: 11x11 Output size: 6x6
            nn.ReLU(),
        )
        # convup formula: (input - 1) * stride - 2*padding + kernel_size + output_padding
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_channels, config.hidden_channels, kernel_size=2, stride=2, padding=1, output_padding=1), # input size: 6x6 Output size: 11x11
            nn.ReLU(),
            nn.ConvTranspose2d(config.hidden_channels, config.hidden_channels, kernel_size=3, stride=2, padding=1, output_padding=2), # input size: 11x11 Output size: 23x23
            nn.ReLU(),
            nn.ConvTranspose2d(config.hidden_channels, config.in_channels, kernel_size=3, stride=1, padding=0, output_padding=1), # input size: 23x23 Output size: 28x28
            nn.Tanh()  
        )
        

    def get_noisy_image_instant(self, x_0, t):
        """
        Adds noise to the image at time step t using the closed-form formula:
        x_t = sqrt(a_bar)*x_0 + sqrt(1 - a_bar) * noise

        returns:
            - x_t: the noisy image at time step t
            - noise: the noise added to the image
        """
        a_t = self.alphas_cumprod[t].view(-1, 1, 1, 1) # Reshape for broadcasting
        noise = torch.randn_like(x_0)
        # Closed-form formula: x_t = sqrt(a_bar)*x_0 + sqrt(1 - a_bar) * noise
        return torch.sqrt(a_t) * x_0 + torch.sqrt(1.0 - a_t) * noise, noise

    def forward(self, x, t, label):
        t_emb = self.time_embedding(t)
        label_emb = self.label_embedding(label)
        x = self.encoder(x)

        # concatenate x, t_emb, label_emb along the channel dimension
        x = x + t_emb.view(-1, 1, 1, 1) + label_emb.view(-1, 1, 1, 1)
        x = self.decoder(x)
        
        # latent vector 
        return x

    def sample(self, num_samples, device):
        

