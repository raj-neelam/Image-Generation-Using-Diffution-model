import torch
import torch.nn as nn

class DDPM_model(nn.Module):
    def __init__(self, config):
        super(DDPM_model, self).__init__()
        self.config = config
        
        self.register_buffer('betas', torch.linspace(1e-4, 0.01, config.T))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.time_embedding = nn.Embedding(config.T, config.hidden_channels)
        self.label_embedding = nn.Embedding(config.num_classes, config.hidden_channels)
        # convdown formula: 
        self.encoder = nn.Sequential(
            nn.Conv2d(config.in_channels, config.hidden_channels, kernel_size=3), # input size: 1x28x28 Output size: 26x26
            nn.MaxPool2d(kernel_size=2, stride=2), # input size: 26x26 Output size: 13x13
            nn.ReLU(),
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=3), # input size: 13x13 Output size: 11x11
            nn.MaxPool2d(kernel_size=2, stride=2), # input size: 11x11 Output size: 55x5
            nn.ReLU(),
        )
        # convup formula: (input - 1) * stride - 2*padding + kernel_size + output_padding
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(config.hidden_channels, config.hidden_channels, kernel_size=2, stride=2, padding=1, output_padding=1), # input size: 6x6 Output size: 9x9
            nn.ReLU(),
            nn.ConvTranspose2d(config.hidden_channels, config.hidden_channels, kernel_size=3, stride=3, padding=1, output_padding=1), # input size: 9x9 Output size: 26x26
            nn.ReLU(),
            nn.ConvTranspose2d(config.hidden_channels, config.hidden_channels, kernel_size=4, stride=1, padding=1, output_padding=0), # input size: 18x18 Output size: 18x18
            nn.ReLU(),
            nn.ConvTranspose2d(config.hidden_channels, config.in_channels, kernel_size=2, stride=1, padding=0, output_padding=0), # input size: 23x23 Output size: 28x28
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
        # x = x.permute((0,2,3,1))
        x = self.encoder(x)

        # concatenate x, t_emb, label_emb along the channel dimension
        value_time = t_emb + label_emb
        # x = x.permute((2,3,0,1))
        # x = x + value_time
        # x = x.permute((2,3,1,0))
        value_time = value_time.view(x.shape[0], x.shape[1], 1, 1)
        x = x + value_time
        x = self.decoder(x)
        # print(x.shape)
        return x

    def sample(self, label, t, img):
        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]
        
        # Predict noise
        t_tensor = torch.tensor([t], device=img.device)
        predicted_noise = self.forward(img, t_tensor, label)
        
        # Reverse formula: x_{t-1} = 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1-alpha_bar_t) * predicted_noise)
        x_prev = (1 / torch.sqrt(alpha_t)) * (img - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise)
        
        # Add stochastic term (except at t=1)
        if t > 1:
            z = torch.randn_like(img)
            x_prev = x_prev + torch.sqrt(beta_t) * z
        
        return x_prev

    def generate(self, label):
        device = next(self.parameters()).device
        noise = torch.randn(1, self.config.in_channels, 
                            self.config.image_size, self.config.image_size, 
                            device=device)
        label_tensor = torch.tensor([label], device=device)
        
        for t in range(self.config.T - 1, 0, -1):
            t_tensor = torch.tensor([t], device=device)
            
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            
            with torch.no_grad():
                predicted_noise = self.forward(noise, t_tensor, label_tensor)
            
            noise = (1 / torch.sqrt(alpha_t)) * (
                noise - (beta_t / torch.sqrt(1 - alpha_bar_t)) * predicted_noise
            )
            
            if t > 1:
                z = torch.randn_like(noise)
                noise = noise + torch.sqrt(beta_t) * z
        
        return noise

