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
        H = config.hidden_channels
        self.enc1 = nn.Sequential(
            nn.Conv2d(config.in_channels, H, kernel_size=3, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        ) # in: [32, 1, 28, 28], out: [32, 64, 13, 13]
        self.enc2 = nn.Sequential(
            nn.Conv2d(H, H*2, kernel_size=3, padding=0),
            nn.ReLU(),
        ) # in: [32, 64, 13, 13], out: [32, 128, 11, 11]
        self.enc3 = nn.Sequential(
            nn.Conv2d(H*2, H*4, kernel_size=3, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        ) # in: [32, 128, 11, 11], out: [32, 256, 4, 4]

        # bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(H*4, H*4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(H*4, H*4, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.project = nn.Linear(H, H*4)
        
        
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(H*4 + H*4, H*2, kernel_size=4, stride=3, padding=1),
            nn.ReLU()
        )  # concat(bottleneck[256], enc3[256]) → [32, 128, 11, 11]
        
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(H*2 + H*2, H, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )  # concat(dec3[128], enc2[128]) → [32, 64, 13, 13]

        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(H + H, H, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(H, config.in_channels, kernel_size=1)
        )  # concat(dec2[64], enc1[64]) → [B, 1, 28, 28]
        

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
        # conditioning embedding
        emb = self.time_embedding(t) + self.label_embedding(label)  # [B, H]

        # encoder — save outputs for skip connections
        e1 = self.enc1(x)   # [B, 64,  28, 28]
        e2 = self.enc2(e1)  # [B, 128, 14, 14]
        e3 = self.enc3(e2)  # [B, 256,  7,  7]
        # bottleneck + inject time/label conditioning
        b = self.bottleneck(e3)  
        emb_proj = self.project(emb).view(-1, b.shape[1])        # [B, 256]
        emb_proj = emb_proj.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 4)
        print(b.shape, emb_proj.shape)
        b = torch.cat([b,  emb_proj], dim=1)                                             # broadcast add
        # decoder — concatenate skip at each level
        d3  = self.dec3(torch.cat([b,  e3], dim=1))   # [B, 128, 8, 8]
        d2  = self.dec2(torch.cat([d3, e2], dim=1))   # [B, 64,  28, 28]
        out = self.dec1(torch.cat([d2, e1], dim=1))   # [B, 1,   28, 28]
        return out

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

