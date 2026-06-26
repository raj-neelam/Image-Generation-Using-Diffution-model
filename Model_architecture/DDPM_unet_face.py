import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_groups=8):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
            nn.SiLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_groups, out_ch),
        )
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.act = nn.SiLU()
        self.proj = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = self.block2(h)
        h = h + self.time_proj(t_emb).view(t_emb.shape[0], -1, 1, 1)
        h = self.act(h)
        return h + self.proj(x)


class SelfAttention(nn.Module):
    def __init__(self, channels, num_heads=4, num_groups=8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads=num_heads, batch_first=True)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).permute(0, 2, 1)   # [B, H*W, C]
        h, _ = self.attn(h, h, h)
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return x + h


class EncoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, use_attn=False):
        super().__init__()
        self.res  = ResBlock(in_ch, out_ch, time_emb_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x, t_emb):
        x = self.res(x, t_emb)
        x = self.attn(x)
        return self.pool(x), x


class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, time_emb_dim, use_attn=False):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.res  = ResBlock(in_ch + skip_ch, out_ch, time_emb_dim)
        self.attn = SelfAttention(out_ch) if use_attn else nn.Identity()

    def forward(self, x, skip, t_emb):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t_emb)
        x = self.attn(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, channels, time_emb_dim):
        super().__init__()
        self.res1 = ResBlock(channels, channels, time_emb_dim)
        self.attn = SelfAttention(channels)
        self.res2 = ResBlock(channels, channels, time_emb_dim)

    def forward(self, x, t_emb):
        x = self.res1(x, t_emb)
        x = self.attn(x)
        x = self.res2(x, t_emb)
        return x


class DDPM_model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        H = config.hidden_channels
        time_emb_dim = H * 4

        betas = torch.linspace(1e-4, 0.02, config.T)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1.0 - betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(1.0 - betas, dim=0))
        self.time_embedding = nn.Embedding(config.T, H)
        self.time_mlp = nn.Sequential(
            nn.Linear(H, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        self.enc1 = EncoderBlock(config.in_channels, H,    time_emb_dim, use_attn=False) 
        self.enc2 = EncoderBlock(H,                  H*2,  time_emb_dim, use_attn=False) 
        self.enc3 = EncoderBlock(H*2,                H*4,  time_emb_dim, use_attn=True)  
        self.enc4 = EncoderBlock(H*4,                H*8,  time_emb_dim, use_attn=True)  
        self.bottleneck = BottleneckBlock(H*8, time_emb_dim)

        self.dec4 = DecoderBlock(H*8, H*8, H*4, time_emb_dim, use_attn=True)   
        self.dec3 = DecoderBlock(H*4, H*4, H*2, time_emb_dim, use_attn=True)   
        self.dec2 = DecoderBlock(H*2, H*2, H,   time_emb_dim, use_attn=False)  
        self.dec1 = DecoderBlock(H,   H,   H,   time_emb_dim, use_attn=False)  

        self.out_conv = nn.Conv2d(H, config.in_channels, kernel_size=1)

    def get_noisy_image_instant(self, x_0, t):
        a_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(x_0)
        x_t   = torch.sqrt(a_bar) * x_0 + torch.sqrt(1.0 - a_bar) * noise
        return x_t, noise

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        # Encoder
        x, s1 = self.enc1(x, t_emb)   # s1: [B,  H, 128, 128]
        x, s2 = self.enc2(x, t_emb)   # s2: [B, 2H,  64,  64]
        x, s3 = self.enc3(x, t_emb)   # s3: [B, 4H,  32,  32]
        x, s4 = self.enc4(x, t_emb)   # s4: [B, 8H,  16,  16]

        # Bottleneck
        x = self.bottleneck(x, t_emb) # [B, 8H, 8, 8]

        # Decoder
        x = self.dec4(x, s4, t_emb)   # [B, 4H, 16, 16]
        x = self.dec3(x, s3, t_emb)   # [B, 2H, 32, 32]
        x = self.dec2(x, s2, t_emb)   # [B,  H, 64, 64]
        x = self.dec1(x, s1, t_emb)   # [B,  H,128,128]

        return self.out_conv(x)        # [B,  3,128,128]

    def sample_step(self, x_t, t):
        beta_t      = self.betas[t]
        alpha_t     = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]

        t_tensor        = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        predicted_noise = self.forward(x_t, t_tensor)

        x_prev = (1.0 / torch.sqrt(alpha_t)) * (
            x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * predicted_noise
        )

        if t > 1:
            x_prev = x_prev + torch.sqrt(beta_t) * torch.randn_like(x_t)

        return x_prev

    @torch.no_grad()
    def generate(self, num_images: int = 1):
        device = next(self.parameters()).device
        x = torch.randn(
            num_images,
            self.config.in_channels,
            self.config.image_size,
            self.config.image_size,
            device=device,
        )
        for t in range(self.config.T - 1, 0, -1):
            x = self.sample_step(x, t)
        return x.clamp(-1, 1)