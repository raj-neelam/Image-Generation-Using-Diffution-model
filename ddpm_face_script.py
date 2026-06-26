import os
import torch
from tqdm import tqdm
from Model_architecture.DDPM_unet_face import DDPM_model
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import math

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


class Config:
    T            = 1000
    image_size   = 128
    in_channels  = 3

    hidden_channels = 64

    batch_size = 16   
    lr_start   = 2e-4   
    lr_end     = 5e-5
    epochs     = 150

    data_root  = r"./data/Human-Faces-Dataset"
    resume_ckpt = None  


config = Config()

writer = SummaryWriter(log_dir="runs/FaceModel_v2")


face_transform = T.Compose([
    T.RandomResizedCrop(
        size=(config.image_size, config.image_size),
        scale=(0.85, 1.0),
        ratio=(1.0, 1.0),
        interpolation=T.InterpolationMode.BICUBIC,
        antialias=True,
    ),
    T.RandomHorizontalFlip(p=0.5),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


class FaceDataset(Dataset):

    VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(self, root: str, transform=None):
        self.transform   = transform
        self.image_paths = []

        folder_path = os.path.join(root, "AI-Generated Images")
        if not os.path.isdir(folder_path):
            raise FileNotFoundError(
                f"[Dataset] AI-Generated Images folder not found at: {folder_path} | Download dataset from https://www.kaggle.com/datasets/kaustubhdhote/human-faces-dataset"
            )

        for fname in os.listdir(folder_path):
            if os.path.splitext(fname)[1].lower() in self.VALID_EXTS:
                self.image_paths.append(os.path.join(folder_path, fname))

        print(f"[Dataset] Found {len(self.image_paths)} AI-generated images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def cosine_lr(optimizer, epoch, total_epochs, lr_start, lr_end):
    progress = epoch / total_epochs                      
    cosine   = 0.5 * (1 + math.cos(math.pi * progress)) 
    lr       = lr_end + (lr_start - lr_end) * cosine
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr

train_data   = FaceDataset(root=config.data_root, transform=face_transform)
train_loader = DataLoader(
    train_data,
    batch_size=config.batch_size,
    shuffle=True
)

model     = DDPM_model(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr_start)
criterion = torch.nn.MSELoss()

start_epoch = 0
global_step = 0

if config.resume_ckpt and os.path.isfile(config.resume_ckpt):
    sd = torch.load(config.resume_ckpt, map_location=device)
    model.load_state_dict(sd)
    try:
        start_epoch = int(config.resume_ckpt.split("_")[-1].replace(".pt", ""))
        global_step = len(train_loader) * start_epoch
        print(f"[Resume] Loaded {config.resume_ckpt} — continuing from epoch {start_epoch}")
    except ValueError:
        print(f"[Resume] Loaded {config.resume_ckpt} — could not parse epoch, starting from 0")
else:
    print("[Train] Starting fresh.")

os.makedirs("Models", exist_ok=True)


for epoch in range(start_epoch, config.epochs):

    current_lr = cosine_lr(optimizer, epoch, config.epochs, config.lr_start, config.lr_end)
    writer.add_scalar("LR", current_lr, epoch)

    model.train()
    loader     = tqdm(train_loader)
    loader.set_description(f"Epoch {epoch+1}/{config.epochs}  lr={current_lr:.2e}")
    epoch_loss = 0.0

    for batch in loader:
        x = batch.to(device)

        t = torch.randint(0, config.T, (x.shape[0],), device=device)
        noise_image, noise = model.get_noisy_image_instant(x, t)
        output = model(noise_image, t)
        loss   = criterion(output, noise)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  
        optimizer.step()

        writer.add_scalar("Loss", loss.item(), global_step)
        epoch_loss += loss.item()
        loader.set_postfix(loss=f"{loss.item():.4f}")
        global_step += 1

    avg_loss = epoch_loss / len(train_loader)
    writer.add_scalar("Loss_avg", avg_loss, epoch)
    print(f"[Epoch {epoch+1:3d}] Avg Loss: {avg_loss:.4f}  LR: {current_lr:.2e}")
    
    model.eval()
    with torch.no_grad():
        img = model.generate(num_images=1)
        img = (img.clamp(-1, 1) + 1) / 2  
        writer.add_image("Generated/face", img[0], epoch)

    torch.save(model.state_dict(), f"Models/face_model_{epoch+1}.pt")

writer.close()