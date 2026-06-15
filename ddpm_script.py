import os
import torch
from DDPM_model import DDPM_model
import torchvision
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu" 
print("Using:", torch.cuda.get_device_name(0))

class Config:
    T = 1000 # total timesteps to sample the model
    image_size = 28 # size of the image
    in_channels = 1 # number of input channels
    hidden_channels = 32
    batch_size = 32
    lr = 1e-4

config = Config()

# download data if not available
if not os.path.exists("data/MNIST/raw"):
    print("Downloading data...")
    torchvision.datasets.MNIST(root="data", download=True)

class Mydataset(Dataset):
    def __init__(self, train=True):
        # using data from data/MNIST\raw
        self.data = torchvision.datasets.MNIST(root="data", train=train, download=False)
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((config.image_size, config.image_size)),
            torchvision.transforms.ToTensor(),
        ])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, label = self.data[idx]
        img = self.transform(img)
        return img, label

train_data = Mydataset(train=True)
train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)

model = DDPM_model(config).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
criterion = torch.nn.MSELoss()
model.compile(optimizer=optimizer, loss=criterion)

