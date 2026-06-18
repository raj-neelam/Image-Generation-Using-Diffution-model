import os
import torch
from tqdm import tqdm
from DDPM_model import DDPM_model
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu" 
print("Using:", torch.cuda.get_device_name(0))

class Config:
    T = 1000 # total timesteps to sample the model
    num_classes = 10
    image_size = 28 # size of the image
    in_channels = 1 # number of input channels
    
    hidden_channels = 64

    batch_size = 32
    lr = 1e-3
    epochs = 100

config = Config()

writer = SummaryWriter(log_dir="runs/Model_mnist")

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
# model.compile(optimizer=optimizer, loss=criterion)
global_step = 0
for epoch in range(config.epochs):
    loader = tqdm(train_loader)
    loader.set_description(f"Epoch {epoch+1}/{config.epochs}")
    for batch in loader:
        x, label = batch
        x = x.to(device)
        label = label.to(device)

        t = torch.randint(0, config.T, (x.shape[0],)).to(device)
        noise_image, noise = model.get_noisy_image_instant(x, t)
        output = model(noise_image, t, label)
        loss = criterion(output, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        writer.add_scalar("Loss", loss.item(), global_step)        
        loader.postfix = f"loss: {loss.item()}"
        # print(loss.item())
        global_step += 1

    model.eval()
    with torch.no_grad():
        for digit in range(10):
            img = model.generate(digit)          # (1, 1, 28, 28)
            img = (img + 1) / 2                  # Tanh → [0,1]
            writer.add_image(f"Generated/digit_{digit}", img[0], epoch)
    model.train()
    torch.save(model.state_dict(), f"Models/model_{epoch+1}.pt")
        
