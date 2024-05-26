import glob
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import matplotlib
import os

from skimage import io
from model import DiffusionModel
from tqdm import tqdm

matplotlib.use("TkAgg")


class RenderingsDataset(torch.utils.data.Dataset):
    """Renderings dataset."""

    def __init__(self, scene_name="cbox"):
        """
        Args:
            root_dir (string): Directory with all the renderings.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_images = sorted(glob.glob(f"../data/{scene_name}/*"))
        self.images = self.all_images
        self.resize_shape = (256, 256)

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path):
        channels = 3
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = np.array(image).reshape((image.shape[0], image.shape[1], channels)).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def __getitem__(self, idx):
        image = self.transform_image(self.images[idx])
        sample = {'image': image, 'idx': idx}
        return sample


def evaluate(model, data_loader, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    for i, (x_data, _) in enumerate(data_loader):
        x_data = x_data.to(device)
        perturbed_input, noise, predicted_noise = model(x_data)
        loss = criterion(predicted_noise, noise)
        print(f"Batch {i} Loss: {loss.item()}")


def train(model, data_loader, epochs=1000, device=torch.device("cpu"), val_loader=None):
    # Move the model to the device
    model.to(device)
    # Initialize the loss function
    criterion = torch.nn.MSELoss()

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

    # Initialize the LR scheduler
    if val_loader:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.86)

    for epoch in range(epochs):
        with tqdm(data_loader, unit="batch") as train_epoch:
            for i, data in enumerate(train_epoch):
                train_epoch.set_description(f"Epoch {epoch}")
                optimizer.zero_grad()

                images = data["image"]
                x_data = images.to(device)
                pertrubed_input, noise, predicted_noise = model(x_data)
                loss = criterion(predicted_noise, noise)

                loss.backward()
                optimizer.step()

                train_epoch.set_postfix({"Noise Prediction Loss": loss.item()})

            # Validation compute loss
            if val_loader:
                val_loss = 0
                for j, (val_images, _) in enumerate(val_loader):
                    val_images = val_images.to(device)
                    perturbed_input, noise, predicted_noise = model(val_images)
                    val_loss += criterion(predicted_noise, noise).item()
                val_loss /= len(val_loader)
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if epoch % 100 == 0:
                print("Saving the model...")
                torch.save(model.state_dict(), f"../results/diffusion_model_CIFAR10_{epoch}.pth")

    print(f"Training complete! {epochs} epochs completed.")
    # Save the model
    print("Saving the model...")
    torch.save(model.state_dict(), f"../results/diffusion_model_CIFAR10_{epochs}.pth")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "../data"
    dataset = "CIFAR10"
    img_res = 256
    img_channels = 3

    # Diffusion Model Parameters
    n_timesteps = 1000
    train_batch_size = 32
    test_batch_size = 32
    lr = 4e-4
    epochs = 1000

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    #transforms = transforms.Compose([ transforms.ToTensor() ])

    train_dataset = RenderingsDataset(scene_name="cbox")

    #train_dataset = torchvision.datasets.CIFAR10(f"{data_path}/{dataset}", transform=transforms, download=False, train=True)
    #test_dataset = torchvision.datasets.CIFAR10(f"{data_path}/{dataset}", transform=transforms, download=False, train=False)
    #train_dataset = torchvision.datasets.Imagenette(f"{data_path}/{dataset}", transform=transforms, download=False, split="train")
    #val_dataset = torchvision.datasets.Imagenette(f"{data_path}/{dataset}", transform=transforms, download=False, split="val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)

    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)
    #val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=1, pin_memory=True)
    #test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DiffusionModel(img_res, img_channels, num_timesteps=n_timesteps)
    model.to(device)

    # Test and visualize Forward Diffusion
    """
    model.eval()
    with torch.no_grad():
        x_data = next(iter(train_loader))[0].to(device)

        all_im = []
        for t in range(0, 100, 25):
            time_tensor = torch.full((x_data.shape[0],), t, dtype=torch.long).to(device)
            noise = torch.randn_like(x_data).to(device)
            diffused_image = model.forward_diffusion(x_data, time_tensor, noise)

            # Reformat range to [0, 1]
            diffused_image = (diffused_image - diffused_image.min()) / (diffused_image.max() - diffused_image.min())
            diffused_image = diffused_image.permute(0, 2, 3, 1).cpu().numpy()
            diffused_image = np.concatenate(diffused_image, axis=0)
            all_im.append(diffused_image)

        diffused_images = np.concatenate(all_im, axis=1)
        plt.imshow(diffused_images)
        plt.show()
    """

    train(model, train_loader, epochs, device)

    """
    model = DiffusionModel(img_res, img_channels, num_timesteps=n_timesteps)
    model.load_state_dict(torch.load("../results/diffusion_model_.pth"))
    model.to(device)

    model.eval()
    with torch.no_grad():
        generated_images = model.sample(n_timesteps, 3)
        # Concatenate and display the image with pyplot
        generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy()
        generated_images = np.concatenate(generated_images, axis=1)
        plt.imshow(generated_images)
        plt.show()
    """


