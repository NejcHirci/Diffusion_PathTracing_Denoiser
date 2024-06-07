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
from model_alternative import DiffusionModelForDenoising
from tqdm import tqdm

#matplotlib.use("TkAgg")


class RenderingsDataset(torch.utils.data.Dataset):
    """Renderings dataset."""

    def __init__(self, res=(256, 256), scene_name="cbox"):
        """
        Args:
            root_dir (string): Directory with all the renderings.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.all_images = sorted(glob.glob(f"../data/{scene_name}/train_*"))
        self.images = self.all_images
        self.resize_shape = res
        self.gt_image = self.transform_image(f"../data/{scene_name}/gt.png")

        self.data = []
        self.load_images()

    def __len__(self):
        return len(self.images)

    def load_images(self):
        for i in range(len(self.images)):
            self.data.append(self.transform_image(self.images[i]))

    def transform_image(self, image_path):
        channels = 3
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        image = np.array(image).reshape((image.shape[0], image.shape[1], channels)).astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))
        image = torch.tensor(image)
        image = (image - 0.5) / 0.5  # Normalize the image to [-1, 1]
        return image

    def __getitem__(self, idx):
        image = self.data[idx]
        sample = (image, idx)
        return sample


def evaluate(model, data_loader, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    for i, (x_data, _) in enumerate(data_loader):
        x_data = x_data.to(device)
        perturbed_input, noise, predicted_noise = model(x_data)
        loss = criterion(predicted_noise, noise)
        print(f"Batch {i} Loss: {loss.item()}")


def train(model, data_loader, epochs=1000, lr=1e-3, device=torch.device("cpu"), gt_image=None):
    # Move the model to the device
    model.to(device)
    model.train()
    # Initialize the loss function
    criterion = torch.nn.MSELoss()

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # We add an empircal regularization loss term
    loss_reg = 0.001

    if gt_image is not None:
        gt_image = gt_image.unsqueeze(0).repeat(data_loader.batch_size, 1, 1, 1).to(device)

    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(data_loader, unit="batch") as train_epoch:
            train_epoch.set_description(f"Epoch {epoch}")
            for images, labels in train_epoch:
                optimizer.zero_grad()
                images = images.cuda()
                if gt_image is not None:
                    pred_gt, _ = model(images, gt_image)
                    loss = torch.sqrt(criterion(pred_gt, gt_image) + loss_reg ** 2)
                else:
                    noisy_x, noise, noise_pred = model(images)
                    loss = criterion(noise_pred, noise)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                train_epoch.set_postfix({"Batch Noise Loss": loss.item(), "Epoch Noise Loss": epoch_loss})

            if epoch % 50 == 0:
                if gt_image is not None:
                    evaluate_special(model, images)
                else:
                    evaluate(model, data_loader, device)
                model.train()

    print(f"Training complete! {epochs} epochs completed.")
    # Save the model
    print("Saving the model...")
    torch.save(model.state_dict(), f"../results/diffusion_model_special2_cbox_final.pth")


def forward_diffusion_test(model, train_loader):
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


def evaluate_special(model, x_real):
    model.eval()
    with torch.no_grad():
        generated_image = model.sample(x_real)
        generated_image = generated_image.permute(0, 2, 3, 1).cpu().numpy()
        temp = x_real.permute(0, 2, 3, 1).cpu().numpy() * 0.5 + 0.5
        images = np.concatenate(generated_image, axis=1)
        temps = np.concatenate(temp, axis=1)
        images = np.concatenate([temps, images], axis=0)
        plt.figure(figsize=(images.shape[1] / 300, images.shape[0] / 300), dpi=300)
        plt.imshow(images)
        plt.axis("off")
        plt.show()


def evaluate_model(model):
    model.eval()
    with torch.no_grad():
        # Perform sampling for 3 images
        generated_images = model.sample(3)
        # Concatenate and display the image with pyplot
        # N, C, H, W -> N, H, W, C
        generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy()
        generated_images = np.concatenate(generated_images, axis=1)
        plt.figure(layout="tight")
        plt.imshow(generated_images)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "../data"
    dataset = "cbox"
    res = (256, 256)
    img_channels = 3

    # Diffusion Model Parameters
    n_timesteps = 70
    train_batch_size = 4
    test_batch_size = 4
    lr = 1e-6
    epochs = 500

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = RenderingsDataset(res=res, scene_name="cbox")
    gt_image = train_dataset.gt_image
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=1, pin_memory=False)

    # Let's continue training the model
    #model = DiffusionModel(res, img_channels, num_timesteps=n_timesteps, scheduler="linear")
    #model.load_state_dict(torch.load("../results/diffusion_model_diffunet_cbox_final.pth"))
    print(torch.cuda.memory.mem_get_info())
    model = DiffusionModelForDenoising(res, img_channels, num_timesteps=n_timesteps, scheduler="linear")
    print(torch.cuda.memory.mem_get_info())
    model.to(device)

    train(model, train_loader, epochs, lr, device, gt_image=gt_image)

    evaluate_model(model)
