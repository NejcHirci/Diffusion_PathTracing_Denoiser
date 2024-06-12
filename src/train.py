import glob
import sys
import logging

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import v2

from model_alternative import DiffusionModelForDenoising
from tqdm import tqdm

#matplotlib.use("TkAgg")


class RenderingsDataset(torch.utils.data.Dataset):
    """Renderings dataset."""

    def __init__(self, res=(256, 256), scene_name="all", spp=None):
        """
        Args:
            root_dir (string): Directory with all the renderings.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.resize_shape = res
        self.data = []
        self.load_images(scene_name=scene_name, spp=spp)
        self.transforms = v2.Compose([v2.RandomHorizontalFlip(0.5), v2.RandomVerticalFlip(0.5)])
        self.scene_name = scene_name

    def __len__(self):
        return len(self.data)

    def load_images(self, scene_name="cbox", spp=None):
        if scene_name == "all":
            scenes = ["bathroom", "kitchen", "cbox", "veach_ajar", "veach_bidir"]
            for i in scenes:
                images = sorted(glob.glob(f"../data/{i}/train_*"))
                gt_image = self.transform_image(f"../data/{i}/gt.png")
                for i in range(len(images)):
                    sample = (self.transform_image(images[i]), gt_image)
                    self.data.append(sample)
        elif scene_name == "all_diff_views" or scene_name == "test":
            scenes = ["bathroom_novel", "kitchen_novel", "veach_ajar_novel", "veach_bidir_novel", "house_novel",
                      "living_room_novel", "cbox_novel"]
            view_ind = range(0, 7, 1) if scene_name == "all_diff_views" else [7]
            spp_set = [1, 2, 4, 8, 16, 32, 64, 128] if spp is None else [spp]
            for spp in spp_set:
                for noise in range(8):
                    for view in view_ind:
                        for scene in scenes:
                            gt_image = self.transform_image(f"../data/{scene}/gt_{view}.png")
                            image = self.transform_image(f"../data/{scene}/train_{noise}_{view}_{spp}.png")
                            self.data.append((image, gt_image))
        else:
            images = sorted(glob.glob(f"../data/{scene_name}/train_*"))
            gt_image = self.transform_image(f"../data/{scene_name}/gt.png")
            for i in range(len(images)):
                sample = (self.transform_image(images[i]), gt_image)
                self.data.append(sample)

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
        image, gt_image = self.data[idx]
        return image, gt_image


def evaluate(model, data_loader, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    for i, (x_data, _) in enumerate(data_loader):
        x_data = x_data.to(device)
        perturbed_input, noise, predicted_noise = model(x_data)
        loss = criterion(predicted_noise, noise)
        print(f"Batch {i} Loss: {loss.item()}")


def train(model, data_loader, epochs=1000, lr=1e-3, device=torch.device("cpu"), scene_name="cbox", save_model=True, test_loader=None):
    # Move the model to the device
    model.to(device)
    model.train()
    # Initialize the loss function
    criterion = torch.nn.MSELoss()

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99))

    # We add an empircal regularization loss term
    loss_reg = 0.001

    # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

    for epoch in range(epochs):
        epoch_loss = 0
        with tqdm(data_loader, unit="batch", file=sys.stdout) as train_epoch:
            train_epoch.set_description(f"Epoch {epoch}")
            for x_data, y_gt in train_epoch:
                optimizer.zero_grad()
                x_data = x_data.cuda()
                y_gt = y_gt.cuda()
                pred_gt, _ = model(x_data, y_gt)
                loss = torch.sqrt(criterion(pred_gt, y_gt) + loss_reg ** 2)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                train_epoch.set_postfix({"Batch Noise Loss": loss.item(), "Epoch Noise Loss": epoch_loss})
            logging.info(f"Epoch {epoch} Batch Noise Loss: {loss.item()} Epoch Loss: {epoch_loss}")
            # lr_scheduler.step()
            # Evaluate model on small sample from test set
            if test_loader:
                logging.info(f"Epoch {epoch}: Evaluating model on test set...")
                model.eval()
                evaluate_model(model, test_loader)
                model.train()
            else:
                # Else just observe results
                evaluate_special(model, x_data)
            model.train()

    # Evaluate the model
    evaluate_special(model, x_data)
    logging.info(f"Training complete! {epochs} epochs completed.")
    # Save the model
    if save_model:
        logging.info("Saving the final model...")
        torch.save(model.state_dict(), f"../results/{folder}/weights.pth")


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


def evaluate_special(model, x_real, save_fig=False, name=""):
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
        if save_fig:
            plt.savefig(f"../results/{folder}/{name}")
            plt.close()
        else:
            plt.show()
            plt.close()


def evaluate_model_basic(model, num_samples=3):
    model.eval()
    with torch.no_grad():
        # Perform sampling for 3 images
        generated_images = model.sample(num_samples)
        # Concatenate and display the image with pyplot
        # N, C, H, W -> N, H, W, C
        generated_images = generated_images.permute(0, 2, 3, 1).cpu().numpy()
        generated_images = np.concatenate(generated_images, axis=1)
        plt.figure(layout="tight")
        plt.imshow(generated_images)
        plt.axis("off")
        plt.show()


def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        total_test_loss = 0
        for x_data, y_gt in test_loader:
            x_data = x_data.cuda()
            y_gt = y_gt.cuda()
            pred_gt, _ = model(x_data, y_gt)
            loss = torch.nn.functional.mse_loss(pred_gt, y_gt)
            total_test_loss += loss.item()
        logging.info(f"Test Loss: {total_test_loss}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = "../data"
    dataset = "all_diff_views"
    res = (256, 256)
    img_channels = 3

    folder = "../results/train_all_novel_lite"

    logging.basicConfig(level=logging.INFO,
                        filename=f"../results/{folder}/output.log",
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        filemode="a+")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    # Diffusion Model Parameters
    n_timesteps = 70
    train_batch_size = 8
    test_batch_size = 8
    lr = 5e-6
    epochs = 100

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = RenderingsDataset(res=res, scene_name=dataset)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2, pin_memory=True)

    test_dataset = RenderingsDataset(res=res, scene_name="test")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size,  shuffle=False, num_workers=0, pin_memory=False)

    # Let's continue training the model
    #model = DiffusionModel(res, img_channels, num_timesteps=n_timesteps, scheduler="linear")
    model = DiffusionModelForDenoising(res, img_channels, num_timesteps=n_timesteps, scheduler="linear", in_dim=32, dim_mults=(1, 2, 4, 8), mode="lite")
    #model.load_state_dict(torch.load("../results/train_all_novel_lite_final/weights.pth"))
    model.to(device)

    train(model, train_loader, epochs, lr, device, scene_name=dataset, test_loader=test_loader)
    #evaluate_model(model, test_loader)
