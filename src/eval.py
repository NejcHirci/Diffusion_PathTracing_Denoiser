import logging

import numpy as np
import torch
from matplotlib import pyplot as plt

from model_alternative import DiffusionModelForDenoising
from train import RenderingsDataset
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import sys

if __name__ == "__main__":
    dataset = RenderingsDataset(scene_name="test")
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, pin_memory=True, num_workers=2)

    def evaluate(model:DiffusionModelForDenoising, data_loader, post_name=""):
        model.eval()

        mse_loss = torch.nn.MSELoss()

        total_mse_loss = 0  # LOWER IS BETTER
        total_ssim_loss = 0  # HIGHER IS BETTER
        total_psnr_loss = 0  # HIGHER IS BETTER

        best_ssim = -100
        best_case = []
        worst_ssim = 100
        worst_case = []

        with torch.no_grad():
            with tqdm(data_loader, unit="batch", file=sys.stdout) as test_epoch:
                for x_data, y_gt in test_epoch:
                    x_data = x_data.cuda()
                    pred_y = model.sample(x_data, denorm=False)
                    pred_y = pred_y.cpu().detach()
                    cur_mse_loss = mse_loss(y_gt, pred_y).item()
                    total_mse_loss += cur_mse_loss
                    y_gt = y_gt.permute(0, 2, 3, 1).numpy()
                    pred_y = pred_y.permute(0, 2, 3, 1).numpy()
                    total_psnr_loss += psnr(y_gt, pred_y)
                    pred_y = (pred_y + 1) / 2
                    y_gt = (y_gt + 1) / 2

                    all_ssim = 0
                    for b in range(x_data.shape[0]):
                        all_ssim += ssim(y_gt[b], pred_y[b], data_range=1, channel_axis=-1)
                    total_ssim_loss += all_ssim

                    if all_ssim > best_ssim:
                        x_in = (x_data.cpu().detach().permute(0, 2, 3, 1).numpy() + 1) / 2
                        best_ssim = all_ssim
                        best_case = [x_in, y_gt, pred_y]
                    if all_ssim < worst_ssim:
                        x_in = (x_data.cpu().detach().permute(0, 2, 3, 1).numpy() + 1) / 2
                        worst_ssim = all_ssim
                        worst_case = [x_in, y_gt, pred_y]

        # Concat input, prediction and ground truth
        worst_in = np.concatenate(worst_case[0], axis=1)
        worst_gt = np.concatenate(worst_case[1], axis=1)
        worst_pred = np.concatenate(worst_case[2], axis=1)
        worst_all = np.concatenate([worst_in, worst_gt, worst_pred], axis=0)
        plt.figure(figsize=(worst_all.shape[1] / 300, worst_all.shape[0] / 300), dpi=300)
        plt.title(f"Worst Case SSIM:{worst_ssim:.4f}")
        plt.imshow(worst_all)
        plt.savefig(f"{base_path}/{post_name}_worst_case.png", dpi=300)
        plt.close()

        # Concat input, prediction and ground truth
        best_in = np.concatenate(best_case[0], axis=1)
        best_gt = np.concatenate(best_case[1], axis=1)
        best_pred = np.concatenate(best_case[2], axis=1)
        best_all = np.concatenate([best_in, best_gt, best_pred], axis=0)
        plt.figure(figsize=(best_all.shape[1] / 300, best_all.shape[0] / 300), dpi=300)
        plt.title(f"Best Case SSIM:{best_ssim:.4f}")
        plt.imshow(best_all)
        plt.savefig(f"{base_path}/{post_name}_best_case.png", dpi=300)
        plt.close()

        logging.info(f"Total MSE Loss: {total_mse_loss / len(data_loader)}")
        logging.info(f"Total SSIM HIGHER BETTER: {total_ssim_loss / len(data_loader)}")
        logging.info(f"Total PSNR HIGHER BETTER: {total_psnr_loss / len(data_loader)}")


    def evaulate_different_spp(model:DiffusionModelForDenoising):
        for spp in [1, 2, 4, 8, 16, 32, 64, 128]:
            spp_dataset = RenderingsDataset(scene_name="test", spp=spp)
            data_loader = torch.utils.data.DataLoader(spp_dataset, batch_size=4, shuffle=False, pin_memory=True, num_workers=2)
            logging.info(f"Evaluating {spp} spp")
            evaluate(model, data_loader, post_name=f"{spp}")


    base_path = "../results/train_all_novel_final"

    logging.basicConfig(level=logging.INFO,
                        filename=f"{base_path}/eval_spp.log",
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        filemode="a+")
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger("").addHandler(console)

    result_path = f"{base_path}/weights.pth"
    model = DiffusionModelForDenoising((256, 256), 3, num_timesteps=70, in_dim=32, mode="full", dim_mults=(1, 2, 2, 4))
    model.load_state_dict(torch.load(result_path))
    logging.info(f"Model Loaded from {result_path}")
    model.to("cuda")
    evaluate(model, data_loader)
