## This code is modified from https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment2-DDIM-CFG/blob/main/image_diffusion_todo/train.py

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import torch
from dataset import QuickDrawDataModule, get_data_iterator, tensor_to_pil_image
from dotmap import DotMap
from model import DiffusionModule
from network import UNet
from pytorch_lightning import seed_everything
from scheduler import DDPMScheduler
from tqdm import tqdm

matplotlib.use("Agg")

def get_current_time():
    now = datetime.now().strftime("%m-%d-%H%M%S")
    return now


def main(args):
    """config"""
    config = DotMap()
    config.update(vars(args))
    config.device = f"cuda:{args.gpu}"

    if config.save_dir_name == "None":
        now = get_current_time()
    else:
        now = config.save_dir_name

    save_dir = Path(f"./stroke_generation/results/diffusion-{args.sample_method}-{now}")
    save_dir.mkdir(exist_ok=True, parents=True)
    print(f"save_dir: {save_dir}")

    seed_everything(config.seed)

    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    image_resolution = config.image_resolution
    ds_module = QuickDrawDataModule(
        config.root_dir,
        config.indices_root_dir,
        batch_size=config.batch_size,
        num_workers=4,
        image_resolution=image_resolution,
        category=config.category,
        coord_length = config.coordinate_length
    )

    train_dl = ds_module.train_dataloader()
    train_it = get_data_iterator(train_dl)

    # Set up the scheduler
    var_scheduler = DDPMScheduler(
        config.num_diffusion_train_timesteps,
        beta_1=config.beta_1,
        beta_T=config.beta_T,
        mode="linear",
    )

    network = UNet(
        T=config.num_diffusion_train_timesteps,
        image_resolution=image_resolution,
        ch=128,
        ch_mult=[1, 2, 2, 2],
        attn=[1],
        num_res_blocks=4,
        dropout=0.1,
    )

    ddpm = DiffusionModule(network, var_scheduler, coord_length=config.coordinate_length)
    ddpm = ddpm.to(config.device)

    optimizer = torch.optim.Adam(ddpm.network.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda t: min((t + 1) / config.warmup_steps, 1.0)
    )

    step = 0
    losses = []
    with tqdm(initial=step, total=config.train_num_steps) as pbar:
        while step < config.train_num_steps:
            if step % config.log_interval == 0:
                ddpm.eval()
                plt.plot(losses)
                plt.savefig(f"{save_dir}/loss.png")
                plt.close()

                # Unconditional training
                samples = ddpm.sample(4, return_traj=False)

                try:
                    pil_images = tensor_to_pil_image(samples)
                    for i, img in enumerate(pil_images):
                        img.save(save_dir / f"step={step}-{i}.png")
                except:
                    print(f"Image Sampling Error at {step}")

                ddpm.save(f"{save_dir}/last.ckpt")
                ddpm.train()

            img, label = next(train_it)
            img, label = img.to(config.device), label.to(config.device)

            # Unconditional training
            loss = ddpm.get_loss(img)
            pbar.set_description(f"Loss: {loss.item():.4f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

            step += 1
            pbar.update(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--root_dir", type=str, default="./data")
    parser.add_argument("--indices_root_dir", type=str, default="./sketch_data")
    parser.add_argument("--category", type=str, default="cat")
    parser.add_argument(
        "--train_num_steps",
        type=int,
        default=100000,
        help="the number of model training steps.",
    )
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--log_interval", type=int, default=200)

    parser.add_argument(
        "--num_diffusion_train_timesteps",
        type=int,
        default=1000,
        help="diffusion Markov chain num steps",
    )
    parser.add_argument("--beta_1", type=float, default=0.0001)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=63)
    parser.add_argument("--image_resolution", type=int, default=64)
    parser.add_argument("--sample_method", type=str, default="ddpm")
    parser.add_argument("--save_dir_name", type=str, default="None")
    parser.add_argument("--coordinate_length", type=int, default=150)
    args = parser.parse_args()
    main(args)
