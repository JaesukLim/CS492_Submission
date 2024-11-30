## This code is modified from https://github.com/KAIST-Visual-AI-Group/Diffusion-Assignment2-DDIM-CFG/blob/main/image_diffusion_todo/model.py

from typing import Optional
import torch
import torch.nn as nn
from tqdm import tqdm


class DiffusionModule(nn.Module):
    def __init__(self, network, var_scheduler, coord_length, **kwargs):
        super().__init__()
        self.network = network
        self.var_scheduler = var_scheduler
        self.coord_length = coord_length

    def get_loss(self, x0, class_label=None, noise=None):

        B = x0.shape[0]
        timestep = self.var_scheduler.uniform_sample_t(B, self.device)

        alpha_prod_t = self.var_scheduler.alphas_cumprod[timestep].view(B, 1, 1, 1).expand(x0.shape)
        eps = torch.randn(x0.shape).to(x0.device)

        xt = alpha_prod_t.sqrt() * x0 + (1 - alpha_prod_t).sqrt() * eps
        eps_theta = self.network(xt, timestep, class_label)
        x0 = (eps - eps_theta) ** 2

        loss = x0.mean()

        return loss
    
    @property
    def device(self):
        return next(self.network.parameters()).device

    @property
    def image_resolution(self):
        return self.network.image_resolution

    @torch.no_grad()
    def sample(
        self,
        batch_size,
        return_traj=False,
        class_label: Optional[torch.Tensor] = None,
    ):
        x_T = torch.randn([batch_size, 3, 1, self.coord_length]).to(self.device)

        traj = [x_T]
        for t in tqdm(self.var_scheduler.timesteps):
            x_t = traj[-1]
            noise_pred = self.network(
                x_t,
                timestep=t.to(self.device),
                class_label=class_label,
            )

            x_t_prev = self.var_scheduler.step(x_t, t, noise_pred)

            traj[-1] = traj[-1].cpu()
            traj.append(x_t_prev.detach())

        if return_traj:
            return traj
        else:
            return traj[-1]

    def save(self, file_path):
        hparams = {
            "network": self.network,
            "var_scheduler": self.var_scheduler,
            } 
        state_dict = self.state_dict()

        dic = {"hparams": hparams, "state_dict": state_dict}
        torch.save(dic, file_path)

    def load(self, file_path):
        dic = torch.load(file_path, map_location="cpu")
        hparams = dic["hparams"]
        state_dict = dic["state_dict"]

        self.network = hparams["network"]
        self.var_scheduler = hparams["var_scheduler"]

        self.load_state_dict(state_dict)
