import torch
from vq_vae_2_pytorch.vqvae import VQVAE

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import distributed as dist

from torchvision import datasets, transforms, utils

@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row

ckpt = torch.load(r"E:\Text2CarsGeneration\Text2CarsGeneration\ImageGeneration\pretrained_autoencoder\vq_vae_2_pytorch\vqvae_560.pt")

if 'args' in ckpt:
    args = ckpt['args']

model = VQVAE()

if 'model' in ckpt:
    ckpt = ckpt['model']

model.load_state_dict(ckpt)
model.to("cuda")
print(model.eval())

save_image(decoded_sample, args.filename, normalize=True, range=(-1, 1))