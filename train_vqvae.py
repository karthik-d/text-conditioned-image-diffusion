import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import json
#from scheduler import CycleScheduler
from torchvision import transforms, utils
from PIL import Image

from tqdm import tqdm

from vqvae import VQVAE

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.data = self.load_metadata()

    def load_metadata(self):
        metadata_file = os.path.join(self.root_dir, 'metadata.jsonl')
        with open(metadata_file, 'r') as f:
            metadata = [json.loads(line) for line in f]
        return metadata

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data[idx]['file_name'])
        image = Image.open(img_name).convert("RGB")
        image=image.resize((256,256))

        text = self.data[idx]['title']

        if self.transform:
            image = self.transform(image)

        return image, text

def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0
    comm1 = []
    print(comm1)
    for i, (img, text_tuple) in enumerate(loader):
        img = img.to(device)
        text_tensors = [torch.tensor([ord(c) for c in text], dtype=torch.long) for text in text_tuple]
        text_tensors_on_device = [tensor.to(device) for tensor in text_tensors]

        model.zero_grad()

        img = img.to(device)

        out = model(img, text_tensors_on_device)
        recon_loss = criterion(out, img)
        loss = recon_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        print(comm)
        comm1.append(comm)
        print(comm1)
        #comm = dist.all_gather(comm)

        for part in comm1:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        lr = optimizer.param_groups[0]["lr"]

        loader.set_description(
            (
                f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                f"avg mse: {mse_sum / mse_n:.5f}; "
                f"lr: {lr:.5f}"
            )
        )
        """
        if i % 100 == 0:
            model.eval()

            sample = img[:sample_size]

            with torch.no_grad():
                out = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),
                f"sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png",
                nrow=sample_size,
                normalize=True,
                range=(-1, 1),
            )"""
        model.train()
            
def main(args):
    device = "cuda"

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = CustomDataset(args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = VQVAE().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

    torch.save(model.state_dict(), f"checkpoint/vqvae_{str(i + 1).zfill(3)}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=560)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    main(args)

