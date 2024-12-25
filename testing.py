import argparse
import os
import torch
from tqdm import tqdm
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import json
from PIL import Image

from vqvae import VQVAE
from pixelsnail import PixelSNAIL
import shutil
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

        text = self.data[idx]['title']

        if self.transform:
            image = self.transform(image)

        return image, text

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


def load_model(model, checkpoint, device):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()
    elif model == 'pixelsnail_top':
        model = PixelSNAIL(
            [32, 32],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )
    elif model == 'pixelsnail_bottom':
        model = PixelSNAIL(
            [64, 64],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )
       
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--output_folder', type=str, default='output')
    args = parser.parse_args()
    
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    # Load your testing dataset
    testing_dataset = CustomDataset("testimages", transform=transform)
   
    model_vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)

    output_folder = args.output_folder
    os.makedirs(output_folder, exist_ok=True)

    for idx, data in enumerate(testing_dataset):
        top_sample = sample_model(model_top, device, args.batch, [32, 32], args.temp)
        bottom_sample = sample_model(
            model_bottom, device, args.batch, [64, 64], args.temp, condition=top_sample
        )

        decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
        decoded_sample = decoded_sample.clamp(-1, 1)
        #print(decoded_sample.shape)
        filename = f"{idx}.jpg"
        save_image(decoded_sample, os.path.join(output_folder, filename), normalize=True, value_range=(-1, 1))