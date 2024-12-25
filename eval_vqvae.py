import torch
import os
import json
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse
from vqvae import VQVAE  # Assuming your VQVAE model is in a file named 'vqvae.py'

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
        image = image.resize((256, 256))

        text = self.data[idx]['title']

        if self.transform:
            image = self.transform(image)

        return image, text

def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def evaluate(model, loader, device):
    model.eval()
    mse_loss = nn.MSELoss()
    psnr_scores = []
    losses = []
    
    with torch.no_grad():
        for img, texts in loader:
            img = img.to(device)
            out = model(img, texts)

            # Compute MSE Loss
            loss = mse_loss(out, img)
            losses.append(loss)

            # Compute PSNR
            for i in range(img.size(0)):
                img_i = img[i]
                out_i = out[i]
                psnr_score = psnr(img_i, out_i)
                psnr_scores.append(psnr_score.item())

    avg_mse_loss = sum(loss.item() for loss in losses) / len(losses)
    avg_psnr_score = sum(psnr_scores) / len(psnr_scores)

    return avg_mse_loss, avg_psnr_score

def main(args):
    transform = transforms.Compose([
        transforms.Resize(args.size),
        transforms.CenterCrop(args.size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    dataset = CustomDataset(args.data_path, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE().to(device)

    # Load model weights
    model.load_state_dict(torch.load(args.model_path))

    # Evaluate the model
    mse_loss, psnr_score = evaluate(model, loader, device)
    print(f"Avg. MSE Loss: {mse_loss:.4f}, Avg. PSNR Score: {psnr_score:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str, help="Path to the trained model checkpoint")
    parser.add_argument("data_path", type=str, help="Path to the evaluation data")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--size", type=int, default=256)

    args = parser.parse_args()
    main(args)
