import time
import datetime

import torch
from torch.optim import AdamW
from hiera_autoencoder import NextMaskedAutoencoderHiera

import argparse
import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_args():
    parser = argparse.ArgumentParser(description='Pre-training MAE with Hiera on ImageNet')
    parser.add_argument('--data-path', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default='./checkpoints')
    parser.add_argument('--logs', type=str, default='log.txt')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(args.data_dir,
                                   transform=transform)

    # Matches Hiera-Tiny
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    model = NextMaskedAutoencoderHiera(embed_dim=96, num_heads=1, stages=(1, 2, 7, 2), q_pool=2)
    model.to(args.device)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    previous_loss = 1000

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(args.device)
            optimizer.zero_grad()

            loss, preds, labels, mask = model(images)

            loss = loss.mean()
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            if loss.item() <= previous_loss:
                for file in os.listdir(args.output_dir):
                    os.remove(os.path.join(args.output_dir, file))
                checkpoint_path = os.path.join(args.output_dir, f"hiera_epoch{epoch + 1}.pth")
                torch.save(model.state_dict(), checkpoint_path)

            previous_loss = loss.item()
            if i % 10 == 0:
                with open(args.logs, "a") as file:
                    file.write(f"Epoch [{epoch + 1}/{args.epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Average Loss: {avg_loss:.4f}")


if __name__ == '__main__':
    main()
