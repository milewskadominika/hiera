import torch
from torch.optim import AdamW
import time
import datetime
from hiera.hiera import hiera_tiny_224

import argparse
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_args():
    parser = argparse.ArgumentParser(description='Finetuning NextHiera on ImageNet')
    parser.add_argument('--data-path', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--mae-ckpt-path', type=str, required=True, help='Path to MAE checkpoint')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default='./finetune_checkpoints')
    parser.add_argument('--logs', type=str, default='finetune_log.txt')
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

    # Dataset i DataLoader
    dataset = datasets.ImageFolder(args.data_dir,
                                   transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

    model = hiera_tiny_224(pretrained=False)
    model.to(args.device)
    model.load_state_dict(torch.load(args.mae_ckpt_path), strict=False)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    criterion = nn.CrossEntropyLoss()

    previous_loss = 1000

    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(args.device)
            targets = targets.to(args.device)

            optimizer.zero_grad()
            output = model(images)

            loss = criterion(output, targets)
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if loss.item() <= previous_loss:
                for file in os.listdir(args.output_dir):
                    os.remove(os.path.join(args.output_dir, file))
                checkpoint_path = os.path.join(args.output_dir, f"hiera_epoch{epoch + 1}_{i}.pth")
                torch.save(model.state_dict(), checkpoint_path)

            previous_loss = loss.item()
            if i % 10 == 0:
                with open(args.logs, "a") as file:
                    file.write(
                        f"Epoch [{epoch + 1}/{args.epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Average Loss: {avg_loss:.4f}")


if __name__ == '__main__':
    main()
