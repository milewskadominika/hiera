import argparse
import time
import torch
from torchvision import datasets, transforms
from hiera import hiera_tiny_224

import nextvit_utils as utils
from timm.utils import accuracy

from torch.utils.data import DataLoader


def get_args():
    parser = argparse.ArgumentParser(description='Pre-training MAE with NextHiera on ImageNet')
    parser.add_argument('--data-path', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--ckpt-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output-dir', type=str, default='./mae_checkpoints')
    parser.add_argument('--logs', type=str, default='mae_log.txt')
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    total_latency = 0.0
    total_images = 0

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        start_time = time.time()

        with torch.cuda.amp.autocast():
            output = model(images)

        torch.cuda.synchronize()
        end_time = time.time()

        latency_ms = (end_time - start_time) * 1000
        total_latency += latency_ms
        total_images += images.size(0)

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()

    avg_latency_per_image = total_latency / total_images if total_images > 0 else 0.0
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} '
          'loss {losses.global_avg:.3f} avg latency/image {latency:.2f} ms'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss,
                  latency=avg_latency_per_image))

    results = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    results['avg_latency_per_image_ms'] = avg_latency_per_image

    return results


def main():
    args = get_args()

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    subset_dataset = utils.SubclassDataset(dataset)
    dataloader = DataLoader(subset_dataset, batch_size=8, shuffle=True)

    model = hiera_tiny_224(pretrained=False)
    model.to(args.device)
    model.load_state_dict(torch.load(args.ckpt_path), strict=False)

    print(evaluate(dataloader, model, args.device))


if __name__ == '__main__':
    main()
